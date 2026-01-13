"""
Route Knowledge API v2

Handles generation and storage of pre-generated narrative texts for audio guides.
Storage: TrackRoute.metadata_json["knowledge"]

Endpoints:
- GET  /tracks/{track_id}/routes/{route_id}/knowledge - Get existing knowledge
- POST /tracks/{track_id}/routes/{route_id}/knowledge/generate - Generate all narratives
- PUT  /tracks/{track_id}/routes/{route_id}/knowledge - Save edited knowledge
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel
import httpx
import json
import logging

from ..database import get_db
from ..models import Track, Waypoint, TrackRoute
from ..auth import get_current_user, User

router = APIRouter()
logger = logging.getLogger(__name__)

# AI API endpoint
INTERNAL_API_KEY = "Inetpass1"
AI_API_BASE = "https://api-ai.arkturian.com"


# ============ Pydantic Models ============

class NarrativeText(BaseModel):
    text: str = ""
    text_original: Optional[str] = None
    edited: bool = False
    audio_storage_id: Optional[int] = None


class KnowledgeConfig(BaseModel):
    persona: str = ""
    target_audience: str = ""
    language: str = "de"
    tone: str = "friendly"
    background_knowledge: str = ""  # Allgemeine Infos zum Ort/Route


class RouteKnowledge(BaseModel):
    version: int = 2
    generated_at: Optional[str] = None
    config: KnowledgeConfig
    route: Dict[str, NarrativeText]  # intro, outro
    segments: Dict[str, Dict[str, Any]]  # seg_id -> {name, entry, exit}
    pois: Dict[str, Dict[str, Any]]  # poi_id -> {name, approaching, at_poi}


class GenerateRequest(BaseModel):
    persona: str = ""
    target_audience: str = ""
    language: str = "de"
    tone: str = "friendly"
    background_knowledge: str = ""  # Allgemeine Infos zum Ort (z.B. Geschichte, Fakten)
    generate_segments: bool = True
    generate_pois: bool = True


# ============ Helper Functions ============

def _load_route_data(db: Session, track_id: int, route_id: int) -> tuple:
    """Load route, segments, and POIs for knowledge generation."""

    # Load all waypoints for this track
    all_waypoints = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.waypoint_type != "gps_track"
    ).all()

    # Filter segment markers (waypoints with metadata_json.segment)
    segment_waypoints = [
        wp for wp in all_waypoints
        if wp.metadata_json and wp.metadata_json.get("segment")
    ]

    # Filter for this route
    route_segment_wps = [
        wp for wp in segment_waypoints
        if wp.metadata_json.get("snap", {}).get("route_id") == route_id
    ]

    # Group by segment name
    segments = {}
    for wp in route_segment_wps:
        segment_meta = wp.metadata_json.get("segment", {})
        seg_name = segment_meta.get("name")
        role = segment_meta.get("role")

        if not seg_name or not role:
            continue

        if seg_name not in segments:
            segments[seg_name] = {"name": seg_name, "start_wp": None, "end_wp": None}

        if role == "start":
            segments[seg_name]["start_wp"] = wp
        elif role == "end":
            segments[seg_name]["end_wp"] = wp

    # Filter POIs (non-segment waypoints)
    pois = [
        wp for wp in all_waypoints
        if not wp.metadata_json or not wp.metadata_json.get("segment")
    ]

    # Filter POIs for this route (snapped to this route)
    route_pois = [
        wp for wp in pois
        if wp.metadata_json and wp.metadata_json.get("snap", {}).get("route_id") == route_id
    ]

    return segments, route_pois


async def _generate_narrative_text(
    narrative_type: str,
    context: Dict[str, Any],
    config: KnowledgeConfig
) -> str:
    """
    Generate narrative text using AI.

    Types:
    - route_intro: Welcome message when starting the route
    - route_outro: Farewell message when completing the route
    - segment_entry: Message when entering a segment
    - segment_exit: Message when leaving a segment
    - poi_approaching: Message when approaching a POI
    - poi_at: Message when at a POI
    """

    # Build common context block
    background = ""
    if config.background_knowledge:
        background = f"""
HINTERGRUNDWISSEN ZUM ORT:
{config.background_knowledge}

Nutze dieses Wissen um die Texte informativer und interessanter zu gestalten.
"""

    # Build the prompt based on type
    prompts = {
        "route_intro": f"""
Schreibe eine Willkommensnachricht für den Start einer Wanderroute.

Route: {context.get('route_name', 'Wanderroute')}
Länge: {context.get('route_length_km', 0):.1f} km
Beschreibung: {context.get('route_description', '')}
{background}
Persona: {config.persona or 'Du bist ein freundlicher Audio-Guide.'}
Zielgruppe: {config.target_audience or 'Wanderer'}
Ton: {config.tone}
Sprache: {config.language}

Schreibe einen kurzen, einladenden Text (2-4 Sätze) der die Wanderer willkommen heißt und neugierig auf die Route macht.
Antworte NUR mit dem Text, keine Erklärungen.
""",
        "route_outro": f"""
Schreibe eine Abschlussnachricht für das Ende einer Wanderroute.

Route: {context.get('route_name', 'Wanderroute')}
Länge: {context.get('route_length_km', 0):.1f} km
{background}
Persona: {config.persona or 'Du bist ein freundlicher Audio-Guide.'}
Zielgruppe: {config.target_audience or 'Wanderer'}
Ton: {config.tone}
Sprache: {config.language}

Schreibe einen kurzen Abschlusstext (2-3 Sätze) der die Wanderer verabschiedet und ihnen für die Wanderung dankt.
Antworte NUR mit dem Text, keine Erklärungen.
""",
        "segment_entry": f"""
Schreibe einen kurzen Begrüßungstext für einen Streckenabschnitt.

Abschnitt: {context.get('segment_name', 'Abschnitt')}
Beschreibung: {context.get('segment_description', '')}
{background}
Persona: {config.persona or 'Du bist ein freundlicher Audio-Guide.'}
Zielgruppe: {config.target_audience or 'Wanderer'}
Ton: {config.tone}
Sprache: {config.language}

Schreibe 1-2 Sätze die den Abschnitt kurz beschreiben. Fokussiere auf das Wesentliche.
Antworte NUR mit dem Text, keine Erklärungen.
""",
        "segment_exit": f"""
Schreibe einen kurzen Übergangstext für das Verlassen eines Streckenabschnitts.

Abschnitt: {context.get('segment_name', 'Abschnitt')}
{background}
Persona: {config.persona or 'Du bist ein freundlicher Audio-Guide.'}
Zielgruppe: {config.target_audience or 'Wanderer'}
Ton: {config.tone}
Sprache: {config.language}

Schreibe 1 Satz der den Übergang zum nächsten Abschnitt einleitet.
Antworte NUR mit dem Text, keine Erklärungen.
""",
        "poi_approaching": f"""
Schreibe eine kurze Ankündigung für einen Point of Interest.

POI: {context.get('poi_name', 'Sehenswürdigkeit')}
Beschreibung: {context.get('poi_description', '')}
Entfernung: ca. 50m
{background}
Persona: {config.persona or 'Du bist ein freundlicher Audio-Guide.'}
Zielgruppe: {config.target_audience or 'Wanderer'}
Ton: {config.tone}
Sprache: {config.language}

Schreibe 1 Satz der den POI ankündigt und neugierig macht.
Antworte NUR mit dem Text, keine Erklärungen.
""",
        "poi_at": f"""
Schreibe eine Beschreibung für einen Point of Interest.

POI: {context.get('poi_name', 'Sehenswürdigkeit')}
Beschreibung vom Autor: {context.get('poi_description', '')}
{background}
Persona: {config.persona or 'Du bist ein freundlicher Audio-Guide.'}
Zielgruppe: {config.target_audience or 'Wanderer'}
Ton: {config.tone}
Sprache: {config.language}

Schreibe 2-4 Sätze die den POI beschreiben. Nutze die Beschreibung als Grundlage, erweitere sie aber mit interessanten Details.
Antworte NUR mit dem Text, keine Erklärungen.
"""
    }

    prompt = prompts.get(narrative_type, "")
    if not prompt:
        return ""

    # Call AI API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AI_API_BASE}/ai/claude",
                json={
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.7
                },
                headers={
                    "X-API-KEY": INTERNAL_API_KEY,
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", "").strip()
            else:
                logger.error(f"AI API returned {response.status_code}: {response.text}")
                return ""

    except Exception as e:
        logger.error(f"AI generation failed: {e}")
        return ""


# ============ API Endpoints ============

@router.get("/{track_id}/routes/{route_id}/knowledge")
def get_route_knowledge(
    track_id: int,
    route_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get existing route knowledge.
    Returns the knowledge JSON from TrackRoute.metadata_json["knowledge"].
    """
    # Load track and route
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.visibility == "private" and track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    route = db.query(TrackRoute).filter(
        TrackRoute.id == route_id,
        TrackRoute.track_id == track_id
    ).first()

    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Get knowledge from metadata
    metadata = route.metadata_json or {}
    knowledge = metadata.get("knowledge")

    if not knowledge:
        return {
            "exists": False,
            "knowledge": None,
            "route_id": route_id,
            "route_name": route.name
        }

    return {
        "exists": True,
        "knowledge": knowledge,
        "route_id": route_id,
        "route_name": route.name
    }


@router.post("/{track_id}/routes/{route_id}/knowledge/generate")
async def generate_route_knowledge(
    track_id: int,
    route_id: int,
    body: GenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate all narrative texts for a route using AI.

    Generates:
    - Route intro/outro
    - Segment entry/exit texts (if generate_segments=True)
    - POI approaching/at texts (if generate_pois=True)

    Returns the complete knowledge JSON (not yet saved).
    """
    # Load track and route
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track creator can generate knowledge")

    route = db.query(TrackRoute).filter(
        TrackRoute.id == route_id,
        TrackRoute.track_id == track_id
    ).first()

    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Load route data
    segments, pois = _load_route_data(db, track_id, route_id)

    # Calculate route length
    gps_points = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.waypoint_type == "gps_track",
        Waypoint.route_id == route_id
    ).count()

    # Rough estimate: ~10m per GPS point average
    route_length_km = (gps_points * 10) / 1000

    # Build config
    config = KnowledgeConfig(
        persona=body.persona,
        target_audience=body.target_audience,
        language=body.language,
        tone=body.tone,
        background_knowledge=body.background_knowledge
    )

    # Initialize knowledge structure
    knowledge = {
        "version": 2,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": config.model_dump(),
        "route": {
            "intro": {"text": "", "text_original": "", "edited": False, "audio_storage_id": None},
            "outro": {"text": "", "text_original": "", "edited": False, "audio_storage_id": None}
        },
        "segments": {},
        "pois": {}
    }

    # Generate route intro/outro
    logger.info(f"Generating route intro for '{route.name}'...")
    intro_text = await _generate_narrative_text(
        "route_intro",
        {
            "route_name": route.name,
            "route_description": route.description or "",
            "route_length_km": route_length_km
        },
        config
    )
    knowledge["route"]["intro"]["text"] = intro_text
    knowledge["route"]["intro"]["text_original"] = intro_text

    logger.info(f"Generating route outro for '{route.name}'...")
    outro_text = await _generate_narrative_text(
        "route_outro",
        {
            "route_name": route.name,
            "route_length_km": route_length_km
        },
        config
    )
    knowledge["route"]["outro"]["text"] = outro_text
    knowledge["route"]["outro"]["text_original"] = outro_text

    # Generate segment texts
    if body.generate_segments:
        for seg_name, seg_data in segments.items():
            logger.info(f"Generating segment texts for '{seg_name}'...")

            # Get description from start waypoint
            start_wp = seg_data.get("start_wp")
            seg_description = start_wp.user_description if start_wp else ""

            entry_text = await _generate_narrative_text(
                "segment_entry",
                {"segment_name": seg_name, "segment_description": seg_description},
                config
            )

            exit_text = await _generate_narrative_text(
                "segment_exit",
                {"segment_name": seg_name},
                config
            )

            knowledge["segments"][seg_name] = {
                "name": seg_name,
                "entry": {"text": entry_text, "text_original": entry_text, "edited": False, "audio_storage_id": None},
                "exit": {"text": exit_text, "text_original": exit_text, "edited": False, "audio_storage_id": None}
            }

    # Generate POI texts
    if body.generate_pois:
        for poi in pois:
            poi_id = str(poi.id)
            poi_name = (poi.metadata_json or {}).get("title", f"POI #{poi.id}")
            poi_description = poi.user_description or ""

            logger.info(f"Generating POI texts for '{poi_name}'...")

            approaching_text = await _generate_narrative_text(
                "poi_approaching",
                {"poi_name": poi_name, "poi_description": poi_description},
                config
            )

            at_poi_text = await _generate_narrative_text(
                "poi_at",
                {"poi_name": poi_name, "poi_description": poi_description},
                config
            )

            knowledge["pois"][poi_id] = {
                "name": poi_name,
                "approaching": {"text": approaching_text, "text_original": approaching_text, "edited": False, "audio_storage_id": None},
                "at_poi": {"text": at_poi_text, "text_original": at_poi_text, "edited": False, "audio_storage_id": None}
            }

    return {
        "success": True,
        "knowledge": knowledge,
        "stats": {
            "segments_count": len(knowledge["segments"]),
            "pois_count": len(knowledge["pois"]),
            "total_texts": 2 + len(knowledge["segments"]) * 2 + len(knowledge["pois"]) * 2
        }
    }


@router.put("/{track_id}/routes/{route_id}/knowledge")
def save_route_knowledge(
    track_id: int,
    route_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Save route knowledge JSON.

    Body should contain the complete knowledge object.
    Stores in TrackRoute.metadata_json["knowledge"].
    """
    # Load track and route
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track creator can save knowledge")

    route = db.query(TrackRoute).filter(
        TrackRoute.id == route_id,
        TrackRoute.track_id == track_id
    ).first()

    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Get knowledge from body
    knowledge = body.get("knowledge")
    if not knowledge:
        raise HTTPException(status_code=400, detail="Knowledge object required")

    # Update metadata
    metadata = route.metadata_json or {}
    metadata["knowledge"] = knowledge

    # Save
    route.metadata_json = metadata
    flag_modified(route, "metadata_json")
    db.commit()
    db.refresh(route)

    return {
        "success": True,
        "route_id": route_id,
        "saved_at": datetime.utcnow().isoformat() + "Z"
    }


@router.delete("/{track_id}/routes/{route_id}/knowledge")
def delete_route_knowledge(
    track_id: int,
    route_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete route knowledge.
    Removes knowledge from TrackRoute.metadata_json.
    """
    # Load track and route
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track creator can delete knowledge")

    route = db.query(TrackRoute).filter(
        TrackRoute.id == route_id,
        TrackRoute.track_id == track_id
    ).first()

    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Remove knowledge from metadata
    metadata = route.metadata_json or {}
    if "knowledge" in metadata:
        del metadata["knowledge"]
        route.metadata_json = metadata
        flag_modified(route, "metadata_json")
        db.commit()

    return {"success": True, "deleted": True}
