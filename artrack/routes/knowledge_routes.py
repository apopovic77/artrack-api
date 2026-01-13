"""
Route Knowledge API v2

Handles generation and storage of pre-generated narrative texts for audio guides.

Storage:
- Route texts (intro/outro) → TrackRoute.metadata_json["knowledge"]
- POI texts (approaching/at_poi) → Waypoint.metadata_json["knowledge"]
- Segment texts (entry/exit) → Segment marker Waypoint.metadata_json["knowledge"]

Endpoints:
- GET  /tracks/{track_id}/routes/{route_id}/knowledge - Get all knowledge for a route
- POST /tracks/{track_id}/routes/{route_id}/knowledge/generate - Generate all narratives
- PUT  /tracks/{track_id}/routes/{route_id}/knowledge - Save all knowledge
- DELETE /tracks/{track_id}/routes/{route_id}/knowledge - Delete all knowledge
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


class GenerateRequest(BaseModel):
    persona: str = ""
    target_audience: str = ""
    language: str = "de"
    tone: str = "friendly"
    background_knowledge: str = ""
    generate_segments: bool = True
    generate_pois: bool = True


# ============ Helper Functions ============

def _load_route_data(db: Session, track_id: int, route_id: int) -> tuple:
    """Load route, segments, and POIs for knowledge generation."""

    # Load all waypoints for this track (except GPS tracks)
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

    # Group by segment name - return waypoint objects for saving
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

    # Filter POIs (non-segment waypoints snapped to this route)
    pois = [
        wp for wp in all_waypoints
        if (not wp.metadata_json or not wp.metadata_json.get("segment"))
        and wp.metadata_json and wp.metadata_json.get("snap", {}).get("route_id") == route_id
    ]

    return segments, pois


def _get_waypoint_knowledge(wp: Waypoint) -> Optional[Dict]:
    """Get knowledge from a waypoint's metadata."""
    if not wp or not wp.metadata_json:
        return None
    return wp.metadata_json.get("knowledge")


def _save_waypoint_knowledge(wp: Waypoint, knowledge: Dict, db: Session):
    """Save knowledge to a waypoint's metadata."""
    if not wp:
        return

    metadata = wp.metadata_json or {}
    metadata["knowledge"] = knowledge
    wp.metadata_json = metadata
    flag_modified(wp, "metadata_json")


async def _generate_narrative_text(
    narrative_type: str,
    context: Dict[str, Any],
    config: KnowledgeConfig
) -> str:
    """Generate narrative text using AI."""

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
    Get all knowledge for a route.

    Aggregates:
    - Route knowledge from TrackRoute.metadata_json["knowledge"]
    - POI knowledge from each Waypoint.metadata_json["knowledge"]
    - Segment knowledge from segment marker Waypoints
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

    # Load segments and POIs
    segments, pois = _load_route_data(db, track_id, route_id)

    # Get route-level knowledge
    route_metadata = route.metadata_json or {}
    route_knowledge = route_metadata.get("knowledge", {})

    # Build aggregated response
    knowledge = {
        "version": 2,
        "config": route_knowledge.get("config", {}),
        "generated_at": route_knowledge.get("generated_at"),
        "route": route_knowledge.get("route", {
            "intro": {"text": "", "edited": False},
            "outro": {"text": "", "edited": False}
        }),
        "segments": {},
        "pois": {}
    }

    # Load segment knowledge from waypoints
    for seg_name, seg_data in segments.items():
        start_wp = seg_data.get("start_wp")
        end_wp = seg_data.get("end_wp")

        # Entry text from start waypoint
        start_knowledge = _get_waypoint_knowledge(start_wp) or {}
        # Exit text from end waypoint
        end_knowledge = _get_waypoint_knowledge(end_wp) or {}

        knowledge["segments"][seg_name] = {
            "name": seg_name,
            "start_waypoint_id": start_wp.id if start_wp else None,
            "end_waypoint_id": end_wp.id if end_wp else None,
            "entry": start_knowledge.get("entry", {"text": "", "edited": False}),
            "exit": end_knowledge.get("exit", {"text": "", "edited": False})
        }

    # Load POI knowledge from waypoints
    for poi in pois:
        poi_knowledge = _get_waypoint_knowledge(poi) or {}
        poi_name = (poi.metadata_json or {}).get("title", f"POI #{poi.id}")

        knowledge["pois"][str(poi.id)] = {
            "waypoint_id": poi.id,
            "name": poi_name,
            "approaching": poi_knowledge.get("approaching", {"text": "", "edited": False}),
            "at_poi": poi_knowledge.get("at_poi", {"text": "", "edited": False})
        }

    # Check if any knowledge exists
    has_route_texts = bool(knowledge["route"].get("intro", {}).get("text") or
                          knowledge["route"].get("outro", {}).get("text"))
    has_segment_texts = any(
        seg.get("entry", {}).get("text") or seg.get("exit", {}).get("text")
        for seg in knowledge["segments"].values()
    )
    has_poi_texts = any(
        poi.get("approaching", {}).get("text") or poi.get("at_poi", {}).get("text")
        for poi in knowledge["pois"].values()
    )

    exists = has_route_texts or has_segment_texts or has_poi_texts

    return {
        "exists": exists,
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

    Returns the generated knowledge (not yet saved).
    Call PUT to save the knowledge.
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
        {"route_name": route.name, "route_length_km": route_length_km},
        config
    )
    knowledge["route"]["outro"]["text"] = outro_text
    knowledge["route"]["outro"]["text_original"] = outro_text

    # Generate segment texts
    if body.generate_segments:
        for seg_name, seg_data in segments.items():
            start_wp = seg_data.get("start_wp")
            end_wp = seg_data.get("end_wp")
            seg_description = start_wp.user_description if start_wp else ""

            logger.info(f"Generating segment texts for '{seg_name}'...")

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
                "start_waypoint_id": start_wp.id if start_wp else None,
                "end_waypoint_id": end_wp.id if end_wp else None,
                "entry": {"text": entry_text, "text_original": entry_text, "edited": False, "audio_storage_id": None},
                "exit": {"text": exit_text, "text_original": exit_text, "edited": False, "audio_storage_id": None}
            }

    # Generate POI texts
    if body.generate_pois:
        for poi in pois:
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

            knowledge["pois"][str(poi.id)] = {
                "waypoint_id": poi.id,
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
    Save all knowledge - distributes to respective objects.

    - Route texts → TrackRoute.metadata_json["knowledge"]
    - POI texts → Waypoint.metadata_json["knowledge"]
    - Segment texts → Segment marker Waypoint.metadata_json["knowledge"]
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

    knowledge = body.get("knowledge")
    if not knowledge:
        raise HTTPException(status_code=400, detail="Knowledge object required")

    # Save route-level knowledge (intro/outro + config)
    route_knowledge = {
        "version": knowledge.get("version", 2),
        "generated_at": knowledge.get("generated_at"),
        "config": knowledge.get("config", {}),
        "route": knowledge.get("route", {})
    }

    route_metadata = route.metadata_json or {}
    route_metadata["knowledge"] = route_knowledge
    route.metadata_json = route_metadata
    flag_modified(route, "metadata_json")

    # Save POI knowledge to individual waypoints
    pois_knowledge = knowledge.get("pois", {})
    for poi_id_str, poi_data in pois_knowledge.items():
        waypoint_id = poi_data.get("waypoint_id") or int(poi_id_str)
        waypoint = db.query(Waypoint).filter(Waypoint.id == waypoint_id).first()

        if waypoint:
            poi_knowledge = {
                "approaching": poi_data.get("approaching", {}),
                "at_poi": poi_data.get("at_poi", {})
            }
            _save_waypoint_knowledge(waypoint, poi_knowledge, db)

    # Save segment knowledge to marker waypoints
    segments_knowledge = knowledge.get("segments", {})
    for seg_name, seg_data in segments_knowledge.items():
        # Save entry to start waypoint
        start_wp_id = seg_data.get("start_waypoint_id")
        if start_wp_id:
            start_wp = db.query(Waypoint).filter(Waypoint.id == start_wp_id).first()
            if start_wp:
                _save_waypoint_knowledge(start_wp, {"entry": seg_data.get("entry", {})}, db)

        # Save exit to end waypoint
        end_wp_id = seg_data.get("end_waypoint_id")
        if end_wp_id:
            end_wp = db.query(Waypoint).filter(Waypoint.id == end_wp_id).first()
            if end_wp:
                _save_waypoint_knowledge(end_wp, {"exit": seg_data.get("exit", {})}, db)

    db.commit()

    return {
        "success": True,
        "route_id": route_id,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "saved_pois": len(pois_knowledge),
        "saved_segments": len(segments_knowledge)
    }


@router.delete("/{track_id}/routes/{route_id}/knowledge")
def delete_route_knowledge(
    track_id: int,
    route_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete all knowledge for a route.

    Removes knowledge from:
    - TrackRoute.metadata_json["knowledge"]
    - All POI waypoints
    - All segment marker waypoints
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

    # Load segments and POIs
    segments, pois = _load_route_data(db, track_id, route_id)

    # Remove route knowledge
    route_metadata = route.metadata_json or {}
    if "knowledge" in route_metadata:
        del route_metadata["knowledge"]
        route.metadata_json = route_metadata
        flag_modified(route, "metadata_json")

    # Remove POI knowledge
    for poi in pois:
        poi_metadata = poi.metadata_json or {}
        if "knowledge" in poi_metadata:
            del poi_metadata["knowledge"]
            poi.metadata_json = poi_metadata
            flag_modified(poi, "metadata_json")

    # Remove segment knowledge
    for seg_name, seg_data in segments.items():
        for wp in [seg_data.get("start_wp"), seg_data.get("end_wp")]:
            if wp:
                wp_metadata = wp.metadata_json or {}
                if "knowledge" in wp_metadata:
                    del wp_metadata["knowledge"]
                    wp.metadata_json = wp_metadata
                    flag_modified(wp, "metadata_json")

    db.commit()

    return {"success": True, "deleted": True}
