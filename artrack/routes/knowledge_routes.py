"""
Track Knowledge API v3

Handles generation and storage of pre-generated narrative texts for audio guides.

Storage:
- Route texts (intro/outro) → TrackRoute.metadata_json["knowledge"]
- POI texts (approaching/at_poi) → Waypoint.metadata_json["knowledge"]
- Segment texts (entry/exit) → Segment marker Waypoint.metadata_json["knowledge"]

Endpoints (Track-Level):
- GET  /tracks/{track_id}/knowledge - Get all knowledge for a track
- POST /tracks/{track_id}/knowledge/generate - Generate all narratives
- POST /tracks/{track_id}/knowledge/audio - Generate TTS audio for single item
- PUT  /tracks/{track_id}/knowledge - Save all knowledge
- DELETE /tracks/{track_id}/knowledge - Delete all knowledge
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
    background_knowledge: str = ""


class GenerateRequest(BaseModel):
    persona: str = ""
    target_audience: str = ""
    language: str = "de"
    tone: str = "friendly"
    background_knowledge: str = ""
    generate_routes: bool = True
    generate_segments: bool = True
    generate_pois: bool = True


class AudioGenerateRequest(BaseModel):
    """Request to generate TTS audio for a single knowledge item."""
    item_type: str  # "route", "segment", "poi"
    item_id: Optional[str] = None  # route_id, segment name, or waypoint_id
    text_type: str  # "intro", "outro", "entry", "exit", "approaching", "at_poi"
    voice: str = "nova"
    add_music: bool = False
    language: str = "de"


# ============ Helper Functions ============

def _load_track_data(db: Session, track_id: int) -> Dict:
    """Load all routes, segments, and POIs for a track."""

    # Load all routes
    routes = db.query(TrackRoute).filter(TrackRoute.track_id == track_id).all()

    # Load all waypoints (except GPS tracks)
    all_waypoints = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.waypoint_type != "gps_track"
    ).all()

    # Separate segment markers and POIs
    segment_waypoints = []
    pois = []

    for wp in all_waypoints:
        if wp.metadata_json and wp.metadata_json.get("segment"):
            segment_waypoints.append(wp)
        else:
            pois.append(wp)

    # Group segments by name
    segments = {}
    for wp in segment_waypoints:
        segment_meta = wp.metadata_json.get("segment", {})
        seg_name = segment_meta.get("name")
        role = segment_meta.get("role")

        if not seg_name or not role:
            continue

        if seg_name not in segments:
            segments[seg_name] = {
                "name": seg_name,
                "start_wp": None,
                "end_wp": None,
                "description": wp.user_description or ""
            }

        if role == "start":
            segments[seg_name]["start_wp"] = wp
            if wp.user_description:
                segments[seg_name]["description"] = wp.user_description
        elif role == "end":
            segments[seg_name]["end_wp"] = wp

    return {
        "routes": routes,
        "segments": segments,
        "pois": pois
    }


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

    background = ""
    if config.background_knowledge:
        background = f"""
HINTERGRUNDWISSEN ZUM ORT:
{config.background_knowledge}

Nutze dieses Wissen um die Texte informativer und interessanter zu gestalten.
"""

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


# ============ API Endpoints (Track-Level) ============

@router.get("/{track_id}/knowledge")
def get_track_knowledge(
    track_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all knowledge for a track.

    Returns:
    - routes: List of routes with their intro/outro
    - segments: All segments with entry/exit texts
    - pois: All POIs with approaching/at_poi texts
    - config: Generation config stored at track level
    """
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.visibility == "private" and track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Load all track data
    data = _load_track_data(db, track_id)

    # Get config from track metadata (or first route)
    track_metadata = track.metadata_json or {}
    config = track_metadata.get("knowledge_config", {})

    # Build response
    knowledge = {
        "version": 3,
        "config": config,
        "routes": {},
        "segments": {},
        "pois": {}
    }

    # Load route knowledge
    for route in data["routes"]:
        route_metadata = route.metadata_json or {}
        route_knowledge = route_metadata.get("knowledge", {})

        knowledge["routes"][str(route.id)] = {
            "id": route.id,
            "name": route.name,
            "description": route.description or "",
            "intro": route_knowledge.get("intro", {"text": "", "edited": False}),
            "outro": route_knowledge.get("outro", {"text": "", "edited": False})
        }

    # Load segment knowledge
    for seg_name, seg_data in data["segments"].items():
        start_wp = seg_data.get("start_wp")
        end_wp = seg_data.get("end_wp")

        start_knowledge = _get_waypoint_knowledge(start_wp) or {}
        end_knowledge = _get_waypoint_knowledge(end_wp) or {}

        knowledge["segments"][seg_name] = {
            "name": seg_name,
            "description": seg_data.get("description", ""),
            "start_waypoint_id": start_wp.id if start_wp else None,
            "end_waypoint_id": end_wp.id if end_wp else None,
            "entry": start_knowledge.get("entry", {"text": "", "edited": False}),
            "exit": end_knowledge.get("exit", {"text": "", "edited": False})
        }

    # Load POI knowledge
    for poi in data["pois"]:
        poi_knowledge = _get_waypoint_knowledge(poi) or {}
        poi_name = (poi.metadata_json or {}).get("title", f"POI #{poi.id}")

        knowledge["pois"][str(poi.id)] = {
            "waypoint_id": poi.id,
            "name": poi_name,
            "description": poi.user_description or "",
            "approaching": poi_knowledge.get("approaching", {"text": "", "edited": False}),
            "at_poi": poi_knowledge.get("at_poi", {"text": "", "edited": False})
        }

    # Check if any knowledge exists
    has_route_texts = any(
        r.get("intro", {}).get("text") or r.get("outro", {}).get("text")
        for r in knowledge["routes"].values()
    )
    has_segment_texts = any(
        s.get("entry", {}).get("text") or s.get("exit", {}).get("text")
        for s in knowledge["segments"].values()
    )
    has_poi_texts = any(
        p.get("approaching", {}).get("text") or p.get("at_poi", {}).get("text")
        for p in knowledge["pois"].values()
    )

    return {
        "exists": has_route_texts or has_segment_texts or has_poi_texts,
        "knowledge": knowledge,
        "track_id": track_id,
        "track_name": track.name
    }


@router.post("/{track_id}/knowledge/generate")
async def generate_track_knowledge(
    track_id: int,
    body: GenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate all narrative texts for a track using AI.

    Uses parallel AI calls for fast generation.

    Generates:
    - Intro/Outro for each route
    - Entry/Exit for each segment
    - Approaching/At for each POI
    """
    import asyncio

    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track creator can generate knowledge")

    # Load all track data
    data = _load_track_data(db, track_id)

    config = KnowledgeConfig(
        persona=body.persona,
        target_audience=body.target_audience,
        language=body.language,
        tone=body.tone,
        background_knowledge=body.background_knowledge
    )

    knowledge = {
        "version": 3,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": config.model_dump(),
        "routes": {},
        "segments": {},
        "pois": {}
    }

    # Collect all generation tasks
    tasks = []
    task_metadata = []  # Track what each task is for

    # Route tasks
    if body.generate_routes:
        for route in data["routes"]:
            gps_count = db.query(Waypoint).filter(
                Waypoint.track_id == track_id,
                Waypoint.waypoint_type == "gps_track",
                Waypoint.route_id == route.id
            ).count()
            route_length_km = (gps_count * 10) / 1000

            # Intro task
            tasks.append(_generate_narrative_text(
                "route_intro",
                {
                    "route_name": route.name,
                    "route_description": route.description or "",
                    "route_length_km": route_length_km
                },
                config
            ))
            task_metadata.append(("route", str(route.id), "intro", route.name, route.id))

            # Outro task
            tasks.append(_generate_narrative_text(
                "route_outro",
                {"route_name": route.name, "route_length_km": route_length_km},
                config
            ))
            task_metadata.append(("route", str(route.id), "outro", route.name, route.id))

    # Segment tasks
    if body.generate_segments:
        for seg_name, seg_data in data["segments"].items():
            # Entry task
            tasks.append(_generate_narrative_text(
                "segment_entry",
                {"segment_name": seg_name, "segment_description": seg_data.get("description", "")},
                config
            ))
            task_metadata.append(("segment", seg_name, "entry", seg_data))

            # Exit task
            tasks.append(_generate_narrative_text(
                "segment_exit",
                {"segment_name": seg_name},
                config
            ))
            task_metadata.append(("segment", seg_name, "exit", seg_data))

    # POI tasks
    if body.generate_pois:
        for poi in data["pois"]:
            poi_name = (poi.metadata_json or {}).get("title", f"POI #{poi.id}")
            poi_description = poi.user_description or ""

            # Approaching task
            tasks.append(_generate_narrative_text(
                "poi_approaching",
                {"poi_name": poi_name, "poi_description": poi_description},
                config
            ))
            task_metadata.append(("poi", str(poi.id), "approaching", poi_name, poi.id))

            # At POI task
            tasks.append(_generate_narrative_text(
                "poi_at",
                {"poi_name": poi_name, "poi_description": poi_description},
                config
            ))
            task_metadata.append(("poi", str(poi.id), "at_poi", poi_name, poi.id))

    # Run tasks sequentially for reliability
    results = []
    logger.info(f"Generating {len(tasks)} texts sequentially...")

    for i, task in enumerate(tasks):
        try:
            result = await task
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(tasks)} texts generated")
        except Exception as e:
            logger.warning(f"Task {i} failed: {e}")
            results.append(e)

    # Process results and count errors
    error_count = 0
    for i, result in enumerate(results):
        meta = task_metadata[i]
        if isinstance(result, Exception):
            error_count += 1
            logger.warning(f"AI generation failed for {meta[0]} {meta[1]} {meta[2]}: {result}")
            text = ""
        else:
            text = result if isinstance(result, str) else ""

        if meta[0] == "route":
            _, route_id, text_type, route_name, rid = meta
            if route_id not in knowledge["routes"]:
                knowledge["routes"][route_id] = {
                    "id": rid,
                    "name": route_name,
                    "intro": {"text": "", "text_original": "", "edited": False},
                    "outro": {"text": "", "text_original": "", "edited": False}
                }
            knowledge["routes"][route_id][text_type] = {
                "text": text,
                "text_original": text,
                "edited": False
            }

        elif meta[0] == "segment":
            _, seg_name, text_type, seg_data = meta
            if seg_name not in knowledge["segments"]:
                knowledge["segments"][seg_name] = {
                    "name": seg_name,
                    "start_waypoint_id": seg_data["start_wp"].id if seg_data.get("start_wp") else None,
                    "end_waypoint_id": seg_data["end_wp"].id if seg_data.get("end_wp") else None,
                    "entry": {"text": "", "text_original": "", "edited": False},
                    "exit": {"text": "", "text_original": "", "edited": False}
                }
            knowledge["segments"][seg_name][text_type] = {
                "text": text,
                "text_original": text,
                "edited": False
            }

        elif meta[0] == "poi":
            _, poi_id, text_type, poi_name, wid = meta
            if poi_id not in knowledge["pois"]:
                knowledge["pois"][poi_id] = {
                    "waypoint_id": wid,
                    "name": poi_name,
                    "approaching": {"text": "", "text_original": "", "edited": False},
                    "at_poi": {"text": "", "text_original": "", "edited": False}
                }
            knowledge["pois"][poi_id][text_type] = {
                "text": text,
                "text_original": text,
                "edited": False
            }

    success_count = len(tasks) - error_count
    logger.info(f"Generation complete: {success_count}/{len(tasks)} texts generated ({error_count} errors)")

    return {
        "success": True,
        "knowledge": knowledge,
        "stats": {
            "routes_count": len(knowledge["routes"]),
            "segments_count": len(knowledge["segments"]),
            "pois_count": len(knowledge["pois"]),
            "total_texts": len(tasks),
            "successful_texts": success_count,
            "failed_texts": error_count
        }
    }


@router.put("/{track_id}/knowledge")
def save_track_knowledge(
    track_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Save all knowledge for a track.

    Distributes knowledge to respective objects:
    - Route texts → TrackRoute.metadata_json["knowledge"]
    - Segment texts → Waypoint.metadata_json["knowledge"]
    - POI texts → Waypoint.metadata_json["knowledge"]
    """
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track creator can save knowledge")

    knowledge = body.get("knowledge")
    if not knowledge:
        raise HTTPException(status_code=400, detail="Knowledge object required")

    # Save config to track
    track_metadata = track.metadata_json or {}
    track_metadata["knowledge_config"] = knowledge.get("config", {})
    track.metadata_json = track_metadata
    flag_modified(track, "metadata_json")

    # Save route knowledge
    routes_knowledge = knowledge.get("routes", {})
    for route_id_str, route_data in routes_knowledge.items():
        route_id = int(route_id_str)
        route = db.query(TrackRoute).filter(TrackRoute.id == route_id).first()

        if route:
            route_metadata = route.metadata_json or {}
            route_metadata["knowledge"] = {
                "intro": route_data.get("intro", {}),
                "outro": route_data.get("outro", {})
            }
            route.metadata_json = route_metadata
            flag_modified(route, "metadata_json")

    # Save segment knowledge
    segments_knowledge = knowledge.get("segments", {})
    for seg_name, seg_data in segments_knowledge.items():
        start_wp_id = seg_data.get("start_waypoint_id")
        if start_wp_id:
            start_wp = db.query(Waypoint).filter(Waypoint.id == start_wp_id).first()
            if start_wp:
                _save_waypoint_knowledge(start_wp, {"entry": seg_data.get("entry", {})}, db)

        end_wp_id = seg_data.get("end_waypoint_id")
        if end_wp_id:
            end_wp = db.query(Waypoint).filter(Waypoint.id == end_wp_id).first()
            if end_wp:
                _save_waypoint_knowledge(end_wp, {"exit": seg_data.get("exit", {})}, db)

    # Save POI knowledge
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

    db.commit()

    return {
        "success": True,
        "track_id": track_id,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "saved_routes": len(routes_knowledge),
        "saved_segments": len(segments_knowledge),
        "saved_pois": len(pois_knowledge)
    }


@router.post("/{track_id}/knowledge/audio")
async def generate_knowledge_audio(
    track_id: int,
    body: AudioGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate TTS audio for a single knowledge item.

    item_type: "route", "segment", "poi"
    item_id: route_id (for route), segment_name (for segment), waypoint_id (for poi)
    text_type: "intro", "outro", "entry", "exit", "approaching", "at_poi"
    """
    import uuid
    import asyncio

    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track creator can generate audio")

    # Validate combinations
    valid_combinations = {
        "route": ["intro", "outro"],
        "segment": ["entry", "exit"],
        "poi": ["approaching", "at_poi"]
    }

    if body.item_type not in valid_combinations:
        raise HTTPException(status_code=400, detail=f"Invalid item_type: {body.item_type}")

    if body.text_type not in valid_combinations[body.item_type]:
        raise HTTPException(status_code=400, detail=f"Invalid text_type for {body.item_type}")

    if not body.item_id:
        raise HTTPException(status_code=400, detail="item_id required")

    # Get text and source object
    text = None
    source_obj = None
    source_id = None

    if body.item_type == "route":
        route_id = int(body.item_id)
        route = db.query(TrackRoute).filter(TrackRoute.id == route_id).first()
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")

        route_metadata = route.metadata_json or {}
        route_knowledge = route_metadata.get("knowledge", {})
        text_data = route_knowledge.get(body.text_type, {})
        text = text_data.get("text", "")
        source_obj = route
        source_id = f"route_{route_id}_{body.text_type}"

    elif body.item_type == "segment":
        # Find segment waypoints
        data = _load_track_data(db, track_id)
        seg_data = data["segments"].get(body.item_id)

        if not seg_data:
            raise HTTPException(status_code=404, detail=f"Segment '{body.item_id}' not found")

        if body.text_type == "entry":
            waypoint = seg_data.get("start_wp")
        else:
            waypoint = seg_data.get("end_wp")

        if not waypoint:
            raise HTTPException(status_code=404, detail=f"Waypoint for segment {body.text_type} not found")

        wp_knowledge = _get_waypoint_knowledge(waypoint) or {}
        text_data = wp_knowledge.get(body.text_type, {})
        text = text_data.get("text", "")
        source_obj = waypoint
        source_id = f"segment_{waypoint.id}_{body.text_type}"

    elif body.item_type == "poi":
        waypoint_id = int(body.item_id)
        waypoint = db.query(Waypoint).filter(Waypoint.id == waypoint_id).first()

        if not waypoint:
            raise HTTPException(status_code=404, detail=f"POI {waypoint_id} not found")

        wp_knowledge = _get_waypoint_knowledge(waypoint) or {}
        text_data = wp_knowledge.get(body.text_type, {})
        text = text_data.get("text", "")
        source_obj = waypoint
        source_id = f"poi_{waypoint_id}_{body.text_type}"

    if not text:
        raise HTTPException(status_code=400, detail="No text found to generate audio from")

    # Generate audio via Dialog API
    job_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    dialog_payload = {
        "id": job_id,
        "type": "speech_request",
        "timestamp": timestamp,
        "content": {
            "text": text,
            "language": body.language,
            "speed": 1.0,
            "voice": body.voice
        },
        "config": {
            "provider": "openai",
            "output_format": "mp3",
            "dialog_mode": True,
            "add_music": body.add_music,
            "add_sfx": False,
            "analyze_only": False,
            "voice_mapping": {"Narrator": body.voice}
        },
        "save_options": {
            "is_public": True,
            "link_id": f"knowledge;{track_id};{source_id}",
            "collection_id": f"artrack-knowledge:{track_id}"
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AI_API_BASE}/ai/dialog/start",
                json=dialog_payload,
                headers={"X-API-KEY": INTERNAL_API_KEY, "Content-Type": "application/json"},
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(f"Dialog API failed: {response.status_code}")
                raise HTTPException(status_code=500, detail="Failed to start audio generation")

            # Poll for completion
            max_wait = 120
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(2)
                elapsed += 2

                status_resp = await client.get(
                    f"{AI_API_BASE}/ai/dialog/status",
                    params={"id": job_id},
                    headers={"X-API-KEY": INTERNAL_API_KEY},
                    timeout=30.0
                )

                if status_resp.status_code != 200:
                    continue

                status_data = status_resp.json()
                phase = status_data.get("phase", "")

                if phase == "done":
                    result = status_data.get("result", {})
                    storage_id = result.get("id")
                    audio_url = result.get("url") or result.get("file_url")
                    duration = result.get("duration_seconds")

                    # Update knowledge with audio_storage_id
                    if storage_id and source_obj:
                        if body.item_type == "route":
                            route_metadata = source_obj.metadata_json or {}
                            if "knowledge" not in route_metadata:
                                route_metadata["knowledge"] = {}
                            if body.text_type not in route_metadata["knowledge"]:
                                route_metadata["knowledge"][body.text_type] = {}
                            route_metadata["knowledge"][body.text_type]["audio_storage_id"] = storage_id
                            source_obj.metadata_json = route_metadata
                            flag_modified(source_obj, "metadata_json")
                        else:
                            wp_metadata = source_obj.metadata_json or {}
                            if "knowledge" not in wp_metadata:
                                wp_metadata["knowledge"] = {}
                            if body.text_type not in wp_metadata["knowledge"]:
                                wp_metadata["knowledge"][body.text_type] = {}
                            wp_metadata["knowledge"][body.text_type]["audio_storage_id"] = storage_id
                            source_obj.metadata_json = wp_metadata
                            flag_modified(source_obj, "metadata_json")

                        db.commit()

                    return {
                        "success": True,
                        "audio_storage_id": storage_id,
                        "audio_url": audio_url,
                        "duration_seconds": duration,
                        "item_type": body.item_type,
                        "item_id": body.item_id,
                        "text_type": body.text_type
                    }

                elif phase == "error":
                    raise HTTPException(status_code=500, detail="Audio generation failed")

            raise HTTPException(status_code=504, detail="Audio generation timed out")

    except httpx.RequestError as e:
        logger.error(f"Dialog API request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{track_id}/knowledge")
def delete_track_knowledge(
    track_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete all knowledge for a track."""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track creator can delete knowledge")

    data = _load_track_data(db, track_id)

    # Remove track config
    track_metadata = track.metadata_json or {}
    if "knowledge_config" in track_metadata:
        del track_metadata["knowledge_config"]
        track.metadata_json = track_metadata
        flag_modified(track, "metadata_json")

    # Remove route knowledge
    for route in data["routes"]:
        route_metadata = route.metadata_json or {}
        if "knowledge" in route_metadata:
            del route_metadata["knowledge"]
            route.metadata_json = route_metadata
            flag_modified(route, "metadata_json")

    # Remove segment knowledge
    for seg_name, seg_data in data["segments"].items():
        for wp in [seg_data.get("start_wp"), seg_data.get("end_wp")]:
            if wp:
                wp_metadata = wp.metadata_json or {}
                if "knowledge" in wp_metadata:
                    del wp_metadata["knowledge"]
                    wp.metadata_json = wp_metadata
                    flag_modified(wp, "metadata_json")

    # Remove POI knowledge
    for poi in data["pois"]:
        poi_metadata = poi.metadata_json or {}
        if "knowledge" in poi_metadata:
            del poi_metadata["knowledge"]
            poi.metadata_json = poi_metadata
            flag_modified(poi, "metadata_json")

    db.commit()

    return {"success": True, "deleted": True}
