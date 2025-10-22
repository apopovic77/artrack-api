from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from ..database import get_db
from ..models import (
    Track, TrackCreate, TrackResponse, TrackStats, User
)
from ..auth import get_current_user

router = APIRouter()

@router.post("/", response_model=TrackResponse)
async def create_track(
    track_data: TrackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new track"""
    
    # Check if track with same client_track_id already exists for this user
    existing_track = db.query(Track).filter(
        Track.client_track_id == track_data.client_track_id,
        Track.created_by == current_user.id
    ).first()
    
    if existing_track:
        # Calculate stats for existing track
        from ..models import Waypoint
        total_waypoints = db.query(Waypoint).filter(Waypoint.track_id == existing_track.id).count()
        processed_waypoints = db.query(Waypoint).filter(
            Waypoint.track_id == existing_track.id,
            Waypoint.processing_state == "published"
        ).count()
        pending_analysis = db.query(Waypoint).filter(
            Waypoint.track_id == existing_track.id,
            Waypoint.processing_state.in_(["pending", "uploading", "uploaded", "analysing"])
        ).count()

        stats = TrackStats(
            total_waypoints=total_waypoints,
            processed_waypoints=processed_waypoints,
            pending_analysis=pending_analysis,
            distance_meters=existing_track.distance_meters,
            duration_seconds=existing_track.duration_seconds
        )

        return TrackResponse(
            id=existing_track.id,
            name=existing_track.name,
            description=existing_track.description,
            visibility=existing_track.visibility,
            track_type=existing_track.track_type,
            tags=existing_track.tags,
            client_track_id=existing_track.client_track_id,
            stats=stats,
            created_at=existing_track.created_at,
            updated_at=existing_track.updated_at,
            metadata_json=existing_track.metadata_json
        )
    
    # Create new track
    db_track = Track(
        name=track_data.name,
        description=track_data.description,
        visibility=track_data.visibility,
        track_type=track_data.track_type,
        tags=track_data.tags,
        client_track_id=track_data.client_track_id,
        created_by=current_user.id,
        storage_object_ids=(track_data.storage_object_ids or []),
        storage_collection=(track_data.storage_collection or {})
    )
    
    db.add(db_track)
    db.commit()
    db.refresh(db_track)
    
    # Build response per OpenAPI
    stats = TrackStats(
        total_waypoints=0,
        processed_waypoints=0,
        pending_analysis=0,
        distance_meters=db_track.distance_meters,
        duration_seconds=db_track.duration_seconds
    )
    return TrackResponse(
        id=db_track.id,
        name=db_track.name,
        description=db_track.description,
        visibility=db_track.visibility,
        track_type=db_track.track_type,
        tags=db_track.tags,
        client_track_id=db_track.client_track_id,
        stats=stats,
        created_at=db_track.created_at,
        updated_at=db_track.updated_at,
        metadata_json=db_track.metadata_json,
        storage_object_ids=getattr(db_track, 'storage_object_ids', None),
        storage_collection=getattr(db_track, 'storage_collection', None)
    )

@router.get("/{track_id}", response_model=TrackResponse)
async def get_track(
    track_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get track details"""
    
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions (user can see their own tracks, or public tracks)
    if track.created_by != current_user.id and track.visibility == "private":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Calculate stats
    from ..models import Waypoint, MediaFile, AnalysisResult
    
    total_waypoints = db.query(Waypoint).filter(Waypoint.track_id == track_id).count()
    
    processed_waypoints = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.processing_state == "published"
    ).count()
    
    pending_analysis = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.processing_state.in_(["pending", "uploading", "uploaded", "analysing"])
    ).count()
    
    stats = TrackStats(
        total_waypoints=total_waypoints,
        processed_waypoints=processed_waypoints,
        pending_analysis=pending_analysis,
        distance_meters=track.distance_meters,
        duration_seconds=track.duration_seconds
    )
    
    # Convert track to response format
    track_response = TrackResponse(
        id=track.id,
        name=track.name,
        description=track.description,
        visibility=track.visibility,
        track_type=track.track_type,
        tags=track.tags,
        client_track_id=track.client_track_id,
        stats=stats,
        created_at=track.created_at,
        updated_at=track.updated_at,
        metadata_json=track.metadata_json,  # Include guide config
        storage_object_ids=getattr(track, 'storage_object_ids', None),
        storage_collection=getattr(track, 'storage_collection', None)
    )

    return track_response

@router.get("/", response_model=List[TrackResponse])
async def list_tracks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    visibility: str = None,
    limit: int = 50,
    offset: int = 0
):
    """List tracks"""
    
    query = db.query(Track)
    
    # Filter by visibility and permissions
    if visibility == "public":
        query = query.filter(Track.visibility == "public")
    elif visibility == "my" or visibility is None:
        query = query.filter(Track.created_by == current_user.id)
    else:
        # For other visibility levels, show only user's own tracks
        query = query.filter(Track.created_by == current_user.id)
    
    # Apply pagination
    tracks = query.offset(offset).limit(limit).all()
    
    # Convert to response format
    track_responses = []
    for track in tracks:
        # Calculate stats for each track
        from ..models import Waypoint
        
        total_waypoints = db.query(Waypoint).filter(Waypoint.track_id == track.id).count()
        processed_waypoints = db.query(Waypoint).filter(
            Waypoint.track_id == track.id,
            Waypoint.processing_state == "published"
        ).count()
        pending_analysis = db.query(Waypoint).filter(
            Waypoint.track_id == track.id,
            Waypoint.processing_state.in_(["pending", "uploading", "uploaded", "analysing"])
        ).count()
        
        stats = TrackStats(
            total_waypoints=total_waypoints,
            processed_waypoints=processed_waypoints,
            pending_analysis=pending_analysis,
            distance_meters=track.distance_meters,
            duration_seconds=track.duration_seconds
        )
        
        track_response = TrackResponse(
            id=track.id,
            name=track.name,
            description=track.description,
            visibility=track.visibility,
            track_type=track.track_type,
            tags=track.tags,
            client_track_id=track.client_track_id,
            stats=stats,
            created_at=track.created_at,
            updated_at=track.updated_at,
            metadata_json=track.metadata_json,
            storage_object_ids=getattr(track, 'storage_object_ids', None),
            storage_collection=getattr(track, 'storage_collection', None)
        )

        track_responses.append(track_response)
    
    return track_responses

@router.delete("/{track_id}")
async def delete_track(
    track_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a track"""
    
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # TODO: Also delete associated waypoints, media files, and analysis results
    # For now, just delete the track
    db.delete(track)
    db.commit()
    
    return {"message": "Track deleted successfully"}

@router.put("/{track_id}")
async def update_track(
    track_id: int,
    track_data: TrackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a track"""
    
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update track
    track.name = track_data.name
    track.description = track_data.description
    track.visibility = track_data.visibility
    track.track_type = track_data.track_type
    track.tags = track_data.tags
    # Optional assets
    try:
        if hasattr(track_data, 'storage_object_ids') and track_data.storage_object_ids is not None:
            track.storage_object_ids = track_data.storage_object_ids
        if hasattr(track_data, 'storage_collection') and track_data.storage_collection is not None:
            track.storage_collection = track_data.storage_collection
    except Exception:
        pass

    track.updated_at = datetime.utcnow()
    track.version += 1
    
    db.commit()
    
    return {"message": "Track updated successfully"}