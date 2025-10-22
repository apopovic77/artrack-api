"""
Collaboration API Routes - Multi-User Track Sharing
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from datetime import datetime, timedelta
import secrets

from ..database import get_db
from ..auth import get_current_user
from ..models import User, Track, Waypoint
from ..collaboration_models import (
    TrackCollaborator, TrackInvitation, TrackActivity,
    TrackCollaboratorCreate, TrackCollaboratorResponse,
    TrackInvitationCreate, TrackInvitationResponse,
    TrackActivityResponse, CollaborativeTrackCreate,
    InviteAcceptRequest, InviteByCodeRequest,
    TrackPermissions, CollaborativeTrackStats,
    get_user_permissions, check_collaboration_limits,
    generate_invite_link, generate_qr_code_data
)
from ..track_geometry import (
    validate_waypoint_proximity, WaypointProximityConfig,
    ProximityPresets, suggest_optimal_tolerance,
    debug_closest_point_calculation, Point, TrackPoint
)

router = APIRouter()

# === Track Collaboration Management ===

@router.post("/{track_id}/collaborate", response_model=dict)
async def enable_collaboration(
    track_id: int,
    collaboration_mode: str = "invite_only",
    max_collaborators: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enable collaboration for a track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track owner can enable collaboration")
    
    # Update track
    track.is_collaborative = True
    track.collaboration_mode = collaboration_mode
    track.max_collaborators = max_collaborators
    
    # Generate invite code if needed
    if not track.invite_code:
        track.invite_code = ''.join(secrets.choice('ABCDEFGHIJKLMNPQRSTUVWXYZ123456789') for _ in range(8))
    
    db.commit()
    
    # Log activity
    activity = TrackActivity(
        track_id=track_id,
        user_id=current_user.id,
        activity_type="collaboration_enabled",
        description=f"Collaboration enabled with mode: {collaboration_mode}"
    )
    db.add(activity)
    db.commit()
    
    return {
        "message": "Collaboration enabled",
        "invite_code": track.invite_code,
        "collaboration_mode": collaboration_mode,
        "max_collaborators": max_collaborators
    }

@router.get("/{track_id}/collaborators", response_model=List[TrackCollaboratorResponse])
async def get_track_collaborators(
    track_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all collaborators for a track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_view:
        raise HTTPException(status_code=403, detail="No permission to view this track")
    
    collaborators = db.query(TrackCollaborator).options(
        joinedload(TrackCollaborator.user)
    ).filter(
        TrackCollaborator.track_id == track_id,
        TrackCollaborator.is_active == True
    ).all()
    
    result = []
    for collab in collaborators:
        result.append(TrackCollaboratorResponse(
            id=collab.id,
            user_id=collab.user_id,
            username=collab.user.username if collab.user else None,
            role=collab.role,
            can_add_waypoints=collab.can_add_waypoints,
            can_edit_waypoints=collab.can_edit_waypoints,
            can_delete_waypoints=collab.can_delete_waypoints,
            can_invite_others=collab.can_invite_others,
            joined_at=collab.joined_at,
            last_activity=collab.last_activity,
            is_active=collab.is_active
        ))
    
    return result

@router.post("/{track_id}/invite", response_model=TrackInvitationResponse)
async def invite_user_to_track(
    track_id: int,
    invitation: TrackInvitationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Invite a user to collaborate on a track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_invite_others and not permissions.is_owner:
        raise HTTPException(status_code=403, detail="No permission to invite users")
    
    # Check collaboration limits
    if not check_collaboration_limits(track, 1):
        raise HTTPException(status_code=400, detail="Maximum collaborators reached")
    
    # Check if user is already invited or collaborating
    existing_invite = db.query(TrackInvitation).filter(
        TrackInvitation.track_id == track_id,
        TrackInvitation.status == "pending"
    )
    
    if invitation.email:
        existing_invite = existing_invite.filter(TrackInvitation.email == invitation.email)
    elif invitation.username:
        existing_invite = existing_invite.filter(TrackInvitation.username == invitation.username)
    
    if existing_invite.first():
        raise HTTPException(status_code=400, detail="User already has pending invitation")
    
    # Create invitation
    expires_at = datetime.utcnow() + timedelta(hours=invitation.expires_in_hours)
    
    new_invitation = TrackInvitation(
        track_id=track_id,
        invited_by=current_user.id,
        email=invitation.email,
        username=invitation.username,
        role=invitation.role,
        can_add_waypoints=invitation.can_add_waypoints,
        can_edit_waypoints=invitation.can_edit_waypoints,
        can_delete_waypoints=invitation.can_delete_waypoints,
        can_invite_others=invitation.can_invite_others,
        expires_at=expires_at,
        personal_message=invitation.personal_message
    )
    
    db.add(new_invitation)
    db.commit()
    db.refresh(new_invitation)
    
    # Log activity
    activity = TrackActivity(
        track_id=track_id,
        user_id=current_user.id,
        activity_type="user_invited",
        description=f"Invited {invitation.email or invitation.username} as {invitation.role}"
    )
    db.add(activity)
    db.commit()
    
    return TrackInvitationResponse(
        id=new_invitation.id,
        track_id=track_id,
        track_name=track.name,
        email=new_invitation.email,
        username=new_invitation.username,
        invite_token=new_invitation.invite_token,
        invite_code=new_invitation.invite_code,
        role=new_invitation.role,
        status=new_invitation.status,
        created_at=new_invitation.created_at,
        expires_at=new_invitation.expires_at,
        personal_message=new_invitation.personal_message,
        inviter_name=current_user.username
    )

@router.post("/invitations/{invite_token}/accept", response_model=dict)
async def accept_invitation(
    invite_token: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Accept a track invitation"""
    invitation = db.query(TrackInvitation).options(
        joinedload(TrackInvitation.track)
    ).filter(
        TrackInvitation.invite_token == invite_token,
        TrackInvitation.status == "pending"
    ).first()
    
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or expired")
    
    if invitation.expires_at < datetime.utcnow():
        invitation.status = "expired"
        db.commit()
        raise HTTPException(status_code=400, detail="Invitation has expired")
    
    # Check if user matches invitation
    if invitation.email and current_user.email != invitation.email:
        raise HTTPException(status_code=403, detail="Invitation is for a different email")
    if invitation.username and current_user.username != invitation.username:
        raise HTTPException(status_code=403, detail="Invitation is for a different username")
    
    # Check if user is already a collaborator
    existing_collab = db.query(TrackCollaborator).filter(
        TrackCollaborator.track_id == invitation.track_id,
        TrackCollaborator.user_id == current_user.id,
        TrackCollaborator.is_active == True
    ).first()
    
    if existing_collab:
        raise HTTPException(status_code=400, detail="You are already a collaborator on this track")
    
    # Check collaboration limits
    if not check_collaboration_limits(invitation.track, 1):
        raise HTTPException(status_code=400, detail="Track has reached maximum collaborators")
    
    # Create collaborator
    collaborator = TrackCollaborator(
        track_id=invitation.track_id,
        user_id=current_user.id,
        role=invitation.role,
        can_add_waypoints=invitation.can_add_waypoints,
        can_edit_waypoints=invitation.can_edit_waypoints,
        can_delete_waypoints=invitation.can_delete_waypoints,
        can_invite_others=invitation.can_invite_others,
        invited_by=invitation.invited_by
    )
    
    db.add(collaborator)
    
    # Update invitation
    invitation.status = "accepted"
    invitation.accepted_at = datetime.utcnow()
    
    db.commit()
    
    # Log activity
    activity = TrackActivity(
        track_id=invitation.track_id,
        user_id=current_user.id,
        activity_type="user_joined",
        description=f"{current_user.username} joined as {invitation.role}"
    )
    db.add(activity)
    db.commit()
    
    return {
        "message": "Invitation accepted successfully",
        "track_id": invitation.track_id,
        "track_name": invitation.track.name,
        "role": invitation.role
    }

@router.post("/join", response_model=dict)
async def join_by_code(
    request: InviteByCodeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Join a track using invite code"""
    # Find track by invite code
    track = db.query(Track).filter(
        Track.invite_code == request.invite_code,
        Track.is_collaborative == True
    ).first()
    
    if not track:
        raise HTTPException(status_code=404, detail="Invalid invite code")
    
    # Check if user is already a collaborator
    existing_collab = db.query(TrackCollaborator).filter(
        TrackCollaborator.track_id == track.id,
        TrackCollaborator.user_id == current_user.id,
        TrackCollaborator.is_active == True
    ).first()
    
    if existing_collab:
        raise HTTPException(status_code=400, detail="You are already a collaborator on this track")
    
    # Check collaboration mode and limits
    if track.collaboration_mode == "invite_only":
        raise HTTPException(status_code=403, detail="This track requires a personal invitation")
    
    if not check_collaboration_limits(track, 1):
        raise HTTPException(status_code=400, detail="Track has reached maximum collaborators")
    
    # Create collaborator with default permissions
    collaborator = TrackCollaborator(
        track_id=track.id,
        user_id=current_user.id,
        role="contributor",
        can_add_waypoints=True,
        can_edit_waypoints=False,
        can_delete_waypoints=False,
        can_invite_others=False,
        invited_by=track.created_by
    )
    
    db.add(collaborator)
    db.commit()
    
    # Log activity
    activity = TrackActivity(
        track_id=track.id,
        user_id=current_user.id,
        activity_type="user_joined",
        description=f"{current_user.username} joined via invite code"
    )
    db.add(activity)
    db.commit()
    
    return {
        "message": "Successfully joined track",
        "track_id": track.id,
        "track_name": track.name,
        "role": "contributor"
    }

@router.get("/{track_id}/permissions", response_model=TrackPermissions)
async def get_user_track_permissions(
    track_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's permissions for a track"""
    track = db.query(Track).options(
        joinedload(Track.collaborators)
    ).filter(Track.id == track_id).first()
    
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    return get_user_permissions(track, current_user.id)

@router.get("/{track_id}/activity", response_model=List[TrackActivityResponse])
async def get_track_activity(
    track_id: int,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get activity log for a track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_view:
        raise HTTPException(status_code=403, detail="No permission to view this track")
    
    activities = db.query(TrackActivity).options(
        joinedload(TrackActivity.user)
    ).filter(
        TrackActivity.track_id == track_id
    ).order_by(
        TrackActivity.created_at.desc()
    ).limit(limit).all()
    
    return [
        TrackActivityResponse(
            id=activity.id,
            track_id=activity.track_id,
            user_id=activity.user_id,
            username=activity.user.username if activity.user else None,
            activity_type=activity.activity_type,
            target_type=activity.target_type,
            target_id=activity.target_id,
            description=activity.description,
            created_at=activity.created_at
        )
        for activity in activities
    ]

@router.get("/{track_id}/stats", response_model=CollaborativeTrackStats)
async def get_track_collaboration_stats(
    track_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get collaboration statistics for a track"""
    track = db.query(Track).options(
        joinedload(Track.collaborators),
        joinedload(Track.waypoints)
    ).filter(Track.id == track_id).first()
    
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_view:
        raise HTTPException(status_code=403, detail="No permission to view this track")
    
    # Calculate stats
    active_collaborators = [c for c in track.collaborators if c.is_active]
    total_collaborators = len(active_collaborators)
    
    # Active in last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)
    active_last_week = len([c for c in active_collaborators if c.last_activity > week_ago])
    
    # Waypoints by user
    waypoints_by_user = {}
    for waypoint in track.waypoints:
        if waypoint.created_by:
            user = db.query(User).filter(User.id == waypoint.created_by).first()
            if user:
                waypoints_by_user[user.username] = waypoints_by_user.get(user.username, 0) + 1
    
    # Most active contributor
    most_active = max(waypoints_by_user.items(), key=lambda x: x[1]) if waypoints_by_user else None
    
    # Last activity
    last_activity = None
    if active_collaborators:
        last_activity = max(c.last_activity for c in active_collaborators)
    
    # Activity last 7 days
    activity_count = db.query(TrackActivity).filter(
        TrackActivity.track_id == track_id,
        TrackActivity.created_at > week_ago
    ).count()
    
    return CollaborativeTrackStats(
        total_collaborators=total_collaborators,
        active_collaborators=active_last_week,
        total_waypoints=len(track.waypoints),
        waypoints_by_user=waypoints_by_user,
        most_active_contributor=most_active[0] if most_active else None,
        last_activity=last_activity,
        activity_last_7_days=activity_count
    )

# === Collaborator Management ===

@router.delete("/{track_id}/collaborators/{collaborator_id}")
async def remove_collaborator(
    track_id: int,
    collaborator_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a collaborator from a track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_manage_collaborators and not permissions.is_owner:
        raise HTTPException(status_code=403, detail="No permission to manage collaborators")
    
    collaborator = db.query(TrackCollaborator).filter(
        TrackCollaborator.id == collaborator_id,
        TrackCollaborator.track_id == track_id
    ).first()
    
    if not collaborator:
        raise HTTPException(status_code=404, detail="Collaborator not found")
    
    # Can't remove track owner
    if collaborator.user_id == track.created_by:
        raise HTTPException(status_code=400, detail="Cannot remove track owner")
    
    # Mark as inactive instead of deleting
    collaborator.is_active = False
    db.commit()
    
    # Log activity
    removed_user = db.query(User).filter(User.id == collaborator.user_id).first()
    activity = TrackActivity(
        track_id=track_id,
        user_id=current_user.id,
        activity_type="user_removed",
        description=f"Removed {removed_user.username if removed_user else 'user'} from track"
    )
    db.add(activity)
    db.commit()
    
    return {"message": "Collaborator removed successfully"}

@router.post("/{track_id}/leave")
async def leave_track(
    track_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Leave a collaborative track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Track owner cannot leave
    if track.created_by == current_user.id:
        raise HTTPException(status_code=400, detail="Track owner cannot leave the track")
    
    collaborator = db.query(TrackCollaborator).filter(
        TrackCollaborator.track_id == track_id,
        TrackCollaborator.user_id == current_user.id,
        TrackCollaborator.is_active == True
    ).first()
    
    if not collaborator:
        raise HTTPException(status_code=400, detail="You are not a collaborator on this track")
    
    # Mark as inactive
    collaborator.is_active = False
    db.commit()
    
    # Log activity
    activity = TrackActivity(
        track_id=track_id,
        user_id=current_user.id,
        activity_type="user_left",
        description=f"{current_user.username} left the track"
    )
    db.add(activity)
    db.commit()
    
    return {"message": "Successfully left the track"}

@router.get("/{track_id}/share-info", response_model=dict)
async def get_track_share_info(
    track_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get share information for a track (invite links, QR codes, etc.)"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_invite_others and not permissions.is_owner:
        raise HTTPException(status_code=403, detail="No permission to share this track")
    
    if not track.is_collaborative:
        raise HTTPException(status_code=400, detail="Track is not collaborative")
    
    # Generate sharing info
    invite_link = f"https://api.arkturian.com/artrack/join/{track.invite_code}"
    qr_data = generate_qr_code_data(track_id, track.invite_code)
    
    return {
        "track_id": track_id,
        "track_name": track.name,
        "invite_code": track.invite_code,
        "invite_link": invite_link,
        "qr_code_data": qr_data,
        "collaboration_mode": track.collaboration_mode,
        "max_collaborators": track.max_collaborators,
        "current_collaborators": len([c for c in track.collaborators if c.is_active])
    }

# === Track Geometry & Waypoint Validation ===

@router.post("/{track_id}/validate-waypoint", response_model=dict)
async def validate_waypoint_location(
    track_id: int,
    waypoint_lat: float,
    waypoint_lon: float,
    tolerance_meters: Optional[float] = None,
    activity_type: str = "hiking",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Validate if a waypoint location is close enough to the track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_add_waypoints:
        raise HTTPException(status_code=403, detail="No permission to add waypoints to this track")
    
    # Get track points (only from track owner's data)
    track_points = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.created_by == track.created_by  # Only owner's GPS track data
    ).order_by(Waypoint.timestamp).all()
    
    if len(track_points) < 2:
        return {
            "is_valid": False,
            "error": "Track has insufficient GPS points for validation",
            "required_points": 2,
            "available_points": len(track_points)
        }
    
    # Convert to format expected by geometry functions
    track_point_data = [
        {
            "latitude": tp.latitude,
            "longitude": tp.longitude,
            "elevation": tp.altitude
        }
        for tp in track_points
    ]
    
    # Determine tolerance
    if tolerance_meters is None:
        # Use preset or calculate suggested tolerance
        if activity_type in ["hiking", "cycling", "driving", "mountaineering"]:
            config = getattr(ProximityPresets, activity_type.upper(), ProximityPresets.HIKING_RELAXED)
            tolerance_meters = config.max_distance_meters
        else:
            tolerance_meters = suggest_optimal_tolerance(
                [{"lat": tp["latitude"], "lon": tp["longitude"]} for tp in track_point_data],
                activity_type
            )
    
    # Create configuration
    config = WaypointProximityConfig(
        max_distance_meters=tolerance_meters,
        use_elevation=False,
        allow_extrapolation=True,
        min_track_points=2
    )
    
    # Validate proximity
    result = validate_waypoint_proximity(
        waypoint_lat=waypoint_lat,
        waypoint_lon=waypoint_lon,
        track_points=track_point_data,
        config=config
    )
    
    return {
        "is_valid": result.is_valid,
        "distance_to_track_meters": result.distance_to_track,
        "tolerance_meters": tolerance_meters,
        "closest_track_point": result.closest_track_point,
        "track_position_km": result.track_position_km,
        "error_message": result.error_message,
        "validation_details": {
            "track_segment_index": result.track_segment_index,
            "projection_ratio": result.projection_ratio,
            "activity_type": activity_type,
            "total_track_points": len(track_point_data)
        }
    }

@router.get("/{track_id}/geometry-info", response_model=dict)
async def get_track_geometry_info(
    track_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get track geometry information and suggested tolerances"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_view:
        raise HTTPException(status_code=403, detail="No permission to view this track")
    
    # Get track points (only owner's GPS data)
    track_points = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.created_by == track.created_by
    ).order_by(Waypoint.timestamp).all()
    
    if len(track_points) < 2:
        return {
            "has_sufficient_data": False,
            "track_point_count": len(track_points),
            "error": "Track needs at least 2 GPS points for geometry analysis"
        }
    
    # Convert to format for analysis
    track_point_data = [
        {
            "lat": tp.latitude,
            "lon": tp.longitude,
            "elevation": tp.altitude
        }
        for tp in track_points
    ]
    
    # Calculate track statistics
    from ..track_geometry import get_track_statistics, TrackPoint
    track_point_objects = [
        TrackPoint(lat=tp["lat"], lon=tp["lon"], elevation=tp.get("elevation"))
        for tp in track_point_data
    ]
    
    stats = get_track_statistics(track_point_objects)
    
    # Suggest tolerances for different activities
    suggested_tolerances = {
        "hiking": suggest_optimal_tolerance(track_point_objects, "hiking"),
        "cycling": suggest_optimal_tolerance(track_point_objects, "cycling"),
        "driving": suggest_optimal_tolerance(track_point_objects, "driving"),
        "mountaineering": suggest_optimal_tolerance(track_point_objects, "mountaineering")
    }
    
    return {
        "has_sufficient_data": True,
        "track_point_count": len(track_points),
        "total_distance_km": stats["total_distance_km"],
        "bounding_box": stats["bounding_box"],
        "suggested_tolerances_meters": suggested_tolerances,
        "preset_configs": {
            "hiking_strict": ProximityPresets.HIKING_STRICT.dict(),
            "hiking_relaxed": ProximityPresets.HIKING_RELAXED.dict(),
            "cycling": ProximityPresets.CYCLING.dict(),
            "driving": ProximityPresets.DRIVING.dict(),
            "mountaineering": ProximityPresets.MOUNTAINEERING.dict()
        },
        "track_metadata": {
            "created_by_owner": True,
            "owner_id": track.created_by,
            "is_collaborative": track.is_collaborative,
            "collaboration_mode": track.collaboration_mode
        }
    }

@router.post("/{track_id}/configure-proximity", response_model=dict)
async def configure_track_proximity_settings(
    track_id: int,
    tolerance_meters: float,
    activity_type: str = "hiking",
    allow_extrapolation: bool = True,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configure proximity validation settings for a collaborative track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Only track owner can configure proximity settings
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only track owner can configure proximity settings")
    
    if not track.is_collaborative:
        raise HTTPException(status_code=400, detail="Track is not collaborative")
    
    # Validate tolerance range
    if not (1.0 <= tolerance_meters <= 10000.0):
        raise HTTPException(status_code=400, detail="Tolerance must be between 1m and 10km")
    
    # Store configuration in track metadata
    current_metadata = track.metadata_json or {}
    current_metadata["proximity_config"] = {
        "tolerance_meters": tolerance_meters,
        "activity_type": activity_type,
        "allow_extrapolation": allow_extrapolation,
        "configured_by": current_user.id,
        "configured_at": datetime.utcnow().isoformat()
    }
    
    track.metadata_json = current_metadata
    db.commit()
    
    # Log activity
    activity = TrackActivity(
        track_id=track_id,
        user_id=current_user.id,
        activity_type="proximity_configured",
        description=f"Set waypoint proximity tolerance to {tolerance_meters}m for {activity_type}"
    )
    db.add(activity)
    db.commit()
    
    return {
        "message": "Proximity settings configured successfully",
        "tolerance_meters": tolerance_meters,
        "activity_type": activity_type,
        "allow_extrapolation": allow_extrapolation,
        "track_id": track_id
    }

@router.get("/{track_id}/proximity-config", response_model=dict)
async def get_track_proximity_config(
    track_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current proximity validation configuration for a track"""
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_view:
        raise HTTPException(status_code=403, detail="No permission to view this track")
    
    # Get proximity config from metadata
    metadata = track.metadata_json or {}
    proximity_config = metadata.get("proximity_config")
    
    if not proximity_config:
        # Return default configuration
        default_config = ProximityPresets.HIKING_RELAXED
        return {
            "has_custom_config": False,
            "tolerance_meters": default_config.max_distance_meters,
            "activity_type": "hiking",
            "allow_extrapolation": default_config.allow_extrapolation,
            "is_default": True
        }
    
    return {
        "has_custom_config": True,
        "tolerance_meters": proximity_config["tolerance_meters"],
        "activity_type": proximity_config["activity_type"],
        "allow_extrapolation": proximity_config["allow_extrapolation"],
        "configured_at": proximity_config.get("configured_at"),
        "is_default": False
    }

@router.post("/{track_id}/debug-proximity", response_model=dict)
async def debug_waypoint_proximity_calculation(
    track_id: int,
    waypoint_lat: float,
    waypoint_lon: float,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Debug-Endpoint: Zeige detaillierte Berechnung der Waypoint-Proximity
    Implementiert den optimierten Algorithmus mit allen Kandidaten
    """
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check permissions
    permissions = get_user_permissions(track, current_user.id)
    if not permissions.can_view:
        raise HTTPException(status_code=403, detail="No permission to view this track")
    
    # Get track points (only owner's GPS data)
    track_points = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.created_by == track.created_by
    ).order_by(Waypoint.timestamp).all()
    
    if len(track_points) < 2:
        return {
            "error": "Track has insufficient GPS points for analysis",
            "required_points": 2,
            "available_points": len(track_points),
            "track_id": track_id
        }
    
    # Convert to TrackPoint objects
    track_point_objects = [
        TrackPoint(
            lat=tp.latitude,
            lon=tp.longitude,
            elevation=tp.altitude,
            index=i
        )
        for i, tp in enumerate(track_points)
    ]
    
    waypoint = Point(waypoint_lat, waypoint_lon)
    
    # Run debug analysis
    debug_result = debug_closest_point_calculation(waypoint, track_point_objects)
    
    # Add track metadata
    debug_result["track_metadata"] = {
        "track_id": track_id,
        "track_name": track.name,
        "owner_id": track.created_by,
        "is_collaborative": track.is_collaborative,
        "gps_points_count": len(track_points)
    }
    
    # Add current proximity config
    metadata = track.metadata or {}
    proximity_config = metadata.get("proximity_config", {})
    debug_result["current_proximity_config"] = {
        "tolerance_meters": proximity_config.get("tolerance_meters", 200.0),
        "activity_type": proximity_config.get("activity_type", "hiking"),
        "allow_extrapolation": proximity_config.get("allow_extrapolation", True)
    }
    
    # Add validation result
    if "best_match" in debug_result:
        tolerance = debug_result["current_proximity_config"]["tolerance_meters"]
        is_valid = debug_result["best_match"]["distance_meters"] <= tolerance
        debug_result["validation_result"] = {
            "is_valid": is_valid,
            "distance_vs_tolerance": f"{debug_result['best_match']['distance_meters']}m / {tolerance}m",
            "margin": tolerance - debug_result["best_match"]["distance_meters"]
        }
    
    return debug_result