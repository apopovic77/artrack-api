from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime

from ..database import get_db
from ..models import (
    Track, Waypoint, MediaFile, AnalysisJob, User,
    SyncStatus, QuotaInfo
)
from ..auth import get_current_user

router = APIRouter()

@router.get("/status", response_model=SyncStatus)
async def get_sync_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get sync status for current user"""
    
    # Count pending uploads (waypoints in pending state)
    pending_uploads = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        Waypoint.processing_state.in_(["pending", "uploading"])
    ).count()
    
    # Count pending analysis (waypoints being analyzed)
    pending_analysis = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        Waypoint.processing_state.in_(["uploaded", "analysing"])
    ).count()
    
    # Count failed uploads
    failed_uploads = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        Waypoint.processing_state == "failed"
    ).count()
    
    # Calculate storage usage
    storage_used = db.query(func.sum(MediaFile.file_size_bytes)).join(Waypoint).join(Track).filter(
        Track.created_by == current_user.id
    ).scalar() or 0
    
    quota_info = QuotaInfo(
        storage_bytes=storage_used,
        storage_limit=(getattr(current_user, 'storage_bytes_limit', 0) or 0),
        uploads_this_month=(getattr(current_user, 'uploads_this_month', 0) or 0),
        upload_limit=(getattr(current_user, 'upload_limit_per_month', 0) or 0),
    )
    
    return SyncStatus(
        user_id=current_user.id,
        last_sync_at=(current_user.last_active_at or datetime.utcnow()),
        pending_uploads=pending_uploads,
        pending_analysis=pending_analysis,
        failed_uploads=failed_uploads,
        quota_used=quota_info
    )

@router.post("/retry-failed")
async def retry_failed_uploads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retry failed uploads"""
    
    # Get failed waypoints for this user
    failed_waypoints = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        Waypoint.processing_state == "failed"
    ).all()
    
    retried_tasks = []
    
    for waypoint in failed_waypoints:
        # Reset waypoint status to pending
        waypoint.processing_state = "pending"
        
        # Get associated media files and reset their status
        media_files = db.query(MediaFile).filter(MediaFile.waypoint_id == waypoint.id).all()
        for media_file in media_files:
            media_file.processing_state = "pending"
        
        retried_tasks.append({
            "task_id": f"waypoint_{waypoint.id}",
            "new_status": "pending",
            "retry_attempt": waypoint.version + 1
        })
        
        waypoint.version += 1
    
    db.commit()
    
    return {
        "retried_tasks": retried_tasks,
        "errors": []
    }

@router.get("/user-stats")
async def get_user_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed user statistics"""
    
    # Track statistics
    total_tracks = db.query(Track).filter(Track.created_by == current_user.id).count()
    public_tracks = db.query(Track).filter(
        Track.created_by == current_user.id,
        Track.visibility == "public"
    ).count()
    
    # Waypoint statistics
    total_waypoints = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id
    ).count()
    
    published_waypoints = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        Waypoint.processing_state == "published"
    ).count()
    
    # Media statistics
    total_media = db.query(MediaFile).join(Waypoint).join(Track).filter(
        Track.created_by == current_user.id
    ).count()
    
    photos = db.query(MediaFile).join(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        MediaFile.media_type == "photo"
    ).count()
    
    videos = db.query(MediaFile).join(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        MediaFile.media_type == "video"
    ).count()
    
    audio = db.query(MediaFile).join(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        MediaFile.media_type == "audio"
    ).count()
    
    # Storage usage
    storage_used = db.query(func.sum(MediaFile.file_size_bytes)).join(Waypoint).join(Track).filter(
        Track.created_by == current_user.id
    ).scalar() or 0
    
    # Recent activity (last 30 days)
    from datetime import timedelta
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    recent_waypoints = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        Waypoint.created_at >= thirty_days_ago
    ).count()
    
    return {
        "user_info": {
            "id": current_user.id,
            "display_name": current_user.display_name,
            "trust_level": current_user.trust_level,
            "member_since": current_user.created_at
        },
        "track_stats": {
            "total_tracks": total_tracks,
            "public_tracks": public_tracks,
            "private_tracks": total_tracks - public_tracks
        },
        "waypoint_stats": {
            "total_waypoints": total_waypoints,
            "published_waypoints": published_waypoints,
            "pending_waypoints": total_waypoints - published_waypoints
        },
        "media_stats": {
            "total_media": total_media,
            "photos": photos,
            "videos": videos,
            "audio": audio
        },
        "storage_stats": {
            "used_bytes": storage_used,
            "used_mb": round(storage_used / (1024 * 1024), 2),
            "limit_bytes": current_user.storage_bytes_limit,
            "limit_mb": round(current_user.storage_bytes_limit / (1024 * 1024), 2),
            "usage_percentage": round((storage_used / current_user.storage_bytes_limit) * 100, 2)
        },
        "activity_stats": {
            "recent_waypoints_30d": recent_waypoints,
            "uploads_this_month": current_user.uploads_this_month,
            "upload_limit": current_user.upload_limit_per_month
        }
    }

@router.post("/cleanup")
async def cleanup_failed_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clean up failed uploads and orphaned data"""
    
    cleanup_results = {
        "deleted_waypoints": 0,
        "deleted_media_files": 0,
        "deleted_analysis_jobs": 0,
        "freed_storage_bytes": 0
    }
    
    # Find waypoints that have been in failed state for more than 24 hours
    from datetime import timedelta
    cleanup_threshold = datetime.utcnow() - timedelta(hours=24)
    
    old_failed_waypoints = db.query(Waypoint).join(Track).filter(
        Track.created_by == current_user.id,
        Waypoint.processing_state == "failed",
        Waypoint.updated_at < cleanup_threshold
    ).all()
    
    from storage import service as storage_service
    
    for waypoint in old_failed_waypoints:
        # Delete associated media files
        media_files = db.query(MediaFile).filter(MediaFile.waypoint_id == waypoint.id).all()
        for media_file in media_files:
            # Delete physical file
            if storage_service.delete_file(media_file.file_path):
                cleanup_results["freed_storage_bytes"] += media_file.file_size_bytes
            
            # Delete from database
            db.delete(media_file)
            cleanup_results["deleted_media_files"] += 1
        
        # Delete failed analysis jobs
        failed_jobs = db.query(AnalysisJob).filter(
            AnalysisJob.media_file_id.in_([mf.id for mf in media_files])
        ).all()
        
        for job in failed_jobs:
            db.delete(job)
            cleanup_results["deleted_analysis_jobs"] += 1
        
        # Delete waypoint
        db.delete(waypoint)
        cleanup_results["deleted_waypoints"] += 1
    
    db.commit()
    
    return cleanup_results

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "artrack-api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }