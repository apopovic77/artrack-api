from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
from datetime import datetime
import math

from ..database import get_db
from ..auth import get_current_user
from ..models import Track, TrackRoute as TrackRouteModel, Waypoint
from pydantic import BaseModel
from .track_report_generator import generate_track_report

router = APIRouter()

# Helper function to calculate distance
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters"""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def _closest_point_on_polyline(poly: List[Tuple[float, float]], lat: float, lon: float) -> Tuple[float, float, float, float]:
    """
    Calculate closest point on polyline to given coordinate.
    Returns (distance_meters, along_meters, snap_lat, snap_lon).

    KORREKTE IMPLEMENTIERUNG - Single Source of Truth:

    1. Sammle alle orthogonalen Projektionen auf Segmente (NUR wo 0 ≤ t ≤ 1)
    2. Sammle alle direkten Distanzen zu GPS-Punkten (Vektoren)
    3. Minimum von allen = Ergebnis
    """
    if len(poly) < 2:
        return (float('inf'), 0.0, lat, lon)

    # Convert lat/lon to approximate meters (good enough for small areas)
    # Using equirectangular approximation
    ref_lat = poly[0][0]
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))

    def to_meters(lat_a, lon_a):
        """Convert lat/lon to meters relative to first point"""
        return (
            (lat_a - poly[0][0]) * meters_per_deg_lat,
            (lon_a - poly[0][1]) * meters_per_deg_lon
        )

    # Point P in meters
    px, py = to_meters(lat, lon)

    # Collect all candidate distances
    candidates = []

    total_along = 0.0

    # 1. Check orthogonal projections on segments
    for i in range(len(poly) - 1):
        # Segment endpoints A and B
        seg_start_lat, seg_start_lon = poly[i]
        seg_end_lat, seg_end_lon = poly[i + 1]

        # Convert to meters
        ax, ay = to_meters(seg_start_lat, seg_start_lon)
        bx, by = to_meters(seg_end_lat, seg_end_lon)

        # Vector AB
        abx = bx - ax
        aby = by - ay

        # Vector AP
        apx = px - ax
        apy = py - ay

        # Segment length squared
        ab_len_sq = abx * abx + aby * aby

        if ab_len_sq >= 0.01:  # Ignore very short segments (< 10cm)
            # Projection parameter t = dot(AP, AB) / dot(AB, AB)
            t = (apx * abx + apy * aby) / ab_len_sq

            # ONLY use projection if WITHIN segment (0 ≤ t ≤ 1)
            if 0 <= t <= 1:
                # Projected point Q = A + t * AB
                qx = ax + t * abx
                qy = ay + t * aby

                # Distance from P to Q (orthogonal distance)
                pqx = px - qx
                pqy = py - qy
                dist = math.sqrt(pqx * pqx + pqy * pqy)

                proj_along = total_along + t * math.sqrt(ab_len_sq)

                # Convert projected point back to lat/lon
                snap_x = qx + poly[0][0] * meters_per_deg_lat / meters_per_deg_lat
                snap_y = qy + poly[0][1] * meters_per_deg_lon / meters_per_deg_lon
                snap_lat = poly[0][0] + qx / meters_per_deg_lat
                snap_lon = poly[0][1] + qy / meters_per_deg_lon

                candidates.append((dist, proj_along, snap_lat, snap_lon))

        # Accumulate distance along polyline
        total_along += math.sqrt(ab_len_sq)

    # 2. Check direct distances to all GPS points (Vektoren)
    total_along = 0.0
    for i, (point_lat, point_lon) in enumerate(poly):
        vx, vy = to_meters(point_lat, point_lon)

        # Direct distance from P to this GPS point
        dx = px - vx
        dy = py - vy
        dist = math.sqrt(dx * dx + dy * dy)

        # Snap position is the GPS point itself
        candidates.append((dist, total_along, point_lat, point_lon))

        # Accumulate for next point
        if i < len(poly) - 1:
            next_lat, next_lon = poly[i + 1]
            nx, ny = to_meters(next_lat, next_lon)
            seg_len = math.sqrt((nx - vx)**2 + (ny - vy)**2)
            total_along += seg_len

    # 3. Find minimum
    if not candidates:
        return (float('inf'), 0.0, lat, lon)

    min_dist, best_along, snap_lat, snap_lon = min(candidates, key=lambda x: x[0])
    return (min_dist, best_along, snap_lat, snap_lon)


def _waypoint_belongs_to_route(wp: Waypoint, route_id: int, track_id: int, db: Session, track: Track, max_snap_distance: float = 50.0) -> bool:
    """
    RUNTIME evaluation: Does this waypoint belong to this route?

    Priority:
    1. Manual override: Check segment.routeId (for segment markers)
    2. Manual override: Check fixedRoutes (for POIs)
    3. Auto-snap: Calculate distance to route polyline using SNAP algorithm

    Single Source of Truth: Uses same snap algorithm as snap_routes.py
    """
    meta = wp.metadata_json or {}

    # Priority 1: segment.routeId (old manual assignment for segment markers)
    if meta.get("segment"):
        seg_route_id = meta.get("segment", {}).get("routeId")
        if seg_route_id == route_id:
            return True

    # Priority 2: Manual override via fixedRoutes
    fixed_routes = meta.get("fixedRoutes", [])
    if fixed_routes:
        return f"route_{route_id}" in fixed_routes

    # Priority 3: Runtime SNAP calculation
    # Check distance to ALL routes and assign to CLOSEST one only
    # (Multi-route only if snap positions are very close together, like shared paths)

    # Get all routes for this track
    all_routes = db.query(TrackRouteModel).filter(TrackRouteModel.track_id == track_id).all()

    if not all_routes:
        return False

    # Get all GPS points once
    all_gps = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.waypoint_type == "gps_track",
        Waypoint.created_by == track.created_by
    ).order_by(Waypoint.timestamp.asc()).all()

    # Calculate distance to each route
    route_distances = []

    for r in all_routes:
        # Filter GPS points for this route
        gps_points = []
        for p in all_gps:
            p_route_id = getattr(p, 'route_id', None)
            if p_route_id == r.id:
                gps_points.append(p)
            elif p.metadata_json:
                meta_route = p.metadata_json.get('route_id')
                if meta_route == r.id:
                    gps_points.append(p)

        if len(gps_points) >= 2:
            polyline = [(p.latitude, p.longitude) for p in gps_points]
            dist_meters, along_meters, snap_lat, snap_lon = _closest_point_on_polyline(polyline, wp.latitude, wp.longitude)

            if dist_meters <= max_snap_distance:
                route_distances.append({
                    'route_id': r.id,
                    'distance': dist_meters,
                    'along_meters': along_meters,
                    'snap_lat': snap_lat,
                    'snap_lon': snap_lon
                })

    if not route_distances:
        # No route within threshold
        return False

    # Sort by distance (closest first)
    route_distances.sort(key=lambda x: x['distance'])

    # Check if this is the closest route
    closest_route_id = route_distances[0]['route_id']
    closest_snap_lat = route_distances[0]['snap_lat']
    closest_snap_lon = route_distances[0]['snap_lon']

    # Multi-route detection: Only if distances from POI to routes are equal
    # If POI is 0.5m from Route 1 and 0.5m from Route 2 → routes overlap → POI on both
    if len(route_distances) > 1:
        # Check if other routes have similar distance from POI (< 2m difference)
        closest_distance = route_distances[0]['distance']

        for rd in route_distances[1:]:
            dist_diff = abs(rd['distance'] - closest_distance)

            if dist_diff < 2.0:
                # POI is equally close to both routes → shared path
                # Return true if our route_id is one of them
                if rd['route_id'] == route_id or closest_route_id == route_id:
                    return True

    # Otherwise: waypoint only belongs to the single closest route
    return closest_route_id == route_id

class RouteCreate(BaseModel):
    name: str
    color: Optional[str] = None
    description: Optional[str] = None
    storage_object_ids: Optional[list[int]] = None
    storage_collection: Optional[dict] = None  # { name: string, owner_email: string }

class RouteUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    storage_object_ids: Optional[list[int]] = None
    storage_collection: Optional[dict] = None

class RouteItem(BaseModel):
    id: int
    track_id: int
    name: str
    color: Optional[str]
    description: Optional[str] = None
    storage_object_ids: Optional[list[int]] = None
    storage_collection: Optional[dict] = None
    created_at: datetime

# Detailed route data response
class WaypointPoint(BaseModel):
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    recorded_at: datetime
    along_meters: float  # Cumulative distance along route

class SegmentDetail(BaseModel):
    id: int
    name: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    waypoints: List[WaypointPoint]
    distance_meters: float
    metadata_json: Optional[dict] = None

class RouteDetail(BaseModel):
    id: int
    track_id: int
    name: str
    color: Optional[str]
    description: Optional[str]
    storage_object_ids: Optional[list[int]]
    storage_collection: Optional[dict]
    created_at: datetime
    segments: List[SegmentDetail]
    total_distance_meters: float
    total_waypoints: int

# Simplified route overview response
class WaypointOverview(BaseModel):
    id: int
    waypoint_type: str
    latitude: float
    longitude: float
    user_description: Optional[str]
    metadata_json: Optional[dict]

class SegmentOverview(BaseModel):
    name: str
    start_waypoint: WaypointOverview
    end_waypoint: WaypointOverview

class RouteOverview(BaseModel):
    id: int
    track_id: int
    name: str
    color: Optional[str]
    description: Optional[str]
    segments: List[SegmentOverview]
    waypoints: List[WaypointOverview]  # All waypoints (POIs) for this route
    total_segments: int
    total_waypoints: int

@router.get("/{track_id}/routes", response_model=List[RouteItem])
async def list_routes(track_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    routes = db.query(TrackRouteModel).filter(TrackRouteModel.track_id == track_id).order_by(TrackRouteModel.id.asc()).all()
    return [RouteItem(id=r.id, track_id=r.track_id, name=r.name, color=r.color, description=r.description, storage_object_ids=getattr(r, 'storage_object_ids', None), storage_collection=getattr(r, 'storage_collection', None), created_at=r.created_at) for r in routes]

@router.get("/{track_id}/routes/{route_id}/overview", response_model=RouteOverview)
async def get_route_overview(
    track_id: int,
    route_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get route overview with all waypoints (POIs) and segments.

    Returns:
    - Route metadata
    - All segments (start/end waypoint pairs)
    - All POIs/waypoints assigned to this route
    """
    # Verify track access
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get route
    route = db.query(TrackRouteModel).filter(
        TrackRouteModel.id == route_id,
        TrackRouteModel.track_id == track_id
    ).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Load all waypoints (we'll filter GPS track points later)
    # Note: Can't use waypoint_type != "gps_track" because NULL != 'gps_track' returns NULL in SQL
    all_waypoints = db.query(Waypoint).filter(
        Waypoint.track_id == track_id
    ).all()

    # Filter out GPS track points in Python
    all_waypoints = [wp for wp in all_waypoints if wp.waypoint_type != "gps_track"]

    # Filter segment markers for this route
    # Priority: 1) segment.routeId (manual), 2) fixedRoutes (manual), 3) RUNTIME SNAP
    segment_waypoints = []
    for wp in all_waypoints:
        if not (wp.metadata_json and wp.metadata_json.get("segment")):
            continue

        # Check if segment belongs to route (same logic as POIs)
        if _waypoint_belongs_to_route(wp, route_id, track_id, db, track):
            segment_waypoints.append(wp)

    # Group segments by name
    segments_by_name = {}
    for wp in segment_waypoints:
        segment_meta = wp.metadata_json.get("segment", {})
        seg_name = segment_meta.get("name")
        role = segment_meta.get("role")

        if not seg_name or not role:
            continue

        if seg_name not in segments_by_name:
            segments_by_name[seg_name] = {}

        segments_by_name[seg_name][role] = wp

    # Build segment overview
    segments_overview = []
    for seg_name, wps in sorted(segments_by_name.items()):
        if "start" not in wps or "end" not in wps:
            continue

        start_wp = wps["start"]
        end_wp = wps["end"]

        segments_overview.append(SegmentOverview(
            name=seg_name,
            start_waypoint=WaypointOverview(
                id=start_wp.id,
                waypoint_type=start_wp.waypoint_type,
                latitude=start_wp.latitude,
                longitude=start_wp.longitude,
                user_description=start_wp.user_description,
                metadata_json=start_wp.metadata_json
            ),
            end_waypoint=WaypointOverview(
                id=end_wp.id,
                waypoint_type=end_wp.waypoint_type,
                latitude=end_wp.latitude,
                longitude=end_wp.longitude,
                user_description=end_wp.user_description,
                metadata_json=end_wp.metadata_json
            )
        ))

    # Filter POIs/waypoints for this route
    # Priority: 1) fixedRoutes (manual override), 2) RUNTIME SNAP calculation
    route_waypoints = []
    for wp in all_waypoints:
        # Skip segment markers
        if wp.metadata_json and wp.metadata_json.get("segment"):
            continue

        # Check if belongs to route (runtime evaluation)
        if _waypoint_belongs_to_route(wp, route_id, track_id, db, track):
            route_waypoints.append(WaypointOverview(
                id=wp.id,
                waypoint_type=wp.waypoint_type,
                latitude=wp.latitude,
                longitude=wp.longitude,
                user_description=wp.user_description,
                metadata_json=wp.metadata_json
            ))

    return RouteOverview(
        id=route.id,
        track_id=route.track_id,
        name=route.name,
        color=route.color,
        description=route.description,
        segments=segments_overview,
        waypoints=route_waypoints,
        total_segments=len(segments_overview),
        total_waypoints=len(route_waypoints)
    )

@router.get("/{track_id}/routes/{route_id}/detail", response_model=RouteDetail)
async def get_route_detail(
    track_id: int,
    route_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get complete route data including all segments and waypoints.

    Returns:
    - Route metadata (name, color, description, storage objects)
    - All segments belonging to this route
    - All GPS waypoints for each segment (ordered by timestamp)
    - Distance calculations
    """
    # Verify track access
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get route
    route = db.query(TrackRouteModel).filter(
        TrackRouteModel.id == route_id,
        TrackRouteModel.track_id == track_id
    ).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # NEW SEGMENT SYSTEM: Load segments from waypoint pairs
    # Segments are defined by two waypoints (start/end) with metadata_json.segment = {name, role, routeId}

    segment_details = []
    total_distance = 0.0
    total_waypoints = 0
    cumulative_distance = 0.0

    # Load all non-GPS waypoints to find segment markers
    all_waypoints = db.query(Waypoint).filter(
        Waypoint.track_id == track_id,
        Waypoint.waypoint_type != "gps_track"
    ).all()

    # Filter segment markers (waypoints with metadata_json.segment)
    segment_waypoints = [
        wp for wp in all_waypoints
        if wp.metadata_json and wp.metadata_json.get("segment")
    ]

    # Filter segment waypoints for this route
    # Priority: 1) segment.routeId (manual assignment), 2) fixedRoutes (auto-snap)
    route_segment_wps = [
        wp for wp in segment_waypoints
        if (wp.metadata_json.get("segment", {}).get("routeId") == route_id
            or f"route_{route_id}" in wp.metadata_json.get("fixedRoutes", []))
    ]

    # Group by segment name
    segments_by_name = {}
    for wp in route_segment_wps:
        segment_meta = wp.metadata_json.get("segment", {})
        seg_name = segment_meta.get("name")
        role = segment_meta.get("role")

        if not seg_name or not role:
            continue

        if seg_name not in segments_by_name:
            segments_by_name[seg_name] = {}

        segments_by_name[seg_name][role] = wp

    # Check if route has segments defined
    if segments_by_name:
        # Route has segments - load GPS waypoints for the route
        all_route_waypoints = db.query(Waypoint).filter(
            Waypoint.track_id == track_id,
            Waypoint.route_id == route_id,
            Waypoint.waypoint_type == "gps_track"
        ).order_by(Waypoint.recorded_at.asc()).all()

        # Build segments from waypoint pairs
        for seg_name, wps in sorted(segments_by_name.items()):
            if "start" not in wps or "end" not in wps:
                continue

            start_wp = wps["start"]
            end_wp = wps["end"]

            # Find waypoints between start and end
            # Simple approach: use timestamps
            segment_waypoints = [
                wp for wp in all_route_waypoints
                if start_wp.recorded_at <= wp.recorded_at <= end_wp.recorded_at
            ]

            waypoint_points = []
            segment_distance = 0.0

            for i, wp in enumerate(segment_waypoints):
                if i > 0:
                    prev = segment_waypoints[i - 1]
                    dist = _haversine(prev.latitude, prev.longitude, wp.latitude, wp.longitude)
                    segment_distance += dist
                    cumulative_distance += dist

                waypoint_points.append(WaypointPoint(
                    latitude=wp.latitude,
                    longitude=wp.longitude,
                    altitude=wp.altitude,
                    recorded_at=wp.recorded_at,
                    along_meters=cumulative_distance
                ))

            total_distance += segment_distance
            total_waypoints += len(segment_waypoints)

            segment_details.append(SegmentDetail(
                id=hash(seg_name) % 10000,  # Virtual ID based on name
                name=seg_name,
                started_at=start_wp.recorded_at,
                ended_at=end_wp.recorded_at,
                waypoints=waypoint_points,
                distance_meters=segment_distance,
                metadata_json={"start_waypoint_id": start_wp.id, "end_waypoint_id": end_wp.id}
            ))
    else:
        # No segments - load all waypoints directly assigned to this route
        waypoints = db.query(Waypoint).filter(
            Waypoint.track_id == track_id,
            Waypoint.route_id == route_id,
            Waypoint.waypoint_type == "gps_track"
        ).order_by(Waypoint.recorded_at.asc()).all()

        if waypoints:
            # Create a virtual segment containing all route waypoints
            waypoint_points = []
            route_distance = 0.0

            for i, wp in enumerate(waypoints):
                if i > 0:
                    prev = waypoints[i - 1]
                    dist = _haversine(prev.latitude, prev.longitude, wp.latitude, wp.longitude)
                    route_distance += dist
                    cumulative_distance += dist

                waypoint_points.append(WaypointPoint(
                    latitude=wp.latitude,
                    longitude=wp.longitude,
                    altitude=wp.altitude,
                    recorded_at=wp.recorded_at,
                    along_meters=cumulative_distance
                ))

            total_distance = route_distance
            total_waypoints = len(waypoints)

            # Create virtual segment with all route waypoints
            segment_details.append(SegmentDetail(
                id=0,  # Virtual segment ID
                name=f"{route.name} (full route)",
                started_at=waypoints[0].recorded_at,
                ended_at=waypoints[-1].recorded_at,
                waypoints=waypoint_points,
                distance_meters=route_distance,
                metadata_json={"virtual": True, "note": "Route has no segments, all waypoints grouped together"}
            ))

    return RouteDetail(
        id=route.id,
        track_id=route.track_id,
        name=route.name,
        color=route.color,
        description=route.description,
        storage_object_ids=getattr(route, 'storage_object_ids', None),
        storage_collection=getattr(route, 'storage_collection', None),
        created_at=route.created_at,
        segments=segment_details,
        total_distance_meters=total_distance,
        total_waypoints=total_waypoints
    )

@router.post("/{track_id}/routes", response_model=RouteItem)
async def create_route(track_id: int, body: RouteCreate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    r = TrackRouteModel(track_id=track_id, created_by=current_user.id, name=body.name, color=body.color, description=body.description, storage_object_ids=body.storage_object_ids or [], storage_collection=body.storage_collection or {})
    db.add(r)
    db.commit()
    db.refresh(r)
    return RouteItem(id=r.id, track_id=r.track_id, name=r.name, color=r.color, description=r.description, storage_object_ids=getattr(r, 'storage_object_ids', None), storage_collection=getattr(r, 'storage_collection', None), created_at=r.created_at)

@router.patch("/{track_id}/routes/{route_id}", response_model=RouteItem)
async def update_route(track_id: int, route_id: int, body: RouteUpdate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    r = db.query(TrackRouteModel).filter(TrackRouteModel.id == route_id, TrackRouteModel.track_id == track_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Route not found")
    if body.name is not None:
        r.name = body.name
    if body.color is not None:
        r.color = body.color
    if body.description is not None:
        r.description = body.description
    if body.storage_object_ids is not None:
        r.storage_object_ids = body.storage_object_ids
    if body.storage_collection is not None:
        # minimal validation
        try:
            sc = body.storage_collection or {}
            name = (sc.get('name') or '').strip()
            owner = (sc.get('owner_email') or '').strip()
            r.storage_collection = {'name': name, 'owner_email': owner}
        except Exception:
            r.storage_collection = body.storage_collection
    db.commit()
    db.refresh(r)
    return RouteItem(id=r.id, track_id=r.track_id, name=r.name, color=r.color, description=r.description, storage_object_ids=getattr(r, 'storage_object_ids', None), storage_collection=getattr(r, 'storage_collection', None), created_at=r.created_at)

@router.delete("/{track_id}/routes/{route_id}", response_model=dict)
async def delete_route(track_id: int, route_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    r = db.query(TrackRouteModel).filter(TrackRouteModel.id == route_id, TrackRouteModel.track_id == track_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Route not found")
    # Also delete GPS points belonging to this route to avoid orphan gps_track dots
    deleted_count = 0
    try:
        points = db.query(Waypoint).filter(
            Waypoint.track_id == track_id,
            Waypoint.waypoint_type == "gps_track",
            Waypoint.created_by == track.created_by
        ).all()
        for p in points:
            meta = p.metadata_json or {}
            pr = getattr(p, 'route_id', None)
            mr = meta.get('route_id') if isinstance(meta, dict) else None
            if pr == route_id or mr == route_id:
                db.delete(p)
                deleted_count += 1
        # Finally delete the route record
        db.delete(r)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"deleted": True, "gps_points_deleted": deleted_count}


@router.get("/{track_id}/structure-report")
async def get_track_structure_report(
    track_id: int,
    full: bool = False,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get a text-based structure report for a track showing routes, segments, POIs, and overlaps.

    Query parameters:
    - full: If true, includes descriptions for track, routes, segments, and POIs
    """
    # Check track permissions
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Generate report using shared logic
    report_text = generate_track_report(
        track_id=track_id,
        show_descriptions=full,
        api_key="Inetpass1",
        base_url="https://api.arkturian.com/artrack"
    )

    return PlainTextResponse(report_text)


@router.get("/{track_id}/routes-overview")
async def get_track_routes_overview(
    track_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get complete routes overview with POI assignments for a track.

    Returns JSON with all routes and their assigned POIs based on runtime snap algorithm.
    This is the single source of truth for POI-to-route assignments.
    """
    # Check track permissions
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get all routes for this track
    routes = db.query(TrackRouteModel).filter(TrackRouteModel.track_id == track_id).all()

    routes_data = []

    for route in routes:
        route_id = route.id

        # Get GPS points count
        gps_count = db.query(Waypoint).filter(
            Waypoint.track_id == track_id,
            Waypoint.route_id == route_id,
            Waypoint.waypoint_type == "gps_track"
        ).count()

        # Get route overview (internally uses the same logic as /overview endpoint)
        all_waypoints = db.query(Waypoint).filter(
            Waypoint.track_id == track_id
        ).all()

        # Filter out GPS track points
        all_waypoints = [wp for wp in all_waypoints if wp.waypoint_type != "gps_track"]

        # Get POIs for this route (using _waypoint_belongs_to_route logic)
        route_pois = []
        for wp in all_waypoints:
            if wp.metadata_json and wp.metadata_json.get("segment"):
                continue  # Skip segment markers
            if _waypoint_belongs_to_route(wp, route_id, track_id, db, track):
                # Get GPS points for distance calculation
                gps_points = db.query(Waypoint).filter(
                    Waypoint.track_id == track_id,
                    Waypoint.route_id == route_id,
                    Waypoint.waypoint_type == "gps_track"
                ).order_by(Waypoint.timestamp.asc()).all()

                polyline = [(p.latitude, p.longitude) for p in gps_points]
                along_meters = 0

                if len(polyline) >= 2:
                    dist, along_meters, _, _ = _closest_point_on_polyline(
                        polyline, wp.latitude, wp.longitude
                    )

                meta = wp.metadata_json or {}
                route_pois.append({
                    "id": wp.id,
                    "title": meta.get("title", "N/A"),
                    "latitude": wp.latitude,
                    "longitude": wp.longitude,
                    "distance_along_route_m": round(along_meters, 1),
                    "user_description": wp.user_description,
                    "metadata": meta
                })

        # Sort POIs by distance along route
        route_pois.sort(key=lambda x: x["distance_along_route_m"])

        # Get segments count
        segment_waypoints = [wp for wp in all_waypoints
                           if wp.metadata_json and wp.metadata_json.get("segment")
                           and _waypoint_belongs_to_route(wp, route_id, track_id, db, track)]

        segments_by_name = {}
        for wp in segment_waypoints:
            segment_meta = wp.metadata_json.get("segment", {})
            seg_name = segment_meta.get("name")
            role = segment_meta.get("role")
            if seg_name and role:
                if seg_name not in segments_by_name:
                    segments_by_name[seg_name] = {}
                segments_by_name[seg_name][role] = wp

        complete_segments = sum(1 for wps in segments_by_name.values()
                              if "start" in wps and "end" in wps)

        routes_data.append({
            "route_id": route_id,
            "route_name": route.name,
            "route_description": route.description,
            "route_color": route.color,
            "gps_points_count": gps_count,
            "segments_count": complete_segments,
            "pois_count": len(route_pois),
            "pois": route_pois
        })

    return {
        "track_id": track_id,
        "track_name": track.name,
        "track_description": track.description,
        "routes_count": len(routes),
        "routes": routes_data
    }


class GenerateIntroRequest(BaseModel):
    dry_run: bool = False
    custom_text: Optional[str] = None

@router.post("/{track_id}/routes/{route_id}/generate-intro")
async def generate_route_intro(
    track_id: int,
    route_id: int,
    body: GenerateIntroRequest = GenerateIntroRequest(),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Generate intro audio for a route.

    Uses the track's guide config to generate a welcome/intro audio that plays when
    the user selects this route in the app.
    """
    from openai import OpenAI
    import requests as req
    from datetime import datetime
    import os

    # Check track permissions
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if track.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get route
    route = db.query(TrackRouteModel).filter(
        TrackRouteModel.id == route_id,
        TrackRouteModel.track_id == track_id
    ).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Get guide config from metadata_json
    metadata = track.metadata_json or {}
    # Fallback to 'guide' if 'guide_config' is missing (supports both old and new format)
    guide_config = metadata.get('guide', metadata.get('guide_config', {}))

    # Initialize OpenAI client
    client = OpenAI()

    # Generate intro text
    if body.custom_text:
        intro_text = body.custom_text
    else:
        # Generate intro text using GPT
        prompt = f"""Du bist ein Wanderführer. Generiere eine kurze, freundliche Begrüßung (max 30 Sekunden Sprechzeit) für die Route "{route.name}".

Track: {track.name}
Route: {route.name}
{f'Beschreibung: {route.description}' if route.description else ''}

Die Begrüßung soll:
- Den Wanderer willkommen heißen
- Die Route kurz vorstellen
- Motivation geben
- Freundlich und einladend sein

Sprich direkt den Wanderer an. Keine Metainformationen, nur den gesprochenen Text."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        intro_text = response.choices[0].message.content

    if body.dry_run:
        return {
            "route_id": route_id,
            "route_name": route.name,
            "intro_text": intro_text,
            "audio_url": None
        }

    # Generate audio using OpenAI TTS
    try:
        voice = guide_config.get('voice', {}).get('style', 'nova')
        if voice == 'warm_guide':
            voice = 'nova'

        speech_response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=intro_text,
            speed=guide_config.get('voice', {}).get('speed', 0.9)
        )

        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"route_{route_id}_intro_{timestamp}.mp3"
        # Use a more robust path handling
        base_path = "/var/www/audio/route_intros"
        filepath = os.path.join(base_path, filename)

        # Ensure directory exists
        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path, exist_ok=True)
                # Set permissions if we created it (best effort)
                try:
                    import shutil
                    os.chmod(base_path, 0o775)
                except:
                    pass
            except OSError as e:
                raise HTTPException(status_code=500, detail=f"Server storage error: Cannot create directory {base_path}. {str(e)}")

        with open(filepath, 'wb') as f:
            f.write(speech_response.content)

        # Ensure file is readable by web server
        try:
            os.chmod(filepath, 0o644)
        except:
            pass

        audio_url = f"https://api.arkturian.com/audio/route_intros/{filename}"

        # Store intro_audio_url in route metadata
        route.intro_audio_url = audio_url
        db.commit()

        return {
            "route_id": route_id,
            "route_name": route.name,
            "intro_text": intro_text,
            "audio_url": audio_url
        }

    except Exception as e:
        # Log the full error
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Audio Generation Error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

