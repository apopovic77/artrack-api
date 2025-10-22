"""
Track Geometry & Waypoint Proximity Validation
Berechnung der kürzesten Distanz von einem Punkt zu einer Track-Polyline
"""

import math
from typing import List, Tuple, Optional, Dict
from pydantic import BaseModel
from dataclasses import dataclass
import numpy as np

# === Data Models ===

@dataclass
class Point:
    """2D Point with latitude/longitude"""
    lat: float
    lon: float
    
    def __post_init__(self):
        # Validate coordinates
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"Invalid latitude: {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"Invalid longitude: {self.lon}")

@dataclass
class TrackPoint:
    """Track point with additional metadata"""
    lat: float
    lon: float
    elevation: Optional[float] = None
    timestamp: Optional[str] = None
    index: Optional[int] = None

@dataclass
class ProjectionResult:
    """Result of point projection onto track segment"""
    closest_point: Point
    distance_meters: float
    track_segment_index: int
    projection_ratio: float  # 0.0 = start of segment, 1.0 = end of segment
    is_valid: bool
    track_position_km: float  # Position along track in kilometers

class WaypointProximityConfig(BaseModel):
    """Configuration for waypoint proximity validation"""
    max_distance_meters: float = 100.0  # Default: 100m tolerance
    use_elevation: bool = False  # Include elevation in distance calculation
    allow_extrapolation: bool = False  # Allow waypoints beyond track start/end
    min_track_points: int = 2  # Minimum track points required
    
class ProximityValidationResult(BaseModel):
    """Result of waypoint proximity validation"""
    is_valid: bool
    distance_to_track: float
    closest_track_point: Dict[str, float]
    track_segment_index: int
    projection_ratio: float
    track_position_km: float
    error_message: Optional[str] = None

# === Core Geometry Functions ===

def haversine_distance(point1: Point, point2: Point) -> float:
    """
    Berechne Haversine-Distanz zwischen zwei GPS-Punkten in Metern
    """
    R = 6371000  # Earth radius in meters
    
    lat1_rad = math.radians(point1.lat)
    lat2_rad = math.radians(point2.lat)
    delta_lat = math.radians(point2.lat - point1.lat)
    delta_lon = math.radians(point2.lon - point1.lon)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def bearing_between_points(point1: Point, point2: Point) -> float:
    """
    Berechne Bearing (Richtung) zwischen zwei Punkten in Grad
    """
    lat1_rad = math.radians(point1.lat)
    lat2_rad = math.radians(point2.lat)
    delta_lon = math.radians(point2.lon - point1.lon)
    
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

def point_to_line_distance_spherical(point: Point, line_start: Point, line_end: Point) -> ProjectionResult:
    """
    Berechne kürzeste Distanz von einem Punkt zu einem Liniensegment auf der Erdkugel
    Verwendet orthogonale Projektion mit sphärischer Geometrie
    """
    
    # Konvertiere zu kartesischen Koordinaten für bessere Genauigkeit
    def to_cartesian(p: Point) -> Tuple[float, float, float]:
        lat_rad = math.radians(p.lat)
        lon_rad = math.radians(p.lon)
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        return (x, y, z)
    
    def from_cartesian(x: float, y: float, z: float) -> Point:
        lat = math.degrees(math.asin(z))
        lon = math.degrees(math.atan2(y, x))
        return Point(lat, lon)
    
    # Konvertiere Punkte zu kartesischen Koordinaten
    p_cart = to_cartesian(point)
    a_cart = to_cartesian(line_start)
    b_cart = to_cartesian(line_end)
    
    # Vektor von A nach B
    ab = (b_cart[0] - a_cart[0], b_cart[1] - a_cart[1], b_cart[2] - a_cart[2])
    
    # Vektor von A nach P
    ap = (p_cart[0] - a_cart[0], p_cart[1] - a_cart[1], p_cart[2] - a_cart[2])
    
    # Projektion von AP auf AB
    ab_length_sq = ab[0]**2 + ab[1]**2 + ab[2]**2
    
    if ab_length_sq == 0:
        # Line segment is a point
        closest_point = line_start
        distance = haversine_distance(point, closest_point)
        return ProjectionResult(
            closest_point=closest_point,
            distance_meters=distance,
            track_segment_index=0,
            projection_ratio=0.0,
            is_valid=True,
            track_position_km=0.0
        )
    
    # Parametrischer Wert t für die Projektion
    t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_length_sq
    
    # Begrenze t auf [0, 1] für Punkt auf dem Segment
    t_clamped = max(0.0, min(1.0, t))
    
    # Berechne den nächsten Punkt auf dem Segment
    closest_cart = (
        a_cart[0] + t_clamped * ab[0],
        a_cart[1] + t_clamped * ab[1],
        a_cart[2] + t_clamped * ab[2]
    )
    
    # Normalisiere für Rückkonvertierung zur Kugeloberfläche
    length = math.sqrt(closest_cart[0]**2 + closest_cart[1]**2 + closest_cart[2]**2)
    closest_cart = (closest_cart[0]/length, closest_cart[1]/length, closest_cart[2]/length)
    
    # Konvertiere zurück zu GPS-Koordinaten
    closest_point = from_cartesian(*closest_cart)
    
    # Berechne Distanz
    distance = haversine_distance(point, closest_point)
    
    return ProjectionResult(
        closest_point=closest_point,
        distance_meters=distance,
        track_segment_index=0,
        projection_ratio=t_clamped,
        is_valid=0.0 <= t <= 1.0,  # True wenn Punkt tatsächlich auf Segment projiziert
        track_position_km=0.0  # Wird später berechnet
    )

def find_closest_point_on_track(waypoint: Point, track_points: List[TrackPoint]) -> ProjectionResult:
    """
    Optimierter Algorithmus für kürzeste Distanz zu Track-Polyline:
    
    1. Direkte Distanz zu Start- und Endpunkt des Tracks
    2. Orthogonale Projektion auf jedes Segment (nur wenn Projektion auf Segment liegt)
    3. Vergleiche alle gültigen Distanzen und wähle minimum
    """
    if len(track_points) < 2:
        raise ValueError("Track muss mindestens 2 Punkte haben")
    
    candidates = []  # Liste aller gültigen Distanz-Kandidaten
    cumulative_distance = 0.0
    
    # 1. DIREKTE DISTANZ zu Start- und Endpunkt
    
    # Start-Punkt
    start_point = Point(track_points[0].lat, track_points[0].lon)
    start_distance = haversine_distance(waypoint, start_point)
    candidates.append({
        'distance': start_distance,
        'closest_point': start_point,
        'track_position_km': 0.0,
        'segment_index': 0,
        'projection_ratio': 0.0,
        'type': 'start_point'
    })
    
    # End-Punkt (berechne total track distance)
    total_track_distance = 0.0
    for i in range(len(track_points) - 1):
        p1 = Point(track_points[i].lat, track_points[i].lon)
        p2 = Point(track_points[i + 1].lat, track_points[i + 1].lon)
        total_track_distance += haversine_distance(p1, p2)
    
    end_point = Point(track_points[-1].lat, track_points[-1].lon)
    end_distance = haversine_distance(waypoint, end_point)
    candidates.append({
        'distance': end_distance,
        'closest_point': end_point,
        'track_position_km': total_track_distance / 1000.0,
        'segment_index': len(track_points) - 2,
        'projection_ratio': 1.0,
        'type': 'end_point'
    })
    
    # 2. ORTHOGONALE PROJEKTION auf alle Segmente
    
    cumulative_distance = 0.0
    for i in range(len(track_points) - 1):
        segment_start = Point(track_points[i].lat, track_points[i].lon)
        segment_end = Point(track_points[i + 1].lat, track_points[i + 1].lon)
        
        # Orthogonale Projektion auf dieses Segment
        projection = point_to_line_distance_spherical(waypoint, segment_start, segment_end)
        
        # 3. NUR GÜLTIGE PROJEKTIONEN (die auf dem Segment landen)
        if projection.is_valid:  # is_valid = True wenn 0.0 <= t <= 1.0
            segment_length = haversine_distance(segment_start, segment_end)
            position_on_segment = segment_length * projection.projection_ratio
            track_position = cumulative_distance + position_on_segment
            
            candidates.append({
                'distance': projection.distance_meters,
                'closest_point': projection.closest_point,
                'track_position_km': track_position / 1000.0,
                'segment_index': i,
                'projection_ratio': projection.projection_ratio,
                'type': 'orthogonal_projection'
            })
        
        # Addiere Segmentlänge für nächste Iteration
        cumulative_distance += haversine_distance(segment_start, segment_end)
    
    # 4. FINDE MINIMUM aus allen gültigen Kandidaten
    if not candidates:
        raise ValueError("Keine gültigen Distanz-Kandidaten gefunden")
    
    best_candidate = min(candidates, key=lambda c: c['distance'])
    
    return ProjectionResult(
        closest_point=best_candidate['closest_point'],
        distance_meters=best_candidate['distance'],
        track_segment_index=best_candidate['segment_index'],
        projection_ratio=best_candidate['projection_ratio'],
        is_valid=True,  # Alle Kandidaten sind per Definition gültig
        track_position_km=best_candidate['track_position_km']
    )

# === High-Level Validation Functions ===

def validate_waypoint_proximity(
    waypoint_lat: float,
    waypoint_lon: float,
    track_points: List[Dict],  # From database: [{"latitude": ..., "longitude": ...}, ...]
    config: WaypointProximityConfig
) -> ProximityValidationResult:
    """
    Hauptfunktion: Validiere ob ein Waypoint nah genug am Track ist
    """
    try:
        # Input validation
        if len(track_points) < config.min_track_points:
            return ProximityValidationResult(
                is_valid=False,
                distance_to_track=float('inf'),
                closest_track_point={"lat": 0, "lon": 0},
                track_segment_index=-1,
                projection_ratio=0.0,
                track_position_km=0.0,
                error_message=f"Track hat zu wenige Punkte ({len(track_points)} < {config.min_track_points})"
            )
        
        # Konvertiere track_points zu TrackPoint objects
        track_point_objects = []
        for i, tp in enumerate(track_points):
            track_point_objects.append(TrackPoint(
                lat=tp["latitude"],
                lon=tp["longitude"],
                elevation=tp.get("elevation"),
                index=i
            ))
        
        waypoint = Point(waypoint_lat, waypoint_lon)
        
        # Finde nächsten Punkt auf Track
        projection = find_closest_point_on_track(waypoint, track_point_objects)
        
        # Prüfe Toleranz
        is_within_tolerance = projection.distance_meters <= config.max_distance_meters
        
        # Prüfe Extrapolation (falls nicht erlaubt)
        is_extrapolation = not projection.is_valid
        if is_extrapolation and not config.allow_extrapolation:
            return ProximityValidationResult(
                is_valid=False,
                distance_to_track=projection.distance_meters,
                closest_track_point={
                    "lat": projection.closest_point.lat,
                    "lon": projection.closest_point.lon
                },
                track_segment_index=projection.track_segment_index,
                projection_ratio=projection.projection_ratio,
                track_position_km=projection.track_position_km,
                error_message=f"Waypoint liegt außerhalb des Track-Bereichs (Extrapolation nicht erlaubt)"
            )
        
        return ProximityValidationResult(
            is_valid=is_within_tolerance,
            distance_to_track=projection.distance_meters,
            closest_track_point={
                "lat": projection.closest_point.lat,
                "lon": projection.closest_point.lon
            },
            track_segment_index=projection.track_segment_index,
            projection_ratio=projection.projection_ratio,
            track_position_km=projection.track_position_km,
            error_message=None if is_within_tolerance else 
                         f"Waypoint ist {projection.distance_meters:.1f}m vom Track entfernt (max: {config.max_distance_meters}m)"
        )
        
    except Exception as e:
        return ProximityValidationResult(
            is_valid=False,
            distance_to_track=float('inf'),
            closest_track_point={"lat": 0, "lon": 0},
            track_segment_index=-1,
            projection_ratio=0.0,
            track_position_km=0.0,
            error_message=f"Fehler bei der Proximity-Validierung: {str(e)}"
        )

# === Configuration Presets ===

class ProximityPresets:
    """Vordefinierte Konfigurationen für verschiedene Anwendungsfälle"""
    
    HIKING_STRICT = WaypointProximityConfig(
        max_distance_meters=50.0,  # 50m Toleranz
        use_elevation=False,
        allow_extrapolation=False,
        min_track_points=3
    )
    
    HIKING_RELAXED = WaypointProximityConfig(
        max_distance_meters=200.0,  # 200m Toleranz
        use_elevation=False,
        allow_extrapolation=True,
        min_track_points=2
    )
    
    CYCLING = WaypointProximityConfig(
        max_distance_meters=100.0,  # 100m Toleranz
        use_elevation=False,
        allow_extrapolation=False,
        min_track_points=5
    )
    
    DRIVING = WaypointProximityConfig(
        max_distance_meters=500.0,  # 500m Toleranz (Parkplätze, etc.)
        use_elevation=False,
        allow_extrapolation=True,
        min_track_points=3
    )
    
    MOUNTAINEERING = WaypointProximityConfig(
        max_distance_meters=30.0,  # 30m Toleranz (präzise)
        use_elevation=True,
        allow_extrapolation=False,
        min_track_points=5
    )

# === Helper Functions ===

def get_track_statistics(track_points: List[TrackPoint]) -> Dict:
    """Berechne Track-Statistiken"""
    if len(track_points) < 2:
        return {"total_distance_km": 0.0, "bounding_box": None}
    
    total_distance = 0.0
    min_lat = min_lon = float('inf')
    max_lat = max_lon = float('-inf')
    
    for i in range(len(track_points) - 1):
        p1 = Point(track_points[i].lat, track_points[i].lon)
        p2 = Point(track_points[i + 1].lat, track_points[i + 1].lon)
        
        total_distance += haversine_distance(p1, p2)
        
        # Update bounding box
        for point in [p1, p2]:
            min_lat = min(min_lat, point.lat)
            max_lat = max(max_lat, point.lat)
            min_lon = min(min_lon, point.lon)
            max_lon = max(max_lon, point.lon)
    
    return {
        "total_distance_km": total_distance / 1000.0,
        "bounding_box": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        },
        "point_count": len(track_points)
    }

def debug_closest_point_calculation(waypoint: Point, track_points: List[TrackPoint]) -> Dict:
    """
    Debug-Funktion: Zeige alle Distanz-Kandidaten für Waypoint-Zuordnung
    Implementiert den optimierten Algorithmus mit detailliertem Output
    """
    if len(track_points) < 2:
        return {"error": "Track muss mindestens 2 Punkte haben"}
    
    candidates = []
    cumulative_distance = 0.0
    
    # 1. Start-Punkt Distanz
    start_point = Point(track_points[0].lat, track_points[0].lon)
    start_distance = haversine_distance(waypoint, start_point)
    candidates.append({
        'type': 'start_point',
        'distance_meters': round(start_distance, 2),
        'track_position_km': 0.0,
        'point': {'lat': start_point.lat, 'lon': start_point.lon},
        'segment_index': 0,
        'valid': True
    })
    
    # 2. End-Punkt Distanz
    total_distance = 0.0
    for i in range(len(track_points) - 1):
        p1 = Point(track_points[i].lat, track_points[i].lon)
        p2 = Point(track_points[i + 1].lat, track_points[i + 1].lon)
        total_distance += haversine_distance(p1, p2)
    
    end_point = Point(track_points[-1].lat, track_points[-1].lon)
    end_distance = haversine_distance(waypoint, end_point)
    candidates.append({
        'type': 'end_point',
        'distance_meters': round(end_distance, 2),
        'track_position_km': round(total_distance / 1000.0, 3),
        'point': {'lat': end_point.lat, 'lon': end_point.lon},
        'segment_index': len(track_points) - 2,
        'valid': True
    })
    
    # 3. Orthogonale Projektionen
    cumulative_distance = 0.0
    for i in range(len(track_points) - 1):
        segment_start = Point(track_points[i].lat, track_points[i].lon)
        segment_end = Point(track_points[i + 1].lat, track_points[i + 1].lon)
        
        projection = point_to_line_distance_spherical(waypoint, segment_start, segment_end)
        
        segment_length = haversine_distance(segment_start, segment_end)
        position_on_segment = segment_length * projection.projection_ratio
        track_position = cumulative_distance + position_on_segment
        
        candidates.append({
            'type': 'orthogonal_projection',
            'distance_meters': round(projection.distance_meters, 2),
            'track_position_km': round(track_position / 1000.0, 3),
            'point': {'lat': projection.closest_point.lat, 'lon': projection.closest_point.lon},
            'segment_index': i,
            'projection_ratio': round(projection.projection_ratio, 3),
            'valid': projection.is_valid,  # Nur wenn 0.0 <= t <= 1.0
            'segment_length_m': round(segment_length, 2)
        })
        
        cumulative_distance += segment_length
    
    # Filter nur gültige Kandidaten
    valid_candidates = [c for c in candidates if c['valid']]
    
    if not valid_candidates:
        return {
            "error": "Keine gültigen Projektionen gefunden",
            "all_candidates": candidates
        }
    
    # Finde besten Kandidaten
    best = min(valid_candidates, key=lambda c: c['distance_meters'])
    
    return {
        "waypoint": {'lat': waypoint.lat, 'lon': waypoint.lon},
        "total_track_distance_km": round(total_distance / 1000.0, 3),
        "total_segments": len(track_points) - 1,
        "all_candidates": candidates,
        "valid_candidates_count": len(valid_candidates),
        "best_match": best,
        "algorithm_steps": [
            "1. Berechne direkte Distanz zu Start-Punkt",
            "2. Berechne direkte Distanz zu End-Punkt", 
            "3. Für jedes Segment: Orthogonale Projektion",
            "4. Filter: Nur Projektionen die auf Segment landen (0.0 <= t <= 1.0)",
            "5. Wähle Kandidat mit minimaler Distanz"
        ]
    }

def suggest_optimal_tolerance(track_points: List[TrackPoint], activity_type: str = "hiking") -> float:
    """
    Schlage optimale Toleranz basierend auf Track-Charakteristiken vor
    """
    if len(track_points) < 2:
        return 100.0
    
    # Berechne durchschnittlichen Punkt-Abstand
    distances = []
    for i in range(len(track_points) - 1):
        p1 = Point(track_points[i].lat, track_points[i].lon)
        p2 = Point(track_points[i + 1].lat, track_points[i + 1].lon)
        distances.append(haversine_distance(p1, p2))
    
    avg_point_distance = sum(distances) / len(distances)
    
    # Basis-Toleranz je nach Aktivität
    base_tolerance = {
        "hiking": 100.0,
        "cycling": 150.0,
        "driving": 300.0,
        "mountaineering": 50.0
    }.get(activity_type, 100.0)
    
    # Anpassung basierend auf Punkt-Dichte
    if avg_point_distance > 100:  # Spärliche Punkte
        base_tolerance *= 1.5
    elif avg_point_distance < 20:  # Dichte Punkte
        base_tolerance *= 0.8
    
    return round(base_tolerance, 1)