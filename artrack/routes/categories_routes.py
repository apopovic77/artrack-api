"""
Categories API - POI and Segment category definitions

Returns the available categories and subcategories for:
- POIs (Points of Interest)
- Segments (Track sections)

These are used by the dashboard for dropdowns and can be
extended without code changes by modifying this file.
"""
from fastapi import APIRouter
from typing import Dict, List, Any
from pydantic import BaseModel

router = APIRouter()


# =============================================================================
# POI Categories
# =============================================================================

POI_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "poi": {
        "bezeichnung": "Sehenswürdigkeit",
        "farbe": "#F59E0B",
        "icon": "📍",
        "unterkategorien": {
            "sight": {"bezeichnung": "Sehenswürdigkeit", "icon": "🏛️"},
            "viewpoint": {"bezeichnung": "Aussichtspunkt", "icon": "👀"},
            "landmark": {"bezeichnung": "Wahrzeichen", "icon": "🗿"},
            "waterfall": {"bezeichnung": "Wasserfall", "icon": "💧"},
            "bridge": {"bezeichnung": "Brücke", "icon": "🌉"},
            "cave": {"bezeichnung": "Höhle", "icon": "🕳️"},
            "monument": {"bezeichnung": "Denkmal", "icon": "🏛️"},
            "church": {"bezeichnung": "Kirche/Kapelle", "icon": "⛪"},
            "castle": {"bezeichnung": "Burg/Schloss", "icon": "🏰"},
        }
    },
    "navigation": {
        "bezeichnung": "Navigation",
        "farbe": "#3B82F6",
        "icon": "🧭",
        "unterkategorien": {
            "junction": {"bezeichnung": "Kreuzung", "icon": "🔀"},
            "trailhead": {"bezeichnung": "Startpunkt", "icon": "🚶"},
            "exit": {"bezeichnung": "Ausgang", "icon": "🚪"},
            "waymark": {"bezeichnung": "Wegmarkierung", "icon": "🔷"},
            "decision": {"bezeichnung": "Entscheidungspunkt", "icon": "❓"},
            "danger": {"bezeichnung": "Gefahrenstelle", "icon": "⚠️"},
            "stairs": {"bezeichnung": "Treppen", "icon": "🪜"},
        }
    },
    "amenity": {
        "bezeichnung": "Einrichtung",
        "farbe": "#10B981",
        "icon": "🏪",
        "unterkategorien": {
            "restaurant": {"bezeichnung": "Restaurant", "icon": "🍽️"},
            "cafe": {"bezeichnung": "Café", "icon": "☕"},
            "kiosk": {"bezeichnung": "Kiosk", "icon": "🏪"},
            "shelter": {"bezeichnung": "Unterstand", "icon": "🛖"},
            "toilet": {"bezeichnung": "WC", "icon": "🚻"},
            "water": {"bezeichnung": "Trinkwasser", "icon": "🚰"},
            "info": {"bezeichnung": "Info-Tafel", "icon": "ℹ️"},
            "ticket": {"bezeichnung": "Ticketschalter", "icon": "🎫"},
            "firstaid": {"bezeichnung": "Erste Hilfe", "icon": "🏥"},
            "shop": {"bezeichnung": "Geschäft", "icon": "🛒"},
        }
    },
    "transport": {
        "bezeichnung": "Transport",
        "farbe": "#8B5CF6",
        "icon": "🚌",
        "unterkategorien": {
            "parking": {"bezeichnung": "Parkplatz", "icon": "🅿️"},
            "bus_stop": {"bezeichnung": "Bushaltestelle", "icon": "🚏"},
            "train": {"bezeichnung": "Bahnhof", "icon": "🚂"},
            "taxi": {"bezeichnung": "Taxi-Stand", "icon": "🚕"},
            "bike_rental": {"bezeichnung": "Fahrradverleih", "icon": "🚲"},
            "charging": {"bezeichnung": "Ladestation", "icon": "🔌"},
        }
    },
    "nature": {
        "bezeichnung": "Natur",
        "farbe": "#22C55E",
        "icon": "🌲",
        "unterkategorien": {
            "tree": {"bezeichnung": "Baum", "icon": "🌳"},
            "flower": {"bezeichnung": "Blume/Pflanze", "icon": "🌸"},
            "animal": {"bezeichnung": "Tier", "icon": "🦌"},
            "rock": {"bezeichnung": "Felsen", "icon": "🪨"},
            "spring": {"bezeichnung": "Quelle", "icon": "💦"},
            "summit": {"bezeichnung": "Gipfel", "icon": "⛰️"},
            "meadow": {"bezeichnung": "Wiese/Alm", "icon": "🌾"},
        }
    },
    "media": {
        "bezeichnung": "Medien",
        "farbe": "#EC4899",
        "icon": "📸",
        "unterkategorien": {
            "photo": {"bezeichnung": "Foto-Spot", "icon": "📷"},
            "video": {"bezeichnung": "Video-Spot", "icon": "🎬"},
            "audio": {"bezeichnung": "Audio-Punkt", "icon": "🎙️"},
            "panorama": {"bezeichnung": "Panorama", "icon": "🌅"},
            "timelapse": {"bezeichnung": "Timelapse", "icon": "⏱️"},
        }
    },
    "system": {
        "bezeichnung": "System",
        "farbe": "#6B7280",
        "icon": "⚙️",
        "unterkategorien": {
            "segment_start": {"bezeichnung": "Segment-Start", "icon": "🟢"},
            "segment_end": {"bezeichnung": "Segment-Ende", "icon": "🔴"},
            "route_point": {"bezeichnung": "Routenpunkt", "icon": "📍"},
            "calibration": {"bezeichnung": "Kalibrierung", "icon": "🎯"},
        }
    },
}


# =============================================================================
# Segment Categories
# =============================================================================

SEGMENT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "terrain": {
        "bezeichnung": "Untergrund",
        "farbe": "#8B4513",
        "icon": "🥾",
        "unterkategorien": {
            "asphalt": {"bezeichnung": "Asphalt", "icon": "🛣️"},
            "gravel": {"bezeichnung": "Schotter", "icon": "🪨"},
            "forest_path": {"bezeichnung": "Waldweg", "icon": "🌲"},
            "grass": {"bezeichnung": "Wiese/Gras", "icon": "🌿"},
            "rock": {"bezeichnung": "Fels", "icon": "🧱"},
            "scree": {"bezeichnung": "Geröll", "icon": "🏔️"},
            "sand": {"bezeichnung": "Sand", "icon": "🏖️"},
            "mud": {"bezeichnung": "Matschig", "icon": "💧"},
            "snow": {"bezeichnung": "Schnee", "icon": "❄️"},
            "stairs": {"bezeichnung": "Stufen", "icon": "🪜"},
            "boardwalk": {"bezeichnung": "Holzsteg", "icon": "🪵"},
        }
    },
    "difficulty": {
        "bezeichnung": "Schwierigkeit",
        "farbe": "#EF4444",
        "icon": "⚡",
        "unterkategorien": {
            "easy": {"bezeichnung": "Leicht", "icon": "🟢"},
            "moderate": {"bezeichnung": "Mittel", "icon": "🔵"},
            "difficult": {"bezeichnung": "Schwer", "icon": "🔴"},
            "expert": {"bezeichnung": "Experte", "icon": "⚫"},
            "via_ferrata": {"bezeichnung": "Klettersteig", "icon": "⛓️"},
            "scramble": {"bezeichnung": "Kraxelei", "icon": "🧗"},
        }
    },
    "condition": {
        "bezeichnung": "Wegzustand",
        "farbe": "#F59E0B",
        "icon": "🔧",
        "unterkategorien": {
            "excellent": {"bezeichnung": "Ausgezeichnet", "icon": "✅"},
            "good": {"bezeichnung": "Gut", "icon": "👍"},
            "fair": {"bezeichnung": "Akzeptabel", "icon": "👌"},
            "poor": {"bezeichnung": "Schlecht", "icon": "👎"},
            "damaged": {"bezeichnung": "Beschädigt", "icon": "🚧"},
            "closed": {"bezeichnung": "Gesperrt", "icon": "🚫"},
            "construction": {"bezeichnung": "Baustelle", "icon": "🏗️"},
        }
    },
    "scenery": {
        "bezeichnung": "Landschaft",
        "farbe": "#22C55E",
        "icon": "🌄",
        "unterkategorien": {
            "forest": {"bezeichnung": "Wald", "icon": "🌲"},
            "alpine": {"bezeichnung": "Alpin", "icon": "🏔️"},
            "meadow": {"bezeichnung": "Wiese/Alm", "icon": "🌾"},
            "gorge": {"bezeichnung": "Schlucht", "icon": "🏞️"},
            "ridge": {"bezeichnung": "Grat", "icon": "⛰️"},
            "lakeside": {"bezeichnung": "Am See", "icon": "🏊"},
            "riverside": {"bezeichnung": "Am Fluss", "icon": "🌊"},
            "urban": {"bezeichnung": "Siedlung", "icon": "🏘️"},
            "panorama": {"bezeichnung": "Panoramastrecke", "icon": "🌅"},
        }
    },
    "exposure": {
        "bezeichnung": "Exposition",
        "farbe": "#3B82F6",
        "icon": "☀️",
        "unterkategorien": {
            "sunny": {"bezeichnung": "Sonnig", "icon": "☀️"},
            "shaded": {"bezeichnung": "Schattig", "icon": "🌳"},
            "mixed": {"bezeichnung": "Gemischt", "icon": "🌤️"},
            "windy": {"bezeichnung": "Windig", "icon": "💨"},
            "sheltered": {"bezeichnung": "Geschützt", "icon": "🛖"},
        }
    },
    "infrastructure": {
        "bezeichnung": "Infrastruktur",
        "farbe": "#6B7280",
        "icon": "🛤️",
        "unterkategorien": {
            "marked": {"bezeichnung": "Markiert", "icon": "🔷"},
            "unmarked": {"bezeichnung": "Unmarkiert", "icon": "❓"},
            "signposted": {"bezeichnung": "Beschildert", "icon": "🪧"},
            "railings": {"bezeichnung": "Geländer", "icon": "🚧"},
            "cables": {"bezeichnung": "Seilsicherung", "icon": "⛓️"},
            "bridge": {"bezeichnung": "Brücke", "icon": "🌉"},
            "tunnel": {"bezeichnung": "Tunnel", "icon": "🚇"},
        }
    },
    "hazard": {
        "bezeichnung": "Gefahr",
        "farbe": "#DC2626",
        "icon": "⚠️",
        "unterkategorien": {
            "rockfall": {"bezeichnung": "Steinschlag", "icon": "🪨"},
            "avalanche": {"bezeichnung": "Lawine", "icon": "❄️"},
            "cliff": {"bezeichnung": "Absturzgefahr", "icon": "🧗"},
            "slippery": {"bezeichnung": "Rutschig", "icon": "💧"},
            "wildlife": {"bezeichnung": "Wildtiere", "icon": "🐻"},
            "traffic": {"bezeichnung": "Verkehr", "icon": "🚗"},
            "hunting": {"bezeichnung": "Jagdgebiet", "icon": "🎯"},
        }
    },
    "activity": {
        "bezeichnung": "Aktivität",
        "farbe": "#8B5CF6",
        "icon": "🎯",
        "unterkategorien": {
            "hiking": {"bezeichnung": "Wandern", "icon": "🥾"},
            "running": {"bezeichnung": "Laufen", "icon": "🏃"},
            "biking": {"bezeichnung": "Radfahren", "icon": "🚴"},
            "mtb": {"bezeichnung": "Mountainbike", "icon": "🚵"},
            "climbing": {"bezeichnung": "Klettern", "icon": "🧗"},
            "skiing": {"bezeichnung": "Skifahren", "icon": "⛷️"},
            "snowshoe": {"bezeichnung": "Schneeschuh", "icon": "🎿"},
        }
    },
}


# =============================================================================
# Response Models
# =============================================================================

class SubcategoryInfo(BaseModel):
    bezeichnung: str
    icon: str


class CategoryInfo(BaseModel):
    bezeichnung: str
    farbe: str
    icon: str
    unterkategorien: Dict[str, SubcategoryInfo]


class CategoriesResponse(BaseModel):
    poi: Dict[str, CategoryInfo]
    segment: Dict[str, CategoryInfo]


class CategoryListItem(BaseModel):
    key: str
    bezeichnung: str
    farbe: str
    icon: str
    subcategories: List[Dict[str, str]]


class CategoriesListResponse(BaseModel):
    poi: List[CategoryListItem]
    segment: List[CategoryListItem]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/", response_model=CategoriesResponse)
async def get_categories():
    """
    Get all POI and Segment categories with their subcategories.

    Returns the full category tree including:
    - bezeichnung (German label)
    - farbe (color hex code)
    - icon (emoji)
    - unterkategorien (subcategories with their labels and icons)
    """
    return {
        "poi": POI_CATEGORIES,
        "segment": SEGMENT_CATEGORIES,
    }


@router.get("/list", response_model=CategoriesListResponse)
async def get_categories_list():
    """
    Get categories as flat lists (easier for dropdowns).

    Each category includes its key, label, color, icon,
    and a list of subcategories with key, label, and icon.
    """
    def to_list(categories: Dict[str, Dict[str, Any]]) -> List[CategoryListItem]:
        result = []
        for key, cat in categories.items():
            subcats = [
                {"key": sub_key, "bezeichnung": sub["bezeichnung"], "icon": sub["icon"]}
                for sub_key, sub in cat["unterkategorien"].items()
            ]
            result.append(CategoryListItem(
                key=key,
                bezeichnung=cat["bezeichnung"],
                farbe=cat["farbe"],
                icon=cat["icon"],
                subcategories=subcats,
            ))
        return result

    return {
        "poi": to_list(POI_CATEGORIES),
        "segment": to_list(SEGMENT_CATEGORIES),
    }


@router.get("/poi")
async def get_poi_categories():
    """Get only POI categories."""
    return POI_CATEGORIES


@router.get("/segment")
async def get_segment_categories():
    """Get only Segment categories."""
    return SEGMENT_CATEGORIES
