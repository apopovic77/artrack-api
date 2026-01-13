# Route Knowledge API v2

AI-generierte Texte für Audio-Guide Narration mit zwei-Phasen Workflow.

**Status:** Implementiert
**Version:** 2.0
**Datum:** 2026-01-13

---

## Übersicht

Route Knowledge speichert vorab generierte Narrative-Texte für Audio-Guides.

**Zwei-Phasen Workflow:**
1. **Text Generation:** AI generiert alle Texte → JSON → Bearbeitung möglich
2. **Audio Production:** Finalisierte Texte → TTS → Audio URLs (Phase 2, noch nicht implementiert)

---

## Storage Design

**Speicherort:** `TrackRoute.metadata_json["knowledge"]`

```python
TrackRoute.metadata_json = {
    "knowledge": {
        "version": 2,
        "generated_at": "2026-01-13T12:00:00Z",
        "config": {
            "persona": "Du bist ein erfahrener Wanderführer...",
            "target_audience": "Familien mit Kindern",
            "language": "de",
            "tone": "friendly"
        },
        "route": {
            "intro": {
                "text": "Willkommen auf der Route...",
                "text_original": "Willkommen auf der Route...",
                "edited": False,
                "audio_storage_id": None
            },
            "outro": { ... }
        },
        "segments": {
            "Wasserfall-Passage": {
                "name": "Wasserfall-Passage",
                "entry": { "text": "...", "text_original": "...", "edited": False },
                "exit": { "text": "...", "text_original": "...", "edited": False }
            }
        },
        "pois": {
            "123": {
                "name": "Tscheppa-Wasserfall",
                "approaching": { "text": "...", "text_original": "...", "edited": False },
                "at_poi": { "text": "...", "text_original": "...", "edited": False }
            }
        }
    }
}
```

**Vorteile:**
- Route-spezifisch (jede Route kann eigene Knowledge haben)
- Keine neuen Tabellen nötig
- Natürlicher Fit mit existierendem Datenmodell
- Einfach zu versionieren/ersetzen

---

## API Endpoints

### 1. Get Route Knowledge

**GET** `/tracks/{track_id}/routes/{route_id}/knowledge`

Holt existierende Knowledge.

**Response:**
```json
{
    "exists": true,
    "knowledge": { /* RouteKnowledge JSON */ },
    "route_id": 42,
    "route_name": "Hauptweg"
}
```

---

### 2. Generate Route Knowledge

**POST** `/tracks/{track_id}/routes/{route_id}/knowledge/generate`

Generiert alle Narrative-Texte mit AI.

**Request:**
```json
{
    "persona": "Du bist ein erfahrener Wanderführer...",
    "target_audience": "Familien mit Kindern",
    "language": "de",
    "tone": "friendly",
    "generate_segments": true,
    "generate_pois": true
}
```

**Response:**
```json
{
    "success": true,
    "knowledge": { /* Vollständiges RouteKnowledge JSON */ },
    "stats": {
        "segments_count": 3,
        "pois_count": 5,
        "total_texts": 18
    }
}
```

**Hinweis:** Die Texte werden generiert aber NICHT automatisch gespeichert.
Der Client kann die Texte bearbeiten und dann mit PUT speichern.

---

### 3. Save Route Knowledge

**PUT** `/tracks/{track_id}/routes/{route_id}/knowledge`

Speichert bearbeitete Knowledge.

**Request:**
```json
{
    "knowledge": { /* RouteKnowledge JSON */ }
}
```

**Response:**
```json
{
    "success": true,
    "route_id": 42,
    "saved_at": "2026-01-13T12:30:00Z"
}
```

---

### 4. Delete Route Knowledge

**DELETE** `/tracks/{track_id}/routes/{route_id}/knowledge`

Löscht Knowledge für eine Route.

**Response:**
```json
{
    "success": true,
    "deleted": true
}
```

---

## Narrative-Typen

### Route Narratives
| Type | Trigger | Beschreibung |
|------|---------|--------------|
| `intro` | Route wird gestartet | "Willkommen auf dem Hauptweg..." |
| `outro` | Route ist abgeschlossen | "Herzlichen Glückwunsch..." |

### Segment Narratives
| Type | Trigger | Beschreibung |
|------|---------|--------------|
| `entry` | Segment betreten | "Du betrittst den Wasserfall-Bereich..." |
| `exit` | Segment verlassen | "Du verlässt den Wasserfall-Bereich..." |

### POI Narratives
| Type | Trigger | Beschreibung |
|------|---------|--------------|
| `approaching` | POI in Reichweite (~50m) | "Du näherst dich dem Wasserfall..." |
| `at_poi` | POI erreicht | "Vor dir stürzt der Wasserfall..." |

---

## AI Prompt Strategie

Die AI generiert Texte basierend auf:
- **Persona:** Wer erzählt? (z.B. "erfahrener Wanderführer")
- **Target Audience:** Wer hört zu? (z.B. "Familien mit Kindern")
- **Language:** Sprache (de, en, it)
- **Tone:** Tonalität (friendly, informative, casual, professional)

### Beispiel-Prompts

**Route Intro:**
```
Schreibe eine Willkommensnachricht für den Start einer Wanderroute.

Route: Hauptweg durch die Tscheppaschlucht
Länge: 2.3 km
Beschreibung: Spektakuläre Klamm mit Wasserfällen

Persona: Du bist ein freundlicher Audio-Guide.
Zielgruppe: Familien mit Kindern
Ton: friendly
Sprache: de

Schreibe einen kurzen, einladenden Text (2-4 Sätze).
```

**POI At:**
```
Schreibe eine Beschreibung für einen Point of Interest.

POI: Tscheppa-Wasserfall
Beschreibung vom Autor: 30m hoher Wasserfall, besonders beeindruckend nach Regen

Persona: Du bist ein freundlicher Audio-Guide.
Zielgruppe: Familien mit Kindern
Ton: friendly
Sprache: de

Schreibe 2-4 Sätze die den POI beschreiben.
```

---

## Frontend Integration

### RouteKnowledgeEditor.tsx

React-Komponente für Text-Bearbeitung.

**Features:**
- Track/Route Auswahl
- Persona & Zielgruppe Eingabe
- Sprache & Ton Auswahl
- Tabs für Route/Segments/POIs
- Text-Editor für jeden Narrative-Typ
- Statistik (Total/Gefüllt/Bearbeitet)
- "Alle Texte generieren" Button
- "Speichern" Button

**Zugang:**
Dashboard → Track ▾ → "Route Knowledge v2..."

---

## Workflow

```
1. Route in Artrack anlegen
   - POIs mit Beschreibungen erstellen
   - Segment-Marker setzen

2. Route Knowledge Editor öffnen
   - Track auswählen
   - Route auswählen
   - Persona + Zielgruppe eingeben
   - "Alle Texte generieren" klicken

3. Texte bearbeiten
   - Generierte Texte prüfen
   - Bei Bedarf anpassen
   - "Speichern" klicken

4. (Zukünftig) Audio generieren
   - TTS für finale Texte
   - Audio URLs im Knowledge JSON
```

---

## Implementierte Dateien

**Backend (artrack-api):**
- `/artrack/routes/knowledge_routes.py` - API Endpoints
- `/main.py` - Router Registration

**Frontend (artrack.arkturian.com):**
- `/src/components/v2/RouteKnowledgeEditor.tsx` - React Editor
- `/src/components/MapEditorV2.tsx` - Integration (Menu Button)

---

## Nächste Schritte

- [ ] Phase 2: Audio Generation (TTS on-demand)
- [ ] Distance Markers (alle X Meter, braucht POI-Kontext)
- [ ] Richtungs-abhängige Texte (forward/backward)
- [ ] Integration mit ios-guide-export
- [ ] Unity Client Anpassung

---

**Autor:** Claude + Alex
**Implementiert:** 2026-01-13
