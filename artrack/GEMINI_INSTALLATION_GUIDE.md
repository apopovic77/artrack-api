# 🤖 ARTrack API - Installation Guide für Gemini

## 🎯 **Mission: ARTrack API in bestehende FastAPI integrieren**

Du sollst die ARTrack API (GPS + AR Tracking mit AI-Analyse) in die bestehende FastAPI auf Port 8001 integrieren.

---

## 📋 **Was du bekommst:**

```
artrack_api/
├── __init__.py
├── config.py              # Konfiguration
├── models.py              # Database Models
├── database.py            # SQLAlchemy Setup  
├── auth.py                # User Authentication
├── storage.py             # File Upload/Storage
├── analysis.py            # AI-Integration (nutzt bestehende /ai/* Endpoints)
├── routes/
│   ├── __init__.py
│   ├── auth_routes.py     # User Registration/Login
│   ├── track_routes.py    # GPS Track Management
│   ├── waypoint_routes.py # Waypoint Upload + Media
│   └── sync_routes.py     # Sync Status/Offline Support
├── requirements.txt       # Dependencies
└── GEMINI_INSTALLATION_GUIDE.md  # Diese Datei
```

---

## 🚀 **Installation Schritt-für-Schritt:**

### **Schritt 1: Dateien auf Server kopieren**

```bash
# Gehe in dein FastAPI-Verzeichnis
cd /var/www/api.arkturian.com/

# Erstelle ARTrack-Ordner
mkdir -p artrack

# Kopiere alle artrack_api/* Dateien nach artrack/
# (außer dieser GEMINI_INSTALLATION_GUIDE.md)
cp -r artrack_api/* artrack/
```

### **Schritt 2: Dependencies installieren**

```bash
# Zusätzliche Packages installieren
pip install sqlalchemy databases[sqlite] python-multipart pydantic python-jose[cryptography] passlib[bcrypt] httpx pillow python-magic aiofiles
```

### **Schritt 3: Upload-Verzeichnisse erstellen**

```bash
# Upload-Ordner für Media-Dateien
mkdir -p /var/www/api.arkturian.com/uploads/artrack/media
mkdir -p /var/www/api.arkturian.com/uploads/artrack/thumbnails
chmod -R 755 /var/www/api.arkturian.com/uploads/artrack/
```

### **Schritt 4: Environment Variables setzen**

```bash
# Setze diese Umgebungsvariablen (oder in .env Datei)
export ARTRACK_DATABASE_URL="sqlite:///./artrack.db"
export ARTRACK_UPLOAD_DIR="/var/www/api.arkturian.com/uploads/artrack"
export ARTRACK_BASE_URL="https://api.arkturian.com"
# API_KEY nutzt den bestehenden Wert
# AI_BASE_URL="http://localhost:8001" (lokale Calls)
```

### **Schritt 5: Deine bestehende main.py erweitern**

Füge diese Imports **OBEN** in deine main.py hinzu:

```python
# ARTrack API Integration
from artrack.routes import auth_routes, track_routes, waypoint_routes, sync_routes
from artrack.database import connect_db, disconnect_db
from artrack.config import settings as artrack_settings
from fastapi.staticfiles import StaticFiles
import os
```

Füge diese Routen zu deiner bestehenden FastAPI app hinzu:

```python
# ARTrack API Routes
app.include_router(
    auth_routes.router,
    prefix="/artrack/auth",
    tags=["artrack-auth"]
)

app.include_router(
    track_routes.router,
    prefix="/artrack/tracks",
    tags=["artrack-tracks"]
)

app.include_router(
    waypoint_routes.router,
    prefix="/artrack",
    tags=["artrack-waypoints"]
)

app.include_router(
    sync_routes.router,
    prefix="/artrack/sync",
    tags=["artrack-sync"]
)

# Static Files für Media-Uploads
upload_dir = artrack_settings.UPLOAD_DIR
if os.path.exists(upload_dir):
    app.mount(
        "/uploads/artrack",
        StaticFiles(directory=upload_dir),
        name="artrack_uploads"
    )
```

Erweitere deine Startup/Shutdown Events:

```python
# Falls du @app.on_event("startup") hast:
@app.on_event("startup")
async def startup():
    # Deine bestehende Startup-Logic...
    
    # ARTrack Database hinzufügen:
    await connect_db()
    print("ARTrack database connected")

@app.on_event("shutdown")
async def shutdown():
    # Deine bestehende Shutdown-Logic...
    
    # ARTrack Database trennen:
    await disconnect_db()
    print("ARTrack database disconnected")

# Falls du lifespan-Funktion nutzt, erweitere sie entsprechend
```

### **Schritt 6: Database initialisieren**

```bash
cd /var/www/api.arkturian.com/
python3 -c "from artrack.database import create_tables; create_tables()"
```

### **Schritt 7: FastAPI-Service neu starten**

```bash
# Stoppe bestehenden Service
pkill -f "uvicorn main:app"

# Starte erweiterte FastAPI
uvicorn main:app --host 0.0.0.0 --port 8001 > api.log 2>&1 &
```

---

## 🧪 **Testing nach Installation:**

### **1. Health Check:**
```bash
curl http://localhost:8001/artrack/health
```
Erwartete Antwort:
```json
{
  "status": "healthy",
  "service": "artrack-api",
  "version": "1.0.0"
}
```

### **2. User Registration testen:**
```bash
curl -X POST http://localhost:8001/artrack/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "testpass123",
    "display_name": "Test User",
    "device_id": "test-device-123"
  }'
```

### **3. API Documentation:**
Öffne in Browser: `http://localhost:8001/docs`
Sollte jetzt ALLE Routen zeigen:
- Deine bestehenden /ai/* Endpoints
- Neue /artrack/* Endpoints

### **4. Bestehende AI-APIs testen:**
```bash
curl http://localhost:8001/ai/gemini \
  -H "X-API-KEY: dein-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test", "image": null}'
```

---

## 🎯 **Resultat nach Installation:**

```
https://api.arkturian.com/
├── ai/                    # Deine bestehenden AI-Services (unverändert)
│   ├── claude, chatgpt, gemini
│   └── gendepth*
├── artrack/               # NEUE ARTrack-APIs
│   ├── auth/register, /login
│   ├── tracks/
│   ├── {track_id}/waypoints
│   └── sync/status
└── uploads/artrack/       # Media Files
```

---

## ⚙️ **Wie ARTrack deine AI-Services nutzt:**

Die ARTrack API ruft **deine bestehenden AI-Endpoints auf DERSELBEN FastAPI-Instanz** auf:

```python
# In artrack/analysis.py:
async def analyze_image(self, image_base64: str):
    # Ruft http://localhost:8001/ai/gemini auf
    response = await self._call_ai_service("/ai/gemini", {
        "text": "Analysiere dieses Outdoor-Bild...",
        "image": f"data:image/jpeg;base64,{image_base64}"
    })
```

**Keine zusätzlichen Ports oder Services nötig!** 🎯

---

## 🛠️ **Troubleshooting:**

### **Import Errors:**
```bash
pip install -r artrack/requirements.txt
```

### **Database Errors:**
```bash
python3 -c "from artrack.database import create_tables; create_tables()"
```

### **Permission Errors:**
```bash
chmod -R 755 /var/www/api.arkturian.com/uploads/
```

### **AI Integration Errors:**
Teste deine bestehenden AI-Endpoints:
```bash
curl http://localhost:8001/ai/gemini -H "X-API-KEY: dein-key" -d '{"text":"test"}'
```

---

## ✅ **Success Indicators:**

- ✅ `curl localhost:8001/artrack/health` → 200 OK
- ✅ `curl localhost:8001/docs` → Zeigt ARTrack + AI Endpoints
- ✅ User Registration funktioniert
- ✅ Bestehende AI-APIs funktionieren weiterhin
- ✅ Logs zeigen "ARTrack database connected"

---

## 📞 **Support:**

Falls Probleme auftreten:
1. **Prüfe Logs:** `tail -f api.log`
2. **Teste AI-Services:** Stelle sicher, dass bestehende /ai/* Endpoints funktionieren
3. **Prüfe Permissions:** Upload-Verzeichnisse beschreibbar?
4. **Database Check:** `ls -la artrack.db` (sollte existieren)

**Nach erfolgreicher Installation läuft ARTrack nahtlos in deiner bestehenden FastAPI! 🚀**