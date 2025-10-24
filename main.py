"""
Artrack API - Track & Collaboration Service

Handles:
- User Authentication & Authorization
- Track Management (CRUD)
- Waypoints & GPS Routes
- Collaboration (Invitations, Sharing)
- Track Segments & Routes
- Audio Features (TTS, Guides)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from artrack.database import engine, Base
from artrack.routes import (
    track_routes,
    waypoint_routes,
    collaboration_routes,
    auth_routes,
    gps_routes,
    snap_routes,
    sync_routes,
    segments_routes,
    routes_routes,
    guide_routes,
)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Artrack API",
    version="1.0.0",
    description="Track management and collaboration service"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_routes.router, prefix="/auth", tags=["Authentication"])
app.include_router(track_routes.router, prefix="/tracks", tags=["Tracks"])
app.include_router(waypoint_routes.router, prefix="", tags=["Waypoints"])
app.include_router(collaboration_routes.router, prefix="/collaboration", tags=["Collaboration"])
app.include_router(gps_routes.router, prefix="/gps", tags=["GPS"])
app.include_router(snap_routes.router, prefix="/snap", tags=["Snap to Road"])
app.include_router(sync_routes.router, prefix="/sync", tags=["Sync"])
app.include_router(segments_routes.router, prefix="/segments", tags=["Segments"])
app.include_router(routes_routes.router, prefix="/tracks", tags=["Routes"])
app.include_router(guide_routes.router, prefix="/guides", tags=["Audio Guides"])

@app.get("/")
def root():
    return {
        "service": "artrack-api",
        "version": "1.0.0",
        "description": "Track management and collaboration service"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "artrack-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
