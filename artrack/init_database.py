#!/usr/bin/env python3
"""
ARTrack Database Initialization Script
=====================================

This script creates a fresh ARTrack database from scratch.
Safe to run multiple times - it will recreate everything.

Usage:
    python3 init_database.py [--reset] [--demo-data]
    
Options:
    --reset      Delete existing database and start fresh
    --demo-data  Create demo users and sample tracks for testing
    
Examples:
    python3 init_database.py                    # Create fresh DB
    python3 init_database.py --reset            # Delete old DB, create new
    python3 init_database.py --demo-data        # Create with demo data
    python3 init_database.py --reset --demo-data # Full reset with demo data
"""

import sqlite3
import os
import sys
import argparse
from datetime import datetime, timedelta
import hashlib
import secrets
import json

# Database configuration
DB_PATH = 'artrack.db'
BACKUP_PATH = 'artrack_backup.db'

def hash_password(password: str) -> str:
    """Hash a password using a simple method (replace with proper bcrypt in production)"""
    # For demo purposes - in production use bcrypt
    return hashlib.sha256(f"salt_{password}".encode()).hexdigest()

def generate_invite_code() -> str:
    """Generate a random 8-character invite code"""
    return secrets.token_urlsafe(6)[:8].upper()

def init_database(reset=False, demo_data=False):
    """Initialize the ARTrack database from scratch"""
    
    print("🗄️ ARTrack Database Initialization")
    print("=" * 50)
    
    # Handle reset option
    if reset and os.path.exists(DB_PATH):
        print(f"🗑️ Deleting existing database: {DB_PATH}")
        os.remove(DB_PATH)
    elif os.path.exists(DB_PATH):
        # Create backup
        print(f"💾 Creating backup: {BACKUP_PATH}")
        import shutil
        shutil.copy2(DB_PATH, BACKUP_PATH)
    
    # Connect to database (creates if not exists)
    print(f"🔌 Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        print("📋 Creating core tables...")
        create_core_tables(cursor)
        
        print("🤝 Creating collaboration tables...")
        create_collaboration_tables(cursor)
        
        print("📊 Creating media and analysis tables...")
        create_media_tables(cursor)
        
        print("📈 Creating performance indexes...")
        create_indexes(cursor)
        
        if demo_data:
            print("🎭 Creating demo data...")
            create_demo_data(cursor)
        
        # Commit all changes
        conn.commit()
        
        print("✅ Database initialization completed successfully!")
        
        # Show summary
        show_database_summary(cursor)
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def create_core_tables(cursor):
    """Create the core ARTrack tables"""
    
    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(255) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        display_name VARCHAR(255) NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        device_id VARCHAR(255),
        is_active BOOLEAN DEFAULT 1,
        is_admin BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        profile_image_url VARCHAR(500),
        preferences JSON,
        metadata JSON
    )
    ''')
    
    # Tracks table with collaboration support
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        created_by INTEGER NOT NULL,
        is_public BOOLEAN DEFAULT 0,
        track_type VARCHAR(50) DEFAULT 'hiking',
        status VARCHAR(20) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        total_distance REAL DEFAULT 0.0,
        total_duration INTEGER DEFAULT 0,
        elevation_gain REAL DEFAULT 0.0,
        elevation_loss REAL DEFAULT 0.0,
        min_elevation REAL,
        max_elevation REAL,
        avg_speed REAL,
        max_speed REAL,
        -- Collaboration fields
        is_collaborative BOOLEAN DEFAULT 0,
        collaboration_mode VARCHAR(20) DEFAULT 'invite_only',
        invite_code VARCHAR(8) UNIQUE,
        max_collaborators INTEGER DEFAULT 10,
        -- Geographic bounds
        bbox_north REAL,
        bbox_south REAL,
        bbox_east REAL,
        bbox_west REAL,
        -- Settings and metadata
        settings JSON,
        metadata JSON,
        FOREIGN KEY (created_by) REFERENCES users (id)
    )
    ''')
    
    # Waypoints table (for GPS points and media waypoints)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS waypoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        created_by INTEGER NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        altitude REAL,
        accuracy REAL,
        timestamp TIMESTAMP NOT NULL,
        waypoint_type VARCHAR(50) DEFAULT 'manual',
        title VARCHAR(255),
        description TEXT,
        is_public BOOLEAN DEFAULT 0,
        is_validated BOOLEAN DEFAULT 1,
        order_index INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        -- GPS-specific fields
        speed REAL,
        course REAL,
        horizontal_accuracy REAL,
        vertical_accuracy REAL,
        -- Classification and tagging
        tags JSON,
        poi_type VARCHAR(50),
        importance_score REAL DEFAULT 0.0,
        -- Processing status
        processing_status VARCHAR(20) DEFAULT 'completed',
        sync_status VARCHAR(20) DEFAULT 'synced',
        -- Metadata
        metadata JSON,
        FOREIGN KEY (track_id) REFERENCES tracks (id),
        FOREIGN KEY (created_by) REFERENCES users (id)
    )
    ''')

def create_collaboration_tables(cursor):
    """Create collaboration-related tables"""
    
    # Track collaborators
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS track_collaborators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        role VARCHAR(50) DEFAULT 'contributor',
        -- Permissions
        can_view BOOLEAN DEFAULT 1,
        can_add_waypoints BOOLEAN DEFAULT 1,
        can_edit_waypoints BOOLEAN DEFAULT 0,
        can_delete_waypoints BOOLEAN DEFAULT 0,
        can_edit_track BOOLEAN DEFAULT 0,
        can_invite_others BOOLEAN DEFAULT 0,
        can_manage_collaborators BOOLEAN DEFAULT 0,
        -- Status
        status VARCHAR(20) DEFAULT 'active',
        invited_by INTEGER,
        joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_activity TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        -- Metadata
        metadata JSON,
        FOREIGN KEY (track_id) REFERENCES tracks (id),
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (invited_by) REFERENCES users (id),
        UNIQUE(track_id, user_id)
    )
    ''')
    
    # Track invitations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS track_invitations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        inviter_id INTEGER NOT NULL,
        -- Invitation methods
        invite_token VARCHAR(64) UNIQUE,
        invite_code VARCHAR(8),
        invite_url VARCHAR(500),
        qr_code_data TEXT,
        -- Target user info
        email VARCHAR(255),
        username VARCHAR(255),
        display_name VARCHAR(255),
        -- Invitation settings
        role VARCHAR(50) DEFAULT 'contributor',
        message TEXT,
        max_uses INTEGER DEFAULT 1,
        current_uses INTEGER DEFAULT 0,
        -- Status and timing
        status VARCHAR(20) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        accepted_at TIMESTAMP,
        accepted_by INTEGER,
        -- Metadata
        metadata JSON,
        FOREIGN KEY (track_id) REFERENCES tracks (id),
        FOREIGN KEY (inviter_id) REFERENCES users (id),
        FOREIGN KEY (accepted_by) REFERENCES users (id)
    )
    ''')
    
    # Track activities (for audit trail)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS track_activities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        activity_type VARCHAR(50) NOT NULL,
        description TEXT,
        details JSON,
        ip_address VARCHAR(45),
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (track_id) REFERENCES tracks (id),
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

def create_media_tables(cursor):
    """Create media and analysis tables"""
    
    # Media files
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS media_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        waypoint_id INTEGER NOT NULL,
        uploaded_by INTEGER NOT NULL,
        -- File information
        file_type VARCHAR(50) NOT NULL,
        mime_type VARCHAR(100),
        file_size INTEGER,
        file_path VARCHAR(500) NOT NULL,
        original_filename VARCHAR(255),
        processed_filename VARCHAR(255),
        thumbnail_path VARCHAR(500),
        -- Processing status
        upload_status VARCHAR(50) DEFAULT 'pending',
        processing_status VARCHAR(50) DEFAULT 'pending',
        analysis_status VARCHAR(50) DEFAULT 'pending',
        -- File metadata
        width INTEGER,
        height INTEGER,
        duration REAL,
        bitrate INTEGER,
        format_info JSON,
        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        uploaded_at TIMESTAMP,
        processed_at TIMESTAMP,
        -- Settings
        is_public BOOLEAN DEFAULT 0,
        compression_level INTEGER DEFAULT 1,
        metadata JSON,
        FOREIGN KEY (waypoint_id) REFERENCES waypoints (id),
        FOREIGN KEY (uploaded_by) REFERENCES users (id)
    )
    ''')
    
    # AI analysis results
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_file_id INTEGER NOT NULL,
        analysis_type VARCHAR(50) NOT NULL,
        ai_model VARCHAR(100),
        model_version VARCHAR(50),
        -- Results
        result_data JSON,
        confidence_score REAL,
        tags JSON,
        description TEXT,
        summary TEXT,
        -- Classifications
        content_type VARCHAR(50),
        safety_rating VARCHAR(20),
        quality_score REAL,
        -- Processing info
        processing_time_ms INTEGER,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'completed',
        error_message TEXT,
        -- Metadata
        metadata JSON,
        FOREIGN KEY (media_file_id) REFERENCES media_files (id)
    )
    ''')
    
    # Content moderation logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS moderation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type VARCHAR(20) NOT NULL, -- 'waypoint', 'media_file', 'track'
        target_id INTEGER NOT NULL,
        -- Moderation info
        moderation_action VARCHAR(50) NOT NULL,
        reason TEXT,
        severity VARCHAR(20),
        confidence_score REAL,
        -- Moderator info
        moderator_id INTEGER,
        moderator_type VARCHAR(20) DEFAULT 'ai', -- 'ai', 'human', 'system'
        automated BOOLEAN DEFAULT 1,
        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        reviewed_at TIMESTAMP,
        resolved_at TIMESTAMP,
        -- Status
        status VARCHAR(20) DEFAULT 'active',
        appeal_status VARCHAR(20),
        -- Metadata
        metadata JSON,
        FOREIGN KEY (moderator_id) REFERENCES users (id)
    )
    ''')

def create_indexes(cursor):
    """Create performance indexes"""
    
    print("   • User indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_active ON users (is_active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_created ON users (created_at)')
    
    print("   • Track indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_created_by ON tracks (created_by)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_public ON tracks (is_public)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_collaborative ON tracks (is_collaborative)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_invite_code ON tracks (invite_code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_type ON tracks (track_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks (status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_created ON tracks (created_at)')
    
    print("   • Waypoint indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_waypoints_track ON waypoints (track_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_waypoints_created_by ON waypoints (created_by)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_waypoints_timestamp ON waypoints (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_waypoints_type ON waypoints (waypoint_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_waypoints_public ON waypoints (is_public)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_waypoints_location ON waypoints (latitude, longitude)')
    
    print("   • GPS-specific indexes...")
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_waypoint_track_gps 
    ON waypoints (track_id, waypoint_type, timestamp) 
    WHERE waypoint_type = 'gps_track'
    ''')
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_waypoint_track_media 
    ON waypoints (track_id, waypoint_type, created_at) 
    WHERE waypoint_type IN ('photo', 'video', 'audio')
    ''')
    
    print("   • Collaboration indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_collaborators_track ON track_collaborators (track_id, is_active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_collaborators_user ON track_collaborators (user_id, is_active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_invitations_track ON track_invitations (track_id, status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_invitations_code ON track_invitations (invite_code, status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_invitations_token ON track_invitations (invite_token)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_activities_track ON track_activities (track_id, created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_activities_user ON track_activities (user_id, created_at)')
    
    print("   • Media indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_waypoint ON media_files (waypoint_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_uploaded_by ON media_files (uploaded_by)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_status ON media_files (upload_status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_type ON media_files (file_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_created ON media_files (created_at)')
    
    print("   • Analysis indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_media ON analysis_results (media_file_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results (analysis_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_processed ON analysis_results (processed_at)')
    
    print("   • Moderation indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_moderation_target ON moderation_logs (target_type, target_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_moderation_action ON moderation_logs (moderation_action)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_moderation_created ON moderation_logs (created_at)')

def create_demo_data(cursor):
    """Create demo users and sample data for testing"""
    
    print("   • Creating demo users...")
    
    # Demo users with different roles
    demo_users = [
        {
            'username': 'testuser',
            'email': 'test@example.com',
            'display_name': 'Test User',
            'password': 'testpass123',
            'is_admin': False
        },
        {
            'username': 'admin',
            'email': 'admin@example.com', 
            'display_name': 'Admin User',
            'password': 'admin123',
            'is_admin': True
        },
        {
            'username': 'hiker1',
            'email': 'hiker1@example.com',
            'display_name': 'Mountain Hiker',
            'password': 'hiker123',
            'is_admin': False
        },
        {
            'username': 'cyclist',
            'email': 'cyclist@example.com',
            'display_name': 'Road Cyclist',
            'password': 'cycle123',
            'is_admin': False
        }
    ]
    
    user_ids = {}
    for user in demo_users:
        cursor.execute('''
        INSERT OR IGNORE INTO users (
            username, email, display_name, password_hash, 
            device_id, is_admin, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user['username'],
            user['email'],
            user['display_name'],
            hash_password(user['password']),
            f"{user['username']}-device-001",
            user['is_admin'],
            datetime.utcnow()
        ))
        
        # Get user ID
        cursor.execute('SELECT id FROM users WHERE username = ?', (user['username'],))
        result = cursor.fetchone()
        if result:
            user_ids[user['username']] = result[0]
    
    print("   • Creating demo tracks...")
    
    # Demo tracks
    if 'testuser' in user_ids:
        # Solo hiking track
        cursor.execute('''
        INSERT OR IGNORE INTO tracks (
            name, description, created_by, is_public, track_type, 
            is_collaborative, collaboration_mode, invite_code,
            created_at, started_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'Tscheppaschlucht Solo Hike',
            'A beautiful solo hiking trail through the gorge',
            user_ids['testuser'],
            True,
            'hiking',
            False,
            'private',
            None,
            datetime.utcnow(),
            datetime.utcnow() - timedelta(hours=2)
        ))
        
        # Collaborative cycling track
        cursor.execute('''
        INSERT OR IGNORE INTO tracks (
            name, description, created_by, is_public, track_type,
            is_collaborative, collaboration_mode, invite_code, max_collaborators,
            created_at, started_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'Group Cycling Adventure',
            'Collaborative cycling track - join the fun!',
            user_ids['testuser'],
            True,
            'cycling',
            True,
            'invite_only',
            generate_invite_code(),
            10,
            datetime.utcnow(),
            datetime.utcnow() - timedelta(hours=1)
        ))
    
    print("   • Creating demo waypoints...")
    
    # Get track IDs for waypoints
    cursor.execute('SELECT id, name FROM tracks WHERE created_by = ?', (user_ids.get('testuser', 1),))
    tracks = cursor.fetchall()
    
    if tracks:
        track_id = tracks[0][0]  # First track
        
        # Demo GPS points (simulating a short trail)
        base_lat, base_lon = 47.1234, 15.6543
        for i in range(10):
            cursor.execute('''
            INSERT INTO waypoints (
                track_id, created_by, latitude, longitude, altitude,
                accuracy, timestamp, waypoint_type, speed, course,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                track_id,
                user_ids['testuser'],
                base_lat + (i * 0.001),  # Move north
                base_lon + (i * 0.0005),  # Move east
                850 + (i * 10),  # Elevation gain
                5.0,
                datetime.utcnow() - timedelta(minutes=30-i*3),
                'gps_track',
                1.2 + (i * 0.1),  # Increasing speed
                90 + (i * 2),  # Changing direction
                json.dumps({
                    'isFromKalmanFilter': True,
                    'realtime_sync': True,
                    'demo_data': True
                })
            ))
        
        # Demo media waypoints
        cursor.execute('''
        INSERT INTO waypoints (
            track_id, created_by, latitude, longitude, altitude,
            accuracy, timestamp, waypoint_type, title, description,
            is_public, poi_type, importance_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            track_id,
            user_ids['testuser'],
            base_lat + 0.005,
            base_lon + 0.003,
            890,
            3.0,
            datetime.utcnow() - timedelta(minutes=15),
            'photo',
            'Beautiful Waterfall',
            'Amazing waterfall view from the trail',
            True,
            'waterfall',
            8.5
        ))

def show_database_summary(cursor):
    """Show a summary of the created database"""
    
    print("\n📊 Database Summary")
    print("=" * 30)
    
    # Tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    print(f"📋 Tables created: {len(tables)}")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"   • {table[0]}: {count} records")
    
    # Indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
    indexes = cursor.fetchall()
    print(f"\n📈 Indexes created: {len(indexes)}")
    
    # Demo users (if any)
    cursor.execute("SELECT username, email, is_admin FROM users")
    users = cursor.fetchall()
    if users:
        print(f"\n👥 Users created: {len(users)}")
        for user in users:
            role = "Admin" if user[2] else "User"
            print(f"   • {user[0]} ({user[1]}) - {role}")
    
    print(f"\n💾 Database file: {DB_PATH}")
    if os.path.exists(DB_PATH):
        size = os.path.getsize(DB_PATH)
        print(f"📏 Database size: {size:,} bytes")

def main():
    """Main function with CLI argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Initialize ARTrack database from scratch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--reset', 
        action='store_true',
        help='Delete existing database and start fresh'
    )
    
    parser.add_argument(
        '--demo-data',
        action='store_true', 
        help='Create demo users and sample tracks for testing'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of existing database before reset'
    )
    
    args = parser.parse_args()
    
    try:
        init_database(reset=args.reset, demo_data=args.demo_data)
        
        print("\n🎉 Database initialization completed!")
        
        if args.demo_data:
            print("\n🔑 Demo Login Credentials:")
            print("   • test@example.com / testpass123 (Regular User)")
            print("   • admin@example.com / admin123 (Admin User)")
            print("   • hiker1@example.com / hiker123 (Hiker)")
            print("   • cyclist@example.com / cycle123 (Cyclist)")
        
        print("\n🚀 Ready to start ARTrack API!")
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()