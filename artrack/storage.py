import os
import aiofiles
import hashlib
try:
    import magic  # requires system libmagic
except Exception:  # pragma: no cover
    magic = None
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import uuid
from datetime import datetime

from .config import settings

class StorageService:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.media_dir = self.upload_dir / "media"
        self.thumbnails_dir = self.upload_dir / "thumbnails"
        self.media_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)

    def _generate_filename(self, original_filename: str, waypoint_id: int) -> str:
        """Generate unique filename"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_uuid = str(uuid.uuid4())[:8]
        extension = Path(original_filename).suffix.lower()
        return f"wp_{waypoint_id}_{timestamp}_{file_uuid}{extension}"

    def _calculate_checksum(self, file_content: bytes) -> str:
        """Calculate SHA256 checksum"""
        return hashlib.sha256(file_content).hexdigest()

    def _detect_mime_type(self, file_content: bytes) -> str:
        """Detect MIME type. Prefer python-magic if available, else fallback."""
        if magic is not None:
            try:
                return magic.from_buffer(file_content, mime=True)
            except Exception:
                pass
        # Fallback when libmagic is unavailable
        return "application/octet-stream"

    def _validate_file_type(self, mime_type: str, expected_type: str) -> bool:
        """Validate file type against allowed types"""
        allowed_types = {
            "image": settings.ALLOWED_IMAGE_TYPES,
            "video": settings.ALLOWED_VIDEO_TYPES,
            "audio": settings.ALLOWED_AUDIO_TYPES
        }
        
        return mime_type in allowed_types.get(expected_type, [])

    def _create_thumbnail(self, image_path: Path, thumbnail_path: Path, size: Tuple[int, int] = (300, 300)) -> bool:
        """Create thumbnail for image"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, "JPEG", quality=85, optimize=True)
                return True
        except Exception as e:
            print(f"Failed to create thumbnail: {e}")
            return False

    async def save_media_file(
        self, 
        file_content: bytes, 
        original_filename: str, 
        waypoint_id: int,
        expected_type: str = "image"
    ) -> dict:
        """Save media file and return file info"""
        
        # Validate file size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes")
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_content)
        
        # Validate file type
        if not self._validate_file_type(mime_type, expected_type):
            raise ValueError(f"File type {mime_type} not allowed for {expected_type}")
        
        # Generate filename and paths
        filename = self._generate_filename(original_filename, waypoint_id)
        file_path = self.media_dir / filename
        thumbnail_path = self.thumbnails_dir / f"thumb_{filename}"
        
        # Calculate checksum
        checksum = self._calculate_checksum(file_content)
        
        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        
        # Create thumbnail for images
        thumbnail_url = None
        if expected_type == "image" and mime_type.startswith("image/"):
            if self._create_thumbnail(file_path, thumbnail_path):
                thumbnail_url = f"{settings.BASE_URL}/uploads/artrack/thumbnails/thumb_{filename}"
        
        # Get image dimensions for images
        width, height = None, None
        if expected_type == "image" and mime_type.startswith("image/"):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
            except:
                pass
        
        file_url = f"{settings.BASE_URL}/uploads/artrack/media/{filename}"
        
        return {
            "file_path": str(file_path),
            "file_url": file_url,
            "thumbnail_url": thumbnail_url,
            "filename": filename,
            "original_filename": original_filename,
            "file_size_bytes": len(file_content),
            "mime_type": mime_type,
            "checksum": checksum,
            "width": width,
            "height": height
        }

    def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                
                # Also delete thumbnail if exists
                if path.parent.name == "media":
                    thumbnail_path = self.thumbnails_dir / f"thumb_{path.name}"
                    if thumbnail_path.exists():
                        thumbnail_path.unlink()
                
                return True
        except Exception as e:
            print(f"Failed to delete file {file_path}: {e}")
        return False

    def get_file_info(self, file_path: str) -> Optional[dict]:
        """Get file information"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            stat = path.stat()
            return {
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "exists": True
            }
        except:
            return None

# Global storage service instance
storage_service = StorageService()