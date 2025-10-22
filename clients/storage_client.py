"""HTTP client for storage-api service."""
import httpx
from typing import Optional, Dict, Any


class StorageClient:
    """Client for communicating with storage-api service."""

    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def save(
        self,
        data: bytes,
        original_filename: str,
        owner_user_id: int,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save file to storage-api."""
        # For now, return a mock response until storage-api is deployed
        # TODO: Implement actual HTTP call to storage-api
        return {
            "object_key": f"mock_{original_filename}",
            "original_filename": original_filename,
            "file_url": f"https://api-storage.arkturian.com/files/mock_{original_filename}",
            "thumbnail_url": None,
            "webview_url": None,
            "mime_type": "application/octet-stream",
            "file_size_bytes": len(data),
            "checksum": "mock_checksum",
        }

    async def save_reference(
        self,
        data: bytes,
        original_filename: str,
        owner_user_id: int,
        context: Optional[str] = None,
        reference_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save reference to existing file in storage-api."""
        # For now, return a mock response
        return {
            "object_key": f"ref_{original_filename}",
            "original_filename": original_filename,
            "file_url": reference_path or f"https://api-storage.arkturian.com/files/ref_{original_filename}",
            "thumbnail_url": None,
            "webview_url": None,
            "mime_type": "application/octet-stream",
            "file_size_bytes": len(data),
            "checksum": "mock_checksum",
        }

    async def update_file(self, object_key: str, data: bytes) -> Dict[str, Any]:
        """Update existing file in storage-api."""
        # For now, return a mock response
        return {
            "file_url": f"https://api-storage.arkturian.com/files/{object_key}",
            "thumbnail_url": None,
            "webview_url": None,
            "mime_type": "application/octet-stream",
            "file_size_bytes": len(data),
            "checksum": "mock_checksum_updated",
        }

    async def enqueue_ai_safety_and_transcoding(self, object_id: int):
        """Enqueue AI processing job for storage object."""
        # For now, do nothing
        # TODO: Implement actual HTTP call to storage-api
        pass


# Global instance
storage_client = StorageClient()


# Mock storage service for backward compatibility
class MockGenericStorage:
    """Mock storage service that delegates to HTTP client."""

    async def save(self, *args, **kwargs):
        return await storage_client.save(*args, **kwargs)

    async def save_reference(self, *args, **kwargs):
        return await storage_client.save_reference(*args, **kwargs)

    async def update_file(self, *args, **kwargs):
        return await storage_client.update_file(*args, **kwargs)


generic_storage = MockGenericStorage()


async def enqueue_ai_safety_and_transcoding(object_id: int):
    """Enqueue AI processing for storage object."""
    await storage_client.enqueue_ai_safety_and_transcoding(object_id)
