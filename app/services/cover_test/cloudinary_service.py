"""Cloudinary Image Upload & Storage Service for RemiCare Cover Test.

Provides secure server-side image upload and cleanup for representative images.
Complies with security constraints:
- Credentials loaded strictly from environment variables (CLOUDINARY_URL or CLOUDINARY_CLOUD_NAME, etc.)
- Secrets are NEVER hard-coded or exposed in logs/responses.
- Automatic cleanup (delete_image) mechanism to prevent orphan images if DB insert fails.
"""

from dataclasses import dataclass
import io
import logging
import os
from typing import Any, Dict, Optional, Union
import cloudinary
import cloudinary.uploader
from cloudinary.exceptions import Error as CloudinaryError

from app.config import (
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_FOLDER,
    CLOUDINARY_UPLOAD_PRESET,
    CLOUDINARY_URL,
)

logger = logging.getLogger("remicare.cloudinary_service")


@dataclass
class CloudinaryUploadResult:
    """Structured result returned from successful Cloudinary image upload."""
    secure_url: str
    public_id: str
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    bytes: Optional[int] = None


class CloudinaryService:
    """Manages Cloudinary configuration, uploads, and cleanups."""

    _instance: Optional["CloudinaryService"] = None

    def __init__(
        self,
        cloudinary_url: Optional[str] = None,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        folder: Optional[str] = None,
    ):
        self.folder = folder or CLOUDINARY_FOLDER or "remicare_cover_test"
        self._configured = False
        self.mock_mode = os.getenv("CLOUDINARY_MOCK_MODE", "false").lower() in ("true", "1", "yes")
        self._init_cloudinary(
            cloudinary_url=cloudinary_url or CLOUDINARY_URL or os.getenv("CLOUDINARY_URL", ""),
            cloud_name=cloud_name or CLOUDINARY_CLOUD_NAME or os.getenv("CLOUDINARY_CLOUD_NAME", ""),
            api_key=api_key or CLOUDINARY_API_KEY or os.getenv("CLOUDINARY_API_KEY", ""),
            api_secret=api_secret or CLOUDINARY_API_SECRET or os.getenv("CLOUDINARY_API_SECRET", ""),
        )

    def _init_cloudinary(
        self,
        cloudinary_url: str,
        cloud_name: str,
        api_key: str,
        api_secret: str,
    ) -> None:
        """Initialize Cloudinary configuration safely without exposing credentials."""
        try:
            if cloudinary_url:
                os.environ["CLOUDINARY_URL"] = cloudinary_url
                cloudinary.reset_config()
                cfg = cloudinary.config()
                if cfg.cloud_name and cfg.api_key:
                    self._configured = True
                    logger.info("[CloudinaryInit] Configured via CLOUDINARY_URL for cloud: %s", cfg.cloud_name)
                    return

            if cloud_name and api_key and api_secret:
                cloudinary.config(
                    cloud_name=cloud_name,
                    api_key=api_key,
                    api_secret=api_secret,
                    secure=True,
                )
                self._configured = True
                logger.info("[CloudinaryInit] Configured via discrete keys for cloud: %s", cloud_name)
                return

            cfg = cloudinary.config()
            if cfg.cloud_name and cfg.api_key:
                self._configured = True
                logger.info("[CloudinaryInit] Configured via existing environment for cloud: %s", cfg.cloud_name)
            else:
                self._configured = False
                logger.warning("[CloudinaryInit] Cloudinary credentials not configured.")
        except Exception as e:
            self._configured = False
            logger.error("[CloudinaryInitError] Failed to initialize Cloudinary: %s", e)

    @property
    def is_configured(self) -> bool:
        """Returns True if Cloudinary credentials are valid and configured."""
        return self._configured

    def upload_image(
        self,
        file_bytes: Union[bytes, io.BytesIO],
        folder: Optional[str] = None,
        public_id: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> CloudinaryUploadResult:
        """Uploads image bytes to Cloudinary.
        
        Args:
            file_bytes: Raw JPEG/PNG image bytes or BytesIO stream.
            folder: Cloudinary directory/folder name.
            public_id: Optional custom public ID.
            tags: Optional tags for indexing.

        Returns:
            CloudinaryUploadResult with secure_url, public_id, format, dimensions.

        Raises:
            RuntimeError: If upload fails or Cloudinary returns an error.
        """
        if not file_bytes:
            raise ValueError("Cannot upload empty image data to Cloudinary.")

        target_folder = folder or self.folder
        # If data is BytesIO, convert or read bytes
        payload = file_bytes.getvalue() if isinstance(file_bytes, io.BytesIO) else file_bytes

        # If in explicit mock mode (e.g. offline testing)
        if self.mock_mode:
            import uuid
            pid = public_id or f"mock_{uuid.uuid4().hex[:12]}"
            cname = cloudinary.config().cloud_name or "dun6cug7p"
            full_pid = f"{target_folder}/{pid}" if not pid.startswith(f"{target_folder}/") else pid
            return CloudinaryUploadResult(
                secure_url=f"https://res.cloudinary.com/{cname}/image/upload/v1790686689/{full_pid}.jpg",
                public_id=full_pid,
                format="jpeg",
                width=256,
                height=256,
                bytes=len(payload),
            )

        if not self._configured:
            raise RuntimeError("Cloudinary is not configured. Missing credentials.")

        upload_options: Dict[str, Any] = {
            "folder": target_folder,
            "resource_type": "image",
            "overwrite": True,
        }
        if public_id:
            upload_options["public_id"] = public_id
        if tags:
            upload_options["tags"] = tags
        if CLOUDINARY_UPLOAD_PRESET:
            upload_options["upload_preset"] = CLOUDINARY_UPLOAD_PRESET

        try:
            response = cloudinary.uploader.upload(payload, **upload_options)
            secure_url = response.get("secure_url")
            out_public_id = response.get("public_id")

            if not secure_url or not out_public_id:
                raise RuntimeError("Cloudinary response missing secure_url or public_id")

            result = CloudinaryUploadResult(
                secure_url=secure_url,
                public_id=out_public_id,
                format=response.get("format", "jpeg"),
                width=response.get("width"),
                height=response.get("height"),
                bytes=response.get("bytes"),
            )
            return result
        except CloudinaryError as ce:
            logger.error("[CloudinaryUploadError] Cloudinary API rejected upload: %s", ce)
            raise RuntimeError(f"Cloudinary upload failed: {ce}") from ce
        except Exception as e:
            logger.error("[CloudinaryUploadError] Unexpected upload failure: %s", e)
            raise RuntimeError(f"Cloudinary upload failed: {e}") from e

    def delete_image(self, public_id: str) -> bool:
        """Deletes an image by public_id (used to cleanup orphan image if DB fails)."""
        if not public_id or not self._configured:
            return False
        try:
            del_result = cloudinary.uploader.destroy(public_id, invalidate=True)
            is_ok = del_result.get("result") in ("ok", "not found")
            logger.info("[CloudinaryCleanup] public_id=%s, result=%s", public_id, del_result.get("result"))
            return is_ok
        except Exception as e:
            logger.warning("[CloudinaryCleanupFailed] Failed to delete image %s: %s", public_id, e)
            return False

    @classmethod
    def get_instance(cls) -> "CloudinaryService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


def get_cloudinary_service() -> CloudinaryService:
    """Convenience getter for singleton CloudinaryService."""
    return CloudinaryService.get_instance()
