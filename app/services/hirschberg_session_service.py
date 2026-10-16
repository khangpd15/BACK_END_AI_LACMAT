"""Local metadata/result persistence for Hirschberg; no images or database connections."""
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional
import uuid

from starlette.concurrency import run_in_threadpool
from app.config import PROJECT_ROOT

logger = logging.getLogger("remicare.hirschberg_session_service")
SESSIONS_DIR = PROJECT_ROOT / "data" / "hirschberg" / "sessions"
SESSION_ID_REGEX = re.compile(r"[a-zA-Z0-9_\-.]{1,100}")


def sanitize_floats(obj: Any) -> Any:
    if isinstance(obj, list):
        return [sanitize_floats(item) for item in obj]
    if isinstance(obj, dict):
        return {key: sanitize_floats(value) for key, value in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class HirschbergSessionService:
    """Save and retrieve sessions from the local filesystem."""

    def __init__(self, sessions_base_dir=None):
        self.sessions_base_dir = str(Path(sessions_base_dir or SESSIONS_DIR).resolve())

    def session_directory(self, session_id: str) -> Path:
        root = Path(self.sessions_base_dir)
        if not SESSION_ID_REGEX.fullmatch(session_id) or session_id in ('.', '..'):
            raise ValueError('Invalid session ID')
        target = (root / session_id).resolve()
        if target == root or not target.is_relative_to(root):
            raise ValueError('Invalid session path')
        return target

    def persist_session_local(self, session_id: str, filename: str, orientation: str,
                              input_kind: str, fixation: str, prediction_result: Dict[str, Any],
                              client_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        root = Path(self.sessions_base_dir)
        root.mkdir(parents=True, exist_ok=True)
        sid = re.sub(r'[^a-zA-Z0-9_.-]', '_', (session_id or '').strip())[:100].strip('.')
        sid = sid or f'hirschberg_{uuid.uuid4()}'
        target = self.session_directory(sid)
        while True:
            try:
                target.mkdir(exist_ok=False)
                break
            except FileExistsError:
                sid = f'{sid[:80]}_{uuid.uuid4().hex[:12]}'
                target = self.session_directory(sid)
        try:
            relative_path = target.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative_path = str(target)
        metadata = {'sessionId': sid, 'testType': 'hirschberg',
                    'createdAt': datetime.now(timezone.utc).isoformat(), 'filename': filename,
                    'orientation': orientation, 'inputKind': input_kind, 'fixation': fixation,
                    'clientMetadata': client_metadata or {}}
        saved = {'sessionId': sid, 'sessionPath': relative_path, 'saved': True}
        result = {**prediction_result, **saved}
        # result.json is written last: readers only list fully committed sessions.
        for name, payload in [('metadata', metadata), ('result', result)]:
            temporary = target / f'.{name}.{uuid.uuid4().hex}.tmp'
            temporary.write_text(json.dumps(sanitize_floats(payload), ensure_ascii=False,
                                            indent=2, allow_nan=False), encoding='utf-8')
            os.replace(temporary, target / f'{name}.json')
        logger.info('[HirschbergSessionLocalSaved] sessionId=%s', sid)
        return saved

    async def persist_session(self, session_id: Optional[str], filename: str, orientation: str,
                              input_kind: str, fixation: str, prediction_result: Dict[str, Any],
                              client_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await run_in_threadpool(self.persist_session_local, session_id or '', filename,
                                      orientation, input_kind, fixation, prediction_result, client_metadata)

    def get_session_local(self, session_id: str):
        folder = self.session_directory(session_id)
        result_path = folder / 'result.json'
        if not result_path.is_file():
            return None
        result = json.loads(result_path.read_text(encoding='utf-8'))
        metadata_path = folder / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
        return {**result, 'sessionId': session_id, 'metadata': metadata,
                'createdAt': metadata.get('createdAt'), 'source': 'filesystem'}

    def list_sessions_local(self, limit=50, offset=0):
        root = Path(self.sessions_base_dir)
        if not root.exists():
            return {'sessions': [], 'count': 0, 'limit': limit, 'offset': offset}
        folders = sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                          and (p / 'result.json').is_file()), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        records = []
        for folder in folders[offset:offset + limit]:
            try:
                record = self.get_session_local(folder.name)
                keys = ('sessionId', 'status', 'predictedClass', 'confidence', 'reason', 'modelId',
                        'latencyMs', 'createdAt', 'sessionPath')
                records.append({key: record.get(key) for key in keys})
            except (OSError, ValueError, TypeError):
                logger.warning('Skipping unreadable local session: %s', folder.name)
        return {'sessions': records, 'count': len(folders), 'limit': limit, 'offset': offset}
