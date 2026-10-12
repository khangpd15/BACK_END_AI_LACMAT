"""Image API for the trained Hirschberg localization and screening pipeline."""
from typing import Literal
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from app.services.trained_hirschberg_service import (
    predict_trained_hirschberg, trained_hirschberg_status,
)

router = APIRouter(prefix='/api/v1/hirschberg', tags=['Hirschberg'])


@router.get('/health')
async def health():
    return trained_hirschberg_status()


@router.post('/predict', summary='Run trained Hirschberg CNNs with landmark and capture QC')
async def predict(
    image: UploadFile = File(...),
    orientation: Literal['auto', 'unmirrored', 'mirrored'] = Form('auto'),
    kind: Literal['auto', 'fullface', 'bilateral'] = Form('auto'),
    fixation: Literal['unknown', 'confirmed', 'offaxis'] = Form('unknown'),
):
    try:
        raw = await image.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, 'Image exceeds 10 MB')
        if not raw:
            raise HTTPException(400, 'Empty image')
        return await run_in_threadpool(predict_trained_hirschberg, raw,
                                      image.filename or 'image', orientation, kind, fixation)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, 'Trained Hirschberg runtime unavailable') from exc
    finally:
        await image.close()
