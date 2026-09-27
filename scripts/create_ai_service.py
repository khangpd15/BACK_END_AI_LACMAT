import os

service_content = '''/**
 * RemiCare Strabismus AI Backend Service
 * 
 * Manages communication between AI_CHECK_LAC frontend and the Python FastAPI
 * AI Backend for Phase 4.2 Research Transfer Inference.
 * 
 * CRITICAL SCIENTIFIC & ARCHITECTURAL PRINCIPLES:
 * 1. Sends RAW time-series sampling only. Feature extraction occurs strictly on backend.
 * 2. PROHIBITS sending clinical labels, target diagnoses, or ground truth (prevents leakage).
 * 3. Does NOT continuously stream camera frames; dispatches exactly ONE request after Cover Test completion.
 * 4. All model outputs are labeled as TRANSFER_EXPERIMENT (domain shift warning active, clinical meaning: null).
 */

const DEFAULT_BACKEND_URL = 'http://localhost:8000';
const DEFAULT_TIMEOUT_MS = 15000;

/**
 * Resolves the configured AI backend URL from environment variables.
 * Fallback to http://localhost:8000 for local development.
 * 
 * @returns {string} Clean base URL without trailing slash
 */
export function getBackendBaseUrl() {
  const envUrl = typeof import.meta !== 'undefined' && import.meta.env?.VITE_AI_BACKEND_URL;
  const rawUrl = envUrl || DEFAULT_BACKEND_URL;
  return rawUrl.replace(/\\/+$/, '');
}

/**
 * Builds a standardized observation-only transfer request payload from Cover Test summary.
 * Strictly excludes clinical labels, ground truth, or verdict predictions.
 * 
 * @param {Object} coverSummary - Summary object containing completed cycles
 * @param {string} [sampleId] - Unique sample UUID
 * @returns {Object} Standardized Cover Test transfer request payload
 */
export function buildTransferPayload(coverSummary, sampleId = null) {
  const effectiveSampleId =
    sampleId ||
    coverSummary?.sampleId ||
    (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : 'sample-' + Date.now());

  const rawCycles = coverSummary?.cycles || coverSummary?.accumulatedCycles || [];

  const cycles = rawCycles.map((cycle, idx) => {
    const cycleNum = cycle.cycleIndex || cycle.cycleNumber || cycle.cycle || (idx + 1);
    const coveredEye = String(cycle.coveredEye || (cycleNum % 2 === 1 ? 'LEFT' : 'RIGHT')).toUpperCase();
    const trackedEye = String(cycle.trackedEye || (cycleNum % 2 === 1 ? 'RIGHT' : 'LEFT')).toUpperCase();

    const rawSamples = cycle.samples || cycle.rawTrajectory || [];
    const formattedSamples = rawSamples.map((s, sIdx) => ({
      index: typeof s.index === 'number' ? s.index : sIdx,
      t: typeof s.t === 'number' ? s.t : (typeof s.timestamp === 'number' ? s.timestamp : 0),
      phase: s.phase || 'BASELINE',
      leftX: typeof s.leftX === 'number' ? s.leftX : (s.left?.x ?? null),
      leftY: typeof s.leftY === 'number' ? s.leftY : (s.left?.y ?? null),
      leftValid: Boolean(s.leftValid ?? s.left?.valid ?? false),
      rightX: typeof s.rightX === 'number' ? s.rightX : (s.right?.x ?? null),
      rightY: typeof s.rightY === 'number' ? s.rightY : (s.right?.y ?? null),
      rightValid: Boolean(s.rightValid ?? s.right?.valid ?? false),
      trackingQuality: typeof s.trackingQuality === 'number' ? s.trackingQuality : 0.95,
    }));

    return {
      cycle: cycleNum,
      coveredEye,
      trackedEye,
      samples: formattedSamples,
    };
  });

  return {
    schemaVersion: '1.0.0',
    sampleId: effectiveSampleId,
    test: 'COVER_TEST',
    source: {
      device: 'WEBCAM',
      tracker: 'MEDIAPIPE_IRIS',
    },
    cycles,
  };
}

/**
 * Sends a single completed Cover Test payload to the FastAPI AI Backend.
 * 
 * @param {Object} payloadOrSummary - Formatted ScreeningRequest or completed coverSummary
 * @param {Object} [options] - Configuration options
 * @param {number} [options.timeoutMs=15000] - Request timeout in milliseconds
 * @returns {Promise<Object>} Structured inference result or error state
 */
export async function analyzeCoverTest(payloadOrSummary, options = {}) {
  const timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;

  // Validate or build standard payload
  let payload;
  if (payloadOrSummary && Array.isArray(payloadOrSummary.cycles) && payloadOrSummary.sampleId && payloadOrSummary.test) {
    // Already structured payload - strip any accidental target leakage
    const sanitized = { ...payloadOrSummary };
    delete sanitized.clinicalLabel;
    delete sanitized.diagnosis;
    delete sanitized.target;
    delete sanitized.groundTruth;
    delete sanitized.verdict;
    payload = sanitized;
  } else if (payloadOrSummary && (payloadOrSummary.cycles || payloadOrSummary.accumulatedCycles)) {
    payload = buildTransferPayload(payloadOrSummary, options.sampleId);
  } else {
    return {
      status: 'INVALID_PAYLOAD',
      error: 'INVALID_PAYLOAD',
      inputCompatible: false,
      message: 'Dữ liệu kiểm tra che mắt không đúng định dạng.',
    };
  }

  const baseUrl = getBackendBaseUrl();
  const endpoint = `${baseUrl}/api/v1/transfer/strabismus`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      if (response.status === 422) {
        return {
          status: 'INPUT_INCOMPATIBLE',
          error: 'INVALID_TIME_SERIES',
          inputCompatible: false,
          message: 'Dữ liệu chuỗi thời gian không hợp lệ hoặc không tương thích mô hình.',
          detail: data?.detail || response.statusText,
        };
      }

      return {
        status: 'SERVER_ERROR',
        error: 'MODEL_ERROR',
        inputCompatible: false,
        message: 'Không thể xử lý suy luận mô hình AI.',
        detail: data?.detail || `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    return data;
  } catch (err) {
    clearTimeout(timeoutId);

    if (err.name === 'AbortError') {
      return {
        status: 'BACKEND_TIMEOUT',
        error: 'TIMEOUT',
        inputCompatible: false,
        message: 'Hệ thống AI phản hồi quá thời gian cho phép (15 giây).',
      };
    }

    return {
      status: 'BACKEND_UNAVAILABLE',
      error: 'BACKEND_UNAVAILABLE',
      inputCompatible: false,
      message: 'Không thể kết nối tới hệ thống AI. Vui lòng thử lại.',
      detail: err.message,
    };
  }
}

/**
 * Diagnostics helper: checks backend operational health status.
 * 
 * @returns {Promise<Object>} Health check status or error
 */
export async function checkBackendHealth() {
  const baseUrl = getBackendBaseUrl();
  const endpoint = `${baseUrl}/health`;
  try {
    const response = await fetch(endpoint, { method: 'GET' });
    if (!response.ok) {
      return { status: 'error', code: response.status };
    }
    return await response.json();
  } catch (err) {
    return { status: 'unavailable', error: err.message };
  }
}
'''

target_path = r'd:\AI_Check_Lac\src\services\aiBackendService.js'
with open(target_path, 'w', encoding='utf-8') as f:
    f.write(service_content)
print(f'Wrote {len(service_content)} chars to {target_path}')
