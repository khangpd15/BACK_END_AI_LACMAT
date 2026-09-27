import sys

content = '''/**
 * AI BACKEND SERVICE TEST CASES
 * Verifies Phase 4.2 Frontend -> FastAPI communication, payload formation,
 * error states, single-call dispatch, and result contracts.
 */

import {
  getBackendBaseUrl,
  buildTransferPayload,
  analyzeCoverTest,
} from './aiBackendService.js';

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

export async function runAiBackendServiceTestCases() {
  const results = [];
  const test = async (name, fn) => {
    try {
      await fn();
      results.push({ name, passed: true });
    } catch (error) {
      results.push({ name, passed: false, error: error.message });
    }
  };

  // Mock sample generator
  const createMockCycle = (cycleNum = 1) => ({
    cycleIndex: cycleNum,
    coveredEye: cycleNum % 2 === 1 ? 'LEFT' : 'RIGHT',
    trackedEye: cycleNum % 2 === 1 ? 'RIGHT' : 'LEFT',
    samples: [
      {
        index: 0,
        t: 0,
        phase: 'BASELINE',
        leftX: 0.44,
        leftY: 0.60,
        leftValid: true,
        rightX: 0.58,
        rightY: 0.60,
        rightValid: true,
        trackingQuality: 0.95,
      },
      {
        index: 1,
        t: 66,
        phase: 'BASELINE',
        leftX: 0.441,
        leftY: 0.601,
        leftValid: true,
        rightX: 0.581,
        rightY: 0.601,
        rightValid: true,
        trackingQuality: 0.95,
      },
    ],
  });

  const mockSummary = {
    sampleId: 'test-sample-uuid-001',
    validCycles: 3,
    cycles: [createMockCycle(1), createMockCycle(2), createMockCycle(3)],
  };

  // 1. Service builds correct request
  await test('1. Service builds correct request schema without target leakage', () => {
    const payload = buildTransferPayload(mockSummary, 'test-sample-uuid-001');
    assert(payload.schemaVersion === '1.0.0', 'Invalid schemaVersion');
    assert(payload.sampleId === 'test-sample-uuid-001', 'sampleId mismatch');
    assert(payload.test === 'COVER_TEST', 'test field must be COVER_TEST');
    assert(payload.source?.device === 'WEBCAM', 'source.device missing');
    assert(payload.source?.tracker === 'MEDIAPIPE_IRIS', 'source.tracker missing');
    assert(payload.cycles.length === 3, 'Must contain 3 cycles');
    assert(payload.cycles[0].cycle === 1, 'Cycle index mismatch');
    assert(payload.cycles[0].coveredEye === 'LEFT', 'Covered eye mismatch');
    assert(payload.cycles[0].trackedEye === 'RIGHT', 'Tracked eye mismatch');
    assert(payload.cycles[0].samples.length === 2, 'Samples missing');
    // Ensure observation only: NO clinical labels
    assert(payload.clinicalLabel === undefined, 'clinicalLabel must not be sent');
    assert(payload.diagnosis === undefined, 'diagnosis must not be sent');
    assert(payload.target === undefined, 'target must not be sent');
    assert(payload.groundTruth === undefined, 'groundTruth must not be sent');
  });

  // 2. Correct backend URL
  await test('2. Correct backend URL resolution', () => {
    const url = getBackendBaseUrl();
    assert(typeof url === 'string' && url.length > 0, 'Invalid backend URL');
    assert(!url.endsWith('/'), 'URL should not have trailing slash');
  });

  // 3. POST request headers and method
  await test('3. POST request structure verification', async () => {
    const originalFetch = globalThis.fetch;
    let interceptedReq = null;
    globalThis.fetch = async (url, init) => {
      interceptedReq = { url, init };
      return {
        ok: true,
        status: 200,
        json: async () => ({ status: 'TRANSFER_EXPERIMENT' }),
      };
    };

    try {
      await analyzeCoverTest(mockSummary);
      assert(interceptedReq !== null, 'Fetch was not called');
      assert(interceptedReq.url.endsWith('/api/v1/transfer/strabismus'), 'Endpoint URL incorrect');
      assert(interceptedReq.init.method === 'POST', 'Must use POST method');
      assert(interceptedReq.init.headers['Content-Type'] === 'application/json', 'Content-Type must be application/json');
      const body = JSON.parse(interceptedReq.init.body);
      assert(body.test === 'COVER_TEST', 'Body payload invalid');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 4. Successful response handling
  await test('4. Successful response parsing', async () => {
    const originalFetch = globalThis.fetch;
    const mockResponse = {
      sampleId: 'test-sample-uuid-001',
      status: 'TRANSFER_EXPERIMENT',
      inputCompatible: true,
      prediction: 'NORMAL',
      classProbability: { NORMAL: 0.85, STRABISMUS: 0.15 },
      domainShiftWarning: true,
      clinicalMeaning: null,
      model: { name: 'korean_shared_model', version: 'shared-v1.0.0' },
    };

    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => mockResponse,
    });

    try {
      const res = await analyzeCoverTest(mockSummary);
      assert(res.status === 'TRANSFER_EXPERIMENT', 'Expected status TRANSFER_EXPERIMENT');
      assert(res.prediction === 'NORMAL', 'Expected prediction NORMAL');
      assert(res.classProbability.NORMAL === 0.85, 'Class probability missing');
      assert(res.domainShiftWarning === true, 'domainShiftWarning must be true');
      assert(res.clinicalMeaning === null, 'clinicalMeaning must be null');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 5. HTTP error handling (422)
  await test('5. HTTP 422 error mapped to INPUT_INCOMPATIBLE', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: async () => ({ detail: 'Cycle 1: non-monotonic timestamps' }),
    });

    try {
      const res = await analyzeCoverTest(mockSummary);
      assert(res.status === 'INPUT_INCOMPATIBLE', 'Expected INPUT_INCOMPATIBLE');
      assert(res.error === 'INVALID_TIME_SERIES', 'Expected INVALID_TIME_SERIES');
      assert(res.inputCompatible === false, 'inputCompatible must be false');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 6. Network error handling
  await test('6. Network error mapped to BACKEND_UNAVAILABLE', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      throw new Error('Failed to fetch');
    };

    try {
      const res = await analyzeCoverTest(mockSummary);
      assert(res.status === 'BACKEND_UNAVAILABLE', 'Expected BACKEND_UNAVAILABLE');
      assert(res.error === 'BACKEND_UNAVAILABLE', 'Expected error code BACKEND_UNAVAILABLE');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 7. Timeout handling
  await test('7. Timeout mapped to BACKEND_TIMEOUT', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      const abortError = new Error('The operation was aborted');
      abortError.name = 'AbortError';
      throw abortError;
    };

    try {
      const res = await analyzeCoverTest(mockSummary, { timeoutMs: 10 });
      assert(res.status === 'BACKEND_TIMEOUT', 'Expected BACKEND_TIMEOUT');
      assert(res.error === 'TIMEOUT', 'Expected TIMEOUT');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 8. No request before Cover Test completion
  await test('8. Reject execution before Cover Test completion or invalid payload', async () => {
    const res = await analyzeCoverTest(null);
    assert(res.status === 'INVALID_PAYLOAD', 'Should reject null payload');
    const emptyRes = await analyzeCoverTest({});
    assert(emptyRes.status === 'INVALID_PAYLOAD', 'Should reject empty payload');
  });

  // 9. Exactly one request after completion guard simulation
  await test('9. Exactly one request after completion guard simulation', async () => {
    let callCount = 0;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      callCount++;
      return {
        ok: true,
        status: 200,
        json: async () => ({ status: 'TRANSFER_EXPERIMENT' }),
      };
    };

    try {
      // Simulate component hasSent guard
      let hasSent = false;
      const onFinished = async (summary) => {
        if (hasSent) return;
        hasSent = true;
        await analyzeCoverTest(summary);
      };

      await onFinished(mockSummary);
      await onFinished(mockSummary); // Second invocation should be ignored
      await onFinished(mockSummary); // Third invocation should be ignored

      assert(callCount === 1, `Expected exactly 1 request, got ${callCount}`);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  // 10. Result rendering data integrity
  await test('10. Result data integrity matches UI requirements', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        sampleId: 'test-sample-uuid-001',
        status: 'TRANSFER_EXPERIMENT',
        inputCompatible: true,
        prediction: 'STRABISMUS',
        classProbability: { NORMAL: 0.28, STRABISMUS: 0.72 },
        domainShiftWarning: true,
        clinicalMeaning: null,
        model: { name: 'korean_shared_model', version: 'shared-v1.0.0' },
      }),
    });

    try {
      const res = await analyzeCoverTest(mockSummary);
      assert(res.prediction === 'NORMAL' || res.prediction === 'STRABISMUS', 'prediction must be NORMAL or STRABISMUS');
      assert(typeof res.classProbability?.NORMAL === 'number', 'NORMAL probability missing');
      assert(typeof res.classProbability?.STRABISMUS === 'number', 'STRABISMUS probability missing');
      assert(res.domainShiftWarning === true, 'Domain shift warning must be true');
      assert(res.clinicalMeaning === null, 'clinicalMeaning must be null');
      assert(res.model?.name === 'korean_shared_model', 'Model name mismatch');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  return {
    passed: results.filter((r) => r.passed).length,
    total: results.length,
    results,
  };
}
'''

target_path = r'd:\AI_Check_Lac\src\services\aiBackendServiceTestCases.js'
with open(target_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Created {target_path}')
