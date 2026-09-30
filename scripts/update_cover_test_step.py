import sys

filepath = r'd:\AI_Check_Lac\src\components\binocular\CoverTestStep.jsx'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add import
target_import = "import { captureScreeningFrame } from '../../services/screeningImageCaptureService.js';"
new_import = """import { captureScreeningFrame } from '../../services/screeningImageCaptureService.js';
import { analyzeCoverTest } from '../../services/aiBackendService.js';"""

if target_import not in content:
    print("ERROR: target_import not found!")
    sys.exit(1)

content = content.replace(target_import, new_import, 1)

# 2. Add AI transfer state and effect
target_state = "  const phaseResolveRef = useRef(null);"
new_state = """  const phaseResolveRef = useRef(null);

  // Phase 4.2: AI Transfer Inference state (research experiment only)
  const [aiTransferState, setAiTransferState] = useState({
    status: 'idle', // 'idle' | 'loading' | 'success' | 'error'
    result: null,
    error: null,
    detail: null,
  });
  const hasSentAiTransferRef = useRef(false);

  const requestAiTransfer = useCallback(async (summary) => {
    if (!summary || hasSentAiTransferRef.current) return;
    hasSentAiTransferRef.current = true;
    setAiTransferState({ status: 'loading', result: null, error: null, detail: null });

    try {
      const response = await analyzeCoverTest(summary, {
        sampleId: sessionIdRef.current,
      });

      if (response && response.status === 'TRANSFER_EXPERIMENT') {
        setAiTransferState({
          status: 'success',
          result: response,
          error: null,
          detail: null,
        });
      } else {
        setAiTransferState({
          status: 'error',
          result: null,
          error: response?.message || 'Không thể nhận kết quả từ mô hình AI.',
          detail: response?.detail || response?.error,
        });
      }
    } catch (err) {
      setAiTransferState({
        status: 'error',
        result: null,
        error: 'Lỗi kết nối tới hệ thống AI. Vui lòng thử lại.',
        detail: err?.message,
      });
    }
  }, []);

  useEffect(() => {
    if (coverState === 'FINISHED' && coverSummary && !hasSentAiTransferRef.current) {
      requestAiTransfer(coverSummary);
    }
  }, [coverState, coverSummary, requestAiTransfer]);"""

if target_state not in content:
    print("ERROR: target_state not found!")
    sys.exit(1)

content = content.replace(target_state, new_state, 1)

# 3. Add reset in startCoverTestProtocol
target_reset = "    currentBaselineRef.current = null;"
new_reset = """    currentBaselineRef.current = null;
    hasSentAiTransferRef.current = false;
    setAiTransferState({ status: 'idle', result: null, error: null, detail: null });"""

if target_reset not in content:
    print("ERROR: target_reset not found!")
    sys.exit(1)

content = content.replace(target_reset, new_reset, 1)

# 4. Add UI Card before the buttons in coverState === 'FINISHED'
target_ui = """          <div style={{ display: 'flex', gap: '14px', maxWidth: '640px', margin: '0 auto' }}>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ flex: 1 }}
              onClick={startCoverTestProtocol}
            >
              🔄 Đo lại Cover Test
            </button>"""

ai_card_ui = """          {/* Phase 4.2: AI Transfer Experiment Card */}
          <div
            className="ai-transfer-experiment-card"
            style={{
              maxWidth: '640px',
              margin: '0 auto 20px',
              background: 'rgba(15, 23, 42, 0.65)',
              border: '1px solid rgba(59, 130, 246, 0.25)',
              borderRadius: '12px',
              padding: '18px 20px',
              boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '1.2rem' }}>🔬</span>
                <h4 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#60a5fa' }}>
                  AI Transfer Experiment
                </h4>
              </div>
              <span
                style={{
                  fontSize: '0.72rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  padding: '3px 8px',
                  borderRadius: '999px',
                  background: 'rgba(59, 130, 246, 0.15)',
                  color: '#93c5fd',
                  border: '1px solid rgba(59, 130, 246, 0.3)',
                }}
              >
                Nghiên cứu / Thử nghiệm
              </span>
            </div>

            {/* Scientific Disclaimer */}
            <div
              style={{
                background: 'rgba(234, 179, 8, 0.1)',
                border: '1px solid rgba(234, 179, 8, 0.3)',
                borderRadius: '8px',
                padding: '10px 14px',
                fontSize: '0.85rem',
                color: '#fde047',
                marginBottom: '14px',
                lineHeight: '1.45',
              }}
            >
              ⚠️ <strong>Lưu ý nghiên cứu:</strong> Kết quả này là đầu ra thử nghiệm của mô hình nghiên cứu (Korean Shared Model), không phải chẩn đoán y khoa.
            </div>

            {/* State rendering */}
            {aiTransferState.status === 'loading' && (
              <div style={{ textAlign: 'center', padding: '16px 0', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                <div style={{ display: 'inline-block', marginBottom: '8px', fontSize: '1.3rem' }}>⏳</div>
                <div>Đang gửi dữ liệu Cover Test thô & thực hiện suy luận mô hình AI...</div>
              </div>
            )}

            {aiTransferState.status === 'error' && (
              <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ color: '#f87171', fontWeight: 600, fontSize: '0.9rem', marginBottom: '4px' }}>
                  {aiTransferState.error || 'Không thể kết nối tới hệ thống AI.'}
                </div>
                <p style={{ margin: '0 0 10px 0', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Vui lòng đảm bảo backend FastAPI đang hoạt động trên cổng được cấu hình (VITE_AI_BACKEND_URL).
                </p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: '0.82rem', padding: '5px 12px' }}
                  onClick={() => {
                    hasSentAiTransferRef.current = false;
                    requestAiTransfer(coverSummary);
                  }}
                >
                  🔄 Thử lại gửi AI
                </button>
              </div>
            )}

            {aiTransferState.status === 'success' && aiTransferState.result && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.86rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 12px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.78rem', marginBottom: '4px' }}>Dự đoán mô hình (Model Prediction):</div>
                    <div style={{ fontSize: '1.15rem', fontWeight: 800, color: aiTransferState.result.prediction === 'NORMAL' ? '#34d399' : '#fbbf24' }}>
                      {aiTransferState.result.prediction}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '10px 12px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
                    <div style={{ color: 'var(--text-dim)', fontSize: '0.78rem', marginBottom: '4px' }}>Xác suất phân lớp (Class Probability):</div>
                    <div style={{ fontSize: '0.86rem', fontWeight: 600 }}>
                      NORMAL: <span style={{ color: '#34d399' }}>{((aiTransferState.result.classProbability?.NORMAL ?? 0) * 100).toFixed(1)}%</span>
                      {' | '}
                      STRABISMUS: <span style={{ color: '#fbbf24' }}>{((aiTransferState.result.classProbability?.STRABISMUS ?? 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '8px 12px', borderRadius: '6px', fontSize: '0.8rem', color: 'var(--text-dim)', lineHeight: '1.6' }}>
                  <div>• <strong>Mô hình:</strong> {aiTransferState.result.model?.name || 'korean_shared_model'} ({aiTransferState.result.model?.version || 'shared-v1.0.0'}, 30 đặc trưng kỹ thuật)</div>
                  <div>• <strong>Domain shift:</strong> <span style={{ color: '#f59e0b', fontWeight: 600 }}>WARNING</span> (Korean eye-tracking research data → RemiCare webcam 10–15 FPS)</div>
                  <div>• <strong>Ý nghĩa lâm sàng:</strong> None (Clinical meaning: null)</div>
                </div>
              </div>
            )}
          </div>

""" + target_ui

if target_ui not in content:
    print("ERROR: target_ui not found!")
    sys.exit(1)

content = content.replace(target_ui, ai_card_ui, 1)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("CoverTestStep.jsx updated successfully!")
