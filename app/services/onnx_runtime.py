"""Share bounded CPU sessions across image screening pipelines."""

import os
from pathlib import Path
import threading

import onnxruntime as ort


_sessions = {}
_lock = threading.Lock()


def get_cpu_session(path):
    resolved = str(Path(path).resolve())
    with _lock:
        if resolved not in _sessions:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = max(1, min(4, int(os.getenv("ONNX_CPU_THREADS", "1"))))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            _sessions[resolved] = ort.InferenceSession(
                resolved, sess_options=options, providers=["CPUExecutionProvider"]
            )
        return _sessions[resolved]
