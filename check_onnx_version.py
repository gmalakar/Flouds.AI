# =============================================================================
# File: check_onnx_version.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import onnx

print("ONNX version:", onnx.__version__)
print("ONNX IR version:", onnx.IR_VERSION)
print("ONNX default opset:", onnx.helper.VERSION_TABLE[-1])
