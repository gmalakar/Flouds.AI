# =============================================================================
# File: torch_repro.py
# Date: 2026-01-13
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# =============================================================================

﻿import torch

print("torch", torch.__version__)
torch.set_num_threads(1)
from torch import nn

print("constructing Linear...")
l = nn.Linear(1024, 1024)
print("ok")
