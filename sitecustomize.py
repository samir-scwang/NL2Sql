"""
Automatically make the project `src` directory importable.

When this repository is on `sys.path` (for example when PyCharm adds
the project root), Python will import this module and we can extend
`sys.path` so in-tree packages such as `myllm` are available without
manual tweaks.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
# print(PROJECT_ROOT) #D:\pythonProject\Text2Sql

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
