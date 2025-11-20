# tests/conftest.py

import os
import sys

# Add the project's src directory to sys.path so tests can import llm_eval_suite.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
