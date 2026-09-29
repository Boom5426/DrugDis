import os
import sys

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drugdis")
for sub in ("decomposition", "splits", ""):
    sys.path.insert(0, os.path.join(CODE, sub))
