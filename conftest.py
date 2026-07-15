# Makes the repo root importable so `import amyg_inv` works without an install step.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
