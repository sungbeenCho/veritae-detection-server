"""pytest configuration - add scripts directory to sys.path for importing standalone CLI scripts."""
import sys
from pathlib import Path

scripts_dir = Path(__file__).parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
