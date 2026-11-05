import sys
from pathlib import Path
agent = Path(__file__).parent / "agent"
if agent.exists():
    sys.path.insert(0, str(agent))
