#!/usr/bin/env python3
"""
Run meeting with file context loaded
"""

import sys
from pathlib import Path

# Load API keys
import os
env_file = Path(__file__).parent.parent / 'research' / 'api_keys.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

sys.path.insert(0, str(Path(__file__).parent))

from run_meeting import MeetingOrchestrator


def main():
    config_path = Path(__file__).parent / 'configs' / 'meeting.yaml'

    # Load the first_meeting_test.md content
    test_file = Path(__file__).parent / 'agents' / 'prompts' / 'first_meeting_test.md'
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    with open(test_file, 'r') as f:
        test_content = f.read()

    print("🚀 Initializing Multi-Agent System with Context...")
    orchestrator = MeetingOrchestrator(config_path)

    # Create detailed topic with full context
    topic = f"""Review the end-to-end pipeline analysis below and provide verdicts on GO/PAUSE_AND_FIX/REJECT for each track.

===== PIPELINE ANALYSIS DOCUMENT =====

{test_content}

===== END OF DOCUMENT =====

Evaluate:
1. **Mistake Map** - Are the identified issues valid and prioritized correctly?
2. **Pipeline Changes** - Are the proposed changes (P0) necessary and sufficient?
3. **Track Verdicts**:
   - V1.0 Production Line - Current status and recommendations
   - CoTRR Lightweight Line - Feasibility and constraints
   - V2.0 Multimodal Research - Rescue plan viability
4. **V2 Rescue Plan Acceptance Gates** - Are the 5 acceptance gates appropriate?
5. **Priority Actions** - Which worklist items (24h/48-72h/Week 1) should be executed first?

Provide your analysis according to your role's perspective and output format.
"""

    # Run meeting with loaded context
    result = orchestrator.run_meeting(
        topic=topic,
        strategy="hierarchical",
        rounds=2
    )

    print(f"\n{'='*60}")
    print("✨ Meeting Complete!")
    print(f"{'='*60}\n")
    print(f"Reports saved in: multi-agent/reports/")


if __name__ == "__main__":
    main()
