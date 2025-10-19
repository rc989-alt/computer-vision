#!/usr/bin/env python3
"""
Run Planning Team Meeting for Multimodal Research Vision
Focuses on: mistake analysis + future pipeline planning
"""

import sys
import os
from pathlib import Path

# Load API keys
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

    # Load the mistake analysis document
    test_file = Path(__file__).parent / 'agents' / 'prompts' / 'first_meeting_test.md'
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    with open(test_file, 'r') as f:
        mistake_analysis = f.read()

    print("🚀 Initializing Planning Team Meeting...")
    print("🎯 Goal: Analyze current mistakes + Plan multimodal research future\n")

    orchestrator = MeetingOrchestrator(config_path)

    # Create planning-focused topic
    topic = f"""# PLANNING TEAM MEETING: Multimodal Research Vision

## CONTEXT & MISSION

You are the **Planning Team** tasked with:
1. **Analyzing current project mistakes** (from document below)
2. **Setting strategic goals** for multimodal computer vision research
3. **Designing the pipeline** for future iterations

## CURRENT MISTAKE ANALYSIS

{mistake_analysis}

## YOUR PLANNING OBJECTIVES

### 1. MISTAKE ANALYSIS & CORRECTION
- Review the Mistake Map (evidence integrity, methodological risks, operational gaps)
- Validate the P0 pipeline changes
- Assess whether proposed fixes are sufficient

### 2. STRATEGIC GOAL SETTING
Based on current state (V1.0 GO, CoTRR constrained, V2.0 paused):
- Define **3-6 month research objectives** for multimodal system
- Establish **acceptance gates** for next generation
- Set **resource allocation** priorities

### 3. FUTURE PIPELINE DESIGN
Design the architecture for multimodal research including:
- **Data requirements**: scale, quality, diversity
- **Model architecture**: fusion strategy, efficiency constraints
- **Evaluation framework**: metrics, baselines, ablations
- **Deployment path**: shadow→production roadmap

## TEAM ROLES FOR THIS MEETING

**Pre-Architect (Claude Opus)**: Define system blueprint, objectives, acceptance gates
**Tech Analysis Team (Claude Sonnet)**: Quantify feasibility, latency, ROI, scan current artifacts
**Data Analyst (Claude Sonnet)**: Assess data quality, identify gaps, power analysis
**Gemini Feasibility**: Research external architectures, literature, precedents
**Critic (GPT-4)**: Challenge assumptions, probe risks, audit integrity
**Moderator (Claude Opus)**: Synthesize, create decision matrix, execution plan

## REQUIRED OUTPUTS

1. **Mistake Analysis Verdict**: Valid? Sufficient? Additional issues?
2. **Strategic Goals**: 3-6 objectives with measurable success criteria
3. **Architecture Blueprint**: High-level system design for multimodal V3.0
4. **Acceptance Gates**: Quantitative thresholds for progression
5. **Execution Plan**:
   - **P0 (24h)**: Critical corrections
   - **P1 (1-2 weeks)**: Foundation building
   - **P2 (1-3 months)**: Full multimodal research iteration
6. **Decision Matrix**: GO/PAUSE_AND_FIX/REJECT with evidence paths

## EVIDENCE RULE
Every precise number must include evidence path: `file#run_id#line`

Provide your analysis according to your role's output format.
"""

    # Run planning meeting with strategy optimized for planning
    result = orchestrator.run_meeting(
        topic=topic,
        strategy="hierarchical",  # Teams → Critic → Moderator
        rounds=1  # Single focused round to avoid rate limits
    )

    print(f"\n{'='*60}")
    print("✨ Planning Meeting Complete!")
    print(f"{'='*60}\n")
    print(f"📊 Reports saved in: multi-agent/reports/")
    print(f"📁 Check reports/ directory for:")
    print(f"   - Strategic goals and objectives")
    print(f"   - Architecture blueprint")
    print(f"   - Execution plan with priorities")


if __name__ == "__main__":
    main()
