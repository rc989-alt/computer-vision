#!/usr/bin/env python3
"""
CVPR 2025 Planning Team Meeting - Week 1 Kickoff
Focus: Assign Week 1 validation tasks for cross-architecture attention collapse testing
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Load API keys from Google Drive
env_file = Path.home() / 'Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
else:
    print(f"⚠️ API keys not found at {env_file}")

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from autonomous_coordinator import AutonomousCoordinator

def main():
    """Run CVPR 2025 Planning Team Meeting - Week 1 Kickoff"""

    print("="*80)
    print("🎯 CVPR 2025 PLANNING TEAM MEETING - WEEK 1 KICKOFF")
    print("="*80)
    print("\n📋 Mission: Assign Week 1 validation tasks for attention collapse research")
    print("🗓️  Timeline: Abstract Nov 6 (23 days), Full Paper Nov 13 (30 days)")
    print("⏰ Week 1 GO/NO-GO Decision: Oct 20 (7 days)\n")

    # Load mission briefing
    briefing_file = parent_dir / 'CVPR_2025_MISSION_BRIEFING_FOR_5_AGENT_SYSTEM.md'
    if not briefing_file.exists():
        print(f"❌ Mission briefing not found: {briefing_file}")
        return

    with open(briefing_file, 'r') as f:
        mission_briefing = f.read()

    # Create meeting topic with CVPR 2025 context
    topic = f"""# CVPR 2025 PLANNING TEAM MEETING: Week 1 Kickoff

## 🎯 MISSION CONTEXT

{mission_briefing}

---

## 📊 TODAY'S MEETING OBJECTIVES

This is the **first Planning Team meeting** for the CVPR 2025 research mission. Your goal is to:

1. **Review and confirm** the CVPR 2025 research strategy
2. **Assign Week 1 tasks** for cross-architecture attention collapse validation
3. **Create pending_actions.json** with HIGH priority experiment tasks for Executive Team

## 🔬 WEEK 1 CRITICAL TASKS (Days 1-2 - TODAY)

The following tasks must be assigned with clear ownership and acceptance criteria:

### **Task 1: Adapt Diagnostic Tools for CLIP**
- Generalize `research/attention_analysis.py` for external models
- Create model-agnostic interfaces
- Build visualization pipeline
- **Owner:** TBD by Planning Team
- **Deadline:** Oct 15 (24 hours)

### **Task 2: Set Up External Model Environments**
- Set up CLIP/OpenCLIP testing environment
- Set up ALIGN model integration (if accessible)
- Prepare Flamingo/BLIP testing (if time allows)
- Ensure GPU access (A100 confirmed available)
- **Owner:** TBD by Planning Team
- **Deadline:** Oct 15 (24 hours)

### **Task 3: Design Cross-Architecture Experiments**
- Define metrics for detecting attention collapse across models
- Design statistical validation framework (CI95, p-values, effect sizes)
- Set up MLflow tracking for cross-model experiments
- **Owner:** TBD by Planning Team
- **Deadline:** Oct 16 (48 hours)

### **Task 4: Run First CLIP Diagnostic**
- Execute attention analysis on CLIP model
- Measure modality contribution scores
- Document findings systematically
- **Owner:** TBD by Planning Team
- **Deadline:** Oct 17 (72 hours)

## 👥 PLANNING TEAM ROLES

**Strategic Leader (Opus 4):**
- Lead this meeting and synthesize team input
- Decide task assignments and priorities
- Create final pending_actions.json

**Empirical Validation Lead (Sonnet 4):**
- Design statistical validation framework
- Define metrics for attention collapse detection
- Specify acceptance criteria for Week 1 experiments

**Critical Evaluator (GPT-4):**
- Challenge assumptions in the research strategy
- Identify methodological risks
- Ensure reproducibility of diagnostic tools

**Gemini Research Advisor (Gemini Flash):**
- Assess accessibility of external models (CLIP, ALIGN, Flamingo)
- Identify recent papers on multimodal fusion attention
- Flag any prior work on attention collapse

## 📤 REQUIRED OUTPUT: pending_actions.json

Create a `pending_actions.json` file with the following structure:

```json
{{
  "meeting_id": "cvpr_planning_week1_kickoff_{{timestamp}}",
  "generated_at": "{{ISO timestamp}}",
  "decisions": [
    {{
      "priority": "HIGH",
      "action": "Adapt attention_analysis.py for CLIP model integration",
      "owner": "ops_commander",
      "rationale": "Required for Week 1 cross-architecture validation",
      "deadline": "2025-10-15T18:00:00Z",
      "acceptance_criteria": [
        "CLIP model loads successfully",
        "Attention weights extracted for all layers",
        "Modality contribution scores computed",
        "Visualization pipeline generates plots"
      ],
      "evidence_paths": ["research/attention_analysis.py", "research/02_v2_research_line/SUMMARY.md#attention-collapse"]
    }},
    {{
      "priority": "HIGH",
      "action": "Set up CLIP/OpenCLIP testing environment on A100 GPU",
      "owner": "ops_commander",
      "rationale": "Infrastructure required for Week 1 experiments",
      "deadline": "2025-10-15T18:00:00Z",
      "acceptance_criteria": [
        "CLIP model downloads successfully",
        "GPU memory sufficient for inference",
        "Test inference completes successfully",
        "Environment documented in README"
      ],
      "evidence_paths": ["research/colab/", "CVPR_2025_MISSION_BRIEFING_FOR_5_AGENT_SYSTEM.md#week-1"]
    }}
  ],
  "acceptance_gates": {{
    "week1_validation": {{
      "min_models_tested": 3,
      "min_models_with_collapse": 2,
      "statistical_threshold": "p<0.05",
      "attention_imbalance_threshold": ">80%"
    }}
  }},
  "next_meeting": {{
    "type": "planning",
    "scheduled": "2025-10-20T10:00:00Z",
    "purpose": "Week 1 GO/NO-GO decision based on validation results"
  }}
}}
```

## ⚠️ CRITICAL SUCCESS FACTORS

**Week 1 GO/NO-GO Decision (Oct 20):**
- **GO Signal:** ≥2 external models show >80% attention imbalance (p<0.05)
- **Pivot Signal:** Only V2 shows severe imbalance (scope to focused case study)
- **Diagnostic Success:** Tools work on all tested models

**Research Philosophy:**
❌ DON'T: Focus only on fixing our V2 model's specific failure
✅ DO: Investigate whether this represents a broader phenomenon
✅ DO: Propose diagnostic methods that work across different architectures
✅ DO: Contribute insights that benefit the entire research community

## 🎤 YOUR TURN

Please provide your analysis according to your role:

**Strategic Leader:** Lead discussion, synthesize input, create final pending_actions.json
**Empirical Validation Lead:** Design statistical framework, define metrics
**Critical Evaluator:** Challenge assumptions, identify risks
**Gemini Research Advisor:** Assess model accessibility, prior work

**Evidence Rule:** Every claim must cite evidence: `file_path#run_id#line_number`
"""

    # Initialize autonomous coordinator with consolidated 5-agent config
    config_path = parent_dir / 'configs' / 'consolidated_5_agent_coordination.yaml'

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return

    print("🤖 Initializing Planning Team (4 agents)...")
    print("   - Strategic Leader (Opus 4)")
    print("   - Empirical Validation Lead (Sonnet 4)")
    print("   - Critical Evaluator (GPT-4)")
    print("   - Gemini Research Advisor (Gemini Flash)\n")

    try:
        coordinator = AutonomousCoordinator(str(config_path))

        # Run planning meeting
        print("🚀 Starting Planning Team meeting...\n")
        result = coordinator.run_planning_meeting(topic)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save transcript
        transcript_dir = parent_dir / 'reports' / 'planning' / 'transcripts'
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = transcript_dir / f'cvpr_week1_kickoff_{timestamp}.md'

        with open(transcript_file, 'w') as f:
            f.write(f"# CVPR 2025 Planning Team Meeting - Week 1 Kickoff\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Meeting ID:** cvpr_planning_week1_kickoff_{timestamp}\n\n")
            f.write("---\n\n")
            f.write(result.get('transcript', 'No transcript available'))

        print(f"\n{'='*80}")
        print("✅ CVPR 2025 Planning Meeting Complete!")
        print(f"{'='*80}\n")
        print(f"📄 Transcript: {transcript_file}")
        print(f"📊 Check reports/handoff/pending_actions.json for Week 1 tasks")
        print(f"\n🎯 Next Step: Run Executive Team meeting to execute Week 1 experiments")

    except Exception as e:
        print(f"\n❌ Error running meeting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
