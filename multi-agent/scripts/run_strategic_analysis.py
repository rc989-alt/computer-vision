#!/usr/bin/env python3
"""
Strategic Analysis Meeting - Comprehensive Project Review
This script runs a focused multi-agent meeting to analyze:
1. Current research progress (V1/V2/CoTRR lines)
2. Data integrity issues identified
3. New pipeline and planning for multimodal vision project
4. Strategic recommendations based on evidence
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Load environment variables
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
    """Run strategic analysis meeting"""

    # Initialize orchestrator
    config_path = Path(__file__).parent / 'configs' / 'meeting.yaml'

    # Verify project root will be correct
    project_root = Path(__file__).parent.parent
    print(f"📁 Project root: {project_root}")
    print(f"📁 Research dir: {project_root / 'research'}")
    print(f"   Exists: {(project_root / 'research').exists()}\n")

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    print("🚀 Initializing Strategic Analysis Meeting...")
    print("=" * 80)

    orchestrator = MeetingOrchestrator(config_path)

    # Define comprehensive strategic meeting topic
    topic = """
# Strategic Analysis Meeting - Computer Vision Multimodal Project

## Context
This is a comprehensive strategic review of our computer vision pipeline project.
All agents have access to local files in the research/ directory and should analyze
actual results and data to inform recommendations.

## Meeting Objectives

### 1. CURRENT STATE ASSESSMENT
**Analyze actual experiment results from local files:**
- V1 Production Line (`research/01_v1_production_line/`)
  - Status: ✅ Deployed (+14.2% compliance improvement)
  - Performance: Stable, proven in production

- V2 Research Line (`research/02_v2_research_line/`)
  - Latest results: `production_evaluation.json` shows thresholds NOT MET
  - Scientific review: `v2_scientific_review_report.json` reveals:
    * ✅ Statistical significance: nDCG +0.011 (CI95 [+0.0101, +0.0119])
    * ❌ Integrity issues: Visual features masked with minimal performance drop
    * ❌ High correlation (0.99+) between V1/V2 scores suggests linear shift only
    * Decision: PAUSE_AND_FIX (not discard, but needs integrity fixes)

- CoTRR Lightweight Line (`research/03_cotrr_lightweight_line/`)
  - Status: Explored 21 optimization approaches
  - Findings: Some promising directions, needs strategic focus

**Key Data Points to Reference:**
- `research/02_v2_research_line/production_evaluation.json`
- `research/02_v2_research_line/v2_scientific_review_report.json`
- `research/02_v2_research_line/V2_Rescue_Plan_Executive_Summary.md`

### 2. DATA INTEGRITY ANALYSIS
**Critical Issues Identified:**
1. **Visual Feature Anomaly**: Masking visual features causes <0.2% performance drop
   - Indicates potential feature redundancy or insufficient multimodal fusion
   - Evidence: `v2_scientific_review_report.json#feature_ablation`

2. **Score Correlation Concern**: V1/V2 correlation 0.99+
   - Suggests V2 may be doing simple linear transformation of V1 scores
   - Not true multimodal innovation
   - Evidence: `v2_scientific_review_report.json#score_verification`

3. **Threshold Failures**:
   - Compliance improvement: 13.8% actual vs 15% target ❌
   - NDCG improvement: 1.14% actual vs 8% target ❌
   - Low margin rate: 0.98 actual vs 0.1 target ❌
   - Evidence: `production_evaluation.json#thresholds_met`

### 3. MISTAKE ANALYSIS
**What went wrong? Each agent should analyze:**
- Why did V2 multimodal fusion show minimal visual feature contribution?
- Were our evaluation methods sufficient to catch these issues early?
- What architectural decisions led to high V1/V2 score correlation?
- How can we prevent similar issues in future multimodal research?

### 4. NEW STRATEGIC PLANNING

**SHORT-TERM GOALS (Next 2-4 weeks):**
- Should we invest in fixing V2's integrity issues or pivot to new approaches?
- What are the highest-ROI improvements for V1 production system?
- How to design better evaluation frameworks for multimodal systems?

**MEDIUM-TERM GOALS (1-3 months):**
- Define next-generation multimodal architecture with proper fusion
- Establish production-ready pipeline with all guardrails
- Set up continuous validation and monitoring systems

**LONG-TERM VISION (3-6 months):**
- True multimodal fusion with proven independent feature contributions
- Candidate generation + reranking pipeline
- Data flywheel and personalization capabilities

### 5. ACTIONABLE DELIVERABLES
Each agent should provide:
- **Specific action items** with priorities (HIGH/MEDIUM/LOW)
- **Success metrics** for each proposed goal
- **Resource estimates** (time, compute, team effort)
- **Risk mitigation** strategies for identified issues
- **Evidence-based recommendations** citing actual data files

## Agent-Specific Focus Areas

**V1 Production Team**: Focus on production stability, incremental improvements, and
proven optimization strategies. Defend the value of stable, working systems.

**V2 Scientific Team**: Analyze integrity issues objectively. Propose fixes or
alternative research directions. Don't be defensive - be scientific.

**CoTRR Team**: Evaluate lightweight optimization opportunities. Consider practical
deployment constraints and efficiency improvements.

**Tech Analysis**: Deep dive into architectural issues. Propose technical solutions
for multimodal fusion problems and evaluation framework improvements.

**Critic**: Challenge assumptions. Question whether we're solving the right problems.
Identify potential blind spots in current analysis.

**Integrity Guardian**: Ensure all claims are backed by evidence. Flag any
inconsistencies. Verify that proposed actions address root causes.

**Data Analyst**: Extract insights from experiment results. Identify patterns in
what worked and what didn't. Propose data-driven decision frameworks.

**Moderator**: Synthesize diverse perspectives into actionable strategy. Highlight
consensus areas and resolve conflicts. Create clear decision framework.

## Expected Output
- Comprehensive analysis of current state
- Clear diagnosis of what went wrong with V2
- Prioritized action plan for next phase
- Strategic roadmap for multimodal vision project
- Decision framework: Fix V2 vs Pivot vs Hybrid approach

---
**Note**: All agents should reference actual files in `research/` directory to support
their analysis. Use specific numbers, file paths, and evidence in recommendations.
"""

    print("\n📋 Meeting Topic:")
    print(topic[:500] + "...\n")

    print(f"⏰ Meeting started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Run the strategic meeting
    result = orchestrator.run_meeting(
        topic=topic,
        strategy="hierarchical",  # Teams → Critic → Integrity → Moderator
        rounds=3  # Multiple rounds for thorough discussion
    )

    print("\n" + "=" * 80)
    print("✨ Strategic Analysis Meeting Complete!")
    print("=" * 80)

    # Summary
    print("\n📊 Meeting Summary:")
    print(f"   - Total Responses: {len(result['responses'])}")
    print(f"   - Actions Identified: {len(result['actions'])}")
    print(f"   - Integrity Check: {'✅ PASSED' if result['integrity']['passed'] else '⚠️ ISSUES'}")
    print(f"   - Consensus Score: {result['integrity']['metrics']['consensus_score']:.2f}")

    print("\n📁 Artifacts saved to: multi-agent/reports/")
    print("   - Full transcript")
    print("   - Action items (prioritized)")
    print("   - Integrity analysis")
    print("   - Executive summary")

    print(f"\n⏰ Meeting ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return result


if __name__ == "__main__":
    result = main()

    # Print top priority actions
    if result and result['actions']:
        print("\n" + "=" * 80)
        print("🎯 TOP PRIORITY ACTIONS")
        print("=" * 80)
        high_priority = [a for a in result['actions'] if a.get('priority') == 'HIGH']
        if high_priority:
            for i, action in enumerate(high_priority[:5], 1):
                print(f"\n{i}. [{action['type']}] {action['description']}")
                print(f"   Source: {action['source']}")
        else:
            print("\nSee full action list in: multi-agent/reports/actions_*.json")
