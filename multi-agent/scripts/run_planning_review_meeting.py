#!/usr/bin/env python3
"""
Planning Team Review Meeting - Post-Execution Cycle

After Executive Team completes tasks, Planning Team:
1. Reviews execution_progress_update.md
2. Assesses progress toward goals
3. Identifies next priorities
4. Generates new pending_actions.json for next cycle
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Load API keys from Google Drive
env_file = Path.home() / 'Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")
else:
    print(f"⚠️ API keys not found at {env_file}")

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def read_execution_results():
    """Read execution results from Executive Team"""
    handoff_dir = parent_dir / 'reports/handoff'

    # Read progress update
    progress_file = handoff_dir / 'execution_progress_update.md'
    if not progress_file.exists():
        print(f"❌ No execution results found at {progress_file}")
        print("⚠️ Executive Team must complete execution first")
        return None

    with open(progress_file, 'r') as f:
        progress_text = f.read()

    # Find most recent JSON results
    results_dir = parent_dir / 'reports/execution/results'
    json_files = list(results_dir.glob('execution_results_*.json'))

    if json_files:
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
        with open(latest_json, 'r') as f:
            results_data = json.load(f)
    else:
        results_data = {}

    return {
        'progress_text': progress_text,
        'results_data': results_data,
        'progress_file': progress_file,
        'results_file': latest_json if json_files else None
    }

def main():
    """Run Planning Team review meeting"""
    print("="*80)
    print("📋 PLANNING TEAM REVIEW MEETING")
    print("="*80)

    print("\n🎯 Purpose: Review execution results and plan next cycle\n")

    # Read execution results
    print("📥 Reading execution results...")
    results = read_execution_results()

    if not results:
        print("\n❌ Cannot run meeting without execution results")
        print("   Run Executive Team cycle first")
        return

    print(f"✅ Found execution results")
    print(f"   Progress: {results['progress_file']}")
    if results['results_file']:
        print(f"   Data: {results['results_file']}")

    # Display summary
    data = results['results_data']
    if data:
        print(f"\n📊 Execution Summary:")
        print(f"   Total tasks: {data.get('total_tasks', 'N/A')}")
        print(f"   Completed: {data.get('completed', 'N/A')} ✅")
        print(f"   Failed: {data.get('failed', 'N/A')} ❌")
        print(f"   Duration: {data.get('total_duration_seconds', 0):.1f}s")

    # Create Planning Team meeting topic
    meeting_topic = f"""# PLANNING TEAM REVIEW MEETING

## 📥 Executive Team Results

{results['progress_text']}

---

## 🎯 Your Task: Review and Plan Next Cycle

As the Planning Team, your job is to:

### 1. **Review Execution Results**
- Which tasks completed successfully?
- Which tasks failed? Why?
- What outputs were generated?
- Is there evidence (MLflow run_ids, file paths)?

### 2. **Assess Progress Toward Goals**

**Week 1 Goal:** Determine if attention collapse is widespread across multimodal architectures

**Key Questions:**
- Are we moving toward the Week 1 GO/NO-GO decision (Oct 20)?
- Do we have statistical evidence? (p<0.05 threshold)
- Have we tested ≥2 models? (CLIP, ALIGN, etc.)
- Are results reproducible? (MLflow tracking)

**Decision Criteria:**
- **GO:** ≥2 models show >80% attention imbalance → Field-wide paper
- **PIVOT:** Only V2 shows collapse → Focused case study
- **DIAGNOSTIC:** Tools work on all models → Framework contribution

### 3. **Identify Next Priorities**

Based on execution results, what should happen in the next cycle?

**If HIGH tasks completed:**
- Move to MEDIUM priority tasks
- Start next phase of validation
- Begin writing paper outline

**If HIGH tasks failed:**
- Debug failures
- Re-run with fixes
- Adjust approach

**Blockers to address:**
- Missing dependencies
- Environment issues
- Resource constraints

### 4. **Generate New pending_actions.json**

Create a detailed plan for the next execution cycle.

**Format:**
```json
{{
  "meeting_id": "cvpr_planning_cycle2_{{timestamp}}",
  "generated_at": "{{ISO timestamp}}",
  "context": "Week 1 - Cycle 2: [Brief description based on results]",
  "decisions": [
    {{
      "priority": "HIGH",
      "action": "[Specific task based on execution results]",
      "owner": "ops_commander",
      "rationale": "[Why this is needed based on previous results]",
      "deadline": "{{ISO timestamp}}",
      "acceptance_criteria": [
        "[Specific, measurable criterion]"
      ],
      "evidence_paths": ["[File paths for verification]"]
    }}
  ],
  "acceptance_gates": {{
    "week1_validation": {{
      "min_models_tested": 3,
      "min_models_with_collapse": 2,
      "statistical_threshold": "p<0.05"
    }}
  }}
}}
```

---

## 🤖 Planning Team Roles

**Strategic Leader (Opus 4):**
- Lead this meeting
- Synthesize team input
- Create final pending_actions.json
- Make GO/PAUSE/PIVOT decisions

**Empirical Validation Lead (Sonnet 4):**
- Assess statistical validity of results
- Verify experimental rigor
- Recommend next experiments

**Critical Evaluator (GPT-4):**
- Challenge claims (is evidence sufficient?)
- Identify methodological risks
- Flag reproducibility issues

**Gemini Research Advisor (Gemini Flash):**
- Provide context from literature
- Suggest alternative approaches
- Assess feasibility of next steps

---

## 📤 Required Output

**1. Assessment of Current Results:**
- What worked?
- What failed?
- What did we learn?

**2. Decision on Next Steps:**
- Continue with current approach?
- Pivot to different strategy?
- Focus on debugging?

**3. New pending_actions.json:**
- 4-7 tasks for next cycle
- Prioritized (HIGH → MEDIUM → LOW)
- Based on execution results
- Clear acceptance criteria

**4. Updated Timeline:**
- Are we on track for Week 1 GO/NO-GO (Oct 20)?
- Do we need to accelerate/adjust?

---

## 🚨 Critical Thinking Required

**Don't just continue blindly:**
- If CLIP diagnostic failed, why? Should we retry or change approach?
- If CLIP shows no attention collapse, what does that mean for our hypothesis?
- If we're behind schedule, what can we cut or parallelize?
- If we're ahead, what stretch goals can we add?

**Evidence-based decisions:**
- Every priority must be justified by execution results
- Every new task must cite what previous results showed
- Every deadline must account for available time

---

## ⏰ Timeline Reminder

**Today:** {datetime.now().strftime('%Y-%m-%d')}
**Week 1 GO/NO-GO:** October 20, 2025 (6 days from now)
**Abstract Deadline:** November 6, 2025
**Full Paper Deadline:** November 13, 2025

Make smart decisions based on time constraints.

---

**Begin your review now. Create a thoughtful, evidence-based plan for the next cycle.**
"""

    print("\n" + "="*80)
    print("🤖 Initializing Planning Team (4 agents)...")
    print("="*80)

    # TODO: Initialize and run Planning Team meeting
    # For now, save the meeting topic
    meeting_file = parent_dir / 'reports/planning/review_meeting_topic.md'
    meeting_file.parent.mkdir(parents=True, exist_ok=True)

    with open(meeting_file, 'w') as f:
        f.write(meeting_topic)

    print(f"\n✅ Planning meeting topic prepared")
    print(f"📄 Saved to: {meeting_file}")

    print("\n" + "="*80)
    print("📋 NEXT STEPS")
    print("="*80)
    print("\n1. ✅ Execution results reviewed")
    print("2. ⏸️ Planning Team meeting topic prepared")
    print("3. 🎯 TODO: Run Planning Team to generate new pending_actions.json")
    print("\n💡 The Planning Team will:")
    print("   - Analyze execution results")
    print("   - Assess progress toward Week 1 goals")
    print("   - Generate new pending_actions.json")
    print("   - Plan next cycle priorities")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
