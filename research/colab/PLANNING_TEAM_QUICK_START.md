# Planning Team Review - Quick Start Guide

**Created:** Complete Colab notebook for Planning Team review meetings
**File:** `planning_team_review_cycle.ipynb`

---

## 🚀 How to Use (In Colab)

### **Step 1: Upload notebook to Colab**

1. Go to: https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Navigate to your Google Drive: `MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/`
4. Open: `planning_team_review_cycle.ipynb`

### **Step 2: Run all cells**

Click "Runtime" → "Run all"

Or run cells one by one (Shift+Enter):

1. **Cell 1:** Mount Google Drive
2. **Cell 2:** Load API keys (smart search)
3. **Cell 3:** Install dependencies
4. **Cell 4:** Read execution results from Cycle 1
5. **Cell 5:** Initialize Planning Team (4 agents)
6. **Cell 6:** Planning Team reviews results
7. **Cell 7:** Strategic Leader generates next cycle plan
8. **Cell 8:** Save new `pending_actions.json`
9. **Cell 9:** Summary

---

## 📋 What It Does

### **Planning Team (4 agents):**

1. **Strategic Leader (Claude Opus 4)**
   - Leads the meeting
   - Makes GO/PAUSE/PIVOT decisions
   - Generates final `pending_actions.json`

2. **Empirical Validation Lead (Claude Sonnet 4)**
   - Validates statistical rigor
   - Recommends next experiments

3. **Critical Evaluator (GPT-4)**
   - Challenges claims
   - Identifies risks

4. **Gemini Research Advisor (Gemini Flash)**
   - Provides literature context
   - Suggests alternatives

### **Output:**

**New file created:**
```
reports/handoff/pending_actions.json
```

This contains the tasks for Cycle 2, such as:
- BLIP diagnostic experiment
- Flamingo diagnostic experiment
- Cross-model statistical comparison
- Paper updates

---

## 🔄 Complete Cycle Flow

```
1. Executive Team (Cycle 1) → cvpr_autonomous_execution_cycle.ipynb
   ├── Executes 7 tasks
   └── Saves: execution_progress_update.md

2. Planning Team Review → planning_team_review_cycle.ipynb ← YOU ARE HERE
   ├── Reads Cycle 1 results
   ├── 4 agents analyze and discuss
   └── Saves: new pending_actions.json

3. Executive Team (Cycle 2) → cvpr_autonomous_execution_cycle.ipynb
   ├── Reads new pending_actions.json
   └── Executes Cycle 2 tasks

4. Repeat...
```

---

## ✅ Expected Output

After running all cells, you should see:

```
✅ PLANNING TEAM REVIEW MEETING COMPLETE
================================================================================

📋 Meeting Summary:
   ✅ Reviewed Cycle 1 execution results
   ✅ All 4 Planning Team agents provided analysis
   ✅ Strategic Leader synthesized input
   ✅ New pending_actions.json generated for Cycle 2

📂 Output Files:
   📄 reports/handoff/pending_actions.json (new cycle tasks)
   📄 reports/planning/pending_actions_history/pending_actions_YYYYMMDD_HHMMSS.json (backup)

🔄 Next Steps:
   1. Review the new pending_actions.json
   2. Run Executive Team execution cycle (Colab notebook)
   3. Execute Cycle 2 tasks
   4. Return here for next Planning Team review

================================================================================
🚀 READY FOR CYCLE 2 EXECUTION
================================================================================
```

---

## 🎯 After Planning Team Finishes

### **Run Executive Team again for Cycle 2:**

1. Open: `cvpr_autonomous_execution_cycle.ipynb`
2. Run all cells
3. It will automatically read the new `pending_actions.json`
4. Execute Cycle 2 tasks (BLIP + Flamingo diagnostics)
5. Save results
6. Return to Planning Team for Cycle 3

---

## 📊 What Planning Team Will Generate

**Example Cycle 2 tasks:**

```json
{
  "decisions": [
    {
      "priority": "HIGH",
      "action": "Run BLIP diagnostic experiment with same protocol as CLIP",
      "owner": "ops_commander",
      "rationale": "CLIP showed MCS=0.73 (p<0.001). Need second model for cross-validation.",
      "acceptance_criteria": [
        "BLIP attention weights extracted successfully",
        "MCS computed with CI95",
        "Statistical comparison to CLIP baseline"
      ]
    },
    {
      "priority": "HIGH",
      "action": "Run Flamingo diagnostic experiment",
      ...
    }
  ]
}
```

---

## 🐛 Troubleshooting

### **If API keys not found:**
- Check that .env file exists in Google Drive
- Verify all 3 keys present (ANTHROPIC, OPENAI, GOOGLE)
- See `ENV_FILE_PATH_FIX.md` for help

### **If agents fail to respond:**
- Check API key validity
- Verify model names are correct:
  - Anthropic: `claude-opus-4-20250514`, `claude-sonnet-4-20250514`
  - OpenAI: `gpt-4-turbo-2024-04-09`
  - Google: `gemini-2.0-flash-exp`

### **If JSON parsing fails:**
- Strategic Leader's response will be shown
- You can manually extract the JSON
- Copy to `reports/handoff/pending_actions.json`

---

## 📝 Files Created

**Main output:**
- `reports/handoff/pending_actions.json` - Next cycle tasks (overwrites previous)

**Backup:**
- `reports/planning/pending_actions_history/pending_actions_YYYYMMDD_HHMMSS.json` - Timestamped backup

**The Executive Team will read the main file (`reports/handoff/pending_actions.json`) for the next cycle.**

---

## 🎉 Ready to Run!

**Just open the notebook in Colab and run all cells!**

The Planning Team will:
1. ✅ Review your Cycle 1 results (CLIP diagnostic success!)
2. ✅ Assess progress toward Week 1 GO/NO-GO
3. ✅ Generate smart next steps (BLIP + Flamingo diagnostics)
4. ✅ Save new `pending_actions.json` for Cycle 2

**Then you can run the Executive Team notebook again for Cycle 2!** 🚀
