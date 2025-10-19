# Quick Start - Updated Autonomous System

## Copy-Paste This Into Colab

### Cell 1: Verify Everything is Ready
```python
# Verification check
exec(open('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/verify_deployment.py').read())
```

**Expected output:** All checks passed ✅

---

### Cell 1.5: Test Gemini Connection (OPTIONAL)
```python
# Test Gemini API - ensures CoTRR Team agent works
exec(open('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/test_gemini_connection.py').read())
```

**Expected output:**
- API key found ✅
- Connection successful ✅
- Models listed ✅
- Test generation works ✅

**Note:** If this fails, the system will still work with 5 agents instead of 6 (Gemini agent will be skipped)

---

### Cell 2: Deploy Updated System
```python
# Deploy with new architecture
exec(open('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/research/colab/deploy_updated_system.py').read())
```

**Expected output:**
- Files copied ✅
- Directories created ✅
- Coordinator started ✅
- System running ✅

---

### Cell 3: Monitor Progress
```python
# Quick status check
import subprocess

print("📊 SYSTEM STATUS")
print("="*60)

# Show recent logs
print("\n🔍 Recent Activity:")
result = subprocess.run(['tail', '-30', '/content/executive.log'],
                       capture_output=True, text=True)
print(result.stdout)

print("\n📁 Planning Reports:")
result = subprocess.run(['ls', '-lh', '/content/cv_project/multi-agent/reports/planning/'],
                       capture_output=True, text=True)
print(result.stdout)

print("\n📁 Execution Reports:")
result = subprocess.run(['ls', '-lh', '/content/cv_project/multi-agent/reports/execution/'],
                       capture_output=True, text=True)
print(result.stdout)

print("\n🤝 Handoff File:")
try:
    with open('/content/cv_project/multi-agent/reports/handoff/pending_actions.json', 'r') as f:
        import json
        data = json.load(f)
        print(f"   Actions: {data.get('count', 0)}")
        print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
except FileNotFoundError:
    print("   ⏳ Waiting for first planning meeting...")
```

---

## Then: Open Monitoring Notebooks

**In Colab file browser, open these in separate tabs:**

1. `research/colab/monitor_planning.ipynb`
   - Run all cells
   - Watch planning meetings and strategic decisions

2. `research/colab/monitor_execution.ipynb`
   - Run all cells
   - Watch executive team execute actions

---

## Timeline

**Minutes 0-5:** System initialization
- Coordinator starts
- Agents load
- Heartbeat begins

**Minutes 5-30:** First planning meeting
- Planning Team discusses V1.0 goals
- Creates strategic recommendations
- Saves handoff file with actions

**Minutes 30+:** Continuous execution
- Executive Team reads actions
- Agents deploy models, run evaluations
- Metrics collected
- Results reported

**Minutes 60:** Second planning meeting
- Reviews execution results
- Adjusts strategy
- New actions issued

**Repeats every 30 minutes** → Progressive deployment toward V1.0

---

## What You'll See

### In monitor_planning.ipynb:
```
📋 Latest Meeting: summary_2025-10-13_23-45
─────────────────────────────────────────────

## Meeting Summary

**Participants:** 5 agents
**Topic:** V1.0 Lightweight Enhancer Deployment
**Duration:** 15 minutes

### Key Decisions:
1. Deploy to shadow environment first
2. Run evaluation on test set
3. Collect baseline metrics
...

🎯 Actions Recommended: 16

1. Deploy V1.0 model to shadow environment...
   Owner: ops_commander
   Priority: high

2. Run comprehensive evaluation suite...
   Owner: latency_analysis
   Priority: high
...
```

### In monitor_execution.ipynb:
```
🤖 Recent Agent Activity:

[23:47:12] 🤖 ops_commander: Deploying V1.0 to shadow
[23:47:45] [TOOL] deploy_model(source=models/v1.0.pth, stage=shadow)
[23:48:01] ✅ Deployment successful: deployment/shadow/v1.0.pth
[23:48:15] 🤖 latency_analysis: Starting evaluation
[23:48:20] [TOOL] run_evaluation(eval_script=research/evaluate_v1.py)
[23:50:33] ✅ Evaluation complete: NDCG@10=0.7234
...

🚀 Deployments:
─────────────────────────────────────────────
   shadow: ✅ 3 files
   5_percent: ❌ Not deployed
   20_percent: ❌ Not deployed
   production: ❌ Not deployed
```

---

## Troubleshooting

### If verification fails:
1. Wait 60 seconds for Drive sync
2. Check API keys loaded: `print(os.getenv('ANTHROPIC_API_KEY')[:20])`
3. Rerun verification

### If deployment hangs:
1. Check logs: `!tail -50 /content/executive.log`
2. Look for errors
3. Restart: `coordinator.stop()` then `coordinator.start()`

### If no planning meeting:
- First meeting takes ~30 minutes
- Check heartbeat: `!grep "HEARTBEAT" /content/executive.log`
- Check meeting schedule: `!grep "meeting" /content/executive.log`

### If no execution:
- Check handoff file exists: `!cat reports/handoff/pending_actions.json`
- Check tools loaded: `!grep "ExecutionTools" /content/executive.log`
- Check agent activity: `!grep "🤖" /content/executive.log`

---

## Success Indicators

You'll know it's working when:

✅ Verification passes all checks
✅ Deployment completes without errors
✅ Logs show heartbeat cycles every 5 minutes
✅ Planning meeting completes after ~30 minutes
✅ Handoff file appears with actions
✅ Executive agents start executing
✅ Tool usage logged: `[TOOL]` entries
✅ Deployment artifacts created
✅ Metrics collected and reported

---

**Ready?** Start with Cell 1 above! 🚀
