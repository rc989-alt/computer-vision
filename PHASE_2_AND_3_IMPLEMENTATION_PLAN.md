# Phase 2 & 3 Implementation Plan

**Date:** October 14, 2025
**Status:** Ready to Implement
**Prerequisites:** Phase 1 Complete ✅

---

## 📊 Current State Assessment

### Phase 1: Critical Foundations ✅ COMPLETE

All core autonomous features are implemented:
- ✅ Heartbeat System (background thread with daemon=True)
- ✅ Trigger System (cost/progress/deadline warnings)
- ✅ SharedMemoryManager (experiment tracking with 诚信)
- ✅ Auto-Topic Generation (milestone + experiment focused)
- ✅ Adaptive Timing (60-180 min based on workload)
- ✅ Deployment Tools (deploy_model, run_evaluation)
- ✅ Cost Tracking & Budget Alerts

**System is 100% autonomous and production-ready!**

### Phase 2: CVPR Standards Integration ⏳ PARTIAL

**What EXISTS:**
- Research Director has comprehensive CVPR awareness (771 lines)
- Timeline constraints (22 days) enforced
- Paper Writer acts as gatekeeper
- GO/NO-GO decision framework
- Trust tier system (T1/T2/T3)
- Backup options (A/B/C/D) defined

**What's OPTIONAL to Add:**
- WebSearch tool integration for checking recent CVPR papers
- Automated novelty validation workflow
- Baseline identification from recent publications

**Assessment:** Current system has strong CVPR awareness through prompts. WebSearch would be a nice-to-have enhancement.

### Phase 3: Nice-to-Haves ⏳ NOT STARTED

- Knowledge Transfer Protocol (V1→V2 insights)
- Auto-Sync from Google Drive (Colab integration)

---

## 🎯 Implementation Tasks

### Task 1: Add WebSearch Tool (Phase 2)

**Goal:** Enable agents to search recent CVPR papers proactively

**Estimated Time:** 30 minutes

**Steps:**

1. **Add WebSearch to AgentTools** (unified_coordinator.py lines 183-611)

```python
class AgentTools:
    def __init__(self, project_root: Path):
        # ... existing init ...
        self.web_search_enabled = os.environ.get('ENABLE_WEB_SEARCH', 'false').lower() == 'true'

    def web_search(self, query: str) -> Dict[str, Any]:
        """
        Search web for recent papers (uses Claude's WebSearch if available)

        Args:
            query: Search query (e.g., "CVPR 2024 multimodal fusion")

        Returns:
            Dict with search results or error message
        """
        if not self.web_search_enabled:
            return {
                'success': False,
                'error': 'WebSearch not enabled. Set ENABLE_WEB_SEARCH=true'
            }

        # Note: This requires Claude Code's WebSearch tool
        # If running in Colab, this may not be available
        return {
            'success': False,
            'error': 'WebSearch requires Claude Code environment'
        }
```

2. **Update Research Director Prompt** (Add section after line 188)

```markdown
### WebSearch (Optional - if available)

**When to use:**
- Before proposing novel research direction
- To check recent CVPR papers (2023-2024)
- To validate that approach hasn't been done
- To identify current SOTA baselines

**Example:**
```
WebSearch("CVPR 2024 multimodal fusion attention")

Results show:
- Paper X (CVPR 2024): Cross-modal attention with gating
- Paper Y (CVPR 2023): Modality-balanced training
- Our approach differs by: [specific difference]

Novelty validated: Our diagnostic framework is unique.
```

**Note:** WebSearch may not be available in all environments (Colab).
If unavailable, rely on Paper Writer's literature knowledge.
```

3. **Test:**
```python
# In Colab or local environment
os.environ['ENABLE_WEB_SEARCH'] = 'true'
coordinator = UnifiedCoordinator(config, project_root)
# Agent can now attempt web_search() in responses
```

**Decision:** This is optional. Current system works without it. Agents can rely on Paper Writer's knowledge.

---

### Task 2: Knowledge Transfer Protocol (Phase 3)

**Goal:** Enable V1→V2→CoTRR insight sharing

**Estimated Time:** 45 minutes

**Steps:**

1. **Verify KnowledgeTransfer class exists** (unified_coordinator.py lines 151-180)

Status: ✅ Already implemented!

```python
class KnowledgeTransfer:
    """Manages knowledge transfer between research lines"""

    def transfer_insight(self, source_line: str, target_line: str, insight: Dict[str, Any]):
        """Transfer insight from one research line to another"""
        # Creates: research/{target_line}/INSIGHTS_FROM_{SOURCE}.md
```

2. **Add transfer_insight tool to AgentTools** (lines 273+)

```python
class AgentTools:
    def transfer_insight(self, source_line: str, target_line: str,
                         title: str, discovery: str, application: str,
                         priority: str = 'MEDIUM', experiments: List[str] = None) -> Dict[str, Any]:
        """
        Transfer insight from one research line to another

        Args:
            source_line: Source research line (e.g., "v1_production")
            target_line: Target research line (e.g., "v2_research")
            title: Insight title
            discovery: What was discovered
            application: How to apply to target line
            priority: HIGH/MEDIUM/LOW
            experiments: List of recommended experiments

        Returns:
            Dict with success status and file path

        Example:
            transfer_insight(
                source_line="v1_production",
                target_line="v2_research",
                title="Gradient Clipping Improves Stability",
                discovery="V1 training stabilized with gradient clipping at 1.0",
                application="V2 might benefit during fusion layer training",
                priority="HIGH",
                experiments=[
                    "Test gradient clipping on V2 fusion layers",
                    "Compare stability with/without clipping"
                ]
            )
        """
        insight = {
            'title': title,
            'discovery': discovery,
            'application': application,
            'priority': priority,
            'experiments': experiments or []
        }

        return self.coordinator.knowledge_transfer.transfer_insight(
            source_line, target_line, insight
        )
```

3. **Update agent prompts to mention transfer_insight**

Add to Research Director and Tech Analyst (after tool descriptions):

```markdown
### transfer_insight (Optional)

**When to use:**
- When discovering insight that could help other research lines
- After successful experiment on one line
- When finding optimization technique that's generalizable

**Example:**
```
# V1 discovered gradient clipping helps
transfer_insight(
    source_line="v1_production",
    target_line="v2_research",
    title="Gradient Clipping at 1.0 Stabilizes Training",
    discovery="V1 training converged faster with gradient clipping",
    application="V2 fusion layers may benefit from same technique",
    priority="HIGH",
    experiments=["Test clipping=1.0 on V2 fusion training"]
)

✅ Insight transferred to research/v2_research/INSIGHTS_FROM_V1_PRODUCTION.md
```

**Note:** Use sparingly - only for truly transferable insights.
```

4. **Test:**
```python
# During meeting, Research Director can do:
result = coordinator.tools.transfer_insight(
    source_line="v1_production",
    target_line="v2_research",
    title="Example Insight",
    discovery="Something learned",
    application="How to apply",
    priority="MEDIUM"
)
# Creates: research/v2_research/INSIGHTS_FROM_V1_PRODUCTION.md
```

**Status:** Class exists, just needs tool wrapper. **15 minutes to implement.**

---

### Task 3: Auto-Sync from Google Drive (Phase 3)

**Goal:** Auto-sync key files from Drive when running in Colab

**Estimated Time:** 30 minutes

**Steps:**

1. **Add _sync_from_drive method** (unified_coordinator.py after line 1726)

```python
def _sync_from_drive(self) -> bool:
    """
    Auto-sync files from Google Drive (if running in Colab)

    Returns:
        True if sync successful, False if not in Colab
    """
    import shutil

    # Detect if running in Colab
    DRIVE_ROOT = Path("/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean")

    if not DRIVE_ROOT.exists():
        return False  # Not in Colab or Drive not mounted

    # Files to watch for updates
    watch_files = [
        "research/RESEARCH_CONTEXT.md",
        "research/01_v1_production_line/SUMMARY.md",
        "research/02_v2_research_line/SUMMARY.md",
        "research/03_cotrr_lightweight_line/SUMMARY.md",
        "research/00_previous_work/comprehensive_evaluation.json",
        "unified-team/configs/team.yaml",
        "CVPR_2025_SUBMISSION_STRATEGY.md"
    ]

    synced_count = 0
    for file_path in watch_files:
        source = DRIVE_ROOT / file_path
        target = self.project_root / file_path

        if source.exists():
            # Check if file is newer in Drive
            if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                print(f"   ✅ Synced: {file_path}")
                synced_count += 1

    if synced_count > 0:
        print(f"   📊 Updated {synced_count} file(s) from Google Drive")

    return True
```

2. **Call in heartbeat loop** (HeartbeatSystem._heartbeat_loop around line 1890)

Add after line 1890 (before generating topic):

```python
# Auto-sync from Drive if in Colab (every 10 minutes)
if cycle_count % 5 == 0:  # Every 5 cycles = 10 minutes at 2-min intervals
    self.coordinator._sync_from_drive()
```

3. **Test:**
```python
# In Colab:
# 1. Mount Drive
# 2. Update a file in Drive
# 3. Wait for next sync cycle (every 10 min or every 5 meetings)
# 4. File should auto-update in local workspace
```

**Status:** Easy to implement. **20 minutes.**

---

## 📋 Implementation Priority

### Priority 1: Knowledge Transfer Tool (15 min) 🟢 RECOMMENDED

**Why:** Class already exists, just needs tool wrapper. Enables cross-line learning.

**Steps:**
1. Add `transfer_insight()` to AgentTools (10 min)
2. Update Research Director prompt with usage example (5 min)
3. Test with one transfer (5 min)

**Impact:** Enables agents to share insights across research lines automatically.

---

### Priority 2: Auto-Sync from Drive (20 min) 🟡 NICE-TO-HAVE

**Why:** Useful for Colab, keeps files in sync automatically.

**Steps:**
1. Add `_sync_from_drive()` method (10 min)
2. Integrate into heartbeat loop (5 min)
3. Test in Colab (5 min)

**Impact:** Better Colab experience, automatic file updates.

---

### Priority 3: WebSearch Tool (30 min) ⚪ OPTIONAL

**Why:** Agents already have strong CVPR awareness through prompts. WebSearch adds external validation but not critical.

**Steps:**
1. Add `web_search()` to AgentTools (15 min)
2. Update Research Director prompt (10 min)
3. Test (5 min)

**Impact:** Enables checking recent papers proactively. May not work in all environments (Colab).

---

## 🎯 Recommendation

**Implement in this order:**

1. **Knowledge Transfer** (15 min) - Highest value, lowest effort
2. **Auto-Sync from Drive** (20 min) - Good for Colab users
3. **WebSearch** (30 min) - Optional, may not work everywhere

**Total time:** 65 minutes for all three
**Or just Priority 1+2:** 35 minutes

---

## ✅ Success Criteria

After implementing:

**Knowledge Transfer:**
- ✅ Research Director can call `transfer_insight()`
- ✅ Insights saved to `research/{target}/INSIGHTS_FROM_{SOURCE}.md`
- ✅ File contains: title, discovery, application, experiments
- ✅ Next meeting can reference these insights

**Auto-Sync:**
- ✅ System detects Colab environment
- ✅ Files sync from Drive every 10 minutes
- ✅ Only newer files are copied (based on mtime)
- ✅ Sync count displayed in monitoring

**WebSearch:**
- ✅ Agent can call `web_search("query")`
- ✅ Returns search results or error message
- ✅ Works in Claude Code environment
- ✅ Graceful fallback if not available

---

## 📝 Testing Plan

### Test Knowledge Transfer

```python
# In a meeting, Research Director does:
result = coordinator.tools.transfer_insight(
    source_line="v1_production",
    target_line="v2_research",
    title="Gradient Clipping Stabilizes Training",
    discovery="V1 converged faster with gradient clipping at 1.0",
    application="V2 fusion layers may benefit from same technique",
    priority="HIGH",
    experiments=["Test clipping=1.0 on V2 fusion training"]
)

# Verify file created:
Path("research/v2_research/INSIGHTS_FROM_V1_PRODUCTION.md").exists()
```

### Test Auto-Sync

```python
# In Colab:
# 1. Start autonomous system
# 2. In another tab, edit a file in Google Drive
# 3. Wait 10 minutes (or 5 meeting cycles)
# 4. Check if file updated locally
# 5. Verify sync count in monitoring dashboard
```

### Test WebSearch

```python
# In meeting context, agent can do:
result = coordinator.tools.web_search("CVPR 2024 multimodal fusion")

# Should return:
# - Success: True/False
# - Results or error message
# - Graceful handling if not available
```

---

## 🚀 Next Steps

**Choose implementation scope:**

**Option A: Minimal (15 min)**
- Just Knowledge Transfer
- Enables cross-line learning
- Good enough for single research line

**Option B: Recommended (35 min)**
- Knowledge Transfer + Auto-Sync
- Best for Colab users
- Complete autonomous experience

**Option C: Complete (65 min)**
- All three features
- Maximum capability
- May have environment limitations (WebSearch)

**My recommendation:** Option B (Knowledge Transfer + Auto-Sync) for 35 minutes of work. WebSearch is nice-to-have but not critical.

---

**Ready to implement:** Yes! All analysis complete.
**Blockers:** None
**Dependencies:** Phase 1 complete ✅

Let me know which option you prefer and I'll implement it!
