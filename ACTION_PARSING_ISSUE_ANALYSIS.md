# Action Parsing Issue Analysis

**Date:** October 14, 2025 03:20
**Status:** 🔴 **CRITICAL ISSUE IDENTIFIED**

---

## 🔴 Problem: Action Parser Extracting Garbage

### Evidence from Latest Meeting (02:59:37)

**Example Bad Extractions:**

```json
{
  "description": "│  ✓   │  ✗   │    ✓     │  ✓   │",
  "agent": "pre_architect"
}
{
  "description": "(AEM)                      │",
  "agent": "pre_architect"
}
{
  "description": "analyzer                      │",
  "agent": "pre_architect"
}
{
  "description": "4 architectures minimum",
  "agent": "pre_architect"
}
{
  "description": "### **Phase 2 (Days 8-14): Empirical Validation & Solution Testing**",
  "agent": "data_analyst"
}
{
  "description": "**Success Probability Matrix:**",
  "agent": "data_analyst"
}
```

**Out of 24 extracted actions:**
- ✅ 3-4 are reasonable (20%)
- ❌ 20+ are garbage (80%)

---

## 🔍 Root Cause Analysis

### Issue 1: Regex Captures Table Content

The regex `r'(?:test|experiment|try|validate|measure|train|run)\s+([^.!?\n]+)'` matches:

**Agent writes:**
```markdown
| Model | Test | Validate | Run |
|-------|------|----------|-----|
|  CLIP |  ✓   |    ✗     |  ✓  |
```

**Regex matches:** `"test"` → captures rest of line → `"│  ✓   │  ✗   │    ✓     │  ✓   │"`

### Issue 2: Regex Captures Markdown Headers

**Agent writes:**
```markdown
### **Phase 2 (Days 8-14): Empirical Validation & Solution Testing**
```

**Regex matches:** `"test"` in "Testing" → captures entire header

### Issue 3: Regex Captures Partial Sentences

**Agent writes:**
```markdown
We should analyze 4 architectures minimum to ensure generalizability.
```

**Regex matches:** `"analyze"` → captures only → `"4 architectures minimum"`

Missing: "We should analyze" (the verb phrase is excluded!)

---

## 💡 Solution Strategy

### Approach: Add Intelligent Filtering

**Keep existing extraction logic BUT add post-processing filters:**

1. **Filter out table content** - Lines containing `│`, `|`, `─`, or multiple consecutive symbols
2. **Filter out markdown headers** - Lines starting with `#` or `###`
3. **Filter out too-short fragments** - Descriptions < 20 chars
4. **Require complete sentence structure** - Must contain subject + verb + object pattern
5. **Validate action completeness** - Description should be self-explanatory

---

## 🔧 Implementation Plan

### Modified `parse_actions.py`

Add filtering function AFTER extraction, BEFORE returning actions:

```python
@staticmethod
def _is_valid_action(description: str) -> bool:
    """Validate that extracted description is a real action, not garbage"""

    # Filter 1: Reject table content
    if any(symbol in description for symbol in ['│', '├', '└', '─']):
        return False
    if description.count('|') > 2:  # More than 2 pipes = likely table
        return False

    # Filter 2: Reject markdown headers
    if description.strip().startswith('#'):
        return False
    if description.strip().startswith('**') and description.strip().endswith('**'):
        return False  # Bold headers like "**Phase 2**"

    # Filter 3: Reject too short
    if len(description) < 20:
        return False

    # Filter 4: Require minimum word count (complete thought)
    word_count = len(description.split())
    if word_count < 4:
        return False

    # Filter 5: Reject if it's just a number or label
    if re.match(r'^[\d\s\-\+\=]+$', description):
        return False

    # Filter 6: Must start with action verb or article + noun
    # Good: "Implement gradient tracking"
    # Good: "The diagnostic framework should be tested"
    # Bad: "4 architectures minimum"
    # Bad: "on All Architectures)"
    first_word = description.split()[0].lower()
    action_verbs = ['implement', 'test', 'run', 'analyze', 'validate',
                    'create', 'build', 'design', 'measure', 'track',
                    'the', 'we', 'our', 'this', 'that']  # Add articles/pronouns

    if not any(verb in first_word for verb in action_verbs):
        # Check if starts with article
        if first_word not in ['the', 'we', 'our', 'a', 'an', 'this', 'that']:
            return False

    return True
```

**Apply in `parse_response()`:**

```python
for action_type, pattern in action_patterns.items():
    matches = re.finditer(pattern, response, re.IGNORECASE)
    for match in matches:
        full_context = ActionParser._extract_full_context(...)

        if ActionParser._is_duplicate(full_context, actions):
            continue

        # NEW: Validate action quality
        if not ActionParser._is_valid_action(full_context):
            continue

        if len(full_context) > 15:
            actions.append(Action(...))
```

---

## 🎯 Expected Results After Fix

### Before:
24 actions extracted, 80% garbage:
- `"│  ✓   │  ✗   │    ✓     │  ✓   │"`
- `"### **Phase 2 (Days 8-14)**"`
- `"4 architectures minimum"`

### After:
6-8 actions extracted, 95% quality:
- `"Implement diagnostic test suite for attention collapse detection"`
- `"Validate the mitigation strategies on at least two architectures"`
- `"Test on 3-4 diverse architectures from the literature"`

---

## 📊 Validation Test Cases

### Test Case 1: Table Content (REJECT)
**Input:** `"│  ✓   │  ✗   │    ✓     │  ✓   │"`
**Expected:** `False` (contains │ symbol)

### Test Case 2: Markdown Header (REJECT)
**Input:** `"### **Phase 2 (Days 8-14): Empirical Validation & Solution Testing**"`
**Expected:** `False` (starts with #)

### Test Case 3: Fragment (REJECT)
**Input:** `"4 architectures minimum"`
**Expected:** `False` (doesn't start with verb/article, too short)

### Test Case 4: Good Action (ACCEPT)
**Input:** `"Implement diagnostic test suite for attention collapse detection across multiple architectures"`
**Expected:** `True` (starts with action verb, complete sentence, >20 chars, >4 words)

### Test Case 5: Good Action with Article (ACCEPT)
**Input:** `"The diagnostic framework should be tested on V2 and CLIP models"`
**Expected:** `True` (starts with article, complete sentence)

---

## ⚡ Priority

**CRITICAL** - This is blocking the entire system from functioning properly:

1. ❌ Planning team generates good discussions
2. ❌ Action parser extracts garbage
3. ❌ Executive team receives nonsense actions
4. ❌ Executive team can't execute meaningful research

**This must be fixed BEFORE restarting the system.**

---

## 🚀 Implementation Steps

1. ✅ Diagnose root cause (DONE - this document)
2. ⚠️ Add `_is_valid_action()` filter to parse_actions.py
3. ⚠️ Test on existing meeting transcripts
4. ⚠️ Sync to Google Drive
5. ⚠️ Restart system and verify

---

*Analysis completed: October 14, 2025 03:20*
*Status: Waiting for fix implementation*
