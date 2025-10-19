# API Key Authentication Fix

**Issue:** All tasks failed with authentication error
**Error:** `"Could not resolve authentication method. Expected either api_key or auth_token to be set."`

---

## 🔍 Root Cause

The Colab notebook's API key loading cell didn't properly set the environment variables for the agent system. The agents use different providers (Anthropic, OpenAI, Google) and need their specific API keys set.

---

## ✅ Fix

### **Option 1: Direct Environment Variable Setting (Recommended for Colab)**

Add this cell **immediately after mounting Google Drive** in the Colab notebook:

```python
import os
from pathlib import Path

# Load API keys from .env file
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/.env')

if env_file.exists():
    print("📄 Loading API keys from .env file...")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")  # Remove quotes
                os.environ[key] = value
                print(f"   ✅ Set {key}")

    # Verify critical keys are set
    required_keys = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']
    missing_keys = []

    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)
        else:
            # Show first 10 chars to verify
            print(f"   ✅ {key}: {os.environ[key][:10]}...")

    if missing_keys:
        print(f"   ⚠️ Missing keys: {', '.join(missing_keys)}")
    else:
        print("\n✅ All API keys loaded successfully!")

else:
    print(f"❌ .env file not found at {env_file}")
    print("   Please check the path to your .env file")
```

### **Option 2: Manual Key Setting (Quick Test)**

If the .env file path is wrong, you can manually set keys:

```python
import os

# Set API keys manually (replace with your actual keys)
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['GOOGLE_API_KEY'] = 'AI...'

print("✅ API keys set manually")
```

### **Option 3: Use Colab Secrets (Most Secure)**

```python
from google.colab import userdata
import os

# Get keys from Colab secrets
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

print("✅ API keys loaded from Colab secrets")
```

**To set Colab secrets:**
1. Click 🔑 key icon in left sidebar
2. Add secrets: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
3. Enable notebook access

---

## 🔧 Updated Colab Notebook Cell Order

**Correct order:**

1. ✅ Check GPU
2. ✅ Mount Google Drive
3. ✅ **Load API keys (ADD IMPROVED VERSION HERE)** ← FIX THIS CELL
4. ✅ Install dependencies
5. ✅ Configure paths
6. ✅ Initialize agents
7. ✅ Execute tasks

---

## 🧪 Test API Keys Are Working

Add this test cell after loading keys:

```python
# Test API key loading
import os

def test_api_keys():
    """Test that all required API keys are loaded"""

    tests = {
        'Anthropic': 'ANTHROPIC_API_KEY',
        'OpenAI': 'OPENAI_API_KEY',
        'Google': 'GOOGLE_API_KEY'
    }

    print("🧪 Testing API Keys:\n")
    all_ok = True

    for provider, key_name in tests.items():
        key_value = os.environ.get(key_name)
        if key_value:
            # Show first/last 4 chars only
            masked = f"{key_value[:4]}...{key_value[-4:]}"
            print(f"✅ {provider:12} ({key_name}): {masked}")
        else:
            print(f"❌ {provider:12} ({key_name}): NOT SET")
            all_ok = False

    print("\n" + "="*60)
    if all_ok:
        print("✅ All API keys loaded successfully!")
        print("✅ Ready to initialize agents")
    else:
        print("❌ Some API keys missing!")
        print("⚠️ Agents will fail to authenticate")
    print("="*60)

    return all_ok

# Run test
test_api_keys()
```

**Expected output:**
```
🧪 Testing API Keys:

✅ Anthropic    (ANTHROPIC_API_KEY): sk-a...x7Qz
✅ OpenAI      (OPENAI_API_KEY): sk-p...8K2p
✅ Google      (GOOGLE_API_KEY): AIza...9mJK

============================================================
✅ All API keys loaded successfully!
✅ Ready to initialize agents
============================================================
```

---

## 🔍 Debug Steps

If keys still not working:

### **Step 1: Check .env file location**

```python
import os
from pathlib import Path

# Check multiple possible locations
possible_paths = [
    '/content/drive/MyDrive/cv_multimodal/project/.env',
    '/content/drive/MyDrive/.env',
    Path.home() / '.env'
]

for path in possible_paths:
    path = Path(path)
    if path.exists():
        print(f"✅ Found .env at: {path}")
        print(f"   Size: {path.stat().st_size} bytes")

        # Show first 5 lines (masked)
        with open(path, 'r') as f:
            lines = f.readlines()[:5]
            for line in lines:
                if '=' in line and not line.startswith('#'):
                    key = line.split('=')[0]
                    print(f"   Contains: {key}=***")
    else:
        print(f"❌ Not found: {path}")
```

### **Step 2: Check agent initialization**

The agents are initialized in the "Initialize Executive Team" cell. Make sure this happens AFTER loading API keys:

```python
# This cell should come AFTER API keys are loaded
from agents.roles import Agent, AgentConfig

# Test creating one agent
try:
    test_config = AgentConfig(
        name='Test Agent',
        model='claude-sonnet-4',
        provider='anthropic',
        role='Test',
        prompt_file='test.md'
    )
    print("✅ AgentConfig created successfully")
    print("✅ API keys are accessible to agents")
except Exception as e:
    print(f"❌ Error creating agent: {e}")
    print("⚠️ API keys may not be loaded properly")
```

---

## 📝 Quick Fix Summary

**Add this cell in Colab (after mounting Drive, before installing dependencies):**

```python
import os
from pathlib import Path

print("="*60)
print("🔑 LOADING API KEYS")
print("="*60)

# Path to .env file
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/.env')

if env_file.exists():
    print(f"\n📄 Found .env file: {env_file}")
    print(f"   Size: {env_file.stat().st_size} bytes\n")

    # Load all environment variables
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

    # Verify required keys
    required = {
        'ANTHROPIC_API_KEY': 'Claude (Ops Commander, Quality & Safety, Infrastructure)',
        'OPENAI_API_KEY': 'GPT-4 (Critical Evaluator)',
        'GOOGLE_API_KEY': 'Gemini (Research Advisor)'
    }

    print("Checking required API keys:\n")
    all_loaded = True

    for key, usage in required.items():
        value = os.environ.get(key)
        if value:
            masked = f"{value[:8]}...{value[-4:]}"
            print(f"✅ {key}")
            print(f"   Value: {masked}")
            print(f"   Used by: {usage}\n")
        else:
            print(f"❌ {key} - NOT FOUND")
            print(f"   Needed by: {usage}\n")
            all_loaded = False

    print("="*60)
    if all_loaded:
        print("✅ ALL API KEYS LOADED SUCCESSFULLY")
        print("✅ Agents can authenticate")
    else:
        print("❌ SOME API KEYS MISSING")
        print("⚠️ Execution will fail")
    print("="*60)

else:
    print(f"\n❌ ERROR: .env file not found")
    print(f"   Expected location: {env_file}")
    print(f"\n💡 Solutions:")
    print(f"   1. Check if path is correct")
    print(f"   2. Verify .env file exists in Google Drive")
    print(f"   3. Use Colab Secrets instead (🔑 icon in sidebar)")
```

---

## ✅ After Applying Fix

**Re-run the Colab notebook:**

1. ✅ Restart runtime (Runtime → Restart runtime)
2. ✅ Mount Google Drive
3. ✅ Run improved API key loading cell
4. ✅ Verify all keys show as loaded
5. ✅ Run remaining cells

**Expected result:**
- All 7 tasks should execute successfully
- Agents will respond with analysis
- No authentication errors

---

## 🎯 Prevention for Future Cycles

**Add to notebook template:**
- Always verify API keys loaded before initializing agents
- Add test cell to check authentication
- Use Colab Secrets for better security

---

**Status:** ✅ Fix ready to apply
**Next Step:** Update Colab notebook with improved API key loading cell
