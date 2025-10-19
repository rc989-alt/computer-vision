# .env File Path Fix

**Issue:** `.env file not found at /content/drive/MyDrive/cv_multimodal/project/.env`
**Reported Location:** `MyDrive/cv_multimodal/project/`

---

## 🔍 Diagnosis

The error message shows:
```
⚠️ .env file not found at /content/drive/MyDrive/cv_multimodal/project/.env
```

But you said the API keys are in: `mydrive/cv_multimodal/project/`

**Possible issues:**
1. File might be named differently (e.g., `api_keys.env`, `.env.txt`, `keys.env`)
2. File might be in a subfolder
3. File might exist but permissions issue
4. Google Drive mount path different

---

## ✅ Quick Fix - Search for .env File

Add this cell in Colab **RIGHT AFTER mounting Google Drive:**

```python
import os
from pathlib import Path

print("="*80)
print("🔍 SEARCHING FOR .env FILE")
print("="*80)

# Check if Drive is mounted
if not os.path.exists('/content/drive/MyDrive'):
    print("❌ Google Drive not mounted!")
    print("   Run the mount cell first")
else:
    print("✅ Google Drive mounted\n")

    # Search in multiple locations
    search_paths = [
        '/content/drive/MyDrive/cv_multimodal/project/.env',
        '/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/.env',
        '/content/drive/MyDrive/cv_multimodal/.env',
        '/content/drive/MyDrive/.env',
        '/content/drive/My Drive/cv_multimodal/project/.env'
    ]

    print("Searching for .env file in common locations:\n")
    found = False

    for path in search_paths:
        if Path(path).exists():
            print(f"✅ FOUND: {path}")
            print(f"   Size: {Path(path).stat().st_size} bytes")
            found = True

            # Show what's inside (first few lines, masked)
            print(f"\n   Contents preview:")
            with open(path, 'r') as f:
                for i, line in enumerate(f, 1):
                    if i > 5:
                        break
                    if '=' in line and not line.startswith('#'):
                        key = line.split('=')[0].strip()
                        print(f"   {i}. {key}=***")
                    else:
                        print(f"   {i}. {line.strip()[:50]}")

            print(f"\n   👉 Use this path: {path}")
            break
        else:
            print(f"❌ Not found: {path}")

    if not found:
        print("\n⚠️ .env file not found in any common location")
        print("\n🔍 Let's search the entire cv_multimodal directory:")

        # Search for any .env files
        base_path = Path('/content/drive/MyDrive/cv_multimodal')
        if base_path.exists():
            env_files = list(base_path.rglob('*.env'))
            env_files += list(base_path.rglob('.env*'))

            if env_files:
                print(f"\n✅ Found {len(env_files)} .env file(s):")
                for f in env_files:
                    print(f"   📄 {f}")
                    print(f"      Size: {f.stat().st_size} bytes")
            else:
                print("\n❌ No .env files found in cv_multimodal directory")
                print("\n💡 Solutions:")
                print("   1. Check if .env file exists in your Google Drive")
                print("   2. Make sure it's named exactly '.env' (with the dot)")
                print("   3. Upload .env file to MyDrive/cv_multimodal/project/")
                print("   4. Or use Colab Secrets instead (🔑 icon in sidebar)")
        else:
            print(f"❌ Base path not found: {base_path}")

print("\n" + "="*80)
```

---

## 🔧 Alternative Fix - List All Files

If the search doesn't find it, list all files in the project directory:

```python
import os
from pathlib import Path

project_dir = Path('/content/drive/MyDrive/cv_multimodal/project')

if project_dir.exists():
    print(f"📁 Listing files in: {project_dir}\n")

    # List all files (not just .env)
    all_files = []
    for item in project_dir.iterdir():
        if item.is_file():
            all_files.append(item)
            # Check for env-related files
            if 'env' in item.name.lower() or 'key' in item.name.lower() or 'api' in item.name.lower():
                print(f"⭐ {item.name} ({item.stat().st_size} bytes)")
            else:
                print(f"   {item.name}")

    print(f"\nTotal files: {len(all_files)}")

    # Check for hidden files
    hidden = [f for f in all_files if f.name.startswith('.')]
    if hidden:
        print(f"\n🔍 Hidden files (starting with .):")
        for f in hidden:
            print(f"   {f.name}")
else:
    print(f"❌ Directory not found: {project_dir}")
```

---

## 📝 Correct Path Template

Once you find the file, update the API key loading cell:

```python
import os
from pathlib import Path

print("="*80)
print("🔑 LOADING API KEYS")
print("="*80)

# UPDATE THIS PATH with the correct location from search above
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/.env')  # ← CHANGE THIS

# Rest of the code stays the same...
if env_file.exists():
    print(f"\n📄 Found .env file: {env_file}")
    print(f"   Size: {env_file.stat().st_size} bytes\n")

    loaded_keys = []
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                loaded_keys.append(key)

    # Verify required keys
    required = {
        'ANTHROPIC_API_KEY': 'Claude Sonnet',
        'OPENAI_API_KEY': 'GPT-4',
        'GOOGLE_API_KEY': 'Gemini'
    }

    print("Checking required API keys:\n")
    all_loaded = True

    for key, usage in required.items():
        value = os.environ.get(key)
        if value and len(value) > 10:
            masked = f"{value[:8]}...{value[-4:]}"
            print(f"✅ {key}")
            print(f"   Value: {masked}")
            print(f"   Used by: {usage}\n")
        else:
            print(f"❌ {key} - NOT FOUND")
            all_loaded = False

    print("="*80)
    if all_loaded:
        print("✅ ALL API KEYS LOADED SUCCESSFULLY")
    else:
        print("❌ SOME API KEYS MISSING")
    print("="*80)
else:
    print(f"\n❌ .env file not found at: {env_file}")
    print("\n💡 Run the search cell above to find the correct path")
    raise FileNotFoundError(f"Missing .env file")
```

---

## 🎯 Common .env File Locations

Try these paths in order:

```python
# Option 1: In project root
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/.env')

# Option 2: In computer-vision-clean subdirectory
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/.env')

# Option 3: In cv_multimodal root
env_file = Path('/content/drive/MyDrive/cv_multimodal/.env')

# Option 4: With space in "My Drive"
env_file = Path('/content/drive/My Drive/cv_multimodal/project/.env')

# Option 5: Different name
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/api_keys.env')
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/keys.env')
env_file = Path('/content/drive/MyDrive/cv_multimodal/project/.env.txt')
```

---

## 🔍 Check File Name

The file MUST be named exactly `.env` (with the dot at the beginning).

**Common mistakes:**
- ❌ `env.txt` (missing dot)
- ❌ `.env.txt` (has .txt extension)
- ❌ `api_keys.env` (different name)
- ❌ `ENV` (uppercase)

**Correct:**
- ✅ `.env` (exactly this)

---

## 💡 Alternative Solution - Use Colab Secrets

If you can't find the .env file, use Colab's built-in secrets feature:

### **Step 1: Add Secrets to Colab**

1. Click 🔑 key icon in Colab left sidebar
2. Click "Add new secret"
3. Add these 3 secrets:
   - Name: `ANTHROPIC_API_KEY`, Value: `sk-ant-...`
   - Name: `OPENAI_API_KEY`, Value: `sk-proj-...`
   - Name: `GOOGLE_API_KEY`, Value: `AIza...`
4. Toggle "Notebook access" ON for each

### **Step 2: Load from Secrets**

Replace the .env loading cell with:

```python
from google.colab import userdata
import os

print("="*80)
print("🔑 LOADING API KEYS FROM COLAB SECRETS")
print("="*80)

try:
    # Load from Colab secrets
    os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
    os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

    print("\nChecking keys:\n")

    for key in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']:
        value = os.environ.get(key)
        if value:
            masked = f"{value[:8]}...{value[-4:]}"
            print(f"✅ {key}: {masked}")
        else:
            print(f"❌ {key}: NOT SET")

    print("\n" + "="*80)
    print("✅ API KEYS LOADED FROM COLAB SECRETS")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error loading from Colab secrets: {e}")
    print("\n💡 Make sure to:")
    print("   1. Click 🔑 icon in left sidebar")
    print("   2. Add all 3 secrets")
    print("   3. Enable 'Notebook access'")
```

---

## 📋 Quick Checklist

**Run these checks in Colab:**

1. **Is Google Drive mounted?**
   ```python
   import os
   print(os.path.exists('/content/drive/MyDrive'))  # Should be True
   ```

2. **Does the directory exist?**
   ```python
   import os
   print(os.path.exists('/content/drive/MyDrive/cv_multimodal/project'))  # Should be True
   ```

3. **List files in that directory:**
   ```python
   import os
   print(os.listdir('/content/drive/MyDrive/cv_multimodal/project'))
   ```

4. **Search for .env:**
   ```python
   from pathlib import Path
   base = Path('/content/drive/MyDrive/cv_multimodal')
   env_files = list(base.rglob('*.env')) + list(base.rglob('.env'))
   print(env_files)
   ```

---

## ✅ Final Solution

**Add this cell to your Colab notebook RIGHT AFTER mounting Google Drive:**

```python
import os
from pathlib import Path

print("="*80)
print("🔍 FINDING AND LOADING API KEYS")
print("="*80)

# Search for .env file
search_paths = [
    '/content/drive/MyDrive/cv_multimodal/project/.env',
    '/content/drive/MyDrive/cv_multimodal/project/computer-vision-clean/.env',
    '/content/drive/MyDrive/cv_multimodal/.env',
    '/content/drive/My Drive/cv_multimodal/project/.env'
]

env_file = None
for path in search_paths:
    if Path(path).exists():
        env_file = Path(path)
        print(f"\n✅ Found .env file: {path}")
        print(f"   Size: {env_file.stat().st_size} bytes\n")
        break
    else:
        print(f"   Checking: {path}... not found")

if not env_file:
    print("\n❌ Could not find .env file in any common location")
    print("\n🔍 Searching entire cv_multimodal directory...")

    base = Path('/content/drive/MyDrive/cv_multimodal')
    all_env = list(base.rglob('*.env')) + list(base.rglob('.env*'))

    if all_env:
        print(f"\n✅ Found .env files:")
        for f in all_env:
            print(f"   📄 {f}")
        env_file = all_env[0]  # Use first one found
    else:
        print("\n❌ No .env files found")
        print("\n💡 Use Colab Secrets instead:")
        print("   1. Click 🔑 icon in left sidebar")
        print("   2. Add ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
        raise FileNotFoundError("No .env file found")

# Load API keys
if env_file:
    loaded_keys = []
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                loaded_keys.append(key)

    # Verify
    required = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']
    all_loaded = True

    print("Verifying required keys:\n")
    for key in required:
        value = os.environ.get(key)
        if value and len(value) > 10:
            print(f"✅ {key}: {value[:8]}...{value[-4:]}")
        else:
            print(f"❌ {key}: MISSING")
            all_loaded = False

    print("\n" + "="*80)
    if all_loaded:
        print("✅ ALL API KEYS LOADED SUCCESSFULLY")
        print(f"   Loaded {len(loaded_keys)} total keys from: {env_file}")
    else:
        print("❌ SOME REQUIRED KEYS MISSING")
    print("="*80)
```

---

**This cell will:**
1. ✅ Search multiple common paths
2. ✅ Find the .env file wherever it is
3. ✅ Load all keys automatically
4. ✅ Verify all required keys present
5. ✅ Show clear error if not found

**Status:** ✅ Fix ready to apply
**Action:** Add this cell to Colab and run it
