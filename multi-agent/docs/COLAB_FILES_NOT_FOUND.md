# Colab "Files Not Found" Troubleshooting

## You're Seeing This Error

```
⚠️ Found folder but no multi-agent: /content/drive/MyDrive/computer-vision
❌ Could not auto-detect project location
```

## What This Means

The notebook found your `computer-vision` folder on Google Drive, but it doesn't see a `multi-agent` subfolder inside it.

## Why This Happens

### Reason 1: Upload Still in Progress (Most Common)
When you drag a folder to Google Drive, it shows the folder immediately but takes time to upload all the files inside.

**Check**:
- Look at bottom-right corner of Drive web interface
- Do you see "Uploading X items"?
- Wait until it says "Upload complete"

**Fix**: Wait 2-5 minutes, then re-run the diagnostic cell in Colab

### Reason 2: Nested Folder Structure
You may have uploaded the folder in a way that created `computer-vision/computer-vision/multi-agent` instead of `computer-vision/multi-agent`.

**Check**: Run the diagnostic cell in the notebook - it will detect this automatically

**Fix**:
```python
# If diagnostic shows nested structure, use this path:
DRIVE_PROJECT = Path("/content/drive/MyDrive/computer-vision/computer-vision")
```

### Reason 3: Uploaded Individual Files Instead of Folder
You may have uploaded files individually rather than the entire folder structure.

**Check**: Use Drive web interface to browse `MyDrive/computer-vision/`

**Fix**: Delete the folder and re-upload the entire `computer-vision` folder from your laptop

### Reason 4: Drive Mount Needs Refresh
Sometimes the Drive mount is cached and doesn't show newly uploaded files.

**Fix**: Remount Drive in Colab
```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive')
```

## Step-by-Step Diagnosis

### Step 1: Run Diagnostic Cell

In your Colab notebook, find and run this cell:
```python
# DIAGNOSTIC TOOL - Run this if you get "files not found" errors
```

This will show you **exactly** what's in your Drive folder.

### Step 2: Check the Output

**Good Output (✅)**:
```
✅ Found folder: /content/drive/MyDrive/computer-vision

Folders (5):
   📁 data/
   📁 docs/
   📁 multi-agent/
      ✅ FOUND IT! This is the multi-agent folder
   📁 research/
   📁 tools/

✅ Perfect! multi-agent/ is directly in computer-vision/
   Path to use: /content/drive/MyDrive/computer-vision
```
→ **Continue to next cell, should work now!**

**Nested Structure (⚠️)**:
```
⚠️ IMPORTANT: Found nested structure!
   Your files are at: computer-vision/computer-vision/
   The correct path is: /content/drive/MyDrive/computer-vision/computer-vision
```
→ **Use the corrected path shown**

**Upload in Progress (⏳)**:
```
Folders (0):
   (empty)

Files (0):
   (empty)

❌ multi-agent/ folder not found!
   Upload may still be in progress
```
→ **Wait 2-5 minutes, then re-run this cell**

**Wrong Folder (❌)**:
```
Folders (15):
   📁 Colab Notebooks/
   📁 Documents/
   📁 Photos/
   (no multi-agent)

❌ multi-agent/ folder not found!
```
→ **Need to upload the correct folder**

## Solutions

### Solution A: Wait for Upload (Most Common)

1. **Check upload progress** in Drive web interface (drive.google.com)
   - Look at bottom-right corner
   - Wait until "Upload complete"

2. **Re-run diagnostic cell** in Colab
   - Should now show `multi-agent/` folder

3. **Continue** to next cell
   - Auto-detection should now work

**Time**: 2-5 minutes for typical project (~50 MB)

### Solution B: Fix Nested Structure

If diagnostic shows nested structure:

1. **Note the corrected path** shown in diagnostic output
   ```
   /content/drive/MyDrive/computer-vision/computer-vision
   ```

2. **When prompted** for manual path, enter the corrected path

3. **Or**: Update the auto-detection list
   ```python
   # In the path setup cell, modify possible_locations:
   possible_locations = [
       DRIVE_BASE / "computer-vision/computer-vision",  # Add this line
       DRIVE_BASE / "computer-vision",
       # ... rest of list
   ]
   ```

### Solution C: Re-upload Correctly

If files are in wrong location or missing:

1. **On your laptop**, verify folder structure:
   ```bash
   cd /Users/guyan/computer_vision
   ls computer-vision/
   # Should show: multi-agent/ research/ docs/ data/ .env

   ls computer-vision/multi-agent/
   # Should show: autonomous_coordinator.py configs/ tools/ agents/
   ```

2. **Delete** the `computer-vision` folder from Google Drive
   - Go to drive.google.com
   - Right-click `computer-vision` → Delete

3. **Upload correctly**:
   - Open drive.google.com in browser
   - Drag the `computer-vision` folder from your laptop
   - **Important**: Drag the folder itself, not just its contents
   - Wait for "Upload complete" in bottom-right

4. **Re-run** diagnostic cell in Colab

### Solution D: Remount Drive

If Drive mount is stale:

```python
# Run this in a Colab cell
from google.colab import drive
drive.flush_and_unmount()
print("Drive unmounted")

# Wait 5 seconds
import time
time.sleep(5)

# Remount
drive.mount('/content/drive')
print("Drive remounted")

# Re-run diagnostic
```

## Verify Your Local Folder Structure

On your laptop, check that you have the right structure:

```bash
cd /Users/guyan/computer_vision/computer-vision

# Should see these folders:
ls -d multi-agent/
ls -d research/
ls -d docs/
ls -d data/

# Should see these files:
ls multi-agent/autonomous_coordinator.py
ls multi-agent/configs/autonomous_coordination.yaml
ls .env  # API keys (create if missing)
```

If any are missing, that's your issue!

## Create Missing .env File

If you're missing the `.env` file:

```bash
cd /Users/guyan/computer_vision/computer-vision

cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here
GOOGLE_API_KEY=AIzaSy-your-key-here
EOF

# Verify it was created
cat .env
```

Then re-upload the folder to Drive (or just upload the .env file).

## Manual Path Entry

If auto-detection keeps failing, you can enter the path manually:

1. **Run the diagnostic cell** to see what's in your Drive

2. **Look for** the folder that contains `multi-agent/`

3. **When prompted**, enter the full path:
   ```
   Path: /content/drive/MyDrive/computer-vision
   ```
   or
   ```
   Path: /content/drive/MyDrive/computer-vision/computer-vision
   ```

4. **Press Enter**

The notebook will verify the path and continue.

## Quick Checklist

Before continuing, verify:

- [ ] `computer-vision` folder exists in Drive
- [ ] Upload shows "Complete" in Drive web interface
- [ ] Diagnostic cell shows `multi-agent/` folder
- [ ] Diagnostic cell shows `.env` file (or you'll enter keys manually)
- [ ] Path is noted (for manual entry if needed)

## Still Not Working?

### Check 1: Folder Permissions
- Ensure the folder wasn't shared with restrictions
- Should be "Only you" have access

### Check 2: File Names
- Check for typos: `multi-agent` not `multiagent` or `multi_agent`
- Case-sensitive on some systems

### Check 3: Browser Cache
- Try opening Drive in an incognito window
- Refresh the page (Ctrl+R or Cmd+R)

### Check 4: Upload Method
- Try using Drive desktop app instead of web upload
- Or use Google Colab's file upload: `files.upload()`

### Check 5: Colab Session
- Try disconnecting and reconnecting runtime
- Runtime → Disconnect and delete runtime → Reconnect

## Expected Timeline

| Step | Time |
|------|------|
| Upload folder to Drive | 5-30 minutes |
| Wait for files to appear | 1-2 minutes |
| Run diagnostic cell | 5 seconds |
| Fix and re-run | 1-2 minutes |
| **Total** | **5-35 minutes** |

## Success Indicators

You'll know it's working when diagnostic shows:

```
✅ Perfect! multi-agent/ is directly in computer-vision/
   Path to use: /content/drive/MyDrive/computer-vision

🔍 Auto-detecting project location...
✅ Found project at: /content/drive/MyDrive/computer-vision

🔍 Checking for required files...
✅ multi-agent/autonomous_coordinator.py (23.4 KB)
✅ multi-agent/configs/autonomous_coordination.yaml (26.8 KB)
```

## Next Steps After Fix

Once diagnostic shows ✅:

1. Continue to next cell (API keys)
2. All subsequent cells should work
3. System will start successfully

## Alternative: Use Standalone Diagnostic Script

If the notebook diagnostic isn't helping, run the standalone script:

1. In Colab, create new cell
2. Copy contents of `multi-agent/colab_diagnostic.py`
3. Paste and run
4. Follow the detailed output

The standalone script provides more verbose diagnostics.

---

**Most Common Issue**: Upload still in progress → Wait 2-5 minutes ⏰

**Second Most Common**: Nested folder → Use corrected path shown in diagnostic

**Quick Test**: Does Drive web interface show `computer-vision/multi-agent/`? If yes, just wait and re-run. If no, re-upload.
