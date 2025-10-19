#!/bin/bash

echo "=================================================="
echo "COPY FILES TO GOOGLE DRIVE"
echo "=================================================="
echo ""

# Find Google Drive folder
GDRIVE=""

# Common locations for Google Drive
if [ -d "$HOME/Google Drive/My Drive" ]; then
    GDRIVE="$HOME/Google Drive/My Drive"
elif [ -d "$HOME/GoogleDrive/My Drive" ]; then
    GDRIVE="$HOME/GoogleDrive/My Drive"
elif [ -d "$HOME/Library/CloudStorage/GoogleDrive" ]; then
    # New Google Drive location on Mac
    GDRIVE_ACCOUNTS=$(ls "$HOME/Library/CloudStorage/" | grep GoogleDrive)
    if [ ! -z "$GDRIVE_ACCOUNTS" ]; then
        FIRST_ACCOUNT=$(echo "$GDRIVE_ACCOUNTS" | head -1)
        GDRIVE="$HOME/Library/CloudStorage/$FIRST_ACCOUNT/My Drive"
    fi
fi

if [ -z "$GDRIVE" ]; then
    echo "❌ Could not find Google Drive folder automatically."
    echo ""
    echo "Please enter the full path to your Google Drive 'My Drive' folder:"
    echo "Example: /Users/guyan/Google Drive/My Drive"
    read -p "Path: " GDRIVE
fi

echo "Google Drive location: $GDRIVE"
echo ""

# Check if path exists
if [ ! -d "$GDRIVE" ]; then
    echo "❌ ERROR: Path does not exist: $GDRIVE"
    exit 1
fi

# Create target folder structure
TARGET="$GDRIVE/cv_multimodal/project/computer-vision"

echo "Creating folder structure..."
mkdir -p "$TARGET/multi-agent/configs"
mkdir -p "$TARGET/research/colab"

echo "✅ Folders created"
echo ""

# Source directory
SOURCE="/Users/guyan/computer_vision/computer-vision"

echo "Copying files..."
echo ""

# Copy .env
if [ -f "$SOURCE/.env" ]; then
    cp "$SOURCE/.env" "$TARGET/.env"
    echo "✅ Copied .env ($(ls -lh "$SOURCE/.env" | awk '{print $5}'))"
else
    echo "⚠️  .env not found"
fi

# Copy autonomous_coordinator.py
if [ -f "$SOURCE/multi-agent/autonomous_coordinator.py" ]; then
    cp "$SOURCE/multi-agent/autonomous_coordinator.py" "$TARGET/multi-agent/"
    echo "✅ Copied autonomous_coordinator.py ($(ls -lh "$SOURCE/multi-agent/autonomous_coordinator.py" | awk '{print $5}'))"
else
    echo "⚠️  autonomous_coordinator.py not found"
fi

# Copy autonomous_coordination.yaml
if [ -f "$SOURCE/multi-agent/configs/autonomous_coordination.yaml" ]; then
    cp "$SOURCE/multi-agent/configs/autonomous_coordination.yaml" "$TARGET/multi-agent/configs/"
    echo "✅ Copied autonomous_coordination.yaml ($(ls -lh "$SOURCE/multi-agent/configs/autonomous_coordination.yaml" | awk '{print $5}'))"
else
    echo "⚠️  autonomous_coordination.yaml not found"
fi

# Copy notebook
if [ -f "$SOURCE/research/colab/autonomous_system_colab.ipynb" ]; then
    cp "$SOURCE/research/colab/autonomous_system_colab.ipynb" "$TARGET/research/colab/"
    echo "✅ Copied autonomous_system_colab.ipynb ($(ls -lh "$SOURCE/research/colab/autonomous_system_colab.ipynb" | awk '{print $5}'))"
else
    echo "⚠️  autonomous_system_colab.ipynb not found"
fi

# Copy reports directory (meeting artifacts)
if [ -d "$SOURCE/multi-agent/reports" ]; then
    echo ""
    echo "Copying meeting reports..."
    mkdir -p "$TARGET/multi-agent/reports"

    # Copy all transcripts, summaries, and JSON files
    REPORT_COUNT=0
    for file in "$SOURCE/multi-agent/reports"/*.{md,json} 2>/dev/null; do
        if [ -f "$file" ]; then
            cp "$file" "$TARGET/multi-agent/reports/"
            REPORT_COUNT=$((REPORT_COUNT + 1))
            echo "✅ Copied $(basename "$file")"
        fi
    done

    # Copy archive folder if it exists
    if [ -d "$SOURCE/multi-agent/reports/archive" ]; then
        mkdir -p "$TARGET/multi-agent/reports/archive"
        cp -r "$SOURCE/multi-agent/reports/archive/"* "$TARGET/multi-agent/reports/archive/" 2>/dev/null
        ARCHIVE_COUNT=$(ls "$SOURCE/multi-agent/reports/archive" | wc -l)
        echo "✅ Copied archive folder ($ARCHIVE_COUNT files)"
    fi

    echo "✅ Total reports copied: $REPORT_COUNT"
else
    echo "⚠️  No reports directory found"
fi

echo ""
echo "=================================================="
echo "✅ DONE! Files copied to Google Drive"
echo "=================================================="
echo ""
echo "Location: $TARGET"
echo ""
echo "Files will sync to Google Drive automatically."
echo "Wait a few minutes for sync to complete, then:"
echo ""
echo "1. Go to drive.google.com"
echo "2. Navigate to cv_multimodal/project/computer-vision/"
echo "3. Verify files are there"
echo "4. Open research/colab/autonomous_system_colab.ipynb in Colab"
echo "5. Run all cells!"
echo ""
