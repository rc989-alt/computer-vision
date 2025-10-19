#!/bin/bash

# Auto-Sync Meeting Reports to Google Drive
# This script automatically syncs new meeting reports to Google Drive

echo "=================================================="
echo "AUTO-SYNC MEETING REPORTS TO GOOGLE DRIVE"
echo "=================================================="
echo ""

# Find Google Drive folder
GDRIVE="/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘"

if [ ! -d "$GDRIVE" ]; then
    echo "❌ Google Drive not found at: $GDRIVE"
    exit 1
fi

# Define paths
SOURCE_DIR="/Users/guyan/computer_vision/computer-vision/multi-agent/reports"
TARGET_DIR="$GDRIVE/cv_multimodal/project/computer-vision-clean/multi-agent/reports"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"
mkdir -p "$TARGET_DIR/archive"

echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo ""

# Function to sync files
sync_reports() {
    echo "🔄 Syncing reports..."

    SYNCED=0
    SKIPPED=0

    # Sync all .md and .json files in reports directory
    for file in "$SOURCE_DIR"/*.{md,json}; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")

            # Check if file exists in target and compare
            if [ -f "$TARGET_DIR/$filename" ]; then
                # Compare file sizes
                SOURCE_SIZE=$(stat -f%z "$file")
                TARGET_SIZE=$(stat -f%z "$TARGET_DIR/$filename")

                if [ "$SOURCE_SIZE" != "$TARGET_SIZE" ]; then
                    cp "$file" "$TARGET_DIR/"
                    echo "   ✅ Updated: $filename"
                    SYNCED=$((SYNCED + 1))
                else
                    SKIPPED=$((SKIPPED + 1))
                fi
            else
                cp "$file" "$TARGET_DIR/"
                echo "   ✅ New: $filename"
                SYNCED=$((SYNCED + 1))
            fi
        fi
    done

    # Sync archive folder
    if [ -d "$SOURCE_DIR/archive" ]; then
        for file in "$SOURCE_DIR/archive"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")

                if [ ! -f "$TARGET_DIR/archive/$filename" ]; then
                    cp "$file" "$TARGET_DIR/archive/"
                    echo "   ✅ Archived: $filename"
                    SYNCED=$((SYNCED + 1))
                fi
            fi
        done
    fi

    echo ""
    echo "   📊 Synced: $SYNCED files"
    echo "   ⏭️  Skipped: $SKIPPED files (unchanged)"
    echo ""
}

# Check if watch mode is requested
if [ "$1" == "--watch" ]; then
    echo "🔍 Watch mode enabled - will sync every 60 seconds"
    echo "   Press Ctrl+C to stop"
    echo ""

    while true; do
        sync_reports
        echo "⏳ Waiting 60 seconds..."
        echo ""
        sleep 60
    done
else
    # One-time sync
    sync_reports

    echo "=================================================="
    echo "✅ SYNC COMPLETE"
    echo "=================================================="
    echo ""
    echo "💡 TIP: Run with --watch to enable continuous syncing:"
    echo "   ./auto_sync_reports.sh --watch"
    echo ""
fi
