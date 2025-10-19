#!/bin/bash

echo "=================================================="
echo "SYNC RECENT UPDATES TO GOOGLE DRIVE"
echo "=================================================="
echo ""

# Google Drive path
GDRIVE="/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/computer-vision-clean"
SOURCE="/Users/guyan/computer_vision/computer-vision"

if [ ! -d "$GDRIVE" ]; then
    echo "❌ ERROR: Google Drive path not found: $GDRIVE"
    exit 1
fi

echo "✅ Google Drive found: $GDRIVE"
echo "✅ Source: $SOURCE"
echo ""

# Function to copy file with size info
copy_with_info() {
    local src="$1"
    local dst="$2"

    if [ -f "$src" ]; then
        cp "$src" "$dst"
        local size=$(ls -lh "$src" | awk '{print $5}')
        echo "  ✅ $(basename "$src") ($size)"
        return 0
    else
        echo "  ⚠️  $(basename "$src") - not found"
        return 1
    fi
}

echo "=================================================="
echo "1. SYNCING RECENT DOCUMENTATION UPDATES"
echo "=================================================="
echo ""

# Recent documentation files
DOCS=(
    "FEATURE_COMPARISON_OLD_VS_CURRENT_GUIDE.md"
    "GUIDE_UPDATE_RECOMMENDATIONS.md"
    "PRIORITY_EXECUTION_SYSTEM_READY.md"
    "AUTONOMOUS_SYSTEM_COMPLETE_GUIDE.md"
    "COMPREHENSIVE_SYSTEM_ANALYSIS.md"
    "COMPLETE_SYSTEM_SUMMARY.md"
    "OLD_VS_NEW_COMPREHENSIVE_COMPARISON.md"
)

for doc in "${DOCS[@]}"; do
    copy_with_info "$SOURCE/$doc" "$GDRIVE/$doc"
done

echo ""
echo "=================================================="
echo "2. SYNCING OLD SYSTEM (TWO-TIER) FILES"
echo "=================================================="
echo ""

# Old system directory
OLD_SYSTEM="$GDRIVE/multi-agent"
mkdir -p "$OLD_SYSTEM/tools"
mkdir -p "$OLD_SYSTEM/agents/prompts/planning_team"
mkdir -p "$OLD_SYSTEM/agents/prompts/executive_team"
mkdir -p "$OLD_SYSTEM/state"
mkdir -p "$OLD_SYSTEM/reports/handoff"

echo "Coordinator and core files:"
copy_with_info "$SOURCE/multi-agent/autonomous_coordinator.py" "$OLD_SYSTEM/autonomous_coordinator.py"
copy_with_info "$SOURCE/multi-agent/AUTONOMOUS_SYSTEM_GUIDE.md" "$OLD_SYSTEM/AUTONOMOUS_SYSTEM_GUIDE.md"
copy_with_info "$SOURCE/multi-agent/AUTONOMOUS_SYSTEM_SUMMARY.md" "$OLD_SYSTEM/AUTONOMOUS_SYSTEM_SUMMARY.md"

echo ""
echo "Enhanced Progress Sync (CRITICAL - Priority Execution):"
copy_with_info "$SOURCE/multi-agent/tools/enhanced_progress_sync.py" "$OLD_SYSTEM/tools/enhanced_progress_sync.py"

echo ""
echo "Planning Team prompts:"
for prompt in "$SOURCE"/multi-agent/agents/prompts/planning_team/*.md; do
    if [ -f "$prompt" ]; then
        copy_with_info "$prompt" "$OLD_SYSTEM/agents/prompts/planning_team/$(basename "$prompt")"
    fi
done

echo ""
echo "Executive Team prompts:"
for prompt in "$SOURCE"/multi-agent/agents/prompts/executive_team/*.md; do
    if [ -f "$prompt" ]; then
        copy_with_info "$prompt" "$OLD_SYSTEM/agents/prompts/executive_team/$(basename "$prompt")"
    fi
done

echo ""
echo "=================================================="
echo "3. SYNCING UNIFIED SYSTEM FILES"
echo "=================================================="
echo ""

# Unified system directory
UNIFIED_SYSTEM="$GDRIVE/unified-team"
mkdir -p "$UNIFIED_SYSTEM/configs"
mkdir -p "$UNIFIED_SYSTEM/state"
mkdir -p "$UNIFIED_SYSTEM/reports"

echo "Coordinator and core files:"
copy_with_info "$SOURCE/multi-agent/unified_coordinator.py" "$UNIFIED_SYSTEM/unified_coordinator.py"
copy_with_info "$SOURCE/multi-agent/UNIFIED_SYSTEM_GUIDE.md" "$UNIFIED_SYSTEM/UNIFIED_SYSTEM_GUIDE.md" 2>/dev/null || echo "  ⚠️  UNIFIED_SYSTEM_GUIDE.md - not found (optional)"

echo ""
echo "Configuration files:"
if [ -f "$SOURCE/multi-agent/configs/unified_coordination.yaml" ]; then
    copy_with_info "$SOURCE/multi-agent/configs/unified_coordination.yaml" "$UNIFIED_SYSTEM/configs/unified_coordination.yaml"
fi

echo ""
echo "=================================================="
echo "4. SYNCING API KEYS AND ENV FILES"
echo "=================================================="
echo ""

if [ -f "$SOURCE/research/api_keys.env" ]; then
    copy_with_info "$SOURCE/research/api_keys.env" "$GDRIVE/research/api_keys.env"
else
    echo "  ⚠️  api_keys.env not found (may be .gitignored)"
fi

if [ -f "$SOURCE/.env" ]; then
    copy_with_info "$SOURCE/.env" "$GDRIVE/.env"
else
    echo "  ⚠️  .env not found (may be .gitignored)"
fi

echo ""
echo "=================================================="
echo "5. VERIFICATION"
echo "=================================================="
echo ""

echo "Checking old system (two-tier):"
if [ -f "$OLD_SYSTEM/autonomous_coordinator.py" ]; then
    echo "  ✅ autonomous_coordinator.py exists"
else
    echo "  ❌ autonomous_coordinator.py MISSING"
fi

if [ -f "$OLD_SYSTEM/tools/enhanced_progress_sync.py" ]; then
    echo "  ✅ enhanced_progress_sync.py exists (priority execution)"
else
    echo "  ❌ enhanced_progress_sync.py MISSING"
fi

PLANNING_PROMPTS=$(ls "$OLD_SYSTEM/agents/prompts/planning_team"/*.md 2>/dev/null | wc -l)
echo "  ✅ Planning prompts: $PLANNING_PROMPTS files"

EXEC_PROMPTS=$(ls "$OLD_SYSTEM/agents/prompts/executive_team"/*.md 2>/dev/null | wc -l)
echo "  ✅ Executive prompts: $EXEC_PROMPTS files"

echo ""
echo "Checking unified system:"
if [ -f "$UNIFIED_SYSTEM/unified_coordinator.py" ]; then
    echo "  ✅ unified_coordinator.py exists"
else
    echo "  ❌ unified_coordinator.py MISSING"
fi

echo ""
echo "=================================================="
echo "✅ SYNC COMPLETE"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "OLD SYSTEM (Two-Tier - Production Deployment):"
echo "  cd $OLD_SYSTEM"
echo "  python3 autonomous_coordinator.py"
echo ""
echo "UNIFIED SYSTEM (Research - CVPR Deadline):"
echo "  cd $UNIFIED_SYSTEM"
echo "  python3 unified_coordinator.py"
echo ""
echo "Files are syncing to Google Drive now..."
echo ""
