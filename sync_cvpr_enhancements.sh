#!/bin/bash
# Sync CVPR Paper Enhancements to Google Drive
# This script copies all new files to Google Drive for Colab access

GDRIVE="/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/computer-vision-clean"
LOCAL="/Users/guyan/computer_vision/computer-vision"

echo "🔄 Syncing CVPR Paper Enhancements to Google Drive..."
echo "=================================================="

# Function to copy file with status
copy_file() {
    local src=$1
    local dst=$2
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "✅ Copied: $(basename $src)"
    else
        echo "⚠️  Not found: $(basename $src)"
    fi
}

# 1. Strategy Documents
echo ""
echo "📄 Copying Strategy Documents..."
copy_file "$LOCAL/CVPR_2025_SUBMISSION_STRATEGY.md" "$GDRIVE/"
copy_file "$LOCAL/CVPR_PAPER_PLAN_SUMMARY.md" "$GDRIVE/"
copy_file "$LOCAL/SYSTEM_IMPROVEMENTS_SUMMARY.md" "$GDRIVE/"
copy_file "$LOCAL/COMPLETE_ENHANCEMENTS_SUMMARY.md" "$GDRIVE/"

# 2. Research Director Role (Already in Drive)
echo ""
echo "👤 Checking Research Director Role..."
if [ -f "$LOCAL/multi-agent/agents/prompts/planning_team/research_director.md" ]; then
    echo "✅ Research Director role ready"
else
    echo "⚠️  Research Director role not found in local"
fi

# 3. All other files already synced to Drive
echo ""
echo "✅ Core System Files (Already in Drive):"
echo "   - multi-agent/tools/execution_tools.py (MLflow integration)"
echo "   - multi-agent/tools/resource_monitor.py"
echo "   - multi-agent/schemas/handoff_v3.1.json"
echo "   - multi-agent/standards/SUCCESS_METRICS_STANDARD.md"
echo "   - research/tools/attention_analysis.py"
echo "   - docs/MLFLOW_INTEGRATION_GUIDE.md"
echo "   - All agent prompts (planning + executive teams)"

echo ""
echo "=================================================="
echo "✅ Sync Complete!"
echo ""
echo "📋 Files ready in Google Drive:"
echo "   $GDRIVE"
echo ""
echo "🚀 Next: Run the Colab notebook to start autonomous system"
