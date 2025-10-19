#!/bin/bash

GDRIVE="/Users/guyan/Library/CloudStorage/GoogleDrive-rc989@cornell.edu/我的云端硬盘/cv_multimodal/project/computer-vision-clean"

echo "=========================================="
echo "GOOGLE DRIVE FILE VERIFICATION"
echo "=========================================="
echo ""
echo "Location: $GDRIVE"
echo ""

# Check essential files
echo "✅ ESSENTIAL FILES:"
echo ""

if [ -f "$GDRIVE/.env" ]; then
    SIZE=$(ls -lh "$GDRIVE/.env" | awk '{print $5}')
    echo "   ✅ .env ($SIZE)"
else
    echo "   ❌ .env - MISSING!"
fi

if [ -f "$GDRIVE/multi-agent/autonomous_coordinator.py" ]; then
    SIZE=$(ls -lh "$GDRIVE/multi-agent/autonomous_coordinator.py" | awk '{print $5}')
    echo "   ✅ autonomous_coordinator.py ($SIZE)"
else
    echo "   ❌ autonomous_coordinator.py - MISSING!"
fi

if [ -f "$GDRIVE/multi-agent/configs/autonomous_coordination.yaml" ]; then
    SIZE=$(ls -lh "$GDRIVE/multi-agent/configs/autonomous_coordination.yaml" | awk '{print $5}')
    echo "   ✅ autonomous_coordination.yaml ($SIZE)"
else
    echo "   ❌ autonomous_coordination.yaml - MISSING!"
fi

if [ -f "$GDRIVE/research/colab/autonomous_system_colab.ipynb" ]; then
    SIZE=$(ls -lh "$GDRIVE/research/colab/autonomous_system_colab.ipynb" | awk '{print $5}')
    echo "   ✅ autonomous_system_colab.ipynb ($SIZE)"
else
    echo "   ❌ autonomous_system_colab.ipynb - MISSING!"
fi

echo ""
echo "📂 FOLDERS:"
echo ""

if [ -d "$GDRIVE/multi-agent/tools" ]; then
    COUNT=$(ls -1 "$GDRIVE/multi-agent/tools" | wc -l)
    echo "   ✅ multi-agent/tools/ ($COUNT files)"
else
    echo "   ❌ multi-agent/tools/ - MISSING!"
fi

if [ -d "$GDRIVE/multi-agent/agents" ]; then
    COUNT=$(ls -1 "$GDRIVE/multi-agent/agents" | wc -l)
    echo "   ✅ multi-agent/agents/ ($COUNT items)"
else
    echo "   ❌ multi-agent/agents/ - MISSING!"
fi

if [ -d "$GDRIVE/research" ]; then
    COUNT=$(find "$GDRIVE/research" -type f | wc -l)
    echo "   ✅ research/ ($COUNT files)"
else
    echo "   ❌ research/ - MISSING!"
fi

echo ""
echo "📊 STATISTICS:"
echo ""

PY_COUNT=$(find "$GDRIVE" -name "*.py" -type f | wc -l)
echo "   Python files: $PY_COUNT"

YAML_COUNT=$(find "$GDRIVE" -name "*.yaml" -type f | wc -l)
echo "   YAML files: $YAML_COUNT"

IPYNB_COUNT=$(find "$GDRIVE" -name "*.ipynb" -type f | wc -l)
echo "   Notebooks: $IPYNB_COUNT"

TOTAL_SIZE=$(du -sh "$GDRIVE" 2>/dev/null | awk '{print $1}')
echo "   Total size: $TOTAL_SIZE"

echo ""
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
