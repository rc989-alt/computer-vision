#!/bin/bash

# Upload Script for Google Drive
# This script shows you which files to upload and where

echo "=================================================="
echo "FILES TO UPLOAD TO GOOGLE DRIVE"
echo "=================================================="
echo ""
echo "Source (your laptop): /Users/guyan/computer_vision/computer-vision/"
echo "Destination (Google Drive): MyDrive/cv_multimodal/project/computer-vision/"
echo ""
echo "=================================================="
echo "Required Files:"
echo "=================================================="
echo ""

# Check if files exist locally
cd /Users/guyan/computer_vision/computer-vision

echo "1. .env file"
if [ -f ".env" ]; then
    size=$(ls -lh .env | awk '{print $5}')
    echo "   ✅ Exists locally ($size)"
    echo "   📤 Upload to: cv_multimodal/project/computer-vision/.env"
else
    echo "   ❌ Not found locally"
fi
echo ""

echo "2. autonomous_coordinator.py"
if [ -f "multi-agent/autonomous_coordinator.py" ]; then
    size=$(ls -lh multi-agent/autonomous_coordinator.py | awk '{print $5}')
    echo "   ✅ Exists locally ($size)"
    echo "   📤 Upload to: cv_multimodal/project/computer-vision/multi-agent/autonomous_coordinator.py"
else
    echo "   ❌ Not found locally"
fi
echo ""

echo "3. autonomous_coordination.yaml"
if [ -f "multi-agent/configs/autonomous_coordination.yaml" ]; then
    size=$(ls -lh multi-agent/configs/autonomous_coordination.yaml | awk '{print $5}')
    echo "   ✅ Exists locally ($size)"
    echo "   📤 Upload to: cv_multimodal/project/computer-vision/multi-agent/configs/autonomous_coordination.yaml"
else
    echo "   ❌ Not found locally"
fi
echo ""

echo "4. autonomous_system_colab.ipynb"
if [ -f "research/colab/autonomous_system_colab.ipynb" ]; then
    size=$(ls -lh research/colab/autonomous_system_colab.ipynb | awk '{print $5}')
    echo "   ✅ Exists locally ($size)"
    echo "   📤 Upload to: cv_multimodal/project/computer-vision/research/colab/autonomous_system_colab.ipynb"
else
    echo "   ❌ Not found locally"
fi
echo ""

echo "=================================================="
echo "UPLOAD METHODS:"
echo "=================================================="
echo ""
echo "Option 1: Drag and Drop (Easiest)"
echo "  1. Open drive.google.com in browser"
echo "  2. Navigate to cv_multimodal/project/computer-vision/"
echo "  3. Drag each file to the correct folder"
echo ""
echo "Option 2: Google Drive Desktop App"
echo "  1. Install Google Drive app if not installed"
echo "  2. Files will sync automatically to:"
echo "     ~/Google Drive/My Drive/cv_multimodal/project/computer-vision/"
echo "  3. Copy files using Finder or command line"
echo ""
echo "Option 3: Command Line (if you have Drive app)"
echo "  Run these commands to copy files:"
echo ""
echo "  cp .env ~/Google\ Drive/My\ Drive/cv_multimodal/project/computer-vision/"
echo "  cp multi-agent/autonomous_coordinator.py ~/Google\ Drive/My\ Drive/cv_multimodal/project/computer-vision/multi-agent/"
echo "  cp multi-agent/configs/autonomous_coordination.yaml ~/Google\ Drive/My\ Drive/cv_multimodal/project/computer-vision/multi-agent/configs/"
echo "  cp research/colab/autonomous_system_colab.ipynb ~/Google\ Drive/My\ Drive/cv_multimodal/project/computer-vision/research/colab/"
echo ""
echo "=================================================="
echo "After uploading, open the notebook in Colab!"
echo "=================================================="
