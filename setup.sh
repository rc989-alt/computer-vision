#!/bin/bash

# Computer Vision Pipeline Setup Script

set -e

echo "🔧 Setting up Computer Vision Pipeline..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [[ $(echo "$python_version < $required_version" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check for CUDA
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "🚀 CUDA detected - GPU acceleration available"
else
    echo "💻 CUDA not detected - using CPU mode"
fi

# Install Node.js dependencies if package.json exists
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{input,output,temp,training}
mkdir -p models
mkdir -p logs
mkdir -p reports

# Download YOLO model if not exists
if [ ! -f "models/yolov8n.pt" ]; then
    echo "📥 Downloading YOLO model..."
    python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.save('models/yolov8n.pt')
print('✅ YOLO model downloaded')
"
fi

# Setup environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "🔑 Please edit .env with your API keys"
fi

echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start:"
echo "  1. Edit .env with your API keys"
echo "  2. Run: python pipeline.py --config config/default.json --input data/input/sample_input.json --output data/output/results.json"
echo "  3. Or try individual components:"
echo "     - YOLO detection: python scripts/image_model.py --help"
echo "     - CLIP probe: python scripts/clip_probe/train_clip_probe_balanced.py --help"
echo "     - Reranking: node scripts/rerank_with_compliance.mjs --help"