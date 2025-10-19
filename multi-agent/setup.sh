#!/bin/bash

# Multi-Agent System Setup Script

echo "🚀 Setting up Multi-Agent Deliberation System..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check API keys
echo ""
echo "🔑 Checking API keys..."

if [ -f "../research/api_keys.env" ]; then
    source ../research/api_keys.env
    echo "✅ API keys loaded from research/api_keys.env"
else
    echo "⚠️  API keys file not found at research/api_keys.env"
    echo "   Please ensure your API keys are configured"
fi

# Create reports directory
mkdir -p reports
echo "✅ Reports directory created"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Customize agent prompts in: agents/prompts/"
echo "2. Add your project data to: data/"
echo "3. Run a test meeting:"
echo ""
echo "   python3 run_meeting.py \"Your question here\""
echo ""
echo "📚 Read the README.md for detailed usage instructions"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
