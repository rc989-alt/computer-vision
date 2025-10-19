#!/usr/bin/env python3
"""
Quick API connectivity test script
Tests OpenAI, Anthropic, and Google Gemini API connections
"""

import os
from pathlib import Path

# Load environment variables from api_keys.env
def load_env_file(env_path):
    """Load environment variables from .env file"""
    if not Path(env_path).exists():
        print(f"❌ Environment file not found: {env_path}")
        return False

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    return True

# Load API keys
env_file = Path(__file__).parent / "api_keys.env"
if not load_env_file(env_file):
    print("Please create api_keys.env with your API keys")
    exit(1)

print("=" * 60)
print("🔍 Testing API Connections")
print("=" * 60)

# Test OpenAI
print("\n1️⃣ Testing OpenAI API...")
try:
    from openai import OpenAI
    client = OpenAI()
    models = client.models.list()
    print(f"✅ OpenAI connected successfully!")
    print(f"   Available models: {len(list(models.data))} models")
except Exception as e:
    print(f"❌ OpenAI connection failed: {str(e)}")

# Test Anthropic
print("\n2️⃣ Testing Anthropic API...")
try:
    from anthropic import Anthropic
    client = Anthropic()
    models = client.models.list()
    print(f"✅ Anthropic connected successfully!")
    print(f"   Available models: {len(models.data)} models")
except Exception as e:
    print(f"❌ Anthropic connection failed: {str(e)}")

# Test Google Gemini
print("\n3️⃣ Testing Google Gemini API...")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models = list(genai.list_models())
    print(f"✅ Google Gemini connected successfully!")
    print(f"   Available models: {len(models)} models")
except Exception as e:
    print(f"❌ Google Gemini connection failed: {str(e)}")

print("\n" + "=" * 60)
print("✨ API Connection Test Complete")
print("=" * 60)
