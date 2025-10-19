"""
Test Gemini API Connection in Colab
Run this to verify Gemini agent can connect properly
"""

import os
import sys

print("="*60)
print("🔍 TESTING GEMINI API CONNECTION")
print("="*60)

# Check if API key is loaded
print("\n1️⃣ Checking API Key...")
gemini_key = os.getenv("GOOGLE_API_KEY")
if gemini_key:
    print(f"✅ GOOGLE_API_KEY found: {gemini_key[:20]}...")
else:
    print("❌ GOOGLE_API_KEY not set!")
    print("\n💡 Load it from .env file:")
    print("""
import os
env_file = "/content/cv_project/.env"
with open(env_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
print("✅ API keys loaded")
    """)
    sys.exit(1)

# Test connection
print("\n2️⃣ Testing Connection...")
try:
    import google.generativeai as genai
    print("✅ google.generativeai imported")
except ImportError:
    print("❌ google-generativeai not installed")
    print("   Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "google-generativeai"],
                   capture_output=True)
    import google.generativeai as genai
    print("✅ google.generativeai installed and imported")

# Configure API
print("\n3️⃣ Configuring API...")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print("✅ API configured")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# List models
print("\n4️⃣ Listing Available Models...")
try:
    models = list(genai.list_models())
    print(f"✅ Found {len(models)} models:")

    # Look for gemini-2.0-flash-exp specifically
    for model in models:
        if "gemini" in model.name.lower():
            print(f"   - {model.name}")
            if "2.0" in model.name and "flash" in model.name:
                print(f"     ⭐ This is the model we'll use!")

except Exception as e:
    print(f"❌ Failed to list models: {e}")
    sys.exit(1)

# Test generation
print("\n5️⃣ Testing Text Generation...")
try:
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content("Say 'Hello from Gemini!' in exactly those words.")
    print(f"✅ Generation successful!")
    print(f"   Response: {response.text[:100]}")
except Exception as e:
    print(f"❌ Generation failed: {e}")
    print(f"\n💡 Error details: {type(e).__name__}")
    if hasattr(e, 'message'):
        print(f"   Message: {e.message}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ GEMINI API FULLY FUNCTIONAL")
print("="*60)
print("\n🎉 The Gemini agent (CoTRR Team) is ready!")
print("   - Model: gemini-2.0-flash-exp")
print("   - Role: Lightweight optimization specialist")
print("   - Status: Connected and operational")
print("\n📋 Next: Deploy system with Gemini agent enabled")
