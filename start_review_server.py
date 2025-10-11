#!/usr/bin/env python3
"""
Local Review Server

Simple HTTP server to test the borderline review UI locally.
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def start_review_server(port=8000):
    """Start local server for review UI."""
    
    # Change to review directory
    review_dir = Path(__file__).parent / "review"
    if not review_dir.exists():
        print("❌ Review directory not found. Run make_borderline_items.py first.")
        return
    
    os.chdir(review_dir)
    
    # Check if items.json exists
    if not Path("items.json").exists():
        print("❌ items.json not found. Run make_borderline_items.py first.")
        return
    
    # Start server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}"
            print(f"🌐 Review UI server started at {url}")
            print(f"📂 Serving from: {review_dir}")
            print(f"🔍 Open browser to start reviewing borderline items")
            print(f"⏹️  Press Ctrl+C to stop server")
            
            # Try to open browser automatically
            try:
                webbrowser.open(url)
            except:
                pass
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n✅ Review server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} already in use. Try a different port.")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start local review server')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    
    args = parser.parse_args()
    start_review_server(args.port)