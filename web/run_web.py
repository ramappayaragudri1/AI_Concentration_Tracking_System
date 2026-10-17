#!/usr/bin/env python3
"""
Lightweight Web Server for the AI Concentration Tracking System.
Runs an HTTP server on port 8000 and automatically opens the browser dashboard.
"""
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
WEB_DIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

def main():
    os.chdir(WEB_DIR)
    url = f"http://localhost:{PORT}"
    print("=" * 60)
    print("  AI Concentration Tracking System - Web Dashboard")
    print("=" * 60)
    print(f"  Serving dashboard at: {url}")
    print("  Press Ctrl+C to stop the web server.\n")

    # Automatically open default web browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
