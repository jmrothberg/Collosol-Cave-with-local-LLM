#!/usr/bin/env python3
"""Local dev server with COOP/COEP headers for crossOriginIsolated mode.

Enables SharedArrayBuffer and WASM multi-threading in the browser,
which ONNX Runtime Web needs for full performance.

Usage:
    python3 scripts/serve-threaded.py [port]   # default 8080
    # Open http://localhost:8080/browser_adventure/adventure.html
"""

import os
import sys
import http.server
import socketserver
import threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

# SimpleHTTPRequestHandler serves the process cwd, not the script's folder. Anchor to repo
# root (parent of scripts/) so /browser_adventure/... works when launched from ~ or anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)


class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


with ThreadedServer(("", PORT), COOPCOEPHandler) as httpd:
    print(f"Serving on http://localhost:{PORT}")
    print(f"  Root: {_REPO_ROOT}")
    print(f"  Game: http://localhost:{PORT}/browser_adventure/adventure.html")
    print(f"  crossOriginIsolated: enabled (COOP + COEP headers)")
    print(f"  Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
