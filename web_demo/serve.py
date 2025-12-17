#!/usr/bin/env python3
"""
Simple HTTP server for the web demo with CORS headers.
Run from project root: python web_demo/serve.py
"""

import http.server
import socketserver
import os

PORT = 8000

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

    def guess_type(self, path):
        if path.endswith('.wasm'):
            return 'application/wasm'
        return super().guess_type(path)

if __name__ == '__main__':
    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print(f"Open http://localhost:{PORT}/web_demo/ in your browser")
        print("\nNote: 512MB model will take time to load!")
        print("Consider quantizing with: python quantize_onnx.py")
        httpd.serve_forever()
