#!/usr/bin/env python3
"""
Mobile Trading Signals Server
Serves the mobile trading signals page at http://localhost:3001
"""

import http.server
import socketserver
import os
import sys
import webbrowser
import threading
import time
import mimetypes
import socket
from pathlib import Path

class MobileSignalsHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for serving mobile signals files"""
    
    def __init__(self, *args, **kwargs):
        # Ensure CSS MIME type is properly set
        mimetypes.add_type('text/css', '.css')
        # Set the directory to serve files from
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for better compatibility
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Ensure proper Content-Type for CSS files
        if self.path.endswith('.css'):
            self.send_header('Content-Type', 'text/css; charset=utf-8')
        
        super().end_headers()
    
    def do_GET(self):
        # Redirect root to mobile_signals.html
        if self.path == '/':
            self.path = '/mobile_signals.html'
        elif self.path == '/mobile_signals' or self.path == '/signals':
            self.path = '/mobile_signals.html'
        
        # Ensure the file exists before serving
        file_path = self.translate_path(self.path)
        if not os.path.exists(file_path) and self.path == '/mobile_signals.html':
            # If mobile_signals.html doesn't exist, generate it first
            print("Generating mobile signals...")
            try:
                import subprocess
                result = subprocess.run([sys.executable, 'mobile_signals.py'], 
                                      capture_output=True, text=True, cwd=os.path.dirname(__file__))
                if result.returncode == 0:
                    print("Mobile signals generated successfully!")
                else:
                    print(f"Error generating signals: {result.stderr}")
            except Exception as e:
                print(f"Error running mobile_signals.py: {e}")
        
        super().do_GET()
    
    def log_message(self, format, *args):
        """Custom log format with better error handling"""
        try:
            print(f"[{time.strftime('%H:%M:%S')}] {format % args}")
        except:
            # Fallback logging if formatting fails
            print(f"[{time.strftime('%H:%M:%S')}] Request processed")
    
    def handle_one_request(self):
        """Handle one HTTP request with better error handling"""
        try:
            super().handle_one_request()
        except ConnectionResetError:
            # Silently handle connection resets (common with mobile devices)
            pass
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Request error: {e}")

def open_browser_delayed():
    """Open browser after a short delay"""
    time.sleep(1)
    webbrowser.open('http://localhost:3001')

def main():
    port = 3001
    
    print(f"Mobile Trading Signals Server")
    print(f"==================================================")
    print(f"Starting server at http://localhost:{port}")
    print(f"Serving directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Check if mobile_signals.html exists
    html_file = Path(__file__).parent / 'mobile_signals.html'
    if not html_file.exists():
        print("Mobile signals HTML not found, will generate on first request...")
    
    try:
        with socketserver.TCPServer(("", port), MobileSignalsHandler) as httpd:
            print(f"Server running at http://localhost:{port}")
            print("Available endpoints:")
            print(f"  http://localhost:{port}/                 - Mobile signals (redirects)")
            print(f"  http://localhost:{port}/mobile_signals    - Mobile signals page")
            print(f"  http://localhost:{port}/mobile_signals.html - Direct HTML file")
            print(f"  http://localhost:{port}/mobile_signals.css  - CSS stylesheet")
            print("\nPress Ctrl+C to stop the server")
            
            # Open browser in a separate thread
            browser_thread = threading.Thread(target=open_browser_delayed)
            browser_thread.daemon = True
            browser_thread.start()
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except OSError as e:
        if "Address already in use" in str(e) or "Only one usage of each socket address" in str(e):
            print(f"Server is already running on port {port}")
            print(f"Access at: http://localhost:{port}")
            # Optionally open browser to existing server
            if len(sys.argv) > 1 and sys.argv[1] == '--open':
                print("Opening browser to existing server...")
                webbrowser.open(f'http://localhost:{port}')
        else:
            print(f"Error starting server: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
