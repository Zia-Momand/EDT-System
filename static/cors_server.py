from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow requests from any origin
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        super().end_headers()

# Run the HTTP server on port 8000
PORT = 8000
httpd = HTTPServer(("0.0.0.0", PORT), CORSRequestHandler)
print(f"Serving at http://localhost:{PORT}")
httpd.serve_forever()
