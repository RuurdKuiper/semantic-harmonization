"""Serve exported browser packs for local development, without loading vectors.

Production can use any static object host configured with the same CORS headers.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="dist/browser-assets")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--origin", default="http://127.0.0.1:5173")
    args = parser.parse_args()

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", args.origin)
            self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-cache" if self.path.endswith("manifest.json") else "public, max-age=31536000, immutable")
            super().end_headers()

    print(f"Terminology manifest: http://127.0.0.1:{args.port}/manifest.json", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=args.directory)).serve_forever()


if __name__ == "__main__":
    main()
