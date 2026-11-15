from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os, webbrowser, threading

ROOT = Path(__file__).resolve().parent
HOST, PORT = '127.0.0.1', 8765

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)
    def log_message(self, fmt, *args):
        print('[AERIS portal] ' + fmt % args)

if __name__ == '__main__':
    url = f'http://{HOST}:{PORT}'
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print('\nAERIS FARMER PORTAL')
    print('Local address:', url)
    print('Only this computer can access this portal.')
    print('Press Ctrl+C to stop.\n')
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nClosing AERIS Farmer Portal...')
    finally:
        server.server_close()
