"""Tiny built-in web dashboard (stdlib only): http://<jetson-ip>:8080"""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Stock Levels</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#222}
header{background:#1f2937;color:#fff;padding:12px 16px;font-size:18px}
main{padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px #0002}
.tray{margin:10px 0}.row{display:flex;justify-content:space-between;font-size:15px}
.bar{height:14px;background:#e5e7eb;border-radius:7px;overflow:hidden;margin-top:4px}
.fill{height:100%;transition:width .5s}
.OK{background:#4caf50}.LOW{background:#ff9800}.EMPTY{background:#f44336}.null{background:#9ca3af}
img{width:100%;border-radius:8px}small{color:#666}
</style></head><body><header>Stock &amp; Display Levels <small id="ts" style="color:#ccc"></small></header>
<main><div class="card" id="trays"></div><div class="card"><img id="img" src="latest.jpg"></div></main>
<script>
async function tick(){
  try{
    const d = await (await fetch('api/levels')).json();
    document.getElementById('ts').textContent = d.updated ? ' updated ' + d.updated : '';
    document.getElementById('trays').innerHTML = d.trays.map(t => {
      const p = t.fill == null ? 0 : Math.round(t.fill * 100);
      return `<div class="tray"><div class="row"><b>${t.label}</b><span>${t.fill == null ? '--' : p + '%'} ${t.state || ''}${t.occluded ? ' (blocked)' : ''}</span></div>
        <div class="bar"><div class="fill ${t.state}" style="width:${p}%"></div></div></div>`;
    }).join('');
    document.getElementById('img').src = 'latest.jpg?t=' + Date.now();
  }catch(e){}
}
tick(); setInterval(tick, 5000);
</script></body></html>"""


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.levels = {"updated": None, "trays": []}
        self.jpeg = b""

    def set(self, levels, jpeg):
        with self.lock:
            self.levels, self.jpeg = levels, jpeg


def start_dashboard(state, host="0.0.0.0", port=8080):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            with state.lock:
                if path == "/":
                    body, ctype = PAGE.encode(), "text/html; charset=utf-8"
                elif path == "/api/levels":
                    body, ctype = json.dumps(state.levels).encode(), "application/json"
                elif path == "/latest.jpg" and state.jpeg:
                    body, ctype = state.jpeg, "image/jpeg"
                else:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Dashboard on http://%s:%d", host, port)
    return server
