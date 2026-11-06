"""
X2 instrumented receiver (public context only): verifies HMAC + key id and aggregates every batch
homomorphically (edge.crypto.alert_encryption.HomomorphicAggregator), recording the wall-clock
receipt time of each alert id. GET /dump returns receipts and counters; POST /reset clears them.

  python x2_receiver.py --repo <repo> --public-context <public.ctx> --port 9300   ($CKKS_HMAC_SECRET)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--public-context", required=True)
    ap.add_argument("--port", type=int, default=9300)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from edge.crypto.alert_encryption import EncryptedBatch, HomomorphicAggregator
    pub = open(a.public_context, "rb").read()
    lock = threading.Lock()
    state = {"agg": HomomorphicAggregator(pub), "receipts": {}, "batches": 0, "alerts": 0, "bytes": 0,
             "rejected": 0, "aggregate_ms": 0.0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            t = time.time()
            if self.path == "/alerts":
                try:
                    b = EncryptedBatch.from_json(body)
                    with lock:
                        t0 = time.perf_counter(); state["agg"].add(b); state["aggregate_ms"] += (time.perf_counter() - t0) * 1e3
                        for i in b.alert_ids:
                            state["receipts"][i] = t
                        state["batches"] += 1; state["alerts"] += b.n_alerts; state["bytes"] += len(body)
                    self._send(200, {"ok": True})
                except Exception as e:  # noqa: BLE001
                    with lock:
                        state["rejected"] += 1
                    self._send(400, {"ok": False, "error": str(e)[:200]})
            elif self.path == "/reset":
                with lock:
                    state.update({"agg": HomomorphicAggregator(pub), "receipts": {}, "batches": 0, "alerts": 0,
                                  "bytes": 0, "rejected": 0, "aggregate_ms": 0.0})
                self._send(200, {"ok": True})
            else:
                self._send(404, {})

        def do_GET(self):
            if self.path == "/dump":
                with lock:
                    out = {k: v for k, v in state.items() if k != "agg"}
                    out["aggregate_n_alerts"] = state["agg"].n_alerts
                self._send(200, out)
            else:
                self._send(404, {})

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"x2 receiver on :{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
