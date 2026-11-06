"""
Diagnostic B: the deployed app (edge.api.gateway:app) wrapped in a pure-ASGI timer that adds
  X-Server-Ms   time from the ASGI call to the response start (handler + framework, excludes the network)
  X-Worker-Pid  the worker process that served the request
  uvicorn diag_app:create --factory
"""
from __future__ import annotations

import os
import time


def create():
    from edge.api.gateway import app as inner
    pid = str(os.getpid()).encode()

    async def timed(scope, receive, send):
        if scope["type"] != "http":
            return await inner(scope, receive, send)
        t0 = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                ms = f"{(time.perf_counter() - t0) * 1e3:.3f}".encode()
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [(b"x-server-ms", ms), (b"x-worker-pid", pid)]
            await send(message)

        await inner(scope, receive, send_wrapper)

    return timed
