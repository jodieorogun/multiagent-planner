import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from core import AgentManager, LLMError, OllamaClient, SchemaError
from tools.workload_model import WorkloadModelError, neural_predict


FORM = """<!doctype html>
<html><head><meta charset="utf-8"><title>Multi-Agent Planner</title>
<style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px}textarea{width:100%;height:120px}button{padding:10px 18px;margin-top:10px}pre{white-space:pre-wrap;background:#f4f4f4;padding:16px;border-radius:8px}</style>
</head><body><h1>Multi-Agent Planner</h1>
<p>Describe your commitments and the local agent pipeline will build a weekly plan.</p>
<form method="post"><textarea name="request" placeholder="Match on Wednesday, 3 study sessions, gym 3 times...">{request}</textarea><br><button type="submit">Plan my week</button></form>{result}</body></html>"""


def render_page(request="", result=""):
    return FORM.replace("{request}", escape(request)).replace("{result}", result)


class PlannerHandler(BaseHTTPRequestHandler):
    manager = None

    def do_GET(self):
        self._send(render_page())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        values = parse_qs(self.rfile.read(length).decode("utf-8"))
        request = values.get("request", [""])[0].strip()
        try:
            plan = self.manager.process(request)
            result = f"<h2>Result</h2><pre>{escape(plan.render())}</pre>"
        except (LLMError, SchemaError, WorkloadModelError) as exc:
            result = f"<p><strong>Error:</strong> {escape(str(exc))}</p>"
        self._send(render_page(request, result))

    def _send(self, body):
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the planner browser demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args(argv)

    client = OllamaClient(model=args.model, timeout_seconds=args.timeout)
    PlannerHandler.manager = AgentManager(client.complete)
    server = ThreadingHTTPServer((args.host, args.port), PlannerHandler)
    print(f"Planner UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
