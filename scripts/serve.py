"""Run the built LAN application and one worker; Ctrl+C stops both."""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cert", help="Trusted HTTPS certificate file")
    parser.add_argument("--key", help="HTTPS private key file")
    parser.add_argument("--no-worker", action="store_true")
    args = parser.parse_args()
    if not (ROOT / "dist" / "client" / "index.html").is_file():
        parser.error("Build the frontend first: npm ci && npm run build")
    if bool(args.cert) != bool(args.key):
        parser.error("Pass both --cert and --key.")
    # Refuse an occupied port before starting any processing work.
    check = socket.socket()
    check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        check.bind((args.host, args.port))
    except OSError:
        parser.error(
            f"Port {args.port} is already in use. Stop the existing server or select --port."
        )
    finally:
        check.close()
    api = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.cert:
        api += [
            "--ssl-certfile",
            str(Path(args.cert).resolve()),
            "--ssl-keyfile",
            str(Path(args.key).resolve()),
        ]
    children = []
    try:
        children.append(subprocess.Popen(api, cwd=ROOT))
        if not args.no_worker:
            children.append(
                subprocess.Popen([sys.executable, "-m", "backend.worker"], cwd=ROOT)
            )
        protocol = "https" if args.cert else "http"
        print(
            f"Open {protocol}://<compute-laptop-LAN-IP>:{args.port} on the capture laptops.",
            flush=True,
        )
        print(
            "Submitted-capture demo. Use trusted HTTPS for browser screen recording; HTTP supports file uploads.",
            flush=True,
        )
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        failed = next(
            (child.returncode for child in children if child.poll() is not None), 0
        )
        if failed:
            raise SystemExit(failed)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
