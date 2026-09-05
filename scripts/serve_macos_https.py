"""Create LAN TLS files and run the macOS MPS app over trusted HTTPS."""

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TLS_DIR = ROOT / "work" / "certs"
PHONE_CERT_DIR = ROOT / "work" / "phone-cert"


def lan_address() -> str:
    for interface in ("en0", "en1"):
        result = subprocess.run(
            ["ipconfig", "getifaddr", interface],
            capture_output=True,
            text=True,
        )
        address = result.stdout.strip()
        if result.returncode == 0 and address:
            return address
    raise RuntimeError("No Wi-Fi LAN address found on en0 or en1.")


def create_certificates(address: str) -> tuple[Path, Path, Path]:
    if not shutil.which("mkcert"):
        raise RuntimeError("mkcert is required. Install it with: brew install mkcert")
    if not shutil.which("openssl"):
        raise RuntimeError("openssl is required to prepare the phone CA certificate.")

    TLS_DIR.mkdir(parents=True, exist_ok=True)
    PHONE_CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert = TLS_DIR / "dnhax-cert.pem"
    key = TLS_DIR / "dnhax-key.pem"
    phone_ca = PHONE_CERT_DIR / "dnhax-rootCA.cer"
    hostname = socket.gethostname()

    subprocess.run(
        [
            "mkcert",
            "-cert-file",
            str(cert),
            "-key-file",
            str(key),
            address,
            "localhost",
            "127.0.0.1",
            hostname,
        ],
        check=True,
    )
    key.chmod(0o600)
    caroot = subprocess.run(
        ["mkcert", "-CAROOT"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        [
            "openssl",
            "x509",
            "-in",
            str(Path(caroot) / "rootCA.pem"),
            "-outform",
            "der",
            "-out",
            str(phone_ca),
        ],
        check=True,
    )
    return cert, key, phone_ca


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--certificate-port", type=int, default=8001)
    parser.add_argument("--max-frames", type=int, default=4)
    args = parser.parse_args()

    address = lan_address()
    cert, key, phone_ca = create_certificates(address)
    env = os.environ.copy()
    env.setdefault("SIMV1_DEVICE", "mps")
    env.setdefault("SIMV1_MAX_FRAMES", str(args.max_frames))

    certificate_server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(args.certificate_port),
            "--bind",
            "0.0.0.0",
            "--directory",
            str(PHONE_CERT_DIR),
        ],
        cwd=ROOT,
    )
    print(
        f"Phone CA certificate: http://{address}:{args.certificate_port}/{phone_ca.name}",
        flush=True,
    )
    print(f"Mac app: https://127.0.0.1:{args.port}", flush=True)
    print(f"Phone app: https://{address}:{args.port}", flush=True)
    if not env.get("VGGT_OMEGA_CHECKPOINT"):
        print(
            "VGGT_OMEGA_CHECKPOINT is unset; sample scenes work, but model reconstruction will not.",
            flush=True,
        )

    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "serve.py"),
                "--port",
                str(args.port),
                "--cert",
                str(cert),
                "--key",
                str(key),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    finally:
        certificate_server.terminate()
        try:
            certificate_server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            certificate_server.kill()
            certificate_server.wait()


if __name__ == "__main__":
    main()
