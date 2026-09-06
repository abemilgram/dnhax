"""OS-level singleton lock, independent of heartbeat freshness."""

import os
from . import store


def acquire():
    store.ROOT.mkdir(parents=True, exist_ok=True)
    stream = (store.ROOT / "worker.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError("A worker already owns this workspace.") from exc
    return stream
