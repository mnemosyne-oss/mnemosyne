"""Explicit embedding readiness check, separate from protocol liveness.

The child bounds downloads/model initialization and suppresses backend diagnostics
that may contain deployment details. It never accesses memory content.
"""
import subprocess
import sys


_INFER = '''
from mnemosyne.core.embeddings import embed_query
vector = embed_query("Mnemosyne embedding readiness probe")
raise SystemExit(0 if vector is not None and len(vector) > 0 else 1)
'''


def main() -> int:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _INFER],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 0:
            print("Embedding inference ready.")
            return 0
    except (OSError, subprocess.TimeoutExpired):
        pass
    print(
        "Embedding inference not ready. Check model download access and cache "
        "permissions; initialization is limited to 120 seconds. Fix the cause "
        "and retry. Ordinary memory calls may still use keyword fallback.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
