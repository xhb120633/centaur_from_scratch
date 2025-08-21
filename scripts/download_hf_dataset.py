import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download, login


def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset repo (e.g., Psych-101-test)")
    parser.add_argument("repo_id", type=str, help="HF dataset repo id, e.g., ORG/ Psych-101-test")
    parser.add_argument("--local-dir", type=str, default="datasets_downloads",
                        help="Target directory to download into (will be created if missing)")
    parser.add_argument("--revision", type=str, default=None, help="Optional git revision (branch, tag, or commit)")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (optional). If omitted, will use HF_TOKEN env var if present.")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token)
        except Exception:
            # Best-effort; continue without login if fails
            pass

    target_dir = Path(args.local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset repo: {args.repo_id}")
    print(f"Destination: {target_dir}")

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=None,
    )

    print(f"Done. Files at: {local_path}")


if __name__ == "__main__":
    main()


