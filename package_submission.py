"""Create a competition zip with Unix executable permissions on run.sh."""

import argparse
from pathlib import Path
import zipfile

REQUIRED = ("run.sh", "main.py", "agent.py", "model.py", "metadata.json", "weights.safetensors")
MAX_ZIP = 50 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="submission")
    parser.add_argument("--output", default="averagejoe_submission.zip")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"missing submission files: {missing}")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in REQUIRED:
            path = source / name
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            mode = 0o100755 if name == "run.sh" else 0o100644
            info.external_attr = mode << 16
            zf.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    if output.stat().st_size > MAX_ZIP:
        output.unlink()
        raise SystemExit("submission zip exceeds the 50 MB limit")
    print(f"Created {output} ({output.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
