import sys
from datetime import datetime
from pathlib import Path

HEADER = """# =============================================================================
# File: {filename}
# Date: {date}
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
"""


def add_header_to_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if "Copyright (c) 2024 Goutam Malakar" in content:
        return  # Already has header
    today = datetime.now().strftime("%Y-%m-%d")
    header = HEADER.format(filename=Path(filepath).name, date=today)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n" + content)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg.endswith(".py"):
            try:
                add_header_to_file(arg)
            except Exception as e:
                print(f"Error processing {arg}: {e}", file=sys.stderr)
                sys.exit(1)
