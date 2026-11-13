from __future__ import annotations

import argparse
from pathlib import Path

from export_tex import DEFAULT_OUTPUT, export_tex


def main() -> None:
    """Parse the output directory and regenerate the four-panel TeX figure.

    Args:
        None.

    Returns:
        None: The generated figure path is printed to standard output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory containing summary_results.csv (default: outputs/)",
    )
    args = parser.parse_args()
    print(export_tex(args.output))


if __name__ == "__main__":
    main()
