#!/usr/bin/env python3
"""Check whether required external CFD tools are available."""

from __future__ import annotations

import shutil
import sys


REQUIRED_TOOLS = {
    "SU2_CFD": "SU2 solver executable",
    "gmsh": "Gmsh mesh generator",
}


def main() -> int:
    missing: list[str] = []

    print("Checking external tools for flow_sketch...\n")

    for executable, description in REQUIRED_TOOLS.items():
        path = shutil.which(executable)
        if path is None:
            missing.append(executable)
            print(f"[missing] {executable}: {description} was not found on PATH.")
        else:
            print(f"[ok]      {executable}: {path}")

    if missing:
        print("\nRequired tool check failed.")
        print("Missing executables:")
        for executable in missing:
            print(f"  - {executable}")
        print("\nInstall the missing tools and make sure they are available on PATH.")
        return 1

    print("\nAll required external tools were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
