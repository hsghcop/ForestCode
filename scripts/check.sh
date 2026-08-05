#!/usr/bin/env bash
# ForestCode local quality-check pipeline for macOS, Linux, and WSL.
# Flow: run all tests -> only on success clean dist/ and rebuild with uv.
#
# Usage:
#   ./scripts/check.sh               # tests + build
#   ./scripts/check.sh --skip-build  # tests only
#
# Keep the test command aligned with the development sections in README.md and README_EN.md.
# Git commits and package publishing remain manual.

set -euo pipefail

skip_build=false

for argument in "$@"; do
    case "$argument" in
        --skip-build)
            skip_build=true
            ;;
        -h|--help)
            printf 'Usage: %s [--skip-build]\n' "$0"
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$argument" >&2
            printf 'Usage: %s [--skip-build]\n' "$0" >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

printf '\n==> Running tests...\n'
if uv run python -m unittest discover -s tests -p "test_*.py"; then
    printf '\nAll tests passed.\n'
else
    test_exit=$?
    printf '\nTests failed (exit %s); skipping build.\n' "$test_exit" >&2
    exit "$test_exit"
fi

if [[ "$skip_build" == true ]]; then
    printf 'Build skipped (--skip-build).\n'
    exit 0
fi

printf '\n==> Cleaning dist/ and rebuilding...\n'
rm -rf -- "$repo_root/dist"

if uv build; then
    printf '\nDone: tests passed and packages were rebuilt. dist/ contains:\n'
else
    build_exit=$?
    printf '\nBuild failed (exit %s).\n' "$build_exit" >&2
    exit "$build_exit"
fi

for artifact in "$repo_root"/dist/*; do
    [[ -e "$artifact" ]] || continue
    printf '  %s\n' "$(basename -- "$artifact")"
done
