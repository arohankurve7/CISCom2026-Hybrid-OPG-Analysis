#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -f ".venv/bin/activate" ]; then
    echo "Error: Project virtual environment not found."
    echo "Create the environment and install requirements first."
    read -r -p "Press Enter to close..."
    exit 1
fi

source ".venv/bin/activate"
python -m dashboard.app
