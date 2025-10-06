#!/bin/bash
set -e

echo "[✓] Converting simulations.ipynb to Markdown..."
# Ensure the .ipynb file exists in docs/
if [ -f "docs/simulations.ipynb" ]; then
  # --to markdown: converts to Markdown
  # --output-dir docs/: saves the output .md file back into the docs/ directory
  # --output simulations: names the output file simulations.md
  pixi run jupyter nbconvert --to markdown --execute docs/simulations.ipynb --output-dir docs/ --output simulations --allow-errors
  echo "[✓] Successfully converted simulations.ipynb to docs/simulations.md"
else
  echo "[!] Warning: docs/simulations.ipynb not found. Skipping notebook conversion."
fi

#echo "[✓] Copying plots to static/evals/..."
#mkdir -p static/evals
#cp ../WindGym-Zoo/results/*.png static/evals/ || echo "No .png plots to copy"
#cp ../WindGym-Zoo/results/*.csv static/evals/ || echo "No .csv files to copy"

