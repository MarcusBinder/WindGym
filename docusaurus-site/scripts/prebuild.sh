#!/bin/bash
set -e

echo "[✓] Cloning WindGym-Zoo..."
if [ ! -d "../WindGym-Zoo" ]; then
  git clone https://github.com/kilojoules/WindGym-Zoo ../WindGym-Zoo
fi

echo "[✓] Generating config..."
python ../WindGym-Zoo/scripts/generate_config.py \
  --template ../WindGym-Zoo/configs/N_turbines_most_sensors.yaml \
  --output ../WindGym-Zoo/configs/2_turbines_most_sensors.yaml \
  --nx 2 

echo "[✓] Running leaderboard evaluation..."
python ../WindGym-Zoo/scripts/eval_leaderboard.py \
  --config-name 2_turbines_most_sensors \
  --configs-dir ../WindGym-Zoo/configs \
  --agents-dir ../WindGym-Zoo/agents \
  --out-dir ../WindGym-Zoo/results

echo "[✓] Converting simulations.ipynb to Markdown..."
# Ensure the .ipynb file exists in docs/
if [ -f "docs/simulations.ipynb" ]; then
  # --to markdown: converts to Markdown
  # --output-dir docs/: saves the output .md file back into the docs/ directory
  # --output simulations: names the output file simulations.md
  jupyter nbconvert --to markdown --execute docs/simulations.ipynb --output-dir docs/ --output simulations --allow-errors
  echo "[✓] Successfully converted simulations.ipynb to docs/simulations.md"
else
  echo "[!] Warning: docs/simulations.ipynb not found. Skipping notebook conversion."
fi

echo "[✓] Copying plots to static/evals/..."
mkdir -p static/evals
cp ../WindGym-Zoo/results/*.png static/evals/ || echo "No .png plots to copy"
cp ../WindGym-Zoo/results/*.csv static/evals/ || echo "No .csv files to copy"

