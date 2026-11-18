#!/usr/bin/env bash
# Container entrypoint (see Dockerfile).
#
# Free hosts wipe local files on every restart, so the demo personas are
# re-seeded at startup. Seeding at run time rather than build time matters:
# the personas' history is dated relative to today, so a week-old image
# would otherwise show stale ACWR zones.
set -euo pipefail

python -m scripts.seed_demo_data

exec streamlit run app.py \
  --server.port="${PORT:-7860}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
