# 🌊 Plastics Hunter Dashboard

Interactive Streamlit dashboard that visualises the optimization model from
`../optimization_model/` in a web UI.

**Live demo:** https://plastic-hunter-dashboard.streamlit.app

---

## Files

```
dashboard/
├── app.py              ← Streamlit UI
├── backend.py          ← Wrapper that downloads the model bundle from
│                         the team Google Drive and exposes query functions
├── requirements.txt    ← Dashboard-only Python deps
├── .streamlit/
│   └── config.toml     ← Theme + server settings
└── data/               ← (auto-created) bundle cache — gitignored
```

## Run locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

First run downloads ~190 MB from Google Drive into `data/`. Subsequent runs are
instant.

## Update the backend model (teammate A)

1. Export a new `optimization_dashboard_bundle.pkl` from the research notebook
   in `../optimization_model/`. The bundle must contain:
   - `meta` (dict: BBOX, PORT_LON, PORT_LAT, n_prod_snapshots, ...)
   - `query` (callable → routes, plastic, fuel, forecast)
   - `find_best` (callable → list of dicts)
2. In the team Google Drive folder, right-click the existing bundle →
   *Manage versions* → *Upload new version*. The file ID stays the same.
3. On https://share.streamlit.io, click *Manage app → Reboot* to force the
   cloud deployment to re-download the new bundle.

## Update the UI (teammate B)

```bash
git pull                # sync latest
# edit app.py ...
streamlit run app.py    # test locally
git add dashboard/app.py
git commit -m "describe the change"
git push                # Streamlit Cloud auto-redeploys in ~1-2 min
```

See the repo root `README.md` for the broader project description.
