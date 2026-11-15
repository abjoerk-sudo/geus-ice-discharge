# Greenland Ice Discharge dashboard — prototype

A GitHub Pages front end for PROMICE/GEUS solid-ice discharge. Defaults to Greenland-wide view, with **fixed-size clickable glacier points** and **monthly and annual discharge rates**. All data must come from GEUS; the project deliberately does not provide fabricated observations.

## Setup

1. Create a **new GitHub repository** and add the entire contents of this directory (including `.github/workflows/update.yml`).
2. Under **Settings → Actions → General**, allow GitHub Actions to have **read and write** repository permissions; the updater needs permission to commit `data/dashboard.json`.
3. Run **Actions → Update PROMICE discharge → Run workflow**. It downloads public NetCDF files from GEUS Dataverse and extracts the Greenland-wide and per-gate series.
4. Under **Settings → Pages**, deploy from the **main** branch, root `/`. You can also choose GitHub Actions-based Pages deployment if preferred.
5. Open the published site. The first run must complete and commit `data/dashboard.json` before map points and measured graphs appear.

GitHub Actions checks once a day. Since the official dataset is updated monthly, this means new versions should be picked up without manual intervention. Historical revisions are handled by full replacement, not appended. The workflow **fails without replacing good previously published data** if GEUS changes file names or NetCDF schema.

## Velocity background

A satellite basemap works immediately. There is also a **velocity** map-layer option, which expects public web Mercator raster tiles in `velocity/{z}/{x}/{y}.png`. Tiles **are not bundled** because preparing an ice-velocity mosaic requires selecting a scientific raster, reprojecting it to EPSG:3857, rendering a color scale and generating tile pyramids. Do not select the velocity layer until you add those tiles (and update the attribution). A Greenland-wide scientific velocity layer can be prepared from MEaSUREs or PROMICE products, subject to checking their licensing and attribution.

## Important scientific choices and limitations

- Monthly figures are **rates** in **Gt/yr**, not monthly discharged mass in Gt.
- Annual figures are **day-weighted means** and are displayed only if **all 12 months** have finite data and no explicitly zero source coverage. This is a preliminary coverage policy and should be refined to use source-provided coverage thresholds after inspecting the downloaded files.
- Annual uncertainty shown in the graph is a simplified propagated estimate assuming independent monthly errors. This may **understate correlated uncertainties**. Review before using the annual envelope scientifically. Monthly errors remain sourced from GEUS.
- Verify that each NetCDF variable has the expected dimensionality and that the gate location/name schema is correct. If not, the Python updater raises an error rather than silently inventing coordinates.
- The graph uses latest available **valid month**; the annual card uses the latest complete calendar year.
- The source code authors advise checking open issues labeled `WARNING` before scientific use: https://github.com/GEUS-Glaciology-and-Climate/ice_discharge/issues?q=label%3AWARNING

## Data links

- Source dataset: https://doi.org/10.22008/promice/data/ice_discharge/d/v02
- Gates dataset: https://doi.org/10.22008/promice/data/ice_discharge/gates/v02
- Source code and output metadata: https://github.com/GEUS-Glaciology-and-Climate/ice_discharge
- Paper: https://doi.org/10.5194/essd-12-1367-2020

## Local testing

```
python -m pip install -r requirements.txt
python scripts/update_data.py
python -m http.server 8000
```

Visit http://localhost:8000 (do not open index.html directly via file:// because the site fetches JSON).

You can test on downloaded official files with `python scripts/update_data.py --source-dir /path/to/downloaded/nc/files` (must contain `GIS.nc` and `gate.nc`).
