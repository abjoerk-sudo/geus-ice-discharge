
#!/usr/bin/env python3
"""Download PROMICE discharge data from GEUS Dataverse.

Export Greenland-wide and individual gate observations
to dashboard.json, including glacier names from gate_meta.csv.
"""

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import xarray as xr


BASE = "https://dataverse.geus.dk/api"

PIDS = [
    "doi:10.22008/promice/data/ice_discharge/d/v02",
    "doi:10.22008/promice/data/ice_discharge/gates/v02",
    "doi:10.22008/promice/data/ice_discharge",
]


def get_manifest(session, pid):
    r = session.get(
        f"{BASE}/datasets/:persistentId/",
        params={"persistentId": pid},
        timeout=60,
    )
    r.raise_for_status()

    obj = r.json()

    if obj.get("status") != "OK":
        raise RuntimeError(f"Dataverse error: {obj}")

    return obj["data"]["latestVersion"]


def download_file(session, file, dst):
    url = f"{BASE}/access/datafile/{file['dataFile']['id']}"

    with session.get(url, timeout=180, stream=True) as r:
        r.raise_for_status()

        with open(dst, "wb") as h:
            for chunk in r.iter_content(chunk_size=1048576):
                if chunk:
                    h.write(chunk)


def pick(ds, names, required=True):
    all_names = list(ds.variables)

    for candidate in names:
        for name in all_names:
            if name.lower() == candidate.lower():
                return name

    for candidate in names:
        for name in all_names:
            if candidate.lower() in name.lower():
                return name

    if required:
        raise ValueError(
            f"Cannot identify {names} in NetCDF: {all_names}"
        )

    return None


def normalized_date(value):
    try:
        return str(pd.Timestamp(value).strftime("%Y-%m-%d"))
    except (ValueError, TypeError):
        return None


def num(value):
    try:
        f = float(value)
        return f if math.isfinite(f) and abs(f) < 1e30 else None
    except (ValueError, TypeError):
        return None


def vector(ds, name, dim, index=None):
    if not name:
        return None

    da = ds[name]

    if index is not None:
        gate_dim, gate_idx = index

        if gate_dim in da.dims:
            da = da.isel({gate_dim: gate_idx})

    if dim not in da.dims:
        raise ValueError(
            f"{name} lacks time dimension {dim}: {da.dims}"
        )

    extras = [d for d in da.dims if d != dim]

    if extras:
        if any(da.sizes[d] != 1 for d in extras):
            raise ValueError(
                f"Ambiguous dimensions {extras} in {name}"
            )

        da = da.squeeze(extras)

    return da.transpose(dim).values


def records(ds, index=None):
    tname = pick(ds, ["time", "date", "timestamp"])
    tcoord = ds[tname]

    if len(tcoord.dims) != 1:
        raise ValueError(
            "Time coordinate must be one-dimensional"
        )

    tdim = tcoord.dims[0]

    d = pick(
        ds, ["D", "discharge", "ice_discharge", "flux"]
    )
    e = pick(
        ds, ["err", "error", "uncertainty"], False
    )
    c = pick(
        ds, ["coverage", "velocity_coverage"], False
    )

    vv = vector(ds, d, tdim, index)
    ee = vector(ds, e, tdim, index)
    cc = vector(ds, c, tdim, index)

    out = []

    for i, t in enumerate(tcoord.values):
        date = normalized_date(t)

        if not date:
            continue

        out.append({
            "date": date,
            "d": num(vv[i]),
            "err": num(ee[i]) if ee is not None else None,
            "coverage": num(cc[i]) if cc is not None else None,
        })

    return sorted(out, key=lambda p: p["date"])


def scalar_gate(ds, field, gate_dim, index):
    if field is None:
        return None

    x = ds[field]

    if gate_dim not in x.dims:
        return None

    v = x.isel({gate_dim: index}).values

    if np.ndim(v) > 0:
        if np.size(v) == 1:
            v = np.ravel(v)[0]

        elif v.dtype.kind in ("S", "U"):
            v = "".join(
                y.decode(errors="ignore")
                if isinstance(y, bytes)
                else str(y)
                for y in np.ravel(v)
            )

        else:
            return None

    if isinstance(v, bytes):
        return v.decode(errors="ignore").strip()

    if isinstance(v, np.generic):
        v = v.item()

    return v


def gate_records(ds):
    dname = pick(
        ds, ["D", "discharge", "ice_discharge", "flux"]
    )
    tname = pick(
        ds, ["time", "date", "timestamp"]
    )

    tdim = ds[tname].dims[0]

    dims = [
        x for x in ds[dname].dims
        if x != tdim
    ]

    if len(dims) != 1:
        raise ValueError(
            f"Expected gate x time data, got {ds[dname].dims}"
        )

    gdim = dims[0]

    latkey = pick(
        ds, ["lat", "latitude", "mean_lat"], False
    )
    lonkey = pick(
        ds, ["lon", "longitude", "mean_lon"], False
    )
    idkey = pick(
        ds, ["gate_id", "gate", "id"], False
    )
    namekey = pick(
        ds,
        [
            "name_Bjork",
            "name_Bjoerk",
            "name_Greenlandic",
            "glacier_name",
            "name",
            "bjork",
        ],
        False,
    )

    if not latkey or not lonkey:
        raise ValueError(
            "No latitude/longitude in gate NetCDF"
        )

    gates = []
    series = {}

    for i in range(ds.sizes[gdim]):
        lat = num(
            scalar_gate(ds, latkey, gdim, i)
        )
        lon = num(
            scalar_gate(ds, lonkey, gdim, i)
        )

        if lat is None or lon is None:
            continue

        if not (55 <= lat <= 85 and -75 <= lon <= -10):
            continue

        gid = (
            scalar_gate(ds, idkey, gdim, i)
            if idkey else i
        )

        gid = str(gid).strip()

        if not gid or gid.lower() == "nan":
            gid = str(i)

        name = scalar_gate(
            ds, namekey, gdim, i
        )

        name = (
            str(name).strip()
            if name is not None else ""
        )

        if not name or name.lower() in (
            "nan", "none", "null"
        ):
            name = "Gate " + gid

        gates.append({
            "id": gid,
            "name": name,
            "lat": lat,
            "lon": lon,
        })

        series[gid] = records(
            ds, (gdim, i)
        )

    if not gates:
        raise ValueError(
            "No valid Greenland gates found"
        )

    return gates, series


def normalize_gate_id(raw):
    """Normalize gate identifiers."""

    if pd.isna(raw):
        return ""

    value = str(raw).strip()

    value = re.sub(
        r"^gate\s*#?\s*",
        "",
        value,
        flags=re.I,
    )

    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".")[0]

    return value


def good_name(value):
    """Format GEUS names without inventing glacier names."""

    name = str(value or "").strip()

    if name.lower() in (
        "", "nan", "none", "null",
        "unknown", "n/a", "-"
    ):
        return None

    if re.fullmatch(
        r"(?:unnamed glacier|gate)\s*#?\s*\d+",
        name,
        re.I,
    ):
        return None

    # Example:
    # JAKOBSHAVN_ISBRAE -> Jakobshavn Isbrae

    # Descriptive identifiers such as SE_NONAME2
    # are retained from the original metadata.

    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)

    return name.strip().title()


def read_gate_names(path):
    """Match actual GEUS metadata names to gate IDs."""

    meta = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        comment="#",
    )

    cols = {
        re.sub(r"[^a-z0-9]", "", c.lower()): c
        for c in meta.columns
    }

    def choose(candidates):
        for value in candidates:
            key = re.sub(
                r"[^a-z0-9]",
                "",
                value.lower(),
            )

            if key in cols:
                return cols[key]

        return None

    idcol = choose([
        "gate",
        "gate_id",
        "gateid",
        "id",
        "gate_number",
        "gate_no",
    ])

    # Actual columns from the uploaded GEUS file:
    # gate, Bjork_2015, Mouginot_2019
    #
    # Bjork_2015 is empty in the supplied file.
    # Mouginot_2019 contains the glacier identifiers.

    preferred = [
        "Bjork_2015",
        "Mouginot_2019",
        "name_Bjork",
        "name_Bjoerk",
        "name_Greenlandic",
        "official_name",
        "glacier_name",
        "name",
        "name_Mouginot",
    ]

    namecols = [
        choose([c])
        for c in preferred
    ]

    namecols = [
        c for c in namecols if c
    ]

    if not namecols:
        namecols = [
            c for c in meta.columns
            if re.search(
                r"name|bjork|bjoerk|glacier",
                c,
                re.I,
            )
        ]

    print("Metadata columns:", list(meta.columns))
    print("Gate ID column:", idcol)
    print("Name columns:", namecols)

    if not idcol or not namecols:
        raise ValueError(
            "Cannot identify gate IDs or glacier names"
        )

    mapping = {}

    for _, row in meta.iterrows():
        gid = normalize_gate_id(
            row[idcol]
        )

        name = next(
            (
                n for col in namecols
                if (n := good_name(row[col]))
            ),
            None,
        )

        if gid and name:
            mapping[gid] = name

    print(
        "Read",
        len(mapping),
        "named gates from gate_meta.csv"
    )

    print(
        "Example IDs:",
        list(mapping)[:8]
    )

    return mapping


def run(source_dir=None):

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        resolved = {}
        version = []

        if source_dir:
            source_dir = Path(source_dir)

            for n in ["GIS.nc", "gate.nc"]:
                p = source_dir / n

                if not p.is_file():
                    raise FileNotFoundError(p)

                resolved[n] = p

            meta_file = source_dir / "gate_meta.csv"

            if meta_file.is_file():
                resolved["gate_meta.csv"] = meta_file

            version = ["local files"]

        else:
            s = requests.Session()

            s.headers.update({
                "User-Agent":
                "GreenlandDischargeDashboard/0.2"
            })

            manifests = []

            for pid in PIDS:
                try:
                    m = get_manifest(s, pid)

                    manifests.append(m)

                    version.append(
                        f"{pid}: "
                        f"{m.get('versionNumber', '?')}."
                        f"{m.get('versionMinorNumber', '?')}"
                    )

                except requests.RequestException as exc:
                    print(
                        "Manifest unavailable:",
                        pid,
                        exc,
                    )

            if not manifests:
                raise RuntimeError(
                    "No GEUS Dataverse manifest accessible"
                )

            for manifest in manifests:
                for f in manifest.get("files", []):
                    name = f["dataFile"]["filename"]

                    if (
                        name in (
                            "GIS.nc",
                            "gate.nc",
                            "gate_meta.csv",
                        )
                        and name not in resolved
                    ):
                        dest = td / name

                        print("Downloading", name)

                        download_file(
                            s, f, dest
                        )

                        resolved[name] = dest

            if not all(
                n in resolved
                for n in ("GIS.nc", "gate.nc")
            ):
                raise RuntimeError(
                    "Required NetCDF files missing: "
                    f"{list(resolved)}"
                )

        # Local fallback for metadata if the remote
        # GEUS dataset does not include gate_meta.csv.

        if "gate_meta.csv" not in resolved:
            local_meta = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "gate_meta.csv"
            )

            if local_meta.is_file():
                resolved["gate_meta.csv"] = local_meta

                print(
                    "Using local data/gate_meta.csv"
                )

        with (
            xr.open_dataset(
                resolved["GIS.nc"],
                decode_times=True,
            ) as gis,
            xr.open_dataset(
                resolved["gate.nc"],
                decode_times=True,
            ) as gate,
        ):
            greenland = records(gis)
            gates, series = gate_records(gate)

        # Apply names from gate_meta.csv

        if "gate_meta.csv" in resolved:
            names = read_gate_names(
                resolved["gate_meta.csv"]
            )

            matched = 0

            for gate in gates:
                name = names.get(
                    normalize_gate_id(gate["id"])
                )

                if name:
                    gate["name"] = name
                    matched += 1

            print(
                f"Matched metadata names to "
                f"{matched} of {len(gates)} gates"
            )

            if matched == 0:
                print(
                    "Sample NetCDF IDs:",
                    [g["id"] for g in gates[:10]]
                )

                print(
                    "Sample CSV IDs:",
                    list(names)[:10]
                )

                raise RuntimeError(
                    "No gate names matched; "
                    "refusing to publish unnamed gates"
                )

        else:
            print(
                "WARNING: gate_meta.csv unavailable"
            )

        if not greenland:
            raise ValueError(
                "Empty Greenland time series"
            )

        named = sum(
            good_name(g["name"]) is not None
            for g in gates
        )

        print(
            f"Gates with recognizable labels: "
            f"{named} of {len(gates)}"
        )

        payload = {
            "version": " | ".join(version),
            "greenland": greenland,
            "gates": gates,
            "series": series,
            "source": "PROMICE / GEUS Dataverse",
            "units": "Gt/yr",
        }

        output = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "dashboard.json"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = output.with_suffix(".json.tmp")

        tmp.write_text(
            json.dumps(
                payload,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        os.replace(tmp, output)

        print(
            f"Wrote {output} with "
            f"{len(gates)} gates and "
            f"{len(greenland)} observations"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source-dir",
        help="Local NetCDF directory for testing",
    )

    args = ap.parse_args()

    run(args.source_dir)
