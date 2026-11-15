#!/usr/bin/env python3
"""Download public PROMICE discharge NetCDF from GEUS Dataverse, export browser JSON.

Source schemas can change; fail loudly rather than publishing mismatched data.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
import requests
import numpy as np
import pandas as pd
import xarray as xr

BASE = 'https://dataverse.geus.dk/api'
PIDS = ['doi:10.22008/promice/data/ice_discharge/d/v02',
        'doi:10.22008/promice/data/ice_discharge/gates/v02',
        'doi:10.22008/promice/data/ice_discharge']

def get_manifest(session, pid):
    r = session.get(f'{BASE}/datasets/:persistentId/', params={'persistentId':pid}, timeout=60)
    r.raise_for_status()
    obj = r.json()
    if obj.get('status') != 'OK': raise RuntimeError(f'Dataverse error: {obj}')
    return obj['data']['latestVersion']

def download_file(session, file, dst):
    url = f"{BASE}/access/datafile/{file['dataFile']['id']}"
    with session.get(url, timeout=180, stream=True) as r:
        r.raise_for_status()
        with open(dst,'wb') as h:
            for chunk in r.iter_content(chunk_size=1048576):
                if chunk:h.write(chunk)

def pick(ds, names, required=True):
    all_names=list(ds.variables)
    for candidate in names:
        for name in all_names:
            if name.lower()==candidate.lower():return name
    for candidate in names:
        for name in all_names:
            if candidate.lower() in name.lower():return name
    if required: raise ValueError(f'Cannot identify {names} in NetCDF variables: {all_names}')
    return None

def normalized_date(value):
    try:return str(pd.Timestamp(value).strftime('%Y-%m-%d'))
    except (ValueError,TypeError):return None

def num(value):
    try:
        f=float(value)
        return f if math.isfinite(f) and abs(f)<1e30 else None
    except (ValueError,TypeError):return None

def vector(ds, name, dim, index=None):
    if not name:return None
    da=ds[name]
    if index is not None:
        gate_dim,gate_idx=index
        if gate_dim in da.dims:da=da.isel({gate_dim:gate_idx})
    if dim not in da.dims:raise ValueError(f'{name} lacks time dimension {dim}: {da.dims}')
    extras=[d for d in da.dims if d!=dim]
    if extras:
        if any(da.sizes[d]!=1 for d in extras):raise ValueError(f'Ambiguous dimensions {extras} in {name}')
        da=da.squeeze(extras)
    return da.transpose(dim).values

def records(ds,index=None):
    tname=pick(ds,['time','date','timestamp']);tcoord=ds[tname]
    if len(tcoord.dims)!=1:raise ValueError('Time coordinate must be one-dimensional')
    tdim=tcoord.dims[0]
    d=pick(ds,['D','discharge','ice_discharge','flux'])
    e=pick(ds,['err','error','uncertainty'],False)
    c=pick(ds,['coverage','velocity_coverage'],False)
    vv=vector(ds,d,tdim,index)
    ee=vector(ds,e,tdim,index)
    cc=vector(ds,c,tdim,index)
    out=[]
    for i,t in enumerate(tcoord.values):
        date=normalized_date(t)
        if not date:continue
        out.append({'date':date,'d':num(vv[i]),'err':num(ee[i]) if ee is not None else None,'coverage':num(cc[i]) if cc is not None else None})
    return sorted(out,key=lambda p:p['date'])

def scalar_gate(ds,field,gate_dim,index):
    if field is None:return None
    x=ds[field]
    if gate_dim not in x.dims:return None
    v=x.isel({gate_dim:index}).values
    if np.ndim(v)>0:
        if np.size(v)==1:v=np.ravel(v)[0]
        elif v.dtype.kind in ('S','U'):
            v=''.join(y.decode(errors='ignore') if isinstance(y,bytes) else str(y) for y in np.ravel(v))
        else:return None
    if isinstance(v,bytes):return v.decode(errors='ignore').strip()
    if isinstance(v,np.generic):v=v.item()
    return v

def gate_records(ds):
    dname=pick(ds,['D','discharge','ice_discharge','flux']);tname=pick(ds,['time','date','timestamp']);tdim=ds[tname].dims[0]
    dims=[x for x in ds[dname].dims if x!=tdim]
    if len(dims)!=1:raise ValueError(f'Expected gate x time data, got {ds[dname].dims}')
    gdim=dims[0]
    latkey=pick(ds,['lat','latitude','mean_lat'],False)
    lonkey=pick(ds,['lon','longitude','mean_lon'],False)
    idkey=pick(ds,['gate_id','gate','id'],False)
    namekey=pick(ds,['name','glacier_name','name_bjork','bjork'],False)
    if not latkey or not lonkey: raise ValueError('No latitude/longitude in gate NetCDF. Gate metadata CSV needed; inspect source.')
    gates=[];series={}
    for i in range(ds.sizes[gdim]):
        lat=num(scalar_gate(ds,latkey,gdim,i));lon=num(scalar_gate(ds,lonkey,gdim,i))
        if lat is None or lon is None or not(55<=lat<=85 and -75<=lon<=-10):continue
        gid=scalar_gate(ds,idkey,gdim,i) if idkey else i
        gid=str(gid).strip()
        if not gid or gid.lower()=='nan':gid=str(i)
        name=scalar_gate(ds,namekey,gdim,i)
        name=str(name).strip() if name is not None else ''
        if not name or name.lower() in ('nan','none','null'):name='Gate '+gid
        gates.append({'id':gid,'name':name,'lat':lat,'lon':lon})
        series[gid]=records(ds,(gdim,i))
    if not gates:raise ValueError('No valid Greenland georeferenced gates found')
    return gates,series

def run(source_dir=None):
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);resolved={};version=[]
        if source_dir:
            source_dir=Path(source_dir)
            for n in ['GIS.nc','gate.nc']:
                p=source_dir/n
                if not p.is_file():raise FileNotFoundError(p)
                resolved[n]=p
            version=['local files']
        else:
            s=requests.Session();s.headers.update({'User-Agent':'GreenlandDischargeDashboard/0.1 research visualization'})
            manifests=[]
            for pid in PIDS:
                try:
                    m=get_manifest(s,pid);manifests.append(m)
                    version.append(f"{pid}: {m.get('versionNumber','?')}.{m.get('versionMinorNumber','?')}")
                except requests.RequestException as exc:
                    print('Manifest unavailable:',pid,exc)
            if not manifests:raise RuntimeError('No GEUS Dataverse manifest accessible. Keep last published data unchanged.')
            for manifest in manifests:
                for f in manifest.get('files',[]):
                    name=f['dataFile']['filename']
                    if name in ('GIS.nc','gate.nc') and name not in resolved:
                        dest=td/name;print('Downloading',name)
                        download_file(s,f,dest);resolved[name]=dest
            if len(resolved)!=2:raise RuntimeError(f'Required GIS.nc and gate.nc missing from manifests; found {list(resolved)}')
        with xr.open_dataset(resolved['GIS.nc'],decode_times=True) as gis, xr.open_dataset(resolved['gate.nc'],decode_times=True) as gate:
            greenland=records(gis);gates,series=gate_records(gate)
        if not greenland:raise ValueError('Empty Greenland time series')
        payload={'version':' | '.join(version),'greenland':greenland,'gates':gates,'series':series,'source':'PROMICE / GEUS Dataverse','units':'Gt/yr'}
        output=Path(__file__).resolve().parents[1]/'data'/'dashboard.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False),encoding='utf-8');os.replace(tmp,output)
        print(f'Wrote {output} with {len(gates)} gates and {len(greenland)} Greenland observations')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',help='Local NetCDF directory for testing/offline operation')
    args=ap.parse_args();run(args.source_dir)
