"""Overlapping first-plane joint search, resumable SQLite tiles, deterministic export."""
from collections import Counter,defaultdict
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.metadata
import json
import math
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from filters.joint_model import detect,grade
from filters.proposals import EnvelopeIndex
from imaging.fits_image import FitsImage
from pipeline.storage import dump_json,serializable,signature,write_csv,read_csv
from pipeline.scan import scan_candidates

ROOT=Path(__file__).resolve().parents[1]
_image=None
_cfg=None
_envelopes=None


def windows(shape,cfg):
    """Core starts cover every image pixel; the context extends beyond edge cores."""
    return [(i,x,y) for i,(x,y) in enumerate(
        (x,y) for y in range(0,shape[0],cfg['stride'])
        for x in range(0,shape[1],cfg['stride']))]


def read_window(image,x,y,cfg):
    size=cfg['context_size'];halo=(size-cfg['search_size'])//2
    x0,y0=x-halo,y-halo
    xa,ya=max(0,x0),max(0,y0)
    xb,yb=min(image.shape[1],x0+size),min(image.shape[0],y0+size)
    raw=np.full((size,size),np.nan,dtype=np.float32)
    raw[ya-y0:yb-y0,xa-x0:xb-x0]=image.read(xa,ya,xb,yb)
    return raw,(x0,y0)


def init_worker(path,cfg,beam_arcsec,envelope_path):
    global _image,_cfg,_envelopes
    threadpool_limits(1)
    _image=FitsImage(path,beam_arcsec);_cfg=cfg
    _envelopes=EnvelopeIndex(read_csv(envelope_path))


def process_window(image,window,cfg,envelopes=None):
    tile_id,x,y=window
    raw,origin=read_window(image,x,y,cfg)
    halo=(cfg['context_size']-cfg['search_size'])//2
    core=raw[halo:halo+cfg['search_size'],halo:halo+cfg['search_size']]
    if np.isfinite(core).sum()<10:
        result=dict(status='no_coverage',candidates=[],proposals=[],proposal_n=0)
    else:
        result=detect(raw,image.beam_fwhm_px,cfg,envelopes,origin)
    # Refined centres may move across an image boundary; do not export them as sky objects.
    result['candidates']=[r for r in result['candidates']
        if 0<=r['x']+origin[0]<image.shape[1] and 0<=r['y']+origin[1]<image.shape[0]]
    result.update(tile_id=tile_id,core_start=[x,y],origin=origin)
    result.setdefault('status','complete')
    return result


def worker(window):
    try:
        return window[0],serializable(process_window(_image,window,_cfg,_envelopes))
    except Exception as exc:
        return window[0],dict(status='error',error=repr(exc),traceback=traceback.format_exc(limit=6),
                              candidates=[],tile_id=window[0],core_start=list(window[1:]))


def records(tile,cfg):
    for index,local in enumerate(tile.get('candidates',[])):
        row={k:v for k,v in local.items() if not k.startswith('_')}
        row.update(grade(local,cfg))
        row.update(rank=tile['tile_id']*cfg['max_refine']+index+1,
                   candidate_id=f"joint-{tile['tile_id']:06d}-{index:02d}",
                   tile_id=tile['tile_id'],local_candidate_index=index,
                   fitted_center_x=local['x']+tile['origin'][0],
                   fitted_center_y=local['y']+tile['origin'][1],
                   fitted_radius_px=local['r'],evidence_score=local['shell_snr'])
        yield row


def deduplicate(rows,beam):
    """Nearby centre/radius matching; spatial buckets avoid a quadratic catalogue pass."""
    ordered=sorted(rows,key=lambda r:(r['tier']!='result',-r['shell_snr'],r['rank']))
    cell=max(beam,1.);buckets=defaultdict(list);kept=[];duplicates={}
    for row in ordered:
        x,y,r=row['fitted_center_x'],row['fitted_center_y'],row['r']
        key=(math.floor(x/cell),math.floor(y/cell));match=None
        for dy in range(-2,3):
            for dx in range(-2,3):
                for other in buckets[(key[0]+dx,key[1]+dy)]:
                    tolerance=min(.45*min(r,other['r']),1.5*beam)
                    if .70<=r/other['r']<=1.40 and math.hypot(
                        x-other['fitted_center_x'],y-other['fitted_center_y'])<=tolerance:
                        match=other;break
                if match is not None:break
            if match is not None:break
        if match is None:
            kept.append(row);buckets[key].append(row)
        else:duplicates[row['rank']]=match['rank']
    return kept,duplicates


def export(db,image,output,cfg,render):
    accepted=[];counts=Counter();reasons=Counter();columns=set();proposals=0;candidates=0
    for payload, in db.execute('SELECT payload FROM tiles ORDER BY id'):
        tile=json.loads(payload);proposals+=tile.get('proposal_n',0)
        for row in records(tile,cfg):
            candidates+=1;columns.update(row);counts[row['tier']]+=1
            if row['tier']=='excluded':reasons[row['reason']]+=1
            else:accepted.append(row)
    kept,duplicates=deduplicate(accepted,image.beam_fwhm_px)
    if kept:
        lon,lat=image.galactic(np.array([r['fitted_center_x'] for r in kept]),
                              np.array([r['fitted_center_y'] for r in kept]))
        for row,l,b in zip(kept,lon,lat):
            row.update(fitted_gal_lon_deg=float(l),fitted_gal_lat_deg=float(b),
                       fitted_radius_arcsec=row['r']*image.pixel_scale_deg*3600)
    catalog_columns=sorted(set().union(*(set(r) for r in kept))) if kept else ['rank','candidate_id','tier']
    write_csv(output/'catalog.csv',kept,catalog_columns)
    for group in ('result','unconfirmed_result'):
        write_csv(output/group/'catalog.csv',[r for r in kept if r['tier']==group],catalog_columns)
    import csv
    with (output/'measurements.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,sorted(columns|{'duplicate_of_rank','final_tier'}),extrasaction='ignore')
        writer.writeheader()
        for payload, in db.execute('SELECT payload FROM tiles ORDER BY id'):
            for row in records(json.loads(payload),cfg):
                row.update(duplicate_of_rank=duplicates.get(row['rank'],''),
                           final_tier='duplicate' if row['rank'] in duplicates else row['tier'])
                writer.writerow(row)
    if render:
        from visualization.plots import save_figure
        # Render each tile once, directly from its saved fitted parameters.
        by_tile=defaultdict(list)
        for row in kept:by_tile[row['tile_id']].append(row)
        done=0
        for tile_id,rows in sorted(by_tile.items()):
            tile=json.loads(db.execute('SELECT payload FROM tiles WHERE id=?',(tile_id,)).fetchone()[0])
            raw,_=read_window(image,*tile['core_start'],cfg)
            for row in rows:
                destination=output/row['tier']/f"rank_{row['rank']}.png"
                if not destination.exists():
                    save_figure(destination,raw,tile['candidates'][row['local_candidate_index']],
                                image.beam_fwhm_px,row,tile['origin'],cfg,image.wcs)
                done+=1
            print(f'render {done}/{len(kept)}',flush=True)
    return dict(proposals=proposals,candidates=candidates,pre_dedup_tiers=dict(counts),
                result=sum(r['tier']=='result' for r in kept),
                unconfirmed_result=sum(r['tier']=='unconfirmed_result' for r in kept),
                duplicates=len(duplicates),exclusion_reasons=dict(reasons),figures_written=render)


def run_image(fits_path,output,config,workers=3,beam_arcsec=None,render=True,tile_ids=None):
    """tile_ids is only a diagnostic subset of the same global grid; CLI searches all tiles."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=config['joint']|dict(wall=config['wall'])
    if (output/'measurement_state.sqlite').exists():
        raise ValueError('Legacy output detected; use a new output directory')
    if (output/'manifest.json').exists():
        previous=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if previous.get('method')!='observed_wall_joint_v5':
            raise ValueError('Previous method detected; use a new output directory')
    environment={k:importlib.metadata.version(k) for k in ('numpy','scipy','astropy','matplotlib')}
    files=[ROOT/p for p in ('filters/joint_model.py','filters/wall.py','filters/proposals.py',
        'pipeline/scan.py','pipeline/joint_search.py',
        'pipeline/storage.py','imaging/fits_image.py','imaging/coordinates.py',
        'visualization/plots.py','settings/schema.py','filters/full_ring.py')]
    selected_ids=sorted(set(tile_ids)) if tile_ids is not None else None
    with FitsImage(fits_path,beam_arcsec) as image:
        metadata=image.metadata();all_windows=windows(image.shape,cfg)
    selected=[w for w in all_windows if selected_ids is None or w[0] in set(selected_ids)]
    if selected_ids is not None and len(selected)!=len(selected_ids):
        raise ValueError('Diagnostic tile IDs do not match the current grid')
    regions=None if selected_ids is None else [(x-64,y-64,x+cfg['search_size']+64,
        y+cfg['search_size']+64) for _,x,y in selected]
    envelope_path=scan_candidates(fits_path,output/'firstcheck',config,workers,beam_arcsec,regions=regions)
    signed=signature(fits_path,dict(joint=cfg,proposals=config['proposals'],environment=environment,metadata=metadata,
                                  beam_arcsec=beam_arcsec,tile_ids=selected_ids),files)
    state=output/'joint_state.sqlite';started=time.time()
    with closing(sqlite3.connect(state)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS tiles (id INTEGER PRIMARY KEY,payload TEXT)')
        saved=db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if saved and saved[0]!=signed:
            raise ValueError('Input/code/config changed; use a new output directory')
        db.execute("INSERT OR REPLACE INTO meta VALUES ('signature',?)",(signed,));db.commit()
        complete={i for i,payload in db.execute('SELECT id,payload FROM tiles')
                  if json.loads(payload)['status']!='error'}
        pending=[w for w in selected if w[0] not in complete]
        base=metadata|dict(method='observed_wall_joint_v5',signature=signed,config=cfg,environment=environment,
            scope='full_image' if selected_ids is None else 'selected_global_grid_tiles',
            tiles_total=len(selected),global_grid_tiles=len(all_windows),diagnostic_tile_ids=selected_ids,
            candidate_origin='fresh_original_fits',old_results_used=False,ph_enabled=False,
            notes='Empirical tiers; conditional OLS amplitude errors use estimated stationary covariance. Not calibrated probabilities.')
        dump_json(output/'manifest.json',base|dict(status='searching',tiles_completed=len(complete)))
        dump_json(output/'configuration.json',dict(signature=signed,joint=cfg))
        print(f'joint: {len(complete)} current-run resumed, {len(pending)} pending / {len(selected)} tiles',flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,
                                     initargs=(str(fits_path),cfg,beam_arcsec,str(envelope_path))) as pool:
                iterator=iter(pending);active={}
                for _ in range(min(len(pending),2*workers)):
                    window=next(iterator);active[pool.submit(worker,window)]=window[0]
                while active:
                    future=next(as_completed(active));active.pop(future)
                    tile_id,payload=future.result()
                    db.execute('INSERT OR REPLACE INTO tiles VALUES (?,?)',
                               (tile_id,json.dumps(payload,allow_nan=False)));db.commit()
                    complete.add(tile_id)
                    elapsed=time.time()-started
                    print(f"joint tiles {len(complete)}/{len(selected)}; last={tile_id} "
                          f"{payload['status']}; {elapsed:.0f}s",flush=True)
                    dump_json(output/'progress.json',dict(stage='joint_search',completed=len(complete),
                        total=len(selected),last_tile=tile_id,last_status=payload['status'],elapsed_seconds=elapsed))
                    window=next(iterator,None)
                    if window is not None:active[pool.submit(worker,window)]=window[0]
        with FitsImage(fits_path,beam_arcsec) as image:
            report=export(db,image,output,cfg,render)
        statuses=Counter(json.loads(p)['status'] for p, in db.execute('SELECT payload FROM tiles'))
        report.update(status='complete_with_errors' if statuses['error'] else 'complete',
                      tiles_completed=len(selected),tile_statuses=dict(statuses),measurement_errors=statuses['error'])
        dump_json(output/'manifest.json',base|report|dict(elapsed_seconds=time.time()-started))
    return report
