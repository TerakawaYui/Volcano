"""Blind first-plane scan. All candidates are generated from the current FITS."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from threadpoolctl import threadpool_limits

from filters.proposals import Candidate, deduplicate, scan_tile, starts
from imaging.fits_image import FitsImage
from pipeline.storage import dump_json, signature, write_csv
from settings import load_config

ROOT = Path(__file__).resolve().parents[1]
_image = None


def resolved_proposals(cfg, beam):
    result = dict(cfg)
    if result.pop("scale_with_beam", False):
        ratio = beam / result["reference_beam_px"]
        for key in ("proposal_sigma", "fragment_sigma", "radius_min", "radius_max"):
            result[key] *= ratio
        for key in ("tile_size", "overlap", "fragment_closing_px"):
            result[key] = max(1, round(result[key] * ratio))
        for key in ("area_min", "area_max"):
            result[key] = max(1, round(result[key] * ratio**2))
    return result


def init_scan(path, beam_arcsec):
    global _image
    threadpool_limits(limits=1)
    _image = FitsImage(path, beam_arcsec)


def scan_batch(batch, cfg):
    found = []
    for tile_id, x, y in batch:
        raw = _image.read(x, y, min(x+cfg["tile_size"], _image.shape[1]),
                          min(y+cfg["tile_size"], _image.shape[0]))
        # Tile-level dedup is independent of worker/batch partition.
        local = deduplicate(scan_tile(raw, x, y, cfg), cfg)
        found.extend(c.row() for c in local)
    return [item[0] for item in batch], found


def scan_candidates(fits_path, output, config, workers=3, beam_arcsec=None, region=None, regions=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FitsImage(fits_path, beam_arcsec) as image:
        metadata = image.metadata()
        cfg = resolved_proposals(config["proposals"], image.beam_fwhm_px)
    shape = metadata["image_shape"]
    windows = [(i, x, y) for i, (x, y) in enumerate(
        (x, y) for y in starts(shape[0], cfg["tile_size"], cfg["overlap"])
        for x in starts(shape[1], cfg["tile_size"], cfg["overlap"]))]
    if region:
        x0, y0, x1, y1 = region
        windows = [(i,x,y) for i,x,y in windows
                   if x < x1 and y < y1 and x+cfg["tile_size"] > x0 and y+cfg["tile_size"] > y0]
    if regions is not None:
        windows = [(i,x,y) for i,x,y in windows if any(
            x < x1 and y < y1 and x+cfg['tile_size'] > x0 and y+cfg['tile_size'] > y0
            for x0,y0,x1,y1 in regions)]
    stage_signature = signature(fits_path, dict(proposals=cfg, beam_arcsec=beam_arcsec,
                                               region=region, regions=regions, metadata=metadata),
                                [Path(__file__), ROOT/"filters/proposals.py",
                                 ROOT/"imaging/fits_image.py"])
    db = sqlite3.connect(output/"scan_state.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS tiles (id INTEGER PRIMARY KEY,payload TEXT)")
    saved = db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
    if saved and saved[0] != stage_signature:
        raise ValueError("Scan inputs/code differ from checkpoint; choose a new output directory")
    db.execute("INSERT OR REPLACE INTO meta VALUES ('signature',?)", (stage_signature,))
    db.commit()
    complete = {i for i, in db.execute("SELECT id FROM tiles")}
    pending = [w for w in windows if w[0] not in complete]
    started = time.time()

    def report(state):
        dump_json(output/"scan_manifest.json",
                  metadata | dict(status=state, signature=stage_signature,
                                  config=cfg, region=region, regions=regions,
                                  tiles_available=len(windows), tiles_completed=len(complete),
                                  elapsed_seconds=round(time.time()-started,1),
                                  candidate_origin="fresh_blind_scan_of_original_first_plane"))

    report("scanning")
    if pending:
        batch_size = 32
        batches = [pending[i:i+batch_size] for i in range(0,len(pending),batch_size)]
        with ProcessPoolExecutor(max_workers=workers, initializer=init_scan,
                                 initargs=(str(fits_path),beam_arcsec)) as pool:
            futures = [pool.submit(scan_batch,b,cfg) for b in batches]
            for index, future in enumerate(as_completed(futures),1):
                ids, found = future.result()
                # Results grouped by their deterministic tile batch; empty tiles are also recorded.
                db.executemany("INSERT INTO tiles VALUES (?,?)",
                               [(i, json.dumps(found if j==0 else [],allow_nan=True))
                                for j,i in enumerate(ids)])
                db.commit()
                complete.update(ids)
                if index % 8 == 0 or index == len(batches):
                    print(f"scan tiles {len(complete)}/{len(windows)} "
                          f"{time.time()-started:.0f}s", flush=True)
                    report("scanning")
    found = []
    for tile_id,payload in db.execute("SELECT id,payload FROM tiles ORDER BY id"):
        found.extend(Candidate(**row) for row in json.loads(payload))
    db.close()
    # Geometry is the final deterministic tie breaker, independent of worker count.
    found.sort(key=lambda c:(-c.score,c.center_y,c.center_x,c.radius_px,c.threshold,c.scale))
    candidates = deduplicate(found,cfg)
    rows = []
    with FitsImage(fits_path,beam_arcsec) as image:
        for rank,candidate in enumerate(candidates,1):
            candidate.rank = rank
            candidate.gal_lon_deg,candidate.gal_lat_deg = image.galactic(candidate.center_x,
                                                                         candidate.center_y)
            candidate.radius_arcmin=candidate.radius_px*image.pixel_scale_deg*60
            row=candidate.row()
            identity=f"{candidate.center_x:.5f}|{candidate.center_y:.5f}|{candidate.radius_px:.5f}"
            row["candidate_id"]="envelope-"+hashlib.sha256(identity.encode()).hexdigest()[:16]
            rows.append(row)
    write_csv(output/"candidates.csv",rows,list(Candidate.__dataclass_fields__)+["candidate_id"])
    report("complete")
    manifest=json.loads((output/"scan_manifest.json").read_text(encoding="utf-8"))
    dump_json(output/"scan_manifest.json",manifest|dict(candidates=len(rows)))
    print(f"scan complete: {len(rows)} fresh candidates",flush=True)
    return output/"candidates.csv"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--beam-arcsec",type=float)
    args=parser.parse_args()
    scan_candidates(args.fits,args.output,load_config(args.config),args.workers,args.beam_arcsec)


if __name__=="__main__":
    main()
