"""Read-only first-frame audit for an archive, including files without a beam."""
import argparse
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from pipeline.storage import dump_json,write_csv
from imaging.coordinates import celestial_wcs


def fits_files(root):
    root=Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and
                  p.name.lower().endswith((".fits",".fit",".fts",".fits.gz")))


def inspect_first_plane(path):
    result=dict(source_fits=str(Path(path).resolve()),status="error")
    try:
        with fits.open(path,memmap=True,do_not_scale_image_data=True,uint=False) as hdul:
            index=next(i for i,h in enumerate(hdul) if h.is_image and h.header.get("NAXIS",0)>=2
                       and h.header.get("NAXIS1",0)>1 and h.header.get("NAXIS2",0)>1)
            hdu=hdul[index]; h=hdu.header
            shape=(int(h["NAXIS2"]),int(h["NAXIS1"]))
            plane=[0]*(int(h["NAXIS"])-2)
            file_size=Path(path).stat().st_size
            data_offset=hdu.fileinfo()["datLoc"]
            plane_bytes=shape[0]*shape[1]*abs(int(h["BITPIX"]))//8
            expected_image_bytes=plane_bytes
            for axis in range(3,int(h["NAXIS"])+1):
                expected_image_bytes*=int(h[f"NAXIS{axis}"])
            # Include the center and a corner to expose missing data and scaling.
            finite_counts=[]; sample=[]
            for x,y in ((max(0,shape[1]//2-16),max(0,shape[0]//2-16)),(0,0)):
                stored=hdu.section[tuple(plane)+(slice(y,min(y+32,shape[0])),
                                                 slice(x,min(x+32,shape[1])))]
                invalid=~np.isfinite(stored)
                if "BLANK" in h and np.issubdtype(stored.dtype,np.integer):
                    invalid |= stored==h["BLANK"]
                values=stored.astype(float)*float(h.get("BSCALE",1))+float(h.get("BZERO",0))
                values[invalid]=np.nan
                finite_counts.append(float(np.isfinite(values).mean()))
                sample.extend(values[np.isfinite(values)].tolist())
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                wcs,repairs=celestial_wcs(h)
                scales=proj_plane_pixel_scales(wcs)
                world=wcs.pixel_to_world(shape[1]/2,shape[0]/2).galactic
            major,minor=h.get("BMAJ"),h.get("BMIN")
            source="BMAJ/BMIN"
            if not(major and minor):
                major,minor=h.get("CLEANBMJ"),h.get("CLEANBMN")
                source="CLEANBMJ/CLEANBMN"
            beam=3600*np.sqrt(float(major)*float(minor)) if major and minor else None
            result.update(status="first_plane_sample_read",
                          ready_for_automatic_header_beam=beam is not None,
                          beam_status=source if beam is not None else "needs_verified_product_beam",
                          beam_arcsec=beam,hdu_index=index,width=shape[1],height=shape[0],
                          file_size=file_size,
                          first_frame_storage_complete=file_size>=data_offset+plane_bytes,
                          full_hdu_storage_complete=file_size>=data_offset+expected_image_bytes,
                          nonspatial_indices=str(plane),bunit=str(h.get("BUNIT","unspecified")),
                          nonspatial_ctype=";".join(str(h.get(f"CTYPE{k}",""))
                                                   for k in range(3,int(h["NAXIS"])+1)),
                          pixel_scale_arcsec=float(np.sqrt(np.prod(scales)))*3600,
                          center_gal_lon_deg=float(world.l.deg),center_gal_lat_deg=float(world.b.deg),
                          center_finite_fraction=finite_counts[0],
                          corner_finite_fraction=finite_counts[1],
                          sample_median=float(np.median(sample)) if sample else None,
                          bscale=float(h.get("BSCALE",1)),bzero=float(h.get("BZERO",0)),
                          blank=h.get("BLANK"),wcs_repairs=";".join(repairs),
                          warnings=";".join(str(m.message) for m in messages))
    except Exception as exc:
        result["error"]=repr(exc)
    return result


def inventory(root,output):
    rows=[]
    files=fits_files(root)
    for i,path in enumerate(files,1):
        rows.append(inspect_first_plane(path))
        if i%20==0 or i==len(files):
            print(f"first-plane sample audit {i}/{len(files)}",flush=True)
    write_csv(output,rows)
    report=dict(files=len(rows),sample_read=sum(r["status"]=="first_plane_sample_read" for r in rows),
                header_beam=sum(bool(r.get("ready_for_automatic_header_beam")) for r in rows),
                need_verified_beam=sum(r.get("beam_status")=="needs_verified_product_beam" for r in rows),
                errors=sum(r["status"]=="error" for r in rows),
                scope="Header and two 32x32 first-plane samples; not a full pixel validation or detector run")
    dump_json(Path(output).with_suffix(".json"),report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(inventory(args.root,args.output))
