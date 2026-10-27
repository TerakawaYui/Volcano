"""Volcano V5: bright envelope -> joint fit -> observed wall -> two-tier catalogue."""
import argparse
from pathlib import Path
from settings import load_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fits',type=Path,help='FITS file or archive; first spatial plane only')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--beam-arcsec',type=float)
    parser.add_argument('--beam-map',type=Path)
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    source=args.fits.resolve(strict=True)
    output=(args.output or Path(__file__).resolve().parent/'outputs'/source.stem).resolve()
    config=load_config(args.config)
    if source.is_dir():
        if args.beam_arcsec:parser.error('Archive runs use --beam-map')
        from pipeline.batch import run_archive
        reports=run_archive(source,output,config,args.workers,args.beam_map,not args.no_figures)
        failed=any(r['status']!='complete' for r in reports)
    else:
        if args.beam_map:parser.error('--beam-map requires an archive directory')
        from pipeline.joint_search import run_image
        report=run_image(source,output,config,args.workers,args.beam_arcsec,not args.no_figures)
        print(report,flush=True);failed=report['status']!='complete'
    if failed:raise SystemExit(2)

if __name__=='__main__':main()
