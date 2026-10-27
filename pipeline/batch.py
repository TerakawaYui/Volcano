"""Archive dispatcher; errors and missing product metadata are explicit."""
import json
from pathlib import Path

from imaging.metadata import fits_files
from pipeline.joint_search import run_image
from pipeline.storage import dump_json,write_csv


def run_archive(root,output,config,workers=3,beam_map=None,render=True):
    root,output=Path(root).resolve(strict=True),Path(output).resolve()
    overrides=json.loads(Path(beam_map).read_text(encoding="utf-8")) if beam_map else {}
    reports=[]
    files=fits_files(root)
    if not files:
        raise ValueError("No FITS files found")
    for source in files:
        relative=source.relative_to(root)
        destination=output/relative.with_suffix("")
        override=overrides.get(str(relative),overrides.get(source.name))
        if override and (not override.get("source") or not override.get("beam_arcsec")):
            raise ValueError(f"Beam override must provide beam_arcsec and source: {source}")
        beam=float(override["beam_arcsec"]) if override else None
        try:
            report=run_image(source,destination,config,workers,beam,render)
            actual=json.loads((destination/"manifest.json").read_text(encoding="utf-8"))
            report.update(status=actual["status"],source_fits=str(source),output=str(destination),
                          beam_override_source=override.get("source") if override else "")
        except Exception as exc:
            report=dict(status="failed",source_fits=str(source),output=str(destination),
                        error=repr(exc))
        reports.append(report)
        write_csv(output/"archive_status.csv",reports)
        dump_json(output/"archive_status.json",reports)
    return reports
