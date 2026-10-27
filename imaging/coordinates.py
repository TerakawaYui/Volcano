"""Conservative in-memory corrections for documented legacy velocity-unit syntax."""
from astropy.wcs import WCS


def celestial_wcs(header):
    corrected=header.copy()
    repairs=[]
    for axis in range(3,max(int(header.get("WCSAXES",2)),int(header.get("NAXIS",2)))+1):
        ctype=str(header.get(f"CTYPE{axis}","")).upper()
        unit=str(header.get(f"CUNIT{axis}","")).strip()
        if ctype.startswith(("VELO","VRAD","VOPT")) and unit in ("m/S","M/S"):
            corrected[f"CUNIT{axis}"]="m/s"
            repairs.append(f"CUNIT{axis}: {unit} -> m/s for {ctype}; original FITS unchanged")
    return WCS(corrected).celestial,repairs
