"""Read only the first spatial plane, scaling bounded windows explicitly."""
import math
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from imaging.coordinates import celestial_wcs


class FitsImage:
    def __init__(self, path, beam_arcsec=None):
        self.path = Path(path).resolve(strict=True)
        self.hdul = fits.open(self.path, memmap=True, lazy_load_hdus=True,
                              do_not_scale_image_data=True, uint=False)
        try:
            self.hdu_index = next(i for i, h in enumerate(self.hdul)
                                  if h.is_image and int(h.header.get("NAXIS", 0)) >= 2
                                  and h.header.get("NAXIS1", 0) > 1
                                  and h.header.get("NAXIS2", 0) > 1)
            self.hdu = self.hdul[self.hdu_index]
            self.header = self.hdu.header.copy()
            self.shape = (int(self.header["NAXIS2"]), int(self.header["NAXIS1"]))
            self.plane = [0] * (int(self.header["NAXIS"]) - 2)
            self.bscale = float(self.header.get("BSCALE", 1))
            self.bzero = float(self.header.get("BZERO", 0))
            self.blank = self.header.get("BLANK")
            with warnings.catch_warnings(record=True) as found:
                warnings.simplefilter("always", FITSFixedWarning)
                self.wcs,self.wcs_repairs = celestial_wcs(self.header)
            self.wcs_warnings = [str(w.message) for w in found]
            scales = np.asarray(proj_plane_pixel_scales(self.wcs), float)
            if len(scales) != 2 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("A valid celestial WCS is required")
            self.pixel_scale_deg = float(np.sqrt(np.prod(scales)))
            self.pixel_axis_scales_deg = scales.tolist()
            major, minor = self.header.get("BMAJ"), self.header.get("BMIN")
            self.beam_source = "BMAJ/BMIN"
            if not (major and minor):
                major, minor = self.header.get("CLEANBMJ"), self.header.get("CLEANBMN")
                self.beam_source = "CLEANBMJ/CLEANBMN"
            if not (major and minor):
                if beam_arcsec is None:
                    raise ValueError("Missing beam: supply a verified --beam-arcsec; no pixel fallback")
                major = minor = float(beam_arcsec) / 3600
                self.beam_source = "explicit_arcsec_override"
            self.beam_major_deg, self.beam_minor_deg = float(major), float(minor)
            if min(self.beam_major_deg, self.beam_minor_deg) <= 0:
                raise ValueError("Beam axes must be positive")
            self.beam_fwhm_px = math.sqrt(float(major) * float(minor)) / self.pixel_scale_deg
            self.beam_pa_deg = float(self.header.get("BPA", self.header.get("CLEANBPA", 0)))
            # Map only the first plane, not an entire cube. This also permits
            # an intact first plane when unused later planes are truncated.
            self.data = None
            compressed=isinstance(self.hdu,fits.CompImageHDU) or self.path.name.lower().endswith(".gz")
            if not compressed:
                dtypes={8:"u1",16:">i2",32:">i4",64:">i8",-32:">f4",-64:">f8"}
                dtype=np.dtype(dtypes[int(self.header["BITPIX"])])
                offset=int(self.hdu.fileinfo()["datLoc"])
                expected=offset+self.shape[0]*self.shape[1]*dtype.itemsize
                if self.path.stat().st_size<expected:
                    raise ValueError("The first spatial frame itself is truncated")
                self.data=np.memmap(self.path,dtype=dtype,mode="r",offset=offset,shape=self.shape)
        except BaseException:
            self.hdul.close()
            raise

    def read(self, x0, y0, x1, y1):
        if self.data is not None:
            stored = self.data[int(y0):int(y1), int(x0):int(x1)]
        else:
            stored=self.hdu.section[tuple(self.plane)+(slice(int(y0),int(y1)),
                                                      slice(int(x0),int(x1)))]
        invalid = ~np.isfinite(stored)
        if self.blank is not None and np.issubdtype(stored.dtype, np.integer):
            invalid = invalid | (stored == self.blank)
        values = np.asarray(stored, dtype=np.float64)
        if self.bscale != 1 or self.bzero != 0:
            values = values * self.bscale + self.bzero
        values = values.astype(np.float32)
        values[invalid] = np.nan
        return values

    def galactic(self, x, y):
        sky = self.wcs.pixel_to_world(x, y).galactic
        if np.ndim(x) == 0:
            return float(sky.l.deg), float(sky.b.deg)
        return np.asarray(sky.l.deg), np.asarray(sky.b.deg)

    def metadata(self):
        return dict(source_fits=str(self.path), image_shape=list(self.shape),
                    hdu_index=self.hdu_index, nonspatial_indices=self.plane,
                    pixel_scale_arcsec=3600*self.pixel_scale_deg,
                    beam_fwhm_px=self.beam_fwhm_px, beam_source=self.beam_source,
                    beam_major_arcsec=3600*self.beam_major_deg,
                    beam_minor_arcsec=3600*self.beam_minor_deg,
                    beam_pa_deg=self.beam_pa_deg,
                    bunit=str(self.header.get("BUNIT", "unspecified")),
                    bscale=self.bscale, bzero=self.bzero, blank=self.blank,
                    wcs_warnings=self.wcs_warnings,wcs_repairs=self.wcs_repairs)

    def close(self):
        mapping=self.data
        self.data = None
        if mapping is not None:
            mapping._mmap.close()
        if self.hdu is not None and self.hdu._data_loaded:
            del self.hdu.data
        self.hdul.close()
        self.hdu = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def normalize(image, low=1.0, high=99.5):
    finite = np.isfinite(image)
    if finite.mean() < 0.8:
        return None
    values = image[finite]
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return None
    return np.clip((np.where(finite, image, np.median(values))-lo)/(hi-lo), 0, 1).astype(np.float32)
