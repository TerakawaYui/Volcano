"""Broad, interior-agnostic proposals of roughly round bright envelopes."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.measure import regionprops
from scipy.spatial import cKDTree

from imaging.fits_image import normalize


class EnvelopeIndex:
    """Geometry index of this run's freshly scanned bright envelopes."""
    def __init__(self, rows):
        self.centres = np.asarray([[float(r['center_x']), float(r['center_y'])]
                                   for r in rows], dtype=float).reshape(-1, 2)
        self.radii = np.asarray([float(r['radius_px']) for r in rows])
        self.tree = cKDTree(self.centres)

    def matches(self, x, y, radius, beam, eta=.45, low=.35, high=1.8):
        nearby = self.tree.query_ball_point([x, y], max(beam, eta*radius))
        return any(low <= radius/self.radii[i] <= high and
                   math.hypot(x-self.centres[i, 0], y-self.centres[i, 1]) <=
                   max(beam, eta*min(radius, self.radii[i])) for i in nearby)


@dataclass
class Candidate:
    center_x: float
    center_y: float
    radius_px: float
    area_px: int
    solidity: float
    eccentricity: float
    contrast: float
    threshold: float
    scale: float
    score: float
    rank: int = 0
    gal_lon_deg: float = math.nan
    gal_lat_deg: float = math.nan
    radius_arcmin: float = math.nan

    def row(self):
        return asdict(self)


def starts(length, size, overlap):
    last = max(0, length - size)
    values = list(range(0, last + 1, size - overlap))
    if not values or values[-1] != last:
        values.append(last)
    return values


def scan_tile(raw, x0, y0, cfg):
    normalized = normalize(raw)
    if normalized is None:
        return []
    found = []
    for sigma, closing in ((cfg["proposal_sigma"], 0),
                           (cfg["fragment_sigma"], cfg["fragment_closing_px"])):
        smooth = gaussian_filter(normalized, sigma)
        finite = smooth[np.isfinite(smooth)]
        try:
            lower = min(float(threshold_triangle(finite)),
                        float(threshold_otsu(finite)))
        except ValueError:
            lower = float(np.percentile(finite, 65))
        lower = float(np.clip(lower, 0.08, cfg["threshold_upper"] - 0.03))
        for threshold in np.linspace(lower, cfg["threshold_upper"],
                                     cfg["threshold_levels"]):
            mask = smooth >= threshold
            if closing:
                mask = binary_closing(mask, iterations=closing)
            labels, _ = label(mask)
            for prop in regionprops(labels):
                if not cfg["area_min"] <= prop.area <= cfg["area_max"]:
                    continue
                y0b, x0b, y1b, x1b = prop.bbox
                if y0b == 0 or x0b == 0 or y1b == raw.shape[0] or x1b == raw.shape[1]:
                    continue
                filled = binary_fill_holes(prop.image)
                area = int(filled.sum())
                radius = math.sqrt(area / math.pi)
                if not cfg["radius_min"] <= radius <= cfg["radius_max"]:
                    continue
                shape = regionprops(filled.astype(np.uint8))[0]
                if (shape.solidity < cfg["min_solidity"]
                        or shape.eccentricity > cfg["max_eccentricity"]):
                    continue
                inside = smooth[y0b:y1b, x0b:x1b][prop.image]
                contrast = float(np.median(inside) - np.median(smooth))
                if contrast < cfg["min_contrast"]:
                    continue
                cy, cx = shape.centroid
                cx += x0b + x0
                cy += y0b + y0
                score = (0.4 * min(contrast / 0.3, 2) + 0.35 * shape.solidity
                         + 0.25 * (1 - shape.eccentricity))
                found.append(Candidate(
                    float(cx), float(cy), radius, area, float(shape.solidity),
                    float(shape.eccentricity), contrast, float(threshold),
                    float(sigma), float(score)))
    return found


def deduplicate(found, cfg):
    """Spatial hash keeps alternative envelope scales without quadratic scans."""
    cell = max(16.0, float(cfg["radius_max"]))
    grid = {}
    kept = []
    for c in sorted(found, key=lambda item: item.score, reverse=True):
        gx, gy = int(c.center_x // cell), int(c.center_y // cell)
        duplicate = False
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                for i in grid.get((gx + dx, gy + dy), ()):
                    k = kept[i]
                    ratio = max(c.radius_px, k.radius_px) / min(c.radius_px, k.radius_px)
                    distance = math.hypot(c.center_x - k.center_x,
                                          c.center_y - k.center_y)
                    if (ratio <= cfg["dedup_radius_ratio"]
                            and distance <= cfg["dedup_center_factor"]
                            * min(c.radius_px, k.radius_px)):
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            grid.setdefault((gx, gy), []).append(len(kept))
            kept.append(c)
    return kept
