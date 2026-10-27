"""Observed wall evidence, measured independently of shell amplitude.

Sample original and source-subtracted transverse profiles. Missing pixels and
unsupported angular gaps remain missing; no interpolation closes an arc.
All brightness differences are in local pixel-scatter units, not p-values.
"""
import math
import numpy as np
from scipy.ndimage import map_coordinates


def angular_support(theta, row):
    mode = row['support_mode']
    if mode == 'arc':
        return np.mod(theta-row['phase'], 2*np.pi) <= row['arc_angle']
    if mode == 'bilateral':
        delta = np.abs(np.angle(np.exp(1j*(theta-row['phase']))))
        return np.minimum(delta, np.pi-delta) <= row['arc_angle']/2
    return np.ones(theta.shape, bool)


def components(raw, row, beam):
    """Reconstruct saved nuisance components without refitting any parameter."""
    x0, y0 = row['_origin']; ny, nx = row['_model_shape']
    seed = row['_seed']; noise = row['noise']
    yy, xx = np.indices((ny, nx)); xx = xx+x0; yy = yy+y0
    half = math.ceil(max(1.8*seed['r'], seed['r']+3*beam))
    out = row['_outside_background_beta']; rout = row['background_outer']
    background = out[0]+out[1]*(xx-seed['x'])/rout+out[2]*(yy-seed['y'])/rout
    beta = row['_joint_background_beta']
    background += noise*(beta[0]+beta[1]*(xx-seed['x'])/half+beta[2]*(yy-seed['y'])/half)
    sources = np.zeros((ny, nx))
    for amplitude, (sx, sy) in zip(row['_coefficients'][:-1], row['_source_positions']):
        sources += noise*amplitude*np.exp(-.5*((xx-sx)**2+(yy-sy)**2)/(beam/2.354820045)**2)
    field = (raw[y0:y0+ny, x0:x0+nx]-background)/noise
    return field, field-sources/noise


def longest_run(good):
    """Number of adjacent supported samples, including a wrap-around arc."""
    best = run = 0
    for value in np.tile(good, 2):
        run = run+1 if value else 0
        best = max(best, run)
    return min(best, len(good))


def measure_wall(raw, row, beam):
    """Normal profiles on the fitted ellipse; two-sided peak support is observed."""
    original, residual = components(raw, row, beam)
    n = max(24, int(math.ceil(4*np.pi*row['major']/beam)))
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    e1, e2, r = row['e1'], row['e2'], row['r']
    radii = r/np.sqrt(1+e1*np.cos(2*theta)+e2*np.sin(2*theta))
    x = radii*np.cos(theta); y = radii*np.sin(theta)
    # The gradient of the ellipse equation gives its outward normal.
    nx = (1+e1)*x+e2*y; ny = e2*x+(1-e1)*y
    norm = np.hypot(nx, ny); nx /= norm; ny /= norm
    dx = np.roll(x, -1)-x; dy = np.roll(y, -1)-y
    ds = np.hypot(dx, dy); perimeter = float(ds.sum())
    d = max(.85*beam, min(1.5*beam, 1.5*row['width']))
    offsets = np.linspace(-1.4*d, 1.4*d, 21)
    x0, y0 = row['_origin']
    profiles = []
    for image in (original, residual):
        samples = []
        for tangent in (-.2*beam, 0., .2*beam):
            xx = row['x']-x0+x[:, None]+nx[:, None]*offsets-ny[:, None]*tangent
            yy = row['y']-y0+y[:, None]+ny[:, None]*offsets+nx[:, None]*tangent
            samples.append(map_coordinates(image, [yy, xx], order=1, mode='constant', cval=np.nan))
        profiles.append(np.mean(samples, axis=0))
    raw_profile, profile = profiles
    yy, xx = np.indices(residual.shape)
    distance = np.hypot(xx+x0-row['x'], yy+y0-row['y'])
    outside = np.isfinite(residual) & (distance >= r+max(1.5*beam, 2*row['width']))
    context = residual[outside]
    context_scatter = 1.
    context_center = 0.
    if context.size >= 30:
        context_center = float(np.median(context))
        context_scatter = max(1., 1.4826*float(np.median(np.abs(context-context_center))),
                              .15*float(np.std(context)))
    central = np.abs(offsets) <= .55*beam
    masked = np.where(central[None, :] & np.isfinite(profile), profile, -np.inf)
    peak_index = np.argmax(masked, axis=1)
    index = np.arange(n)
    peak = profile[index, peak_index]
    raw_peak = raw_profile[index, peak_index]
    inner = np.median(profile[:, offsets <= -.8*d], axis=1)
    outer = np.median(profile[:, offsets >= .8*d], axis=1)
    inner_drop = peak-inner; outer_drop = peak-outer
    local_peak = ((peak_index > 0) & (peak_index < len(offsets)-1) &
                  (peak >= profile[index, np.maximum(0, peak_index-1)]) &
                  (peak >= profile[index, np.minimum(len(offsets)-1, peak_index+1)]))
    supported = angular_support(theta, row)
    finite = np.isfinite(profile).all(axis=1) & np.isfinite(raw_profile).all(axis=1)
    good = supported & finite & local_peak & (peak >= 1.) & (inner_drop >= .25) & (outer_drop >= .5)
    run = longest_run(good)
    # Approximate physical run length; sampling is uniform in angle, not arclength.
    run_beams = run/n*perimeter/beam
    angle = min(np.pi, 2*np.pi*run/n)
    sagitta = r*(1-np.cos(angle/2))/beam
    use = supported & finite
    def median(values):
        return float(np.median(values[use])) if use.any() else 0.
    retention = np.maximum(peak, 0)/np.maximum(raw_peak, 1e-8)
    return dict(wall_fraction=float(np.sum(ds[good])/perimeter),
                wall_run_beams=float(run_beams), wall_sagitta_beams=float(sagitta),
                wall_peak=median(peak), wall_inner_drop=median(inner_drop),
                wall_outer_drop=median(outer_drop), wall_retention=median(np.clip(retention, 0, 1)),
                wall_context_contrast=(median(peak)-context_center)/context_scatter if context.size>=30 else 0.,
                wall_context_scatter=float(context_scatter), wall_context_pixels=int(context.size),
                wall_radius_width_ratio=float(r/max(row['width'],1e-8)),
                wall_valid_fraction=float(finite.mean()),
                _wall_angles=theta.tolist(), _wall_good=good.tolist())
