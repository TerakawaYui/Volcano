"""One configuration for the current envelope/joint/wall pipeline."""
import copy,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def merge(base,override):
    for key,value in override.items():
        if key not in base:raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(base[key],dict):
            if not isinstance(value,dict):raise ValueError(f'{key} must be a mapping')
            merge(base[key],value)
        else:base[key]=value

def load_config(path=None):
    cfg=json.loads((ROOT/'settings/defaults.json').read_text(encoding='utf-8'))
    if path:merge(cfg,json.loads(Path(path).read_text(encoding='utf-8')))
    joint=cfg['joint'];proposals=cfg['proposals']
    angular=joint['full_ring']
    if type(angular['max_per_window']) is not int or not 1<=angular['max_per_window']<=joint['max_refine']:
        raise ValueError('Invalid angular refinement budget')
    for key in ('min_original_fraction','min_original_weak_fraction','min_cv_disk_gain'):
        if not isinstance(angular[key],(int,float)) or not 0<=angular[key]<=1:
            raise ValueError('Invalid angular fraction/gain')
    if not math.isfinite(angular['min_original_cavity']) or angular['min_original_cavity']<0:
        raise ValueError('Invalid original cavity threshold')
    for key in ('context_size','search_size','stride','mesh','max_coarse','max_refine','geometry_nfev','arc_nfev'):
        if not isinstance(joint[key],int) or isinstance(joint[key],bool) or joint[key]<1:
            raise ValueError(f'joint.{key} must be a positive integer')
    if not joint['search_size']<=joint['context_size']<=512:
        raise ValueError('search/context sizes must fit the FFT grid')
    if (joint['context_size']-joint['search_size'])%2 or joint['stride']>joint['search_size']:
        raise ValueError('Symmetric halo and covering stride required')
    if joint['max_refine']>joint['max_coarse']:raise ValueError('Invalid refinement budget')
    if not joint['radii'] or not all(isinstance(r,(int,float)) and 2.7<=r<=32 for r in joint['radii']):
        raise ValueError('Proposal radii must lie in [2.7,32] pixels')
    for field in ('snr','arc_beams'):
        if not 0<joint['weak_'+field]<=joint['high_'+field]:raise ValueError('Non-nested joint thresholds')
    for field in ('flat_max','clump_max'):
        if joint['high_'+field]>joint['weak_'+field]:raise ValueError('Non-nested model thresholds')
    if not 0<=proposals['overlap']<proposals['tile_size']:raise ValueError('Invalid proposal overlap')
    for key,value in cfg['wall']['weak'].items():
        high=cfg['wall']['high'][key]
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in (value,high)) or high<value:
            raise ValueError(f'Non-nested wall threshold: {key}')
        if key in ('fraction','retention') and high>1:raise ValueError(f'{key} must not exceed one')
    return copy.deepcopy(cfg)
