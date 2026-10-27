"""Full-ring refinement; no point-source subtraction or gap completion.

Only independently observed, near-complete walls get the extra model. Ring and
filled disk share the same two angular parameters and block holdout comparison.
These are empirical operating criteria, not calibrated detection probabilities.
"""
import copy
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from filters.joint_model import LocalFit,grade,shell_template
from filters.wall import measure_wall,components


def cavity(original,row):
    yy,xx=np.indices(original.shape);x0,y0=row['_origin']
    dx=xx+x0-row['x'];dy=yy+y0-row['y']
    rho=np.sqrt(np.maximum((1+row['e1'])*dx*dx+2*row['e2']*dx*dy+(1-row['e1'])*dy*dy,0))
    inner=original[(rho<.3*row['r'])&np.isfinite(original)]
    wall=original[(rho>.8*row['r'])&(rho<1.2*row['r'])&np.isfinite(original)]
    return float(np.median(wall)-np.median(inner)) if len(inner)>=3 and len(wall)>=10 else 0.


def original_evidence(raw,row,beam):
    before=copy.deepcopy(row)
    before['_coefficients']=[0.]*len(row['_coefficients'][:-1])+[row['_coefficients'][-1]]
    measured=measure_wall(raw,before,beam)
    original,_=components(raw,row,beam)
    return dict(original_wall_fraction=measured['wall_fraction'],
                original_wall_inner_drop=measured['wall_inner_drop'],
                original_wall_outer_drop=measured['wall_outer_drop'],
                original_cavity_depth=cavity(original,row))


def fit_alternative(raw,b,s,emission,seed,beam,power,cfg,initial):
    local=LocalFit(raw,b,s,emission,seed,beam,power,cfg)
    clump_loss=local.clump_loss()
    local.sources=np.empty((len(local.y),0));local.Sp=local.sources
    local.loss0=float(local.z@local.z)
    X,y=local.B,local.y;valid=local.valid;acfg=cfg['full_ring']
    lo,hi=local.bounds();lo=np.r_[lo,0.,-np.pi];hi=np.r_[hi,.95,np.pi]
    start=np.r_[initial,.3,0.]
    block=max(1,int(np.ceil(beam)))
    folds=((local.xx[valid]//block+2*(local.yy[valid]//block))%4).astype(int)

    def template(p,disk):
        if not disk:return shell_template(local.xx,local.yy,*p[:6],beam,modulation=p[6:])
        cx,cy,r,e1,e2,width,m,phase=p;dx=local.xx-cx;dy=local.yy-cy
        rho=np.sqrt(np.maximum((1+e1)*dx*dx+2*e2*dx*dy+(1-e1)*dy*dy,0))
        t=1/(1+np.exp(np.clip((rho-r)/width,-80,80)))
        t*=1+m*rho/np.sqrt(rho*rho+beam*beam)*np.cos(np.arctan2(dy,dx)-phase)
        t=gaussian_filter(t,beam/2.354820045,mode='constant')
        return t/max(t.max(),1e-20)

    def fit(disk,training):
        Q=np.linalg.qr(X[training],mode='reduced')[0]
        yp=y[training]-Q@(Q.T@y[training])
        def solve(p,details=False):
            t=template(p,disk);v=t[valid];vt=v[training]
            vp=vt-Q@(Q.T@vt)
            amplitude=max(0.,float(vp@yp)/max(float(vp@vp),1e-20))
            beta=np.linalg.lstsq(X[training],y[training]-amplitude*vt,rcond=None)[0]
            residual=y-X@beta-amplitude*v
            return (residual,amplitude,beta,t) if details else residual[training]
        opt=least_squares(solve,np.clip(start,lo+1e-8,hi-1e-8),bounds=(lo,hi),
            max_nfev=45,ftol=2e-3,xtol=2e-3,gtol=2e-3)
        return opt.x,solve(opt.x,True)

    training=np.ones(len(y),bool)
    p,(res,amplitude,beta,t)=fit(False,training)
    _,(diskres,_,_,_)=fit(True,training)
    heldout=[]
    for disk in (False,True):
        total=0.
        for k in range(4):
            _,(error,_,_,_)=fit(disk,folds!=k)
            total+=float(error[folds==k]@error[folds==k])
        heldout.append(total)
    gain=1-heldout[0]/max(heldout[1],1e-20)
    if not np.isfinite(gain) or gain<acfg['min_cv_disk_gain']:return None
    loss=float(res@res)
    row=local.score(p[:6],(loss,np.array([amplitude]),res,t),('full',0.,0.))
    if row is None:return None
    improvement=max(local.loss0-loss,1.)
    row.update(angular_modulation=float(p[6]),angular_phase=float(p[7]),
        model_family='modulated_full_ring',cv_ring_sse=heldout[0],cv_disk_sse=heldout[1],cv_disk_gain=gain,
        flat_advantage=float((loss-diskres@diskres)/max(float(diskres@diskres),1.)),
        flat_advantage_legacy=float((loss-diskres@diskres)/improvement),
        clump_advantage=float((loss-clump_loss)/improvement),central_amplitude=0.,source_count=0,
        _source_positions=[],proposal_score=seed['proposal_score'],proposal_branch=seed['branch'],
        full_circle_snr=row['shell_snr'])
    return row


def select_model(raw,b,s,emission,best,beam,power,cfg,envelopes,origin,allow_extra):
    full=best.pop('_full_fit',None);choices=[best];attempted=False
    for row in choices+([full] if full is not None else []):
        row.update(original_evidence(raw,row,beam))
    if full is not None:
        full['envelope_supported']=bool(envelopes is not None and envelopes.matches(
            full['x']+origin[0],full['y']+origin[1],full['r'],beam))
        full.update(measure_wall(raw,full,beam));choices.append(full)
    acfg=cfg['full_ring']
    # This gate checks the full ellipse in original, not source-subtracted pixels.
    reference=full if full is not None else best
    if (allow_extra and reference['envelope_supported'] and reference['shell_snr']>=cfg['high_snr']
        and reference['original_wall_fraction']>=acfg['min_original_fraction']
        and reference['original_wall_inner_drop']>=cfg['wall']['weak']['inner_drop']
        and reference['original_wall_outer_drop']>=cfg['wall']['weak']['outer_drop']
        and reference['original_cavity_depth']>=acfg['min_original_cavity']):
        attempted=True
        extra=fit_alternative(raw,b,s,emission,best['_seed'],beam,power,cfg,reference['_parameters'])
        if extra is not None:
            extra['envelope_supported']=bool(envelopes is not None and envelopes.matches(
                extra['x']+origin[0],extra['y']+origin[1],extra['r'],beam))
            extra.update(measure_wall(raw,extra,beam));extra.update(original_evidence(raw,extra,beam))
            if (extra['original_wall_fraction']>=acfg['min_original_fraction'] and
                extra['original_cavity_depth']>=acfg['min_original_cavity']):choices.append(extra)
    # Original data must independently support even a legacy PSF-subtracted wall.
    for row in choices:
        if row['original_wall_fraction']<acfg['min_original_weak_fraction'] or row['original_wall_inner_drop']<.25:
            row['original_wall_veto']=True
    level={'excluded':0,'unconfirmed_result':1,'result':2}
    def quality(row):
        tier='excluded' if row.get('original_wall_veto',False) else grade(row,cfg)['tier']
        return level[tier],row.get('model_family')=='modulated_full_ring',row['shell_snr']
    chosen=max(choices,key=quality)
    chosen['alternative_model_summary']=[dict(family=r.get('model_family',r['support_mode']),
        tier='excluded' if r.get('original_wall_veto',False) else grade(r,cfg)['tier'],
        snr=r['shell_snr']) for r in choices]
    return chosen,attempted
