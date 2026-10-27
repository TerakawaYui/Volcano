"""Joint shell/PSF search in an image window; no reference labels or positions.

Fits original pixels with joint nonnegative shell/core amplitudes. Fits use OLS;
conditional errors use an empirical stationary covariance (sandwich variance).
This is not full GLS and scores are not calibrated p-values.
"""
import math,time
from functools import lru_cache
import numpy as np
from scipy import fft
from scipy.ndimage import (gaussian_filter,maximum_filter,binary_dilation,
                           binary_fill_holes,label,map_coordinates,find_objects,distance_transform_edt)
from scipy.optimize import nnls,least_squares
from filters.wall import measure_wall

DEFAULT=dict(context_size=384,search_size=128,mesh=64,proposal_floor=2.5,
             radii=[2.7,3.3,3.9,4.5,5.2,6.,7.,8.,9.5,11.,13.,15.5,18.5,22.,26.5,32.],
             max_coarse=96,max_refine=16,geometry_nfev=18,arc_nfev=10,
             kernel_lag_beams=4.,seed=20261019)

def robust_sigma(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    if len(a)<10:return float('nan')
    s=1.4826022185*np.median(np.abs(a-np.median(a)))
    return float(max(s,.15*np.std(a),1e-30))

def clipped_stats(a):
    v=np.asarray(a);v=v[np.isfinite(v)]
    if v.size<10:return float("nan"),float("nan")
    for _ in range(5):
        m=float(np.median(v));s=robust_sigma(v)
        good=np.abs(v-m)<3*s
        if good.all() or good.sum()<30:break
        v=v[good]
    return float(np.median(v)),robust_sigma(v)

def background_maps(raw,beam,cfg):
    n,m=raw.shape;step=cfg['mesh'];ny=math.ceil(n/step);nx=math.ceil(m/step)
    bm=np.zeros((ny,nx));sm=np.zeros_like(bm)
    for iy in range(ny):
        for ix in range(nx):
            bm[iy,ix],sm[iy,ix]=clipped_stats(raw[iy*step:min(n,(iy+1)*step),ix*step:min(m,(ix+1)*step)])
    grid_good=np.isfinite(bm)&np.isfinite(sm)&(sm>0)
    if not grid_good.any():raise ValueError("No usable background grid cells")
    if not grid_good.all():
        nearest=distance_transform_edt(~grid_good,return_distances=False,return_indices=True)
        bm=bm[tuple(nearest)];sm=sm[tuple(nearest)]
    yy,xx=np.indices(raw.shape)
    coords=[(yy+.5)/step-.5,(xx+.5)/step-.5]
    b=map_coordinates(bm,coords,order=1,mode='nearest')
    s=map_coordinates(sm,coords,order=1,mode='nearest')
    valid=np.isfinite(raw)
    z=np.where(valid,(raw-b)/s,0.)
    emission=binary_dilation((z>3.8)|(~valid),iterations=max(1,round(.6*beam)))
    good=valid&~emission&(np.abs(z)<4)
    # A second mesh pass estimates fluctuations outside obvious emission.
    for iy in range(ny):
        for ix in range(nx):
            sl=(slice(iy*step,min(n,(iy+1)*step)),slice(ix*step,min(m,(ix+1)*step)))
            values=raw[sl][good[sl]]
            if len(values)>=80:bm[iy,ix],sm[iy,ix]=clipped_stats(values)
    b=map_coordinates(bm,coords,order=1,mode='nearest')
    s=map_coordinates(sm,coords,order=1,mode='nearest')
    z=np.where(valid,(raw-b)/s,0.)
    emission=binary_dilation((z>3.8)|(~valid),iterations=max(1,round(.6*beam)))
    good=valid&~emission&(np.abs(z)<4)
    return b,s,z,emission,good

def spectrum_from_background(z,mask,beam,shape=(512,512)):
    """Masked empirical ACF -> tapered PSD -> positive spectral floor."""
    lag=int(math.ceil(4*beam));center=np.asarray(z.shape)-1
    field=np.where(mask,z-np.median(z[mask]),0.)
    den=fft.irfftn(fft.rfftn(mask.astype(float),s=(2*z.shape[0]-1,2*z.shape[1]-1))*
                  np.conj(fft.rfftn(mask.astype(float),s=(2*z.shape[0]-1,2*z.shape[1]-1))),
                  s=(2*z.shape[0]-1,2*z.shape[1]-1))
    num=fft.irfftn(np.abs(fft.rfftn(field,s=den.shape))**2,s=den.shape)
    offsets=np.arange(-lag,lag+1)
    acf=num[np.ix_(offsets%den.shape[0],offsets%den.shape[1])]/np.maximum(
        den[np.ix_(offsets%den.shape[0],offsets%den.shape[1])],1.)
    acf=acf/max(acf[lag,lag],1e-20)
    taper=(1-np.abs(offsets)/(lag+1))
    acf*=taper[:,None]*taper[None,:]
    arr=np.zeros(shape)
    arr[np.ix_(offsets%shape[0],offsets%shape[1])]=acf
    power=fft.rfftn(arr).real
    positive=power[power>0]
    floor=.04*np.median(positive) if positive.size else 1e-8
    power=np.maximum(power,max(floor,1e-8))
    weights=np.full(power.shape[1],2.);weights[0]=weights[-1]=1.
    power/=np.sum(power*weights[None,:])/np.prod(shape)
    return power.astype(np.float32)

def kernel_fft(kernel,shape):
    h=np.zeros(shape,np.float32);h[:kernel.shape[0],:kernel.shape[1]]=kernel
    h=np.roll(h,(-(kernel.shape[0]//2),-(kernel.shape[1]//2)),axis=(0,1))
    return fft.rfftn(h)

def spectral_variance(h,power,shape):
    H=kernel_fft(h,shape)
    weights=np.full(power.shape[1],2.);weights[0]=weights[-1]=1.
    return float(np.sum(np.abs(H)**2*power*weights[None,:])/np.prod(shape))

def shell_template(xx,yy,cx,cy,r,e1,e2,width,beam,support=('full',0.,0.),modulation=None):
    dx,dy=xx-cx,yy-cy
    rho=np.sqrt(np.maximum((1+e1)*dx*dx+2*e2*dx*dy+(1-e1)*dy*dy,0))
    angle=np.mod(np.arctan2(dy,dx),2*np.pi)
    v=np.exp(-.5*((rho-r)/width)**2)
    if modulation is not None:
        strength,phase=modulation
        v*=1+strength*rho/np.sqrt(rho*rho+beam*beam)*np.cos(angle-phase)
    mode,phase,length=support
    if mode=='arc':v*=np.mod(angle-phase,2*np.pi)<=length
    if mode=='bilateral':
        delta=np.abs(np.angle(np.exp(1j*(angle-phase))))
        delta2=np.abs(np.angle(np.exp(1j*(angle-phase-np.pi))))
        v*=np.minimum(delta,delta2)<=length/2
    v=gaussian_filter(v,beam/2.354820045,mode='constant')
    return v/max(v.max(),1e-20)

@lru_cache(maxsize=3)
def filter_bank(beam_key,radii_key,shape):
    beam=float(beam_key);items=[]
    for r in radii_key:
        half=int(math.ceil(max(1.8*r,r+3*beam)))
        yy,xx=np.indices((2*half+1,2*half+1));xx=xx-half;yy=yy-half
        rr=np.hypot(xx,yy);angle=np.arctan2(yy,xx);region=rr<=half
        core=np.exp(-.5*(rr/(beam/2.354820045))**2)
        B=np.column_stack((np.ones(region.sum()),xx[region]/half,yy[region]/half,core[region]))
        Q=np.linalg.qr(B,mode='reduced')[0]
        for width in (max(.5,.12*r),max(.9,.32*r)):
            base=shell_template(xx,yy,0,0,r,0,0,width,beam)
            specs=[]
            for harmonic in (np.ones_like(xx),np.cos(angle),np.sin(angle),np.cos(2*angle),np.sin(2*angle)):
                v=(base*harmonic)[region];v=v-Q@(Q.T@v)
                h=np.zeros_like(base);h[region]=v
                H=kernel_fft(h,shape)
                specs.append((H,np.abs(H)**2))
            items.append((r,width,specs))
    return items

def propose(raw,z,beam,power,cfg):
    n=raw.shape[0];core=cfg['search_size'];lo=(n-core)//2;hi=lo+core
    shape=(512,512);Z=fft.rfftn(np.clip(z,-12,40).astype(np.float32),s=shape)
    rw=np.full(power.shape[1],2.);rw[0]=rw[-1]=1.
    seeds=[]
    for r,width,specs in filter_bank(round(beam,10),tuple(cfg['radii']),shape):
        responses=[]
        for H,H2 in specs:
            var=float(np.sum(H2*power*rw[None,:])/np.prod(shape))
            response=fft.irfftn(Z*H,s=shape)[lo:hi,lo:hi]/math.sqrt(max(var,1e-12))
            responses.append(response)
        a,b,c,d,e=responses
        score=np.maximum.reduce([a,(a+np.hypot(b,c))/math.sqrt(2),
                                 (a+np.hypot(d,e))/math.sqrt(2)])
        peaks=(score==maximum_filter(score,size=max(3,round(beam))))&(score>=cfg['proposal_floor'])
        ys,xs=np.nonzero(peaks)
        for i in np.argsort(score[ys,xs])[-16:][::-1]:
            seeds.append(dict(x=float(xs[i]+lo),y=float(ys[i]+lo),r=float(r),
                              width=float(width),proposal_score=float(score[ys[i],xs[i]]),branch='shell_filter'))
    # Background-normalized intensity islands provide alternative centroids and scales.
    for smoothing in (0.,.3*beam):
        field=z if smoothing==0 else gaussian_filter(z,smoothing)
        for threshold in (2.5,4.,6.):
            labels,count=label(field>threshold)
            for k,slc in enumerate(find_objects(labels),1):
                if slc is None:continue
                localmask=labels[slc]==k
                ys,xs=np.nonzero(localmask)
                if len(xs)<8 or len(xs)>4000:continue
                ys=ys+slc[0].start;xs=xs+slc[1].start
                cy,cx=float(np.mean(ys)),float(np.mean(xs))
                if not(lo<=cx<hi and lo<=cy<hi):continue
                r=math.sqrt(binary_fill_holes(localmask).sum()/np.pi)
                if not(2.5<=r<=32):continue
                seeds.append(dict(x=cx,y=cy,r=r,width=max(.6,.18*r),
                                  proposal_score=float(np.mean(field[ys,xs])*math.sqrt(min(len(xs),50))/3),
                                  branch='intensity_island'))
    seeds.sort(key=lambda d:-d['proposal_score'])
    kept=[];buckets={}
    for s in seeds:
        key=(int((s['x']-lo)//16),int((s['y']-lo)//16))
        bucket=buckets.setdefault(key,[])
        if any(math.hypot(s['x']-t['x'],s['y']-t['y'])<.6*beam and
               abs(s['r']-t['r'])<max(.45*beam,.15*min(s['r'],t['r'])) for t in bucket):continue
        if len(bucket)<3:bucket.append(s)
    for depth in range(3):
        layer=sorted([v[depth] for v in buckets.values() if len(v)>depth],key=lambda s:-s['proposal_score'])
        for s in layer:
            kept.append(s)
            if len(kept)>=cfg['max_coarse']:return kept
    return kept

def local_background(raw,bmap,smap,emission,cx,cy,r,beam):
    yy,xx=np.indices(raw.shape);dx,dy=xx-cx,yy-cy;rr=np.hypot(dx,dy)
    r0=max(1.5*r,r+1.5*beam);r1=max(3.5*r,r+7*beam)
    valid=np.isfinite(raw);mask=valid&(rr>=r0)&(rr<=r1)&~emission
    fallback=False
    if mask.sum()<80:
        mask=valid&(rr>=r0)&(rr<=r1);fallback=True
    X=np.column_stack((np.ones(mask.sum()),dx[mask]/r1,dy[mask]/r1));values=raw[mask].astype(float)
    beta=np.linalg.lstsq(X,values,rcond=None)[0]
    for _ in range(5):
        e=values-X@beta;s=robust_sigma(e)
        weights=np.minimum(1.,1.5*s/np.maximum(np.abs(e),1e-30))
        beta=np.linalg.lstsq(X*np.sqrt(weights[:,None]),values*np.sqrt(weights),rcond=None)[0]
    plane=beta[0]+beta[1]*dx/r1+beta[2]*dy/r1
    noise=max(robust_sigma((raw-plane)[mask]),.7*float(np.median(smap[mask])))
    return plane,noise,dict(background_pixels=int(mask.sum()),background_fallback=fallback,
                            background_inner=r0,background_outer=r1,
                            _outside_background_beta=beta.tolist(),
                            background_masked_fraction=float(1-mask.sum()/max(1,np.sum(valid&(rr>=r0)&(rr<=r1)))))

class LocalFit:
    def __init__(self,raw,bmap,smap,emission,seed,beam,power,cfg):
        self.seed=seed;self.beam=beam;self.cfg=cfg;self.power=power;self.shape=(512,512)
        r=seed['r'];cx,cy=seed['x'],seed['y']
        plane,self.noise,self.bginfo=local_background(raw,bmap,smap,emission,cx,cy,r,beam)
        half=int(math.ceil(max(1.8*r,r+3*beam)))
        self.x0=max(0,round(cx)-half);self.y0=max(0,round(cy)-half)
        x1=min(raw.shape[1],round(cx)+half+1);y1=min(raw.shape[0],round(cy)+half+1)
        self.raw=raw[self.y0:y1,self.x0:x1].astype(float)
        self.plane=plane[self.y0:y1,self.x0:x1]
        yy,xx=np.indices(self.raw.shape);self.xx=xx+self.x0;self.yy=yy+self.y0
        self.valid=np.isfinite(self.raw)&(np.hypot(self.xx-cx,self.yy-cy)<=half)
        self.B=np.column_stack((np.ones(self.valid.sum()),(self.xx[self.valid]-cx)/half,(self.yy[self.valid]-cy)/half))
        self.Q=np.linalg.qr(self.B,mode='reduced')[0]
        self.y=(self.raw[self.valid]-self.plane[self.valid])/self.noise
        self.z=self.y-self.Q@(self.Q.T@self.y)
        clean=np.nan_to_num((self.raw-self.plane)/self.noise)
        maxima=clean==maximum_filter(clean,size=max(3,round(beam)))
        radius=np.hypot(self.xx-cx,self.yy-cy)
        possible=np.where(maxima&(radius<max(.35*r,.6*beam))&(clean>4),clean,-np.inf)
        if np.isfinite(possible).any():
            i=np.unravel_index(np.argmax(possible),possible.shape)
            self.core=(float(self.xx[i]),float(self.yy[i]))
        else:self.core=(cx,cy)
        self.external=[]
        allowed=maxima&(radius>1.4*r+beam)&(clean>5)
        ys,xs=np.nonzero(allowed)
        for j in np.argsort(clean[ys,xs])[-2:][::-1]:
            self.external.append((float(self.xx[ys[j],xs[j]]),float(self.yy[ys[j],xs[j]])))
        self.source_arrays=[self.psf(*p) for p in [self.core]+self.external]
        self.sources=np.column_stack([a[self.valid] for a in self.source_arrays])
        self.Sp=self.sources-self.Q@(self.Q.T@self.sources)
        self.c0,_=nnls(self.Sp,self.z,maxiter=150)
        self.null_residual=self.z-self.Sp@self.c0
        self.loss0=float(self.null_residual@self.null_residual)

    def psf(self,cx,cy):
        return np.exp(-.5*((self.xx-cx)**2+(self.yy-cy)**2)/(self.beam/2.354820045)**2)

    def evaluate(self,p,support=('full',0.,0.),flat=False):
        cx,cy,r,e1,e2,width=p
        if flat:
            dx,dy=self.xx-cx,self.yy-cy
            rho=np.sqrt(np.maximum((1+e1)*dx*dx+2*e2*dx*dy+(1-e1)*dy*dy,0))
            t=gaussian_filter(1/(1+np.exp(np.clip((rho-r)/.3,-80,80))),self.beam/2.354820045)
            t/=max(t.max(),1e-20)
        else:t=shell_template(self.xx,self.yy,cx,cy,r,e1,e2,width,self.beam,support)
        v=t[self.valid];tp=v-self.Q@(self.Q.T@v)
        S=np.column_stack((self.Sp,tp));coef,_=nnls(S,self.z,maxiter=150)
        res=self.z-S@coef;loss=float(res@res)
        return loss,coef,res,t

    def bounds(self):
        s=self.seed;r=s['r'];delta=max(.65*self.beam,.18*r)
        lo=[s['x']-delta,s['y']-delta,max(2.35,.5*r),-.34,-.34,.3]
        hi=[s['x']+delta,s['y']+delta,min(34.,1.65*r),.34,.34,max(1.,min(.5*r,2.2*self.beam))]
        return np.array(lo),np.array(hi)

    def optimize(self,p,support=('full',0.,0.),flat=False,nfev=None):
        lo,hi=self.bounds();p=np.clip(p,lo+1e-9,hi-1e-9)
        result=least_squares(lambda v:self.evaluate(v,support,flat)[2],p,bounds=(lo,hi),
                             max_nfev=nfev or self.cfg['geometry_nfev'],
                             diff_step=np.array([1e-4,1e-4,2e-3,2e-3,2e-3,2e-3]),
                             ftol=2e-3,xtol=2e-3,gtol=2e-3)
        return result.x,self.evaluate(result.x,support,flat)

    def score(self,p,fit,support):
        loss,coef,res,t=fit
        active=coef[:-1]>max(1e-10,float(np.max(coef))*1e-7)
        nuisance=np.column_stack((self.B,self.sources[:,active]))
        tv=t[self.valid];u=tv-nuisance@np.linalg.lstsq(nuisance,tv,rcond=None)[0]
        norm=float(u@u)
        if norm<1e-9:return None
        h=np.zeros(t.shape);h[self.valid]=u/norm
        variance=spectral_variance(h,self.power,self.shape)
        sigma=math.sqrt(max(variance,1e-20))
        snr=float(coef[-1]/sigma)
        beta=np.linalg.lstsq(self.B,self.y-self.sources@coef[:-1]-tv*coef[-1],rcond=None)[0]
        field=self.raw-self.plane
        background=np.zeros(t.shape);background[self.valid]=self.B@beta*self.noise
        core=np.zeros(t.shape);core[self.valid]=self.sources@coef[:-1]*self.noise
        retained=field-background-core
        cx,cy,r,e1,e2,width=p
        theta=np.linspace(0,2*np.pi,72,endpoint=False)
        rad=r/np.sqrt(1+e1*np.cos(2*theta)+e2*np.sin(2*theta))
        mode,phase,length=support
        select=np.ones(len(theta),bool)
        if mode=='arc':select=np.mod(theta-phase,2*np.pi)<=length
        if mode=='bilateral':select=np.minimum(np.abs(np.angle(np.exp(1j*(theta-phase)))),np.abs(np.angle(np.exp(1j*(theta-phase-np.pi)))))<=length/2
        at=map_coordinates(retained,[cy-self.y0+rad*np.sin(theta),cx-self.x0+rad*np.cos(theta)],order=1)
        rawwall=map_coordinates(field-background,[cy-self.y0+rad*np.sin(theta),cx-self.x0+rad*np.cos(theta)],order=1)
        observed=select&np.isfinite(at)
        angular_weights=np.where(observed,np.maximum(np.nan_to_num(at/self.noise,nan=0.),0),0.)
        participation=float(angular_weights.sum()**2/max(float(angular_weights@angular_weights),1e-20))
        effective_beams=float(2*np.pi*np.mean(rad)/self.beam*participation/len(theta))
        e=math.hypot(e1,e2);major=r/math.sqrt(1-e);minor=r/math.sqrt(1+e)
        return dict(x=float(cx),y=float(cy),r=float(r),major=float(major),minor=float(minor),
                    q=float(minor/major),width=float(width),e1=float(e1),e2=float(e2),
                    support_mode=mode,phase=float(phase),arc_angle=float(length),
                    coverage=1. if mode=='full' else float(length/(2*np.pi)*(2 if mode=='bilateral' else 1)),
                    shell_snr=snr,shell_amplitude=float(coef[-1]*self.noise),
                    amplitude_error=float(sigma*self.noise),noise=float(self.noise),
                    wall_excess=float(np.median(rawwall[select&np.isfinite(rawwall)])/self.noise)
                        if np.any(select&np.isfinite(rawwall)) else float('nan'),
                    joint_residual_wall=float(np.median(at[observed])/self.noise)
                        if observed.any() else float('nan'),
                    angular_participation=participation,effective_arc_beams=effective_beams,
                    source_count=len(self.external),central_amplitude=float(coef[0]*self.noise),
                    loss_base=self.loss0,loss_shell=loss,variance_factor=float(variance/np.sum(h*h)),
                    background_center_shift=float(beta[0]),**self.bginfo,
                    _parameters=list(map(float,p)),_coefficients=coef.tolist(),_origin=[self.x0,self.y0],
                    _model_shape=list(t.shape),_seed=self.seed,
                    _source_positions=[list(self.core)]+[list(p) for p in self.external],
                    _joint_background_beta=beta.tolist())

    def clump_loss(self):
        clean=np.nan_to_num((self.raw-self.plane)/self.noise)
        maxima=clean==maximum_filter(clean,size=max(3,round(.8*self.beam)))
        rr=np.hypot(self.xx-self.seed['x'],self.yy-self.seed['y'])
        allowed=maxima&(rr<max(1.4*self.seed['r'],2*self.beam))&(clean>2.5)
        yy,xx=np.nonzero(allowed);points=[]
        for j in np.argsort(clean[yy,xx])[::-1]:
            p=(float(self.xx[yy[j],xx[j]]),float(self.yy[yy[j],xx[j]]))
            if math.hypot(p[0]-self.core[0],p[1]-self.core[1])<.6*self.beam:continue
            if any(math.hypot(p[0]-q[0],p[1]-q[1])<.8*self.beam for q in points):continue
            points.append(p)
            if len(points)>=3:break
        if not points:return self.loss0
        def residual(v):
            vals=np.column_stack([self.psf(v[2*i],v[2*i+1])[self.valid] for i in range(len(points))])
            vals=vals-self.Q@(self.Q.T@vals)
            design=np.column_stack((self.Sp,vals));coef,_=nnls(design,self.z,maxiter=150)
            return self.z-design@coef
        p=np.asarray(points).ravel();delta=.6*self.beam
        result=least_squares(residual,p,bounds=(p-delta,p+delta),max_nfev=12,ftol=.003,xtol=.003)
        res=residual(result.x)
        return float(res@res)

def fit_deep(raw,bmap,smap,emission,seed,beam,power,cfg):
    local=LocalFit(raw,bmap,smap,emission,seed,beam,power,cfg)
    p=np.array([seed['x'],seed['y'],seed['r'],0.,0.,seed['width']])
    p,fit=local.optimize(p)
    full=local.score(p,fit,('full',0.,0.))
    if full is None:return None
    trials=[(full,p,fit,('full',0.,0.))]
    for length in (2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3):
        for phase in np.arange(0,2*np.pi,np.pi/4):
            support=('arc',float(phase),float(length));f=local.evaluate(p,support)
            result=local.score(p,f,support)
            if result:trials.append((result,p,f,support))
    for length in (np.pi/3,np.pi/2,2*np.pi/3):
        for phase in np.arange(0,np.pi,np.pi/4):
            support=('bilateral',float(phase),float(length));f=local.evaluate(p,support)
            result=local.score(p,f,support)
            if result:trials.append((result,p,f,support))
    best=max(trials,key=lambda t:t[0]['shell_snr'])
    if best[3][0]!='full':
        pp,ff=local.optimize(best[1],best[3],nfev=cfg['arc_nfev'])
        rr=local.score(pp,ff,best[3])
        if rr and rr['shell_snr']>best[0]['shell_snr']:best=(rr,pp,ff,best[3])
    result,p,fit,support=best
    _,flatfit=local.optimize(p,flat=True,nfev=14)
    clump=local.clump_loss()
    improvement=max(local.loss0-result['loss_shell'],1.)
    result.update(delta_flat=float(flatfit[0]-result['loss_shell']),
                  delta_clump=float(clump-result['loss_shell']),
                  flat_advantage=float((result['loss_shell']-flatfit[0])/max(flatfit[0],1.)),
                  flat_advantage_legacy=float((result['loss_shell']-flatfit[0])/improvement),
                  clump_advantage=float((result['loss_shell']-clump)/improvement),
                  proposal_score=seed['proposal_score'],proposal_branch=seed['branch'],
                  full_circle_snr=full['shell_snr'])
    fp=np.asarray(full['_parameters']);_,ff=local.optimize(fp,flat=True,nfev=14)
    gain=max(local.loss0-full['loss_shell'],1.)
    full.update(flat_advantage=float((full['loss_shell']-ff[0])/max(ff[0],1.)),
                flat_advantage_legacy=float((full['loss_shell']-ff[0])/gain),
                clump_advantage=float((full['loss_shell']-clump)/gain),
                proposal_score=seed['proposal_score'],proposal_branch=seed['branch'],
                full_circle_snr=full['shell_snr'])
    result['_full_fit']=dict(full)
    return result

def detect(raw,beam,cfg=None,envelopes=None,origin=(0,0)):
    if cfg is None:
        from settings import load_config
        config=load_config();cfg=config['joint']|dict(wall=config['wall'])
    start=time.time();b,s,z,emission,good=background_maps(raw,beam,cfg)
    if good.sum()<80:
        return dict(candidates=[],proposals=[],proposal_n=0,refined_n=0,
                    status='insufficient_background',elapsed_seconds=time.time()-start)
    power=spectrum_from_background(z,good,beam)
    proposals=propose(raw,z,beam,power,cfg)
    if envelopes is not None:
        proposals=[p for p in proposals if envelopes.matches(
            p['x']+origin[0],p['y']+origin[1],p['r'],beam,eta=.70,low=.30,high=2.)]
    coarse=[]
    for seed in proposals:
        local=LocalFit(raw,b,s,emission,seed,beam,power,cfg)
        best=None
        for factor in ((1.,.72) if seed['branch']=='intensity_island' else (1.,)):
            p=[seed['x'],seed['y'],max(2.4,seed['r']*factor),0.,0.,max(.5,seed['width']*factor)]
            fit=local.evaluate(p);score=local.score(p,fit,('full',0.,0.))
            if score and score['effective_arc_beams']>=1.8 and (best is None or score['shell_snr']>best[0]):best=(score['shell_snr'],seed|dict(r=p[2],width=p[-1]))
        if best:coarse.append(best)
    coarse.sort(key=lambda p:-p[0]);selected=[]
    buckets={}
    for score,seed in coarse:
        key=(int(seed['x']//16),int(seed['y']//16))
        buckets.setdefault(key,[]).append((score,seed))
    diversified=[]
    for depth in range(3):
        diversified.extend(sorted([v[depth] for v in buckets.values() if len(v)>depth],key=lambda p:-p[0]))
    for score,seed in diversified:
        if any(math.hypot(seed['x']-t['x'],seed['y']-t['y'])<.9*beam and
               abs(seed['r']-t['r'])<max(.7*beam,.2*seed['r']) for t in selected):continue
        selected.append(seed)
        if len(selected)>=cfg['max_refine']:break
    fitted=[]
    for seed in selected:
        result=fit_deep(raw,b,s,emission,seed,beam,power,cfg)
        if result:
            result['envelope_supported']=bool(envelopes is not None and envelopes.matches(
                result['x']+origin[0],result['y']+origin[1],result['r'],beam))
            result.update(measure_wall(raw,result,beam))
            fitted.append(result)
    fitted.sort(key=lambda p:-p['shell_snr']);kept=[]
    from filters.full_ring import select_model
    budget=cfg['full_ring']['max_per_window']
    updated=[]
    for row in fitted:
        chosen,attempted=select_model(raw,b,s,emission,row,beam,power,cfg,envelopes,origin,budget>0)
        budget-=int(attempted);updated.append(chosen)
    fitted=updated
    for row in fitted:
        if any(math.hypot(row['x']-t['x'],row['y']-t['y'])<.6*beam and abs(row['r']-t['r'])<max(.5*beam,.18*row['r']) for t in kept):continue
        kept.append(row)
    return dict(candidates=kept,proposals=proposals,
                proposal_n=len(proposals),refined_n=len(fitted),
                source_mask_fraction=float(emission.mean()),background_valid_fraction=float(good.mean()),
                elapsed_seconds=time.time()-start,
                limitations=['OLS fit; empirical stationary covariance for conditional amplitude errors.',
                             'Search scores are uncalibrated until compared with full local-search controls.',
                             'Finite proposal/refinement budget within each overlapping search window.'])


def grade(row,cfg):
    """Empirical tiers: unchanged weak disk gate; relative evidence for high tier."""
    def finite(value):
        return isinstance(value,(int,float,np.integer,np.floating)) and np.isfinite(value)
    keys=('shell_snr','effective_arc_beams','flat_advantage','clump_advantage',
          'background_center_shift','flat_advantage_legacy','wall_context_scatter')
    if not all(finite(row.get(key)) for key in keys):
        return dict(tier='excluded',reason='invalid_measurement')
    if not row.get('envelope_supported',False):
        return dict(tier='excluded',reason='no_bright_envelope')
    if row.get('original_wall_veto',False):
        return dict(tier='excluded',reason='no_original_wall')
    for key,minimum in cfg['wall']['weak'].items():
        value=row.get('wall_'+key)
        if not finite(value):return dict(tier='excluded',reason='invalid_observed_wall')
        if value<minimum:return dict(tier='excluded',reason='wall_'+key)
    for key,limit,minimum in (
        ('effective_arc_beams',cfg['weak_arc_beams'],True),
        ('flat_advantage_legacy',cfg['weak_flat_max'],False),
        ('clump_advantage',cfg['weak_clump_max'],False),
        ('shell_snr',cfg['weak_snr'],True)):
        if (row[key]<limit if minimum else row[key]>limit):
            return dict(tier='excluded',reason=key)
    shift=float(row['background_center_shift']/max(1.,row['wall_context_scatter']))
    original_keys=('fraction','inner_drop','outer_drop')
    original_ok=all(finite(row.get('original_wall_'+key)) and
        row['original_wall_'+key]>=cfg['wall']['high'][key] for key in original_keys)
    strong=(original_ok and row['shell_snr']>=cfg['high_snr'] and
            row['effective_arc_beams']>=cfg['high_arc_beams'] and
            row['flat_advantage']<=cfg['high_flat_max'] and
            row['clump_advantage']<=cfg['high_clump_max'] and
            abs(shift)<=cfg['high_background_shift_max'] and
            all(row['wall_'+key]>=minimum for key,minimum in cfg['wall']['high'].items()))
    return dict(tier='result' if strong else 'unconfirmed_result',
                reason='observed_wall_and_joint_evidence' if strong else 'requires_visual_review',
                background_context_shift=shift,original_high_wall_pass=bool(original_ok))
