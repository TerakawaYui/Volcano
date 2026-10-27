"""Galactic-WCS result images and fixed-angle radial evidence."""
import math
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from filters.joint_model import background_maps,shell_template
from filters.wall import components as nuisance_fields,angular_support


def sky_axes(fig,position,wcs):
    ax=fig.add_subplot(*position,projection=wcs)
    if any(str(c).upper().startswith('GLON') for c in wcs.wcs.ctype):
        coords=ax.coords
    else:
        for c in ax.coords:
            c.set_ticks_visible(False);c.set_ticklabel_visible(False);c.set_axislabel('')
        coords=ax.get_coords_overlay('galactic')
    for c in coords:
        c.set_format_unit(u.deg);c.set_major_formatter('d.ddd');c.set_ticks(number=4)
        c.set_ticklabel(size=8,exclude_overlapping=True)
    coords[0].set_coord_type('longitude',coord_wrap=360*u.deg)
    coords[0].set_major_formatter('d.ddd')
    coords[0].set_ticks_position('b');coords[0].set_ticklabel_position('b');coords[0].set_axislabel_position('b')
    coords[1].set_ticks_position('l');coords[1].set_ticklabel_position('l');coords[1].set_axislabel_position('l')
    coords[0].set_axislabel('Galactic longitude l [deg]')
    coords[1].set_axislabel('Galactic latitude b [deg]')
    coords.grid(color='white',alpha=.2,linestyle=':')
    return ax


def components(raw,row,beam):
    x0,y0=row['_origin'];ny,nx=row['_model_shape'];seed=row['_seed'];noise=row['noise']
    yy,xx=np.indices((ny,nx));xx+=x0;yy+=y0
    half=math.ceil(max(1.8*seed['r'],seed['r']+3*beam))
    modulation=(row['angular_modulation'],row['angular_phase']) if 'angular_modulation' in row else None
    shell=noise*row['_coefficients'][-1]*shell_template(xx,yy,*row['_parameters'],beam,
        (row['support_mode'],row['phase'],row['arc_angle']),modulation=modulation)
    _,retained=nuisance_fields(raw,row,beam)
    return retained*noise,shell,np.hypot(xx-seed['x'],yy-seed['y'])<=half


def radial_samples(values,fit,angles,radii):
    x0,y0=fit['_origin']
    rr=radii[:,None]/np.sqrt(1+fit['e1']*np.cos(2*angles)+fit['e2']*np.sin(2*angles))[None,:]
    coords=[fit['y']-y0+rr*np.sin(angles),fit['x']-x0+rr*np.cos(angles)]
    return map_coordinates(values,coords,order=1,mode='constant',cval=np.nan)


def save_figure(destination,raw,fit,beam,row,origin,cfg,wcs):
    cx,cy=fit['x'],fit['y'];noise=fit['noise'];half=max(24,math.ceil(2.5*fit['major']))
    b,s,_,_,_=background_maps(raw,beam,cfg)
    xa,xb=max(0,round(cx)-half),min(raw.shape[1],round(cx)+half+1)
    ya,yb=max(0,round(cy)-half),min(raw.shape[0],round(cy)+half+1)
    window_wcs=wcs.slice((slice(origin[1],None),slice(origin[0],None)))
    pixel_arcsec=float(np.sqrt(np.prod(proj_plane_pixel_scales(wcs)))*3600)
    fig=plt.figure(figsize=(11,9.5),dpi=120)
    axes=[sky_axes(fig,(2,2,i),window_wcs) for i in (1,2,3)]
    profile=fig.add_subplot(2,2,4)
    bg=float(np.nanmedian(b[ya:yb,xa:xb]));finite=raw[ya:yb,xa:xb]
    high=max(bg+8*noise,float(np.nanpercentile(finite,99.5)))
    axes[0].imshow(raw,origin='lower',cmap='gray',vmin=bg-2*noise,vmax=high)
    axes[0].set_title('Original first-plane intensity',fontsize=10)
    z=(raw-b)/s;upper=max(8.,float(np.nanpercentile(z[ya:yb,xa:xb],99.5)))
    axes[1].imshow(z,origin='lower',cmap='gray',vmin=-2,vmax=upper)
    axes[1].set_title('Cyan: fitted support; orange: observed wall',fontsize=10)
    theta=np.linspace(0,2*np.pi,600);radii=fit['r']/np.sqrt(1+fit['e1']*np.cos(2*theta)+fit['e2']*np.sin(2*theta))
    selected=angular_support(theta,fit)
    axes[1].plot(np.where(selected,cx+radii*np.cos(theta),np.nan),np.where(selected,cy+radii*np.sin(theta),np.nan),c='cyan',lw=.7,ls=':')
    aa=np.asarray(fit['_wall_angles']);good=np.asarray(fit['_wall_good'],bool)
    aa=np.r_[aa,aa[0]+2*np.pi];good=np.r_[good,good[0]]
    rr=fit['r']/np.sqrt(1+fit['e1']*np.cos(2*aa)+fit['e2']*np.sin(2*aa))
    axes[1].plot(np.where(good,cx+rr*np.cos(aa),np.nan),np.where(good,cy+rr*np.sin(aa),np.nan),c='orange',lw=1.6)
    for sx,sy in fit['_source_positions']:axes[1].plot(sx,sy,'rx',ms=5)
    retained,shell,mask=components(raw,fit,beam);x0,y0=fit['_origin']
    extent=(x0-.5,x0+retained.shape[1]-.5,y0-.5,y0+retained.shape[0]-.5)
    axes[2].imshow(np.where(mask,retained/noise,np.nan),origin='lower',cmap='gray',extent=extent,
        vmin=-2,vmax=max(8.,float(np.nanpercentile(retained[mask]/noise,99.5))))
    axes[2].set_title('Data minus joint background and point sources',fontsize=10)
    for ax in axes:
        ax.set_xlim(xa-.5,xb-.5);ax.set_ylim(ya-.5,yb-.5)
        ax.plot([xa+3,xa+3+beam],[ya+3,ya+3],c='cyan',lw=3)
    angles=np.linspace(0,2*np.pi,96,endpoint=False);angles=angles[angular_support(angles,fit)]
    grid=np.linspace(.1,1.7*fit['r'],60)
    for a,label,color in ((retained,'Data after nuisance subtraction','C0'),(shell,'Ring model','C1')):
        profile.plot(grid*pixel_arcsec,np.nanmedian(radial_samples(a/noise,fit,angles,grid),axis=1),label=label,c=color)
    profile.axvline(fit['r']*pixel_arcsec,c='gray',ls=':');profile.legend(fontsize=8)
    profile.set_xlabel('Elliptical radius [arcsec]');profile.set_ylabel('Intensity / local pixel scatter')
    fig.suptitle(f"rank {row['rank']} | {row['tier']} | {fit['support_mode']} | l={row['fitted_gal_lon_deg']:.4f}, b={row['fitted_gal_lat_deg']:.4f}\n"
        f"conditional SNR={fit['shell_snr']:.2f}; original wall={fit.get('original_wall_fraction',0):.0%}; observed wall={fit['wall_fraction']:.0%}; contiguous={fit['wall_run_beams']:.2f} beams",fontsize=10)
    fig.subplots_adjust(left=.10,right=.97,bottom=.08,top=.88,wspace=.38,hspace=.38)
    destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination);plt.close(fig)
    save_directions(destination.with_name(destination.stem+'_directions.png'),raw,fit,beam,row,pixel_arcsec)


def save_directions(destination,raw,fit,beam,row,pixel_arcsec):
    before,after=nuisance_fields(raw,fit,beam)
    angles=np.linspace(0,2*np.pi,96,endpoint=False);grid=np.linspace(.1,1.7*fit['r'],60)
    data=[radial_samples(a,fit,angles,grid) for a in (before,after)]
    fig,axs=plt.subplots(2,2,figsize=(11,8),dpi=120)
    for a,label in zip(data,('Before PSF subtraction','After PSF subtraction')):
        axs[0,0].plot(grid*pixel_arcsec,np.nanmedian(a,axis=1),label=label)
    axs[0,0].legend(fontsize=8);axs[0,0].set_title('Whole-circle radial medians')
    for k in range(8):axs[0,1].plot(grid*pixel_arcsec,np.nanmedian(data[0][:,k*12:(k+1)*12],axis=1),label=f'{45*k}-{45*(k+1)} deg')
    axs[0,1].legend(fontsize=6,ncol=2);axs[0,1].set_title('8 fixed sectors before point-source subtraction')
    for ax in axs[0]:ax.set_xlabel('Elliptical radius [arcsec]');ax.set_ylabel('Intensity / local pixel scatter')
    finite=data[0][np.isfinite(data[0])];upper=max(8.,float(np.percentile(finite,99.5))) if finite.size else 8.
    for ax,a,title in zip(axs[1],data,('Before point-source subtraction','After point-source subtraction')):
        im=ax.imshow(a,origin='lower',aspect='auto',extent=(0,360,grid[0]*pixel_arcsec,grid[-1]*pixel_arcsec),vmin=-2,vmax=upper,cmap='gray')
        ax.axhline(fit['r']*pixel_arcsec,c='orange',ls=':');ax.set_title(title)
        ax.set_xlabel('Angle in native image plane [deg]');ax.set_ylabel('Elliptical radius [arcsec]')
        fig.colorbar(im,ax=ax,label='Intensity / local pixel scatter')
    fig.suptitle(f"l={row['fitted_gal_lon_deg']:.4f}, b={row['fitted_gal_lat_deg']:.4f} | rank {row['rank']} | directions are correlated")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(destination);plt.close(fig)
