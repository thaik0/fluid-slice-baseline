"""Physical-unit diagnostics of xy cuts, not a 3D shell spectrum."""
import numpy as np


def metrics(a, patch, spacing):
    a = np.asarray(a,dtype='float64')
    if a.ndim != 4 or not np.isfinite(a).all() or spacing<=0: raise ValueError('Invalid fields/spacing')
    n,c,h,w = a.shape
    f = a-a.mean((-2,-1),keepdims=True)
    # Full FFT with Parseval normalization: sum spectrum == fluctuation energy.
    power = 0.5*(abs(np.fft.fft2(f,axes=(-2,-1)))**2).sum(1).mean(0)/(h*w)**2
    ky = np.fft.fftfreq(h,d=spacing)*2*np.pi
    kx = np.fft.fftfreq(w,d=spacing)*2*np.pi
    dk = min(2*np.pi/(h*spacing),2*np.pi/(w*spacing))
    bins = np.rint(np.hypot(ky[:,None],kx[None,:])/dk).astype(int)
    spectrum = np.bincount(bins.ravel(),weights=power.ravel())
    jumps=[]; wraps=[]
    for ax,size in [(-2,h),(-1,w)]:
        d = np.mean(np.diff(a,axis=ax)**2,axis=tuple(i for i in range(4) if i != ax%4))
        phase = [float(d[np.arange(len(d))%patch==p].mean()) for p in range(patch)]
        jumps.append(max(phase)/(float(d.mean())+1e-30))
        wraps.append(float(np.mean((np.take(a,0,axis=ax)-np.take(a,-1,axis=ax))**2))/(float(d.mean())+1e-30))
    def derivative(x,axis):
        return (np.roll(x,-1,axis)-np.roll(x,1,axis))/(2*spacing)
    div = derivative(a[:,0],-1)+derivative(a[:,1],-2)
    omega = derivative(a[:,1],-1)-derivative(a[:,0],-2)
    s2 = {str(r):float(np.mean((np.roll(a,r,-1)-a)**2)) for r in [1,2,4,8,16] if r<w//2}
    return dict(mean=a.mean((0,2,3)).tolist(),std=a.std((0,2,3)).tolist(),energy=float(.5*np.mean(np.sum(a*a,axis=1))),fluctuation_energy=float(.5*np.mean(np.sum(f*f,axis=1))),spectrum=spectrum.tolist(),k=(np.arange(len(spectrum))*dk).tolist(),low_band_energy=float(spectrum[1:5].sum()),high_band_energy=float(spectrum[max(1,min(h,w)//4):].sum()),max_phase_seam_ratio=jumps,periodic_wrap_ratio=wraps,in_plane_divergence_rms=float(np.sqrt(np.mean(div**2))),normal_vorticity_rms=float(np.sqrt(np.mean(omega**2))),component_averaged_x_structure_function=s2)
