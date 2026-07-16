import os, numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

OUT='/mnt/data/prd_kernel_final/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12, 'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

def hessian(f,x,h=1e-4):
    x=np.asarray(x,float); n=len(x); H=np.zeros((n,n)); fx=f(x)
    for i in range(n):
        ei=np.zeros(n); ei[i]=h
        H[i,i]=(f(x+ei)-2*fx+f(x-ei))/h**2
        for j in range(i+1,n):
            ej=np.zeros(n); ej[j]=h
            H[i,j]=H[j,i]=(f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*h*h)
    return H

def U_s2cp(x,branch,A=1.,B=1.,C=.05,q=1.):
    u,v=x; a1=np.exp(u); a2=np.exp(v); m,n=branch; N=m*(n*n-1)/8
    return (-A*(2/a1**4/a2**4+24/a1**2/a2**6)
            +B*(m*m/(2*a1**6*a2**4)+q*n*n/(a1**2*a2**8))
            -C*N*N/(a1**2*a2**4))
G24=np.array([[4.,4.],[4.,12.]])
branches=[(3,3),(1,5)]
results={}
for br in branches:
    r=minimize(lambda z: U_s2cp(z,br),[0,0],bounds=[(-1.6,1.6),(-1.6,1.6)],method='L-BFGS-B')
    H=hessian(lambda z:U_s2cp(z,br),r.x)
    masses=np.sort(np.real(np.linalg.eigvals(np.linalg.solve(G24,H))))
    results[str(br)]={'x':r.x,'radii':np.exp(r.x),'U':r.fun,'m2':masses}

def U_s3(x,f=(3,3),A=1.,B=1.,D=.2):
    u,v=x; b1=np.exp(u); b2=np.exp(v); f1,f2=f
    return (-A*(6/b1**5/b2**3+6/b1**3/b2**5)
            +B*(f1*f1/(2*b1**9*b2**3)+f2*f2/(2*b1**3*b2**9))
            +D/(b1**3*b2**3))
r=minimize(lambda z:U_s3(z),[0,0],bounds=[(-1.6,1.6),(-1.6,1.6)],method='L-BFGS-B')
G33=np.array([[7.5,4.5],[4.5,7.5]])
Hh=hessian(lambda z:U_s3(z),r.x)
masses=np.sort(np.real(np.linalg.eigvals(np.linalg.solve(G33,Hh))))
results['S3xS3']={'x':r.x,'radii':np.exp(r.x),'U':r.fun,'m2':masses}

# Figs 1-2: separate, ~20% larger than previous half-panels
for idx,br in enumerate(branches, start=1):
    x0=results[str(br)]['x']
    us=np.linspace(x0[0]-.65,x0[0]+.65,220); vs=np.linspace(x0[1]-.65,x0[1]+.65,220)
    UU,VV=np.meshgrid(us,vs); ZZ=np.vectorize(lambda u,v:U_s2cp((u,v),br))(UU,VV)
    zmin=np.nanmin(ZZ); levels=zmin+np.geomspace(.15,max(.2,np.nanpercentile(ZZ-zmin,92)),18)
    fig,ax=plt.subplots(figsize=(6.2,4.8),constrained_layout=True)
    cs=ax.contour(UU,VV,ZZ,levels=levels,linewidths=1.0)
    ax.plot(x0[0],x0[1],'o',ms=7,label='stable extremum')
    ax.set_title(rf'$S^2\times\mathbb{{CP}}^2$: flux branch $({br[0]},{br[1]})$')
    ax.set_xlabel(r'$u=\ln a_1$'); ax.set_ylabel(r'$v=\ln a_2$')
    ax.legend(frameon=False,loc='best')
    fig.savefig(f'{OUT}/benchmark_{br[0]}{br[1]}.pdf',bbox_inches='tight')
    fig.savefig(f'{OUT}/benchmark_{br[0]}{br[1]}.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

# Fig 3: observational amplitude relation, fonts 50% larger than previous
As=2.10e-9; rr=np.logspace(-4,np.log10(.05),300); HH=np.sqrt(np.pi*As*rr)/4
with plt.rc_context({'font.size':15,'axes.titlesize':16,'axes.labelsize':17,'legend.fontsize':13,'xtick.labelsize':14,'ytick.labelsize':14}):
    fig,ax=plt.subplots(figsize=(6.1,4.6),constrained_layout=True)
    ax.loglog(rr,HH,lw=2.3,label=r'$A_s=2.10\times10^{-9}$')
    ax.axvspan(.036,.05,alpha=.12,label=r'excluded by $r_{0.05}<0.036$')
    ax.axvline(.036,ls='--',lw=1.4)
    for rv,off in [(1e-3,(8,8)),(1e-2,(8,-22)),(.036,(-78,10))]:
        hv=np.sqrt(np.pi*As*rv)/4
        ax.plot(rv,hv,'o',ms=6)
        ax.annotate(rf'$r={rv:g}$',xy=(rv,hv),xytext=off,textcoords='offset points',fontsize=13)
    ax.set_xlabel(r'tensor-to-scalar ratio $r$'); ax.set_ylabel(r'$H_*/M_{\rm P}$')
    ax.set_title('CMB amplitude constraint')
    ax.legend(frameon=False,loc='upper left')
    fig.savefig(f'{OUT}/amplitude_curve.pdf',bbox_inches='tight')
    fig.savefig(f'{OUT}/amplitude_curve.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

# Fig 4: Schur response, fonts 50% larger
H2=10.; gamma=.4
models=[('(3,3)',results['(3, 3)']['m2'][0]),('(1,5)',results['(1, 5)']['m2'][0]),(r'$S^3\times S^3$',results['S3xS3']['m2'][0])]
x=np.logspace(-2,1.2,360)
with plt.rc_context({'font.size':15,'axes.titlesize':16,'axes.labelsize':17,'legend.fontsize':13,'xtick.labelsize':14,'ytick.labelsize':14}):
    fig,ax=plt.subplots(figsize=(6.2,4.7),constrained_layout=True)
    for label,m2 in models:
        mu2=m2/H2; R=1/(1-gamma**2/(mu2+x**2))
        ax.semilogx(x,R,label=label,lw=2.2)
    ax.axhline(1,ls='--',lw=1.2)
    ax.set_xlabel(r'$k/k_*$'); ax.set_ylabel(r'$\Delta_\zeta^2/\Delta_{\zeta,0}^2$')
    ax.set_title('One-mode Schur-complement response')
    ax.legend(frameon=False,loc='upper right')
    ax.set_ylim(.98,1.18); ax.margins(x=.03)
    fig.savefig(f'{OUT}/schur_corrections.pdf',bbox_inches='tight')
    fig.savefig(f'{OUT}/schur_corrections.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

# Fig 5: comparison to actual Planck constraints on primordial scalar spectrum
kp=.05; ns=.9649; sns=.0042; slnAs=.014
k=np.logspace(-4,np.log10(.2),500)
P0=As*(k/kp)**(ns-1)
Plo=np.exp(np.log(As)-slnAs)*(k/kp)**((ns-sns)-1)
Phi=np.exp(np.log(As)+slnAs)*(k/kp)**((ns+sns)-1)
# Ensure pointwise envelope ordering away from pivot
lower=np.minimum(Plo,Phi); upper=np.maximum(Plo,Phi)
curves={}; verdict={}
for label,m2 in models:
    mu2=m2/H2
    xx=k/kp
    R=1/(1-gamma**2/(mu2+xx**2))
    Rp=1/(1-gamma**2/(mu2+1.0))
    P=P0*R/Rp  # all normalized to measured A_s at pivot
    curves[label]=P
    inside=np.logical_and(P>=lower,P<=upper)
    verdict[label]=100*np.mean(inside)

fig,ax=plt.subplots(figsize=(7.0,4.8),constrained_layout=True)
ax.fill_between(k,lower,upper,alpha=.22,label=r'Planck 2018 $1\sigma$ parameter envelope')
ax.loglog(k,P0,lw=2.3,label='Planck best-fit power law')
for label,P in curves.items(): ax.loglog(k,P,lw=1.9,label=label+' benchmark')
ax.axvline(kp,ls=':',lw=1.2)
ax.text(kp*1.08,lower.min()*1.03,r'$k_*=0.05\ {\rm Mpc}^{-1}$',fontsize=10,rotation=90,va='bottom')
ax.set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$'); ax.set_ylabel(r'$\Delta_\zeta^2(k)$')
ax.set_title('Benchmark compactifications versus the observed scalar spectrum')
ax.legend(frameon=False,fontsize=9,loc='lower left')
ax.set_xlim(k.min(),k.max())
fig.savefig(f'{OUT}/observational_comparison.pdf',bbox_inches='tight')
fig.savefig(f'{OUT}/observational_comparison.png',dpi=240,bbox_inches='tight')
plt.close(fig)

with open('/mnt/data/prd_kernel_final/benchmark_values.tex','w') as f:
    for key,val in results.items(): f.write(f'% {key}: radii={val["radii"]}, U={val["U"]}, m2={val["m2"]}\n')
    f.write('% fraction of plotted k grid inside Planck 1sigma parameter envelope\n')
    for key,val in verdict.items(): f.write(f'% {key}: {val:.1f} percent\n')
print(results); print('verdict',verdict)
