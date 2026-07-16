import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

OUT='/mnt/data/prd_kernel_revised/figures'

# ---------- common helpers ----------
def hessian(f,x,h=1e-4):
    x=np.asarray(x,float); n=len(x); H=np.zeros((n,n)); fx=f(x)
    for i in range(n):
        ei=np.zeros(n); ei[i]=h
        H[i,i]=(f(x+ei)-2*fx+f(x-ei))/h**2
        for j in range(i+1,n):
            ej=np.zeros(n); ej[j]=h
            H[i,j]=H[j,i]=(f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*h*h)
    return H

# ---------- S2 x CP2 benchmark ----------
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

# ---------- S3 x S3 benchmark ----------
def U_s3(x,f=(3,3),A=1.,B=1.,D=.2):
    u,v=x; b1=np.exp(u); b2=np.exp(v); f1,f2=f
    return (-A*(6/b1**5/b2**3+6/b1**3/b2**5)
            +B*(f1*f1/(2*b1**9*b2**3)+f2*f2/(2*b1**3*b2**9))
            +D/(b1**3*b2**3))
r=minimize(lambda z:U_s3(z),[0,0],bounds=[(-1.6,1.6),(-1.6,1.6)],method='L-BFGS-B')
G33=np.array([[7.5,4.5],[4.5,7.5]])
H=hessian(lambda z:U_s3(z),r.x)
masses=np.sort(np.real(np.linalg.eigvals(np.linalg.solve(G33,H))))
results['S3xS3']={'x':r.x,'radii':np.exp(r.x),'U':r.fun,'m2':masses}

# ---------- Figure 1: potential contours ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),constrained_layout=True)
for ax,br in zip(axs,branches):
    x0=results[str(br)]['x']
    us=np.linspace(x0[0]-.65,x0[0]+.65,180)
    vs=np.linspace(x0[1]-.65,x0[1]+.65,180)
    UU,VV=np.meshgrid(us,vs)
    ZZ=np.vectorize(lambda u,v:U_s2cp((u,v),br))(UU,VV)
    zmin=np.nanmin(ZZ)
    levels=zmin+np.geomspace(.15,max(.2,np.nanpercentile(ZZ-zmin,92)),16)
    ax.contour(UU,VV,ZZ,levels=levels,linewidths=.8)
    ax.plot(x0[0],x0[1],'o',ms=5)
    ax.set_title(rf'$({br[0]},{br[1]})$ branch')
    ax.set_xlabel(r'$u=\ln a_1$')
    ax.set_ylabel(r'$v=\ln a_2$')
fig.suptitle(r'Benchmark $S^2\times\mathbb{CP}^2$ potentials')
fig.savefig(f'{OUT}/benchmark_potentials.pdf',bbox_inches='tight')
fig.savefig(f'{OUT}/benchmark_potentials.png',dpi=220,bbox_inches='tight')
plt.close(fig)

# ---------- Figure 2: observational amplitude relation ----------
As=2.1e-9
rr=np.logspace(-4,np.log10(.036),240)
H=np.sqrt(np.pi*As*rr)/4 # unreduced M_P
fig,ax=plt.subplots(figsize=(4.8,3.35),constrained_layout=True)
ax.loglog(rr,H,lw=1.7)
ax.axvline(.036,ls='--',lw=1)
for rv in [1e-3,1e-2,.036]:
    hv=np.sqrt(np.pi*As*rv)/4
    ax.plot(rv,hv,'o',ms=4)
    ax.annotate(rf'$r={rv:g}$',xy=(rv,hv),xytext=(5,5),textcoords='offset points',fontsize=8)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$H_*/M_{\rm P}$')
ax.set_title(r'Observed $A_s$ fixes the $H_*$--$r$ curve')
ax.text(.00013,1.3e-6,r'$A_s=2.1\times10^{-9}$',fontsize=9)
fig.savefig(f'{OUT}/amplitude_curve.pdf',bbox_inches='tight')
fig.savefig(f'{OUT}/amplitude_curve.png',dpi=220,bbox_inches='tight')
plt.close(fig)

# ---------- Figure 3: illustrative Schur response ----------
# Common H^2=10 in the dimensionless benchmark; fixed dimensionless mixing gamma=0.4.
H2=10.; gamma=.4
models=[('(3,3)',results['(3, 3)']['m2'][0]),('(1,5)',results['(1, 5)']['m2'][0]),(r'$S^3\times S^3$',results['S3xS3']['m2'][0])]
x=np.logspace(-2,1.2,300)
fig,ax=plt.subplots(figsize=(5.0,3.45),constrained_layout=True)
for label,m2 in models:
    mu2=m2/H2
    R=1/(1-gamma**2/(mu2+x**2))
    ax.semilogx(x,R,label=label,lw=1.5)
ax.axhline(1,ls='--',lw=.8)
ax.set_xlabel(r'$k/k_*$')
ax.set_ylabel(r'$\Delta_\zeta^2/\Delta_{\zeta,0}^2$')
ax.set_title('Illustrative one-mode Schur correction')
ax.legend(frameon=False,fontsize=8)
ax.set_ylim(.98,1.18)
fig.savefig(f'{OUT}/schur_corrections.pdf',bbox_inches='tight')
fig.savefig(f'{OUT}/schur_corrections.png',dpi=220,bbox_inches='tight')
plt.close(fig)

# ---------- write numerical data ----------
with open('/mnt/data/prd_kernel_revised/benchmark_values.tex','w') as f:
    for key,val in results.items():
        f.write(f'% {key}: radii={val["radii"]}, U={val["U"]}, m2={val["m2"]}\n')
print(results)
