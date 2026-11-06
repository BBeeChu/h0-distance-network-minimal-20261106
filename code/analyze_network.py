#!/usr/bin/env python3
"""Deterministic joint GLS inference from supplied literal observations; no imputation."""
import ast, json, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import chi2
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'outputs'; F=ROOT/'report/images'
O.mkdir(exist_ok=True); F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180})
def dump(name,x):
    (O/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load():
    tree=ast.parse((ROOT/'data/H0DN_MinimalDataset.txt').read_text()); d={}
    for node in tree.body:
        if isinstance(node,ast.Assign): d[node.targets[0].id]=ast.literal_eval(node.value)
    return d
D=load(); K=5/np.log(10)
def distance_term(z,law='cosmography'):
    z=np.asarray(z); q=-.55; j=1.
    if law=='linear': f=np.ones_like(z); deriv=1/z
    elif law=='exact_LCDM':
        # D_L * H0 in km/s; flat LCDM with Omega_m=0.3.
        v=np.array([(1+t)*quad(lambda u:1/np.sqrt(.3*(1+u)**3+.7),0,t)[0] for t in z])
        integ=v/(1+z); dv=integ+(1+z)/np.sqrt(.3*(1+z)**3+.7)
        return 5*np.log10(D['c_km']*v)+25,K*dv/v
    else:
        f=1+(1-q)*z/2-(1-q-3*q*q+j)*z*z/6
        deriv=1/z+((1-q)/2-(1-q-3*q*q+j)*z/3)/f
    return 5*np.log10(D['c_km']*z*f)+25,K*deriv

def fit(name='Default',method=None,anchor=None,drop_host=None,with_sbf=True,covmode='full',pv=250,zmin=0,law='cosmography',scatter=0):
    prim=[r for r in D['host_measurements'] if (method is None or r[1]==method) and (anchor is None or r[2]==anchor)]
    hosts=list(dict.fromkeys(r[0] for r in prim)); sns=[r for r in D['sneia_calibrators'] if r[0] in hosts and r[0]!=drop_host]
    flow=[r for r in D['hubble_flow_sneia'] if r[0]>zmin]
    groups=list(dict.fromkeys(D['host_group'].values())) if with_sbf else []
    pars=['mu:'+h for h in hosts]+['M_Ia','h']+(['M_SBF']+['mu_group:'+g for g in groups] if with_sbf else [])
    pmap={p:i for i,p in enumerate(pars)}; rows=[]; ys=[]; variances=[]; labels=[]; types=[]
    def add(y,v,terms,label,typ):
        row=np.zeros(len(pars))
        for p,c in terms.items(): row[pmap[p]]=c
        rows.append(row);ys.append(y);variances.append(v);labels.append(label);types.append(typ)
    for h,m,a,mu,e in prim: add(mu,e*e,{'mu:'+h:1},h+' '+m+' '+a,'primary')
    for h,m,e in sns: add(m,e*e+scatter*scatter,{'mu:'+h:1,'M_Ia':1},h,'SN calibrator')
    for z,m,e,v in flow:
        term,deriv=distance_term([z],law); var=e*e+(deriv[0]*pv/D['c_km'])**2+scatter*scatter
        add(m-term[0],var,{'M_Ia':1,'h':-1},f'z={z:.3f}','SN flow')
    if with_sbf:
        for h,m,e in D['sbf_calibrators']:
            add(m,e*e+D['depth_scatter']**2,{'M_SBF':1,'mu_group:'+D['host_group'][h]:1},h,'SBF calibrator')
        for z,m,e,v in D['hubble_flow_sbf']:
            term,deriv=distance_term([z],law); var=e*e+(deriv[0]*pv/D['c_km'])**2
            add(m-term[0],var,{'M_SBF':1,'h':-1},f'SBF z={z:.3f}','SBF flow')
    X=np.array(rows);y=np.array(ys);n=len(y);nd=len(prim)
    comps={'independent':np.diag(variances)}; vectors=[]
    for a in dict.fromkeys(r[2] for r in prim):
        v=np.zeros(n); v[:nd]=[D['anchors'][a]['err'] if r[2]==a else 0 for r in prim]
        comps['anchor:'+a]=np.outer(v,v);vectors.append(v)
    for m,a in dict.fromkeys((r[1],r[2]) for r in prim):
        v=np.zeros(n);v[:nd]=[D['method_anchor_err'].get((m,a),0) if (r[1],r[2])==(m,a) else 0 for r in prim]
        comps['method:'+m+':'+a]=np.outer(v,v);vectors.append(v)
    if covmode=='none': comps={k:(v if k=='independent' else np.zeros_like(v)) for k,v in comps.items()}
    if covmode=='diagonal': comps={k:np.diag(np.diag(v)) for k,v in comps.items()}
    C=sum(comps.values());L=np.linalg.cholesky(C);A=solve_triangular(L,X,lower=True);b=solve_triangular(L,y,lower=True)
    theta,_,rank,s=np.linalg.lstsq(A,b,rcond=None)
    if rank<len(pars): raise ValueError('Rank deficient selected fit '+name)
    V=np.linalg.inv(A.T@A);res=y-X@theta;white=solve_triangular(L,res,lower=True)
    ih=pmap['h'];h=theta[ih];sh=np.sqrt(V[ih,ih]);H=10**(h/5);sig=H*sh/K
    W=V@A.T@solve_triangular(L,np.eye(n),lower=True);w=W[ih]
    dof=n-rank;ch=float(white@white)
    result=dict(variant=name,H0=float(H),sigma_H0=float(sig),lower68=float(10**((h-sh)/5)),upper68=float(10**((h+sh)/5)),precision_percent=float(sig/H*100),h=float(h),sigma_h=float(sh),M_Ia=float(theta[pmap['M_Ia']]),sigma_M_Ia=float(np.sqrt(V[pmap['M_Ia'],pmap['M_Ia']])),chi2=ch,dof=int(dof),reduced_chi2=ch/dof,p_value=float(chi2.sf(ch,dof)),n=n,rank=int(rank),n_primary=len(prim),n_calibrators=len(sns),n_flow=len(flow),law=law,pv_km_s=pv,intrinsic_scatter_mag=scatter,covariance=covmode)
    return dict(result=result,X=X,y=y,C=C,L=L,A=A,b=b,theta=theta,V=V,res=res,white=white,pars=pars,labels=labels,types=types,components=comps,vectors=vectors,prim=prim,W=W)

def pilot():
    b=fit();dump('pilot.json',b['result']);print(json.dumps(b['result'],indent=2))

def main():
    b=fit();r=b['result']; configs=[('Default',{}),('No SBF',dict(with_sbf=False)),('Diagonal covariance',dict(covmode='diagonal')),('No shared errors',dict(covmode='none')),('Cepheid only',dict(method='Cepheid')),('TRGB only',dict(method='TRGB')),('N4258 only',dict(anchor='N4258')),('LMC only',dict(anchor='LMC')),('PV 0 km/s',dict(pv=0)),('PV 150 km/s',dict(pv=150)),('PV 400 km/s',dict(pv=400)),('z > 0.04',dict(zmin=.04)),('Linear distance law',dict(law='linear')),('Exact flat LCDM',dict(law='exact_LCDM')),('SN scatter 0.10 mag',dict(scatter=.1)),('SN scatter 0.30 mag',dict(scatter=.3))]
    fits=[fit(n,**cfg) for n,cfg in configs];tab=pd.DataFrame([v['result'] for v in fits]);tab.to_csv(O/'variants.csv',index=False)
    loo=[fit('Omit '+row[0],drop_host=row[0]) for row in D['sneia_calibrators']];pd.DataFrame([v['result'] for v in loo]).to_csv(O/'leave_one_out.csv',index=False)
    np.savez_compressed(O/'network_matrices.npz',X=b['X'],y=b['y'],C=b['C'],theta=b['theta'],parameter_covariance=b['V'],residuals=b['res'],whitened_residuals=b['white'],parameters=np.array(b['pars']),labels=np.array(b['labels']),row_types=np.array(b['types']))
    pd.DataFrame(dict(row=b['labels'],type=b['types'],y=b['y'],model=b['X']@b['theta'],residual_mag=b['res'],marginal_error_mag=np.sqrt(np.diag(b['C'])),whitened_residual=b['white'])).to_csv(O/'residuals.csv',index=False)
    # Independent GLS normal equations and analytic two-stage calibration/intercept.
    normal=np.linalg.solve(b['X'].T@np.linalg.solve(b['C'],b['X']),b['X'].T@np.linalg.solve(b['C'],b['y']))
    n=len(b['prim']);hosts=[p[3:] for p in b['pars'] if p.startswith('mu:')];Xp=b['X'][:n,:len(hosts)];Cp=b['C'][:n,:n];Vp=np.linalg.inv(Xp.T@np.linalg.solve(Cp,Xp));mp=Vp@Xp.T@np.linalg.solve(Cp,b['y'][:n])
    mcal=np.array([r[1] for r in D['sneia_calibrators']]);inds=[hosts.index(r[0]) for r in D['sneia_calibrators']];Cal=Vp[np.ix_(inds,inds)]+np.diag([r[2]**2 for r in D['sneia_calibrators']]);one=np.ones(len(inds));vm=1/(one@np.linalg.solve(Cal,one));M=vm*one@np.linalg.solve(Cal,mcal-mp[inds])
    fi=np.where(np.array(b['types'])=='SN flow')[0]; cf=b['C'][np.ix_(fi,fi)];of=np.ones(len(fi));vb=1/(of@np.linalg.solve(cf,of));beta=vb*of@np.linalg.solve(cf,b['y'][fi]);htwo=M-beta
    # Shared calibration priors expressed as latent offsets: independent check of marginal covariance.
    U=np.array(b['vectors']).T;sd=np.sqrt(np.diag(b['components']['independent'])); B=np.column_stack([b['X'],U]);aug=np.vstack([B/sd[:,None],np.column_stack([np.zeros((U.shape[1],len(b['pars']))),np.eye(U.shape[1])])]);yy=np.r_[b['y']/sd,np.zeros(U.shape[1])];tn=np.linalg.lstsq(aug,yy,rcond=None)[0][:len(b['pars'])]
    rng=np.random.default_rng(20261106); noise=b['L']@rng.normal(size=(len(b['y']),50000));sim=b['W']@noise;mc=np.std(sim[b['pars'].index('h')],ddof=1)
    # SBF-only design h, M_SBF, Fornax, Virgo; null vector [1,1,-1,-1].
    mask=np.array([t.startswith('SBF') for t in b['types']]);cols=[b['pars'].index(p) for p in ['h','M_SBF','mu_group:Fornax','mu_group:Virgo']];S=b['X'][np.ix_(mask,cols)];null=np.array([1,1,-1,-1]); no=fits[1]['result']
    w=b['W'][b['pars'].index('h')];contrib={k:float(w@v@w) for k,v in b['components'].items()};frac={k:v/r['sigma_h']**2 for k,v in contrib.items()}
    # Data diagnostics from independent primary-only host distances.
    dm=mcal-mp[inds];es=np.sqrt(np.diag(Cal));pd.DataFrame(dict(host=[v[0] for v in D['sneia_calibrators']],mu_primary=mp[inds],M_Ia=dm,error=es,standardized_deviation=(dm-M)/np.sqrt(np.diag(Cal)-vm))).to_csv(O/'calibrator_diagnostics.csv',index=False)
    validation=dict(normal_equation_max_parameter_delta=float(np.max(np.abs(normal-b['theta']))),two_stage_h=float(htwo),two_stage_h_delta=float(htwo-r['h']),two_stage_sigma_h=float(np.sqrt(vm+vb)),two_stage_sigma_delta=float(np.sqrt(vm+vb)-r['sigma_h']),latent_nuisance_max_parameter_delta=float(np.max(np.abs(tn-b['theta']))),monte_carlo_n=50000,seed=20261106,mc_sigma_h=float(mc),analytic_sigma_h=r['sigma_h'],mc_ratio=float(mc/r['sigma_h']),full_design_rank=r['rank'],n_parameters=len(b['pars']),whitened_condition=float(np.linalg.cond(b['A'])),covariance_min_eigenvalue=float(np.linalg.eigvalsh(b['C']).min()),sbf_only_rank=int(np.linalg.matrix_rank(S)),sbf_only_n_parameters=4,sbf_null_vector=null.tolist(),sbf_null_residual=float(np.linalg.norm(S@null)),sbf_inclusion_delta_H0=r['H0']-no['H0'],sbf_inclusion_delta_sigma=r['sigma_H0']-no['sigma_H0'],variance_h_components=contrib,variance_h_fractions=frac,independent_variance_h_by_type={typ:float(np.sum((w[np.array(b['types'])==typ]**2)*np.diag(b['components']['independent'])[np.array(b['types'])==typ])) for typ in dict.fromkeys(b['types'])},fixed_instruction_reference_profile_chi2=float(r['chi2']+(5*np.log10(73.5)-r['h'])**2/r['sigma_h']**2),calibration_magnitudes_range=[float(dm.min()),float(dm.max())],unit_sign_checks=['primary mu values treated as already absolute: no anchor central modulus added again','m_cal=mu_host+M_Ia','m_flow-F(z)=M_Ia-5log10H0','distance in Mpc; c in km/s; H0 in km/s/Mpc','redshift sigma_z=250/c, no shared-flow covariance supplied'],assumptions=['all mB interpreted as standardized magnitudes; not specified in input','err_meas interpreted as independent and additional method/anchor terms added once','method-anchor offsets independent, shared within pair; anchor offset shared across methods','baseline expansion q0=-0.55, j0=1; velocities independent','no unlisted group memberships imposed'],reference_disagreement_checks=['Cholesky GLS matches independent normal-equation solver','analytic primary-only calibration plus independent flow intercept matches joint fit','latent calibration-offset prior formulation matches covariance formulation','exact LCDM integration compared to cubic cosmography','raw calibration magnitudes and Hubble diagram inspected without alteration'],fit_rejected=bool(r['p_value']<.001),diagnostic_birge_scaled_sigma_H0=float(r['sigma_H0']*np.sqrt(r['reduced_chi2'])))
    dump('validation.json',validation)
    cmb=67.4;scmb=.5
    comparisons=[]
    for name,h,s in [('Computed default (formal)',r['H0'],r['sigma_H0']),('Instruction reference',73.5,.81)]: comparisons.append(dict(name=name,H0=h,sigma=s,CMB_H0=cmb,CMB_sigma=scmb,delta=h-cmb,gaussian_difference_sigma=(h-cmb)/np.hypot(s,scmb),interpretation='Descriptive arithmetic only: local model rejected' if name.startswith('Computed') else 'Instruction reference only; not reproduced'))
    dump('cmb_comparison.json',dict(model='Planck 2018 base spatially flat six-parameter LCDM, CMB temperature/polarization/lensing',assumed_independent=True,source='https://arxiv.org/abs/1807.06209',comparisons=comparisons))
    dump('headline_results.json',dict(default=r,stated_reference=dict(H0=73.5,sigma_H0=.81,precision_percent=.81/73.5*100,origin='INSTRUCTIONS.md; not fitted or reproduced'),CMB=dict(H0=cmb,sigma_H0=scmb,model='Planck 2018 base LCDM'),interpretation='Numerically identifiable subset fit, rejected by fit quality; does not establish requested robust full-network consensus.',full_consensus_established=False,total_systematics_established=False,public_release=False))
    makefigs(b,tab,loo,dm,es,M,frac)
    manifest=dict(input_sha256=hashlib.sha256((ROOT/'data/H0DN_MinimalDataset.txt').read_bytes()).hexdigest(),python=__import__('sys').executable,versions={k:__import__(k).__version__ for k in ['numpy','scipy','pandas','matplotlib']},command='python code/analyze_network.py',files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [ROOT/'code',O,F] for p in folder.glob('*') if p.is_file() and p.name not in ['reproduction_manifest.json'] and p.suffix in ['.py','.csv','.npz','.png','.json'] and (folder!=O or p.name in ['headline_results.json','validation.json','variants.csv','leave_one_out.csv','residuals.csv','calibrator_diagnostics.csv','network_matrices.npz','cmb_comparison.json'])})
    dump('reproduction_manifest.json',manifest)
    print(tab[['variant','H0','sigma_H0','precision_percent','chi2','dof']].to_string(index=False));print(json.dumps(validation,indent=2))

def makefigs(b,tab,loo,dm,es,M,frac):
    # Availability explicitly encodes actual supplied rows; no decorative edges imply absent calibration links.
    labels=['N4258 anchor-linked rows','LMC anchor-linked rows','MW anchor-linked rows','SMC anchor-linked rows','Cepheid host distances','TRGB host distances','Mira host distances','JAGB host distances','SN Ia calibrators / flow','SBF calibrators / flow','SN II / FP / TF rows'];vals=[9,2,0,0,7,4,0,0,12,6,0]
    fig,ax=plt.subplots(figsize=(10,5.5));y=np.arange(len(labels));ax.barh(y,vals,color=['#247a96' if v else '#c8ccd0' for v in vals]);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,15);ax.set_xlabel('Number of supplied rows (categories overlap)');ax.set_title('Input coverage: SBF has no independent primary-distance tie')
    for i,v in enumerate(vals): ax.text(v+.15,i,str(v) if v else 'absent / unused',va='center')
    fig.tight_layout();fig.savefig(F/'network.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,8)); names=list(tab.variant)+['Instruction reference','Planck base LCDM'];hs=list(tab.H0)+[73.5,67.4];ss=list(tab.sigma_H0)+[.81,.5];y=np.arange(len(names));ax.axvspan(73.5-.81,73.5+.81,color='#d48b24',alpha=.15);ax.errorbar(hs,y,xerr=ss,fmt='o',color='#247a96',capsize=3);ax.scatter(hs[-2:],y[-2:],c=['#d48b24','#7751a8'],zorder=4);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$], propagated 1$\sigma$');ax.set_title('Subset GLS estimates and references (local fit is rejected)');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(F/'h0_comparison.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,5),gridspec_kw={'width_ratios':[1.25,1,1.3]});ax=axs[0];y=np.arange(7);ax.errorbar(dm,y,xerr=es,fmt='o',capsize=3);ax.axvline(M,color='#c14e43',ls='--',label='GLS common $M_B$');ax.set_yticks(y,[r[0] for r in D['sneia_calibrators']]);ax.invert_yaxis();ax.set_xlabel('Primary-only calibrated $M_B$ [mag]');ax.set_title('a. SN calibrator mismatch');ax.legend(fontsize=8)
    ax=axs[1];z=np.array([r[0] for r in D['hubble_flow_sneia']]);m=np.array([r[1] for r in D['hubble_flow_sneia']]);e=np.array([r[2] for r in D['hubble_flow_sneia']]);zgrid=np.linspace(.03,.09,100);Fg,_=distance_term(zgrid);ax.errorbar(z,m,yerr=e,fmt='o',capsize=3,label='supplied flow SNe');ax.plot(zgrid,Fg+b['result']['M_Ia']-b['result']['h'],label='default GLS');ax.invert_yaxis();ax.set_xlabel('Redshift z');ax.set_ylabel('mB [mag]');ax.set_title('b. Hubble-flow relation');ax.legend(fontsize=8)
    ax=axs[2];colors={'primary':'#247a96','SN calibrator':'#c14e43','SN flow':'#d48b24','SBF calibrator':'#7751a8','SBF flow':'#438b50'}
    for typ,c in colors.items():
        ix=np.where(np.array(b['types'])==typ)[0];ax.scatter(ix,b['white'][ix],s=27,c=c,label=typ)
    ax.axhline(0,color='k',lw=.8);ax.axhspan(-3,3,color='gray',alpha=.12);ax.set_xlabel('Observation row in released matrix');ax.set_ylabel('Cholesky-whitened residual');ax.set_title('c. Covariance-aware residuals');ax.legend(fontsize=7);fig.suptitle(f"Default goodness of fit: chi-square={b['result']['chi2']:.1f} / {b['result']['dof']} dof",y=1.01);fig.tight_layout();fig.savefig(F/'residuals.png',bbox_inches='tight');plt.close(fig)
    n=len(b['prim']);C=b['C'][:n,:n];corr=C/np.sqrt(np.outer(C.diagonal(),C.diagonal()));fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(corr,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(n),b['labels'][:n],rotation=65,ha='right',fontsize=8);ax.set_yticks(range(n),b['labels'][:n],fontsize=8);ax.set_title('Primary-distance correlation: shared anchor + method/anchor errors');fig.colorbar(im,ax=ax,label='Correlation coefficient');fig.tight_layout();fig.savefig(F/'covariance.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5));a=axs[0];yy=np.arange(len(loo));hh=[f['result']['H0'] for f in loo];ee=[f['result']['sigma_H0'] for f in loo];a.errorbar(hh,yy,xerr=ee,fmt='o',capsize=3);a.axvline(b['result']['H0'],color='#c14e43',ls='--',label='default');a.set_yticks(yy,[f['result']['variant'] for f in loo]);a.invert_yaxis();a.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$], 1$\sigma$');a.set_title('a. Leave one SN calibrator out');a.legend();a=axs[1];keys=list(frac);a.barh(range(len(keys)),[100*frac[k] for k in keys],color='#247a96');a.set_yticks(range(len(keys)),[k.replace('method:','method/anchor: ') for k in keys]);a.invert_yaxis();a.set_xlabel('Fraction of modeled Var(5 log H0) [%]');a.set_title('b. Modeled variance (not full systematics)');fig.tight_layout();fig.savefig(F/'influence.png');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    pilot() if args.pilot else main()
