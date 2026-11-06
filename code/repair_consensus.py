"""Fit missing SN scatter and quantify precision limits; never supply missing data."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2
import analyze_network as net

O, F = net.O, net.F
b = net.fit()
X, y, C = b['X'], b['y'], b['C']
ih = b['pars'].index('h')
cal = np.array(b['types']) == 'SN calibrator'
flow = np.array(b['types']) == 'SN flow'

def solve(scatter, fixed_h=None, reml=False):
    cov = C + np.diag(cal * scatter[0]**2 + flow * scatter[1]**2)
    fac = cho_factor(cov, lower=True)
    design = X if fixed_h is None else np.delete(X, ih, axis=1)
    obs = y if fixed_h is None else y - X[:, ih] * fixed_h
    normal = design.T @ cho_solve(fac, design)
    rhs = design.T @ cho_solve(fac, obs)
    theta = np.linalg.solve(normal, rhs)
    residual = obs - design @ theta
    chisq = float(residual @ cho_solve(fac, residual))
    objective = chisq + 2*np.log(np.diag(fac[0])).sum()
    if reml:
        objective += np.linalg.slogdet(normal)[1]
    return float(objective), theta, np.linalg.inv(normal), chisq

def optimize(fixed_h=None, reml=False):
    candidates = [minimize(lambda s: solve(s, fixed_h, reml)[0], s,
                           bounds=[(0., 5.), (0., 5.)], method='L-BFGS-B',
                           options={'ftol': 1e-12, 'gtol': 1e-7})
                  for s in ([.1, .1], [.4, .1], [.0, .0])]
    opt = min(candidates, key=lambda z: z.fun)
    assert opt.success or np.linalg.norm(opt.jac) < 1e-3, opt.message
    assert np.all(opt.x < 4.99), 'Scatter upper bound reached'
    return opt, solve(opt.x, fixed_h, reml)

out = {}
for name, reml in [('maximum_likelihood', False), ('restricted_maximum_likelihood', True)]:
    opt, (_, theta, V, ch) = optimize(reml=reml)
    H = 10**(theta[ih]/5)
    out[name] = dict(H0=float(H), scatter_calibrator_mag=float(opt.x[0]),
        scatter_flow_mag=float(opt.x[1]), conditional_sigma_H0=float(H*np.log(10)/5*np.sqrt(V[ih, ih])),
        chi2=ch, nominal_residual_dof=17,
        caution='Scatter is estimated; conditional errors omit its estimation uncertainty. '
                'Chi-square p values with fitted variance components are not calibrated.')

ml_opt, (minimum, _, _, _) = optimize()
best = out['maximum_likelihood']['H0']
def profile(H):
    opt, result = optimize(5*np.log10(H))
    return result[0] - minimum, opt.x
interval = [brentq(lambda H: profile(H)[0]-1, 30, best),
            brentq(lambda H: profile(H)[0]-1, best, 250)]
rows = []
for H in np.unique(np.r_[np.linspace(60,180,121), best, interval]):
    delta, s = profile(H)
    rows.append(dict(H0=H, delta_minus2logL=delta, scatter_calibrator_mag=s[0], scatter_flow_mag=s[1]))
pd.DataFrame(rows).to_csv(O/'scatter_profile.csv', index=False)
out['maximum_likelihood']['profile_68_interval'] = interval
out['maximum_likelihood']['profile_interval_interpretation'] = 'Likelihood-ratio delta(-2 log L)=1, asymptotic 68.27%; small-sample coverage not guaranteed.'
out['maximum_likelihood']['profile_half_width_percent'] = 100*(interval[1]-interval[0])/(2*best)
out['precision_limits'] = {}
for pv in [0,250]:
    test = net.fit(pv=pv)
    idx = np.array(test['types']) == 'SN flow'
    variance = 1/np.sum(1/np.diag(test['C'])[idx])
    out['precision_limits'][str(pv)] = dict(known_absolute_magnitude_sigma_h=float(np.sqrt(variance)),
        minimum_percent=float(100*np.log(10)/5*np.sqrt(variance)),
        assumption='Exactly known SN absolute magnitude, zero intrinsic scatter; supplied five flow SNe only. '
                   'This optimistic lower bound is conditional on this branch, not on future measurements.')

# A uniform extra primary zero-point shift is degenerate with the absolute scale.
# The sensitivity grid illustrates an unidentified uncertainty, not an empirical prior.
scenarios = []
primary = (np.array(b['types']) == 'primary').astype(float)
for tau in np.linspace(0,.1,51):
    cov = C + tau**2 * np.outer(primary,primary)
    V = np.linalg.inv(X.T @ np.linalg.solve(cov,X))
    scenarios.append(dict(assumed_common_zero_point_sigma_mag=tau,
        conditional_precision_percent=100*np.log(10)/5*np.sqrt(V[ih,ih])))
pd.DataFrame(scenarios).to_csv(O/'systematic_sensitivity.csv',index=False)
out['common_zero_point_sensitivity'] = dict(assumption='Hypothetical independent shared primary offset. '
    'Magnitude not identifiable from these data; scenarios are not a completed systematic budget.',
    analytic_variance_increment_verified=bool(all(np.isclose((r['conditional_precision_percent']/100*net.K)**2,
        b['V'][ih,ih]+r['assumed_common_zero_point_sigma_mag']**2) for r in scenarios)))
out['missing_inputs'] = ['MW parallaxes', 'SMC detached-eclipsing-binary measurements',
    'Miras', 'JAGB', 'SNe II', 'FP', 'TF', 'Independent primary-to-SBF calibration',
    'SN standardization and full systematic covariance']
out['full_consensus_established'] = False
out['reason'] = 'Fitted extra scatter broadens precision; it cannot create absent observations, identify common scale systematics, or restore independent SBF calibration.'
net.dump('consensus_repair.json',out)

fig, ax = net.plt.subplots(1,2,figsize=(12,4.5))
t = pd.DataFrame(rows).sort_values('H0')
ax[0].plot(t.H0,t.delta_minus2logL,label='Profile over two SN scatter terms')
ax[0].axhline(1,color='gray',ls='--',label='Asymptotic 68% threshold')
ax[0].axvspan(*interval,color='C0',alpha=.12)
ax[0].axvline(73.5,color='C1',ls=':',label='Instruction reference')
ax[0].set(xlabel='H₀ [km s⁻¹ Mpc⁻¹]',ylabel='Δ(−2 log likelihood)',ylim=(0,12))
ax[0].legend(fontsize=8)
s = pd.DataFrame(scenarios)
ax[1].plot(s.assumed_common_zero_point_sigma_mag,s.conditional_precision_percent,
           label='Default + hypothetical common offset')
ax[1].axhline(1.1,color='C1',ls=':',label='Instruction precision ≈1.10%')
ax[1].axhline(out['precision_limits']['250']['minimum_percent'],color='C2',ls='--',
              label='Five flow SNe; calibration exactly known')
ax[1].set(xlabel='Assumed common zero-point uncertainty [mag]',ylabel='Conditional σ(H₀)/H₀ [%]')
ax[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(F/'consensus_repair.png');net.plt.close(fig)
assert out['common_zero_point_sensitivity']['analytic_variance_increment_verified']
print(json.dumps(out,indent=2))
