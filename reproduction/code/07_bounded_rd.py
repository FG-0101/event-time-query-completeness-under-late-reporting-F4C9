"""Fractional-logit sensitivity using equal-day weights, alongside the original local-linear descriptive RDiT.
Retained-snapshot daily shares, local linear index, separate trends, weekday indicators; HAC 7-lag delta-method SE.
All endpoint predictions constrained to [0,1]. This does not identify a causal effect of the migration.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
D=pd.read_csv('results/rd_series.csv',parse_dates=['a']);cut=pd.Timestamp('2025-07-01')
from scipy.optimize import minimize
from math import ceil
RIDGE=1e-4; B=400; BLOCK=7; rng=np.random.default_rng(20261010)
def fit_ridge(X,y,penalty=RIDGE):
 def fun(beta):
  eta=X@beta
  return np.mean(np.logaddexp(0,eta)-y*eta)+penalty*np.dot(beta,beta)/2, X.T@(expit(eta)-y)/len(y)+penalty*beta
 z=minimize(fun,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-13,'gtol':1e-8})
 assert z.success,(z.message,z.fun)
 return z.x
rows=[]
for name,z in D.groupby('p'):
 z=z.sort_values('a');t=(z.a-cut).dt.days.to_numpy(float)/28;post=(t>=0).astype(float)
 X=np.column_stack([np.ones(len(z)),post,t,t*post,*[(z.a.dt.dayofweek.to_numpy()==j).astype(float) for j in range(1,7)]])
 x0=np.zeros((7,X.shape[1]));x0[:,0]=1
 for j in range(1,7):x0[j,3+j]=1
 x1=x0.copy();x1[:,1]=1
 jump=lambda beta:float((expit(x1@beta)-expit(x0@beta)).mean())
 for outcome in ['w0','w7','w30']:
  y=z[outcome].to_numpy(float); beta=fit_ridge(X,y); pred_b=expit(X@beta)
  bs=[]
  for rep in range(B):
   starts=rng.integers(0,len(z),size=ceil(len(z)/BLOCK))
   idx=np.concatenate([(start+np.arange(BLOCK))%len(z) for start in starts])[:len(z)]
   bs.append(jump(fit_ridge(X[idx],y[idx])))
  low,high=np.quantile(bs,[.025,.975]);ols=sm.OLS(y,X).fit();pred=ols.predict(X)
  grid=[jump(fit_ridge(X,y,p)) for p in [1e-5,1e-4,1e-3]]
  rows.append(dict(p=name,outcome=outcome,n_days=len(z),bounded_jump=jump(beta),se=float(np.std(bs,ddof=1)),ci_low=float(low),ci_high=float(high),ridge_min=float(min(grid)),ridge_max=float(max(grid)),ols_jump=float(ols.params[1]),ols_prediction_min=float(pred.min()),ols_prediction_max=float(pred.max()),ols_outside=int(((pred<0)|(pred>1)).sum()),glm_pred_min=float(pred_b.min()),glm_pred_max=float(pred_b.max())))
r=pd.DataFrame(rows);r.to_csv('results/bounded_rd.csv',index=False)
summary={'model':'ridge fractional logit with equal-day weights, scaled time, lambda=0.0001, circular seven-day moving-block pairs bootstrap, 400 draws, seed 20261010','n_comparisons':len(r),'outside_ols_cells':int(r.ols_outside.sum()),'bounded_all':bool(((r.glm_pred_min>=0)&(r.glm_pred_max<=1)).all()),'direction_agreements':int((np.sign(r.bounded_jump)==np.sign(r.ols_jump)).sum())}
json.dump(summary,open('results/bounded_rd.json','w'),indent=2)
q=r[(r.outcome=='w7') & r.p.isin(['All platforms','Without Google Shopping','Platform average','TikTok','X','Shein'])]
with open('tex/bounded_rd.tex','w') as f:
 f.write('\\subsection{Bounded-share sensitivity}\\label{sec:bounded}\nThe local-linear estimator is a descriptive approximation to a daily share. A fractional-logit sensitivity uses the same 28-day windows, separate time trends and weekday indicators, with equal weights per observed day and time scaled by 28 days. A quadratic penalty of $10^{-4}$ on all coefficients makes the fit finite when observed shares are separated; sensitivity uses penalties of $10^{-5}$ and $10^{-3}$. We report the fitted discontinuity averaged across the seven weekdays and a 95\\% percentile interval from 400 circular seven-day moving-block pairs bootstrap draws. The interval describes variation in this regularized fit and does not establish nominal causal coverage. The fitted shares stay between zero and one. These estimates describe the retained snapshot; concurrent changes and reporting revisions remain possible explanations.\n\\begin{table*}[!tbp]\\centering\\small\\begin{tabular}{lrrr}\\toprule\nSeries & Linear jump & Fractional-logit jump & Bootstrap 95\\% interval\\\\\\midrule\n')
 for x in q.itertuples():f.write(f'{x.p} & {x.ols_jump:.3f} & {x.bounded_jump:.3f} & [{x.ci_low:.3f}, {x.ci_high:.3f}]\\\\\n')
 f.write('\\bottomrule\\end{tabular}\\caption{Same-week reporting shares around the migration, linear and bounded-share estimates.}\\end{table*}\n')
print(summary);print(q.to_string(index=False))
