"""Step 15b: E4 control for the cross-sensor experiment. Leave-one-subject-out calibration within GalaxyPPG, alone or pooled with the development participants. Output: results/e4_control.csv"""

import numpy as np, pandas as pd
from ppgs.conformal import METHODS, N_FOLDS, HR_RANGE, ALPHA, weighted_quantile, person_folds
F3=['snr_db','motion','disagree']
GM={"baseline":"rest","rest":"rest","meditation":"rest","adaptation":"rest","screen reading":"rest","standing":"rest","TSST preparation":"stress","TSST speech":"stress","SSST preparation":"stress","SSST singing":"stress","keyboard typing":"typing","mobile typing":"typing","walking":"walking","jogging":"jogging","running":"running"}
def qw(s): return weighted_quantile(s,np.ones(len(s)),min(1,(1-ALPHA)*(1+1/len(s))))
def qp(s,subj):
    _,inv,c=np.unique(subj,return_inverse=True,return_counts=True); return weighted_quantile(s,1/c[inv],(1-ALPHA)*(1+1/len(c)))
dev=pd.read_csv('results/windows.csv'); gal=pd.read_csv('results/windows_galaxy.csv'); gal['act']=gal.condition.map(GM)
out=[]
for setting in ['galaxy only','dev + galaxy']:
  for p in gal.subject.unique():
    cal=gal[gal.subject!=p]; te=gal[gal.subject==p]
    if setting=='dev + galaxy': cal=pd.concat([dev,cal],ignore_index=True)
    X,e=cal[F3].to_numpy(),(cal.hr_ref-cal.hr_hat).to_numpy(); s=cal.subject.to_numpy()
    f=person_folds(s,cal.dataset.to_numpy()); oof=np.empty(len(e))
    for k in range(N_FOLDS): oof[f==k]=METHODS['normalized']().fit(X[f!=k],e[f!=k]).score(X[f==k],e[f==k])
    sig=METHODS['normalized']().fit(X,e).sigma(te[F3].to_numpy())
    for r,q in [('window',qw(oof)),('person',qp(oof,s))]:
      lo=np.maximum(te.hr_hat-q*sig,40); hi=np.minimum(te.hr_hat+q*sig,200)
      out.append(pd.DataFrame({'setting':setting,'rule':r,'subject':p,'act':te.act,'cv':(te.hr_ref>=lo)&(te.hr_ref<=hi),'w':hi-lo}))
o=pd.concat(out)
o.to_csv('results/e4_control.csv', index=False)
per = o.groupby(['setting', 'rule', 'subject']).cv.mean()
print((per >= 0.9).groupby(['setting', 'rule']).sum())
print(o.groupby(['setting', 'rule']).w.median().round(1))
print((o.groupby(['setting', 'rule', 'act']).cv.mean() * 100).round(0).unstack())
