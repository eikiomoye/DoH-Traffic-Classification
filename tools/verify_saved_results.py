import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
root=Path(__file__).resolve().parents[1] / 'results' / 'original'
frames={n:pd.read_csv(root/(n+'.csv')) for n in ['cv_fold_metrics','test_metrics_by_seed','cv_metrics_summary','feature_summary','test_metrics_summary','summed_confusion_matrices','permutation_importance']}
a=json.loads((root/'dataset_audit.json').read_text())
cv=frames['cv_fold_metrics']; test=frames['test_metrics_by_seed']; summary=frames['test_metrics_summary']; cs=frames['cv_metrics_summary']
checks=[]
def check(name,value): checks.append({'check':name,'passed':bool(value)})
def close(a,b): return np.allclose(a,b,atol=1e-10,rtol=1e-9,equal_nan=True)
models={'Logistic Regression','Random Forest','XGBoost'}
check('test seeds/models: exactly nine unique combinations',set(test.seed)=={42,52,62} and set(test.model)==models and len(test)==9 and not test.duplicated(['seed','model']).any())
check('CV: three unique folds per seed and model',len(cv)==27 and not cv.duplicated(['seed','model','fold']).any() and set(cv.seed)=={42,52,62} and set(cv.model)==models and all(set(g.fold)=={1,2,3} for _,g in cv.groupby(['seed','model'])))
check('held-out partition sizes',test.development_records.eq(815293).all() and test.test_records.eq(203824).all())
check('development class counts',test.development_non_doh.eq(600364).all() and test.development_doh.eq(214929).all())
check('XGBoost class ratio',close(test.loc[test.model=='XGBoost','xgboost_scale_pos_weight'],600364/214929))
check('all confusion totals equal held-out size',test[['tn','fp','fn','tp']].sum(axis=1).eq(test.test_records).all())
check('confusion class totals match test partition',((test.tn+test.fp)==150092).all() and ((test.tp+test.fn)==53732).all())
check('confusion values are nonnegative integers',((test[['tn','fp','fn','tp']]>=0)&(test[['tn','fp','fn','tp']]%1==0)).all().all())
for col,calc in {'accuracy':(test.tp+test.tn)/test.test_records,'precision_doh':test.tp/(test.tp+test.fp),'recall_doh':test.tp/(test.tp+test.fn),'false_positive_rate':test.fp/(test.fp+test.tn),'balanced_accuracy':.5*(test.tp/(test.tp+test.fn)+test.tn/(test.tn+test.fp)),'macro_f1':.5*(2*test.tp/(2*test.tp+test.fp+test.fn)+2*test.tn/(2*test.tn+test.fp+test.fn)),'mcc':(test.tp.astype(float)*test.tn-test.fp.astype(float)*test.fn)/np.sqrt((test.tp+test.fp).astype(float)*(test.tp+test.fn)*(test.tn+test.fp)*(test.tn+test.fn))}.items(): check(col+' independently recomputed from confusion counts',close(test[col],calc))
check('summary contains all model/metric combinations exactly once',len(summary)==33 and not summary.duplicated(['model','metric']).any() and set(summary.model)==models and set(summary.metric)=={'accuracy','balanced_accuracy','precision_doh','recall_doh','macro_f1','mcc','pr_auc','roc_auc','false_positive_rate','training_time_seconds','prediction_time_seconds'})
for col in ['mean','standard_deviation','ci95_lower','ci95_upper','n_repetitions']:
 okay=True
 for row in summary.itertuples():
  vals=test.loc[test.model==row.model,row.metric];mean=vals.mean();sd=vals.std(ddof=1);half=t.ppf(.975,len(vals)-1)*sd/np.sqrt(len(vals))
  exp={'mean':mean,'standard_deviation':sd,'ci95_lower':mean-half,'ci95_upper':mean+half,'n_repetitions':len(vals)}[col]
  okay &= close(getattr(row,col),exp)
 check('test summary '+col+' reconciles',okay)
for suffix in ['mean','std']:
 okay=True
 for row in cs.to_dict('records'):
  g=cv[cv.model==row['model']]
  for out,value in row.items():
   if out.endswith('_'+suffix):
    key=out[:-len(suffix)-1];expected=g[key].mean() if suffix=='mean' else g[key].std(ddof=1);okay &= close(value,expected)
 check('CV summary '+suffix+' reconciles across nine folds per model',okay)
summed=frames['summed_confusion_matrices'].set_index('model').sort_index()
expected=test.groupby('model')[['tn','fp','fn','tp']].sum().sort_index()
check('summed confusion matrices reconcile',np.array_equal(summed[['summed_tn','summed_fp','summed_fn','summed_tp']],expected))
f=frames['feature_summary']; p=frames['permutation_importance']
features=set(f.feature)
check('29 unique feature summaries',len(f)==29 and len(features)==29 and f['count'].eq(1019117).all())
check('feature names match audit',features==set(a['dtypes'])-{'Label'})
check('no constant features in supplied summary',f.unique_values.gt(1).all())
check('all reported feature ranges are finite',np.isfinite(f[['min','max','mean','std']]).all().all())
check('87 unique permutation entries for the same 29 features',len(p)==87 and not p.duplicated(['model','feature']).any() and set(p.model)==models and all(set(g.feature)==features for _,g in p.groupby('model')))
check('permutation protocol: seed42 sample20000 repeats3 AP',p.seed.eq(42).all() and p.held_out_sample_records.eq(20000).all() and p.permutation_repeats.eq(3).all() and p.scoring.eq('average_precision').all())
check('tree leading importance feature PacketLengthMode',all(p[p.model==m].sort_values('importance_mean',ascending=False).iloc[0].feature=='PacketLengthMode' for m in ['Random Forest','XGBoost']))
check('audit dimensions and class counts',a['rows']==1019117 and a['columns']==30 and a['predictor_count']==29 and a['label_counts']=={'NonDoH':750456,'DoH':268661})
check('audit reports clean validation',a['validation_status']=='PASSED' and not a['validation_failures'] and not a['non_numeric_predictors'] and not a['constant_features'] and not a['near_constant_features_99_9_percent'] and all(a[k]==0 for k in ['missing_cells','exact_duplicate_rows','infinite_numeric_cells']))
check('recorded dataset hash well formed',len(a['sha256'])==64 and all(c in '0123456789abcdef' for c in a['sha256']))
check('metric ranges',test[['accuracy','balanced_accuracy','precision_doh','recall_doh','macro_f1','pr_auc','roc_auc','false_positive_rate']].ge(0).all().all() and test[['accuracy','balanced_accuracy','precision_doh','recall_doh','macro_f1','pr_auc','roc_auc','false_positive_rate']].le(1).all().all())
check('model times positive',test[['training_time_seconds','prediction_time_seconds']].gt(0).all().all())
report={'scope':'arithmetic reconciliation of supplied saved outputs, not a new training run or verification of dataset bytes','checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks),'dataset_sha256_recorded':a['sha256']}
(root.parent / 'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print(summary.pivot(index='model',columns='metric',values='mean')[['macro_f1','mcc','pr_auc','false_positive_rate','training_time_seconds']].to_string())
if report['passed']!=report['total']: raise SystemExit(1)
