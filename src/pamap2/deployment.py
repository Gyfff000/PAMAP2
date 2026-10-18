"""Fit an explicitly non-evaluation deployment model; export a truly held-out demo."""
import json
import shutil
import time
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import ROOT, LABEL_NAMES, SUBJECTS, load_data, save_json, write_csv, sha256
from .experiment import fit_nn, make_loader, predict_loader
from .models import make_model

def main():
    cfg=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    summary=json.loads((ROOT/'outputs/model_summary_seed42.json').read_text(encoding='utf-8'))
    candidates=[r for r in summary if r['model'] in ('logistic','rf_all','cnn','fusion')]
    if len(candidates)!=4:
        raise RuntimeError('Complete all main models before exporting a final model.')
    chosen=max(candidates,key=lambda r:r['macro_f1'])['model']
    x,y,g,features,metadata=load_data()
    out=ROOT/'artifacts/final'
    out.mkdir(parents=True,exist_ok=True)
    demo=ROOT/'artifacts/demo'
    demo.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    selected_epochs=None
    if chosen in ('cnn','fusion'):
        epochs=[json.loads((ROOT/f'outputs/folds/{chosen}_seed42/{s}/metrics.json').read_text(encoding='utf-8'))['selected_epochs'] for s in SUBJECTS]
        selected_epochs=max(1,int(np.median(epochs)))
        model,norm,history,_=fit_nn(chosen,x,features,y,np.arange(len(y)),None,cfg,42,selected_epochs)
        torch.save({'name':chosen,'state_dict':model.state_dict(),'selected_epochs':selected_epochs},out/'model.pt')
        np.savez(out/'normalization.npz',**norm)
        write_csv(ROOT/'outputs/final_training_history.csv',history)
        shutil.copy2(ROOT/f'models/folds/{chosen}_seed42/101/model.pt',demo/'model.pt')
        shutil.copy2(ROOT/f'models/folds/{chosen}_seed42/101/normalization.npz',demo/'normalization.npz')
    else:
        if chosen=='logistic':
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=42))
        else:
            model=RandomForestClassifier(n_estimators=cfg['rf_trees'],min_samples_leaf=cfg['rf_min_samples_leaf'],
                    max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=6)
        model.fit(features,y)
        joblib.dump({'model':model,'feature_columns':np.arange(150)},out/'model.joblib',compress=3)
        shutil.copy2(ROOT/f'models/folds/{chosen}_seed42/101/model.joblib',demo/'model.joblib')
    # Fixed temporal quantiles, independent of model predictions; avoid only showing recording onsets.
    selected=[]
    for k in range(12):
        available=np.flatnonzero((g==101)&(y==k))
        selected.extend(available[np.minimum((np.array([.25,.75])*len(available)).astype(int),len(available)-1)])
    selected=np.array(selected,dtype=np.int64)
    with np.load(ROOT/f'outputs/folds/{chosen}_seed42/101/predictions.npz') as original:
        lookup=dict(zip(original['indices'].tolist(),original['y_pred'].tolist()))
        expected_prediction=np.array([lookup[int(i)] for i in selected])
    np.savez_compressed(demo/'sample_windows.npz',windows=np.asarray(x[selected]),features=features[selected],
                        y_true=y[selected],window_indices=selected,subject=g[selected],expected_prediction=expected_prediction)
    save_json(out/'manifest.json',{'model':chosen,'trained_on_subjects':SUBJECTS,'training_windows':len(y),
              'selected_epochs':selected_epochs,'selection_rule':'Highest pooled LOSO macro-F1 among four prespecified main models.',
              'evaluation_warning':'Refitted on all eight subjects for demonstration/deployment. No new independent test score is claimed.',
              'fit_seconds':time.monotonic()-begin,'label_names':LABEL_NAMES,
              'config_sha256':sha256(ROOT/'configs/experiment.json'),
              'data_audit_sha256':sha256(ROOT/'outputs/data_audit.json'),
              'files':{p.name:sha256(p) for p in out.iterdir() if p.name!='manifest.json'}})
    save_json(demo/'manifest.json',{'model':chosen,'test_subject':101,
              'trained_on_subjects':[s for s in SUBJECTS if s!=101],
              'samples':len(selected),'description':'Two temporal-quantile windows (25%, 75%) per class from subject 101; no filtering by prediction correctness. Paired outer-fold model; illustrative replay only.',
              'source_dataset':'PAMAP2, UCI, DOI:10.24432/C5NW2H, CC BY 4.0',
              'label_names':LABEL_NAMES,'files':{p.name:sha256(p) for p in demo.iterdir() if p.name!='manifest.json'}})
    print(f'Exported final {chosen} model and a subject-disjoint demo.',flush=True)

def predict_demo():
    from .prepare import features as extract_features
    directory=ROOT/'artifacts/demo'
    meta=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in meta['files'].items():
        if sha256(directory/name)!=digest:
            raise RuntimeError(f'Artifact integrity mismatch: {name}')
    with np.load(directory/'sample_windows.npz') as data:
        x=data['windows'];f=data['features'];y=data['y_true'];idx=data['window_indices']
        expected_prediction=data['expected_prediction']
    # Independent feature recomputation catches incompatible preprocessing changes.
    np.testing.assert_allclose(extract_features(x),f,rtol=1e-6,atol=1e-6)
    if meta['model'] in ('cnn','fusion'):
        saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
        model=make_model(meta['model']);model.load_state_dict(saved['state_dict']);model.eval()
        with np.load(directory/'normalization.npz') as n:
            norm={k:n[k] for k in n.files}
        pred=predict_loader(model,make_loader(x,f,y,np.arange(len(y)),norm,128,False,42))
    else:
        saved=joblib.load(directory/'model.joblib')
        pred=saved['model'].predict(f[:,saved['feature_columns']])
    np.testing.assert_array_equal(pred,expected_prediction,err_msg='Reloaded model predictions must match the original outer-fold records.')
    print(f'模型：{meta["model"]}；演示受试者：101（未参与该演示模型训练）')
    rows=[]
    for i in range(len(y)):
        print(f'window {idx[i]:4d} | true: {LABEL_NAMES[y[i]]:<18} | predicted: {LABEL_NAMES[pred[i]]}')
        rows.append({'window_index':int(idx[i]),'subject':101,'true_activity':LABEL_NAMES[y[i]],
                     'predicted_activity':LABEL_NAMES[pred[i]],'correct':bool(y[i]==pred[i])})
    write_csv(ROOT/'outputs/demo_predictions.csv',rows)
    print(f'演示正确 {int((y==pred).sum())}/{len(y)}。这组少量演示样本不能替代完整 LOSO 评估。')

if __name__=='__main__':
    main()
