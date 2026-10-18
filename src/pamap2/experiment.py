import argparse
import copy
import csv
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .models import make_model
from .common import ROOT, SUBJECTS, LABEL_NAMES, load_data, save_json, write_csv, seed_all, sha256

def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(12), average='macro', zero_division=0)),
            'macro_recall': float(recall_score(y, pred, labels=range(12), average='macro', zero_division=0)),
            'supported_macro_f1': float(f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),
            'supported_classes': int(len(np.unique(y)))}

def split_subjects(groups, test_subject):
    i = SUBJECTS.index(test_subject)
    val_subject = SUBJECTS[(i + 1) % len(SUBJECTS)]
    inner_train = np.flatnonzero((groups != test_subject) & (groups != val_subject))
    val = np.flatnonzero(groups == val_subject)
    outer_train = np.flatnonzero(groups != test_subject)
    test = np.flatnonzero(groups == test_subject)
    assert not (set(groups[inner_train]) & set(groups[val]))
    assert not (set(groups[outer_train]) & set(groups[test]))
    return inner_train, val, outer_train, test, val_subject

def feature_indices(name):
    lookup = {'rf_wrist': [0], 'rf_chest': [1], 'rf_ankle': [2],
              'rf_no_wrist': [1,2], 'rf_no_chest': [0,2], 'rf_no_ankle': [0,1]}
    return np.concatenate([np.arange(i*50, (i+1)*50) for i in lookup.get(name, [0,1,2])])

def normalization(x, features, indices):
    subset = np.asarray(x[indices], dtype=np.float32)
    mean = subset.mean(axis=(0,2), dtype=np.float64).astype(np.float32)[None,:,None]
    std = subset.std(axis=(0,2), dtype=np.float64).astype(np.float32)[None,:,None]
    fmean = features[indices].mean(0, dtype=np.float64).astype(np.float32)
    fstd = features[indices].std(0, dtype=np.float64).astype(np.float32)
    return {'mean': mean, 'std': np.maximum(std, 1e-6),
            'fmean': fmean, 'fstd': np.maximum(fstd, 1e-6)}

def make_loader(x, features, y, indices, norm, batch_size, shuffle, seed):
    tx = torch.from_numpy(((np.asarray(x[indices])-norm['mean'])/norm['std']).copy())
    tf = torch.from_numpy(((features[indices]-norm['fmean'])/norm['fstd']).copy())
    ty = torch.from_numpy(y[indices].copy())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(tx,tf,ty), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)

def predict_loader(model, loader):
    model.eval()
    results = []
    with torch.inference_mode():
        for xb, fb, _ in loader:
            results.append(model(xb,fb).argmax(1).numpy())
    return np.concatenate(results)

def fit_nn(name, x, features, y, train_idx, val_idx, config, seed, epochs=None):
    seed_all(seed)
    model = make_model(name)
    norm = normalization(x,features,train_idx)
    train_loader = make_loader(x,features,y,train_idx,norm,config['batch_size'],True,seed)
    val_loader = None if val_idx is None else make_loader(x,features,y,val_idx,norm,256,False,seed)
    counts = np.bincount(y[train_idx], minlength=12)
    weights = 1 / np.sqrt(np.maximum(counts,1))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
    history = []
    best, selected, stale = -1., 1, 0
    for epoch in range(1,(epochs or config['max_epochs'])+1):
        begin = time.monotonic()
        model.train()
        total, n = 0., 0
        for xb,fb,yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb,fb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total += float(loss.detach())*len(yb)
            n += len(yb)
        row = {'epoch':epoch,'train_loss':total/n,'seconds':time.monotonic()-begin}
        if val_loader is not None:
            score = metrics(y[val_idx],predict_loader(model,val_loader))['macro_f1']
            row['validation_macro_f1'] = score
            if score > best+1e-8:
                best,selected,stale = score,epoch,0
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch%5 == 0:
            print(f'  {name} epoch {epoch}: loss={total/n:.4f}'+
                  (f' val F1={row["validation_macro_f1"]:.4f}' if val_loader is not None else ''),flush=True)
        if val_loader is not None and stale >= config['patience']:
            break
    return model, norm, history, selected

def run_fold(name,subject,x,y,g,features,metadata,config,seed):
    folder = ROOT / 'outputs/folds' / f'{name}_seed{seed}' / str(subject)
    model_dir = ROOT / 'models/folds' / f'{name}_seed{seed}' / str(subject)
    artifact=model_dir/('model.pt' if name in ('cnn','fusion') else 'model.joblib')
    if (folder/'metrics.json').exists() and artifact.exists():
        previous=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if previous['config_sha256']!=sha256(ROOT/'configs/experiment.json'):
            raise RuntimeError('Existing results use a different configuration; preserve them and use a new output directory.')
        with np.load(folder/'predictions.npz') as prediction:
            expected=np.flatnonzero(g==subject)
            if not np.array_equal(prediction['indices'],expected) or not np.array_equal(prediction['y_true'],y[expected]):
                raise RuntimeError('Existing predictions do not match the current processed dataset.')
        return previous
    folder.mkdir(parents=True,exist_ok=True)
    model_dir.mkdir(parents=True,exist_ok=True)
    inner,val,train,test,val_subject = split_subjects(g,subject)
    begin=time.monotonic()
    fold_seed=seed+subject
    print(f'[{name}] outer test={subject}, train={len(train)}, test={len(test)}',flush=True)
    selected_epoch=None
    if name in ('cnn','fusion'):
        _,_,history,selected_epoch=fit_nn(name,x,features,y,inner,val,config,fold_seed)
        write_csv(folder/'inner_history.csv',history)
        model,norm,history,_=fit_nn(name,x,features,y,train,None,config,fold_seed,selected_epoch)
        write_csv(folder/'refit_history.csv',history)
        test_loader=make_loader(x,features,y,test,norm,256,False,fold_seed)
        prediction=predict_loader(model,test_loader)
        torch.save({'state_dict':model.state_dict(),'name':name,'selected_epochs':selected_epoch},model_dir/'model.pt')
        np.savez(model_dir/'normalization.npz',**norm)
        parameter_count=sum(p.numel() for p in model.parameters())
    else:
        columns=feature_indices(name)
        if name=='logistic':
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=fold_seed))
        else:
            model=RandomForestClassifier(n_estimators=config['rf_trees'],min_samples_leaf=config['rf_min_samples_leaf'],
                max_features=config['rf_max_features'],class_weight='balanced_subsample',random_state=fold_seed,n_jobs=6)
        model.fit(features[train][:,columns],y[train])
        prediction=model.predict(features[test][:,columns])
        joblib.dump({'model':model,'feature_columns':columns},model_dir/'model.joblib',compress=3)
        parameter_count=None
    results=metrics(y[test],prediction)
    results.update({'model':name,'seed':seed,'test_subject':subject,
                    'train_subjects':[s for s in SUBJECTS if s!=subject],
                    'inner_validation_subject':val_subject if name in ('cnn','fusion') else None,
                    'selected_epochs':selected_epoch,'train_windows':len(train),'test_windows':len(test),
                    'seconds':time.monotonic()-begin,'parameter_count':parameter_count,
                    'config_sha256':sha256(ROOT/'configs/experiment.json')})
    np.savez_compressed(folder/'predictions.npz',indices=test,y_true=y[test],y_pred=prediction,
                        subject=g[test],segment=metadata[test,0],start_seconds=metadata[test,1],end_seconds=metadata[test,2])
    save_json(folder/'metrics.json',results)
    print(f'[{name}] subject {subject}: F1={results["macro_f1"]:.4f}, accuracy={results["accuracy"]:.4f}, {results["seconds"]:.1f}s',flush=True)
    return results

def summarize(config,seed):
    rows=[]
    summaries=[]
    for name in config['models']:
        folders=[ROOT/'outputs/folds'/f'{name}_seed{seed}'/str(s) for s in SUBJECTS]
        if not all((f/'metrics.json').exists() for f in folders):
            continue
        fold_rows=[json.loads((f/'metrics.json').read_text(encoding='utf-8')) for f in folders]
        rows.extend(fold_rows)
        predictions=[np.load(f/'predictions.npz') for f in folders]
        y=np.concatenate([p['y_true'] for p in predictions])
        pred=np.concatenate([p['y_pred'] for p in predictions])
        summary={'model':name,'seed':seed,'folds':len(fold_rows),'windows':len(y),**metrics(y,pred)}
        for key in ('accuracy','macro_f1','macro_recall'):
            values=np.array([r[key] for r in fold_rows])
            summary['subject_mean_'+key]=float(values.mean())
            summary['subject_std_'+key]=float(values.std(ddof=1))
        summary['total_fit_seconds']=sum(r['seconds'] for r in fold_rows)
        summaries.append(summary)
        save_json(ROOT/f'outputs/reports/{name}_seed{seed}.json',classification_report(y,pred,labels=range(12),target_names=LABEL_NAMES,output_dict=True,zero_division=0))
        np.savetxt(ROOT/f'outputs/reports/{name}_seed{seed}_confusion.csv',confusion_matrix(y,pred,labels=range(12)),fmt='%d',delimiter=',')
    # Lists are serialized for audit-friendly flat CSVs.
    for row in rows:
        row['train_subjects']=' '.join(map(str,row['train_subjects']))
    write_csv(ROOT/f'outputs/fold_metrics_seed{seed}.csv',rows)
    write_csv(ROOT/f'outputs/model_summary_seed{seed}.csv',summaries)
    save_json(ROOT/f'outputs/model_summary_seed{seed}.json',summaries)
    return summaries

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+')
    parser.add_argument('--subjects',type=int,nargs='+')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    seed_all(args.seed)
    (ROOT/'outputs/reports').mkdir(parents=True,exist_ok=True)
    save_json(ROOT/'outputs/environment.json',{'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
        'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'torch_threads':torch.get_num_threads(),
        'config_sha256':sha256(ROOT/'configs/experiment.json'),
        'run_started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    x,y,g,features,metadata=load_data()
    for name in (args.models or config['models']):
        if name not in config['models']:
            raise ValueError(f'Unknown model {name}')
        for subject in (args.subjects or SUBJECTS):
            run_fold(name,subject,x,y,g,features,metadata,config,args.seed)
        summarize(config,args.seed)

if __name__=='__main__':
    main()
