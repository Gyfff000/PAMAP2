"""Training-only three-fold selection, then fixed 30% test evaluation."""
import argparse
import json
import platform
import sys
import time
import joblib
import numpy as np
import scipy
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import ROOT, load_data, seed_all, save_json, write_csv, sha256
from .experiment import metrics, feature_indices, fit_nn, make_loader, predict_loader
from .protocol import OUT, load_split

MAIN_MODELS = ['logistic','rf_all','cnn','fusion']


def run_fit(name, stage, fit, evaluate, x, y, subjects, features, metadata, cfg, seed):
    directory = OUT/stage/name
    model_dir = ROOT/'models/holdout'/stage/name
    model_file = model_dir/('model.pt' if name in ('cnn','fusion') else 'model.joblib')
    score_file = directory/'metrics.json'
    prediction_file = directory/'predictions.npz'
    config_hash = sha256(ROOT/'configs/experiment.json')
    split_hash = sha256(OUT/'split_assignments.csv')
    if score_file.exists() and model_file.exists() and prediction_file.exists():
        previous = json.loads(score_file.read_text(encoding='utf-8'))
        if previous['config_sha256'] != config_hash or previous['split_sha256'] != split_hash:
            raise RuntimeError('Configuration changed; preserve results before starting a new run.')
        assert previous['model_sha256'] == sha256(model_file)
        assert previous['predictions_sha256'] == sha256(prediction_file)
        with np.load(prediction_file) as saved:
            np.testing.assert_array_equal(saved['indices'], evaluate)
            np.testing.assert_array_equal(saved['y_true'], y[evaluate])
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    print(f'[{stage}/{name}] train={len(fit)}, evaluate={len(evaluate)}', flush=True)
    if name in ('cnn','fusion'):
        # Epoch count is fixed in advance; neither CV nor test is used for early stopping.
        model, norm, history, _ = fit_nn(name,x,features,y,fit,None,cfg,seed,cfg['max_epochs'])
        write_csv(directory/'training_history.csv', history)
        pred = predict_loader(model,make_loader(x,features,y,evaluate,norm,256,False,seed))
        torch.save({'state_dict':model.state_dict(),'name':name,'epochs':cfg['max_epochs']},model_file)
        np.savez(model_dir/'normalization.npz',**norm)
    else:
        columns = feature_indices(name)
        if name == 'logistic':
            model = make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',
                max_iter=2000,random_state=seed))
        else:
            model = RandomForestClassifier(n_estimators=cfg['rf_trees'],min_samples_leaf=cfg['rf_min_samples_leaf'],
                max_features=cfg['rf_max_features'],class_weight='balanced_subsample',random_state=seed,n_jobs=6)
        model.fit(features[fit][:,columns],y[fit])
        pred = model.predict(features[evaluate][:,columns])
        joblib.dump({'model':model,'feature_columns':columns},model_file,compress=3)
    np.savez_compressed(prediction_file,indices=evaluate,y_true=y[evaluate],y_pred=pred,
        subject=subjects[evaluate],segment=metadata[evaluate,0],
        start_seconds=metadata[evaluate,1],end_seconds=metadata[evaluate,2])
    result = {'model':name,'stage':stage,'seed':seed,**metrics(y[evaluate],pred),
        'train_windows':len(fit),'evaluation_windows':len(evaluate),
        'train_segments':sorted(np.unique(metadata[fit,0].astype(int)).tolist()),
        'evaluation_segments':sorted(np.unique(metadata[evaluate,0].astype(int)).tolist()),
        'subject_disjoint':False, 'epochs':cfg['max_epochs'] if name in ('cnn','fusion') else None,
        'seconds':time.monotonic()-begin, 'config_sha256':config_hash,'split_sha256':split_hash,
        'model_sha256':sha256(model_file),'predictions_sha256':sha256(prediction_file)}
    save_json(score_file,result)
    print(f'[{stage}/{name}] F1={result["macro_f1"]:.4f}, accuracy={result["accuracy"]:.4f}',flush=True)
    return result


def read_result(stage, name):
    return json.loads((OUT/stage/name/'metrics.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--cv-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    selected = args.models or cfg['models']
    if not set(selected) <= set(cfg['models']):
        raise ValueError('Unknown model configuration.')
    seed_all(cfg['seed'])
    x,y,g,f,metadata = load_data()
    train,test,folds = load_split(y,g,metadata)
    save_json(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
        'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'torch_threads':torch.get_num_threads(),'config_sha256':sha256(ROOT/'configs/experiment.json')})
    for name in selected:
        for fold,(fit,val) in enumerate(folds,1):
            run_fit(name,f'cv{fold}',fit,val,x,y,g,f,metadata,cfg,cfg['seed']+fold)
    if not all((OUT/f'cv{k}'/name/'metrics.json').exists()
               for name in cfg['models'] for k in range(1,4)):
        print('Partial CV finished. All configured CV runs are required before testing.',flush=True)
        return
    cv_rows = []
    for name in cfg['models']:
        scores = [read_result(f'cv{k}',name) for k in range(1,4)]
        cv_rows.append({'model':name,'folds':3,'validation_windows':len(train),
            'cv_mean_f1':float(np.mean([r['macro_f1'] for r in scores])),
            'cv_std_f1':float(np.std([r['macro_f1'] for r in scores],ddof=1)),
            **{f'fold{k}_f1':scores[k-1]['macro_f1'] for k in range(1,4)}})
    write_csv(OUT/'cv_summary.csv',cv_rows)
    chosen = max((r for r in cv_rows if r['model'] in MAIN_MODELS),key=lambda r:r['cv_mean_f1'])['model']
    selection = {'selected_model':chosen,'selection_metric':'Mean 12-class macro-F1 over training-only three-fold group CV',
        'candidates':MAIN_MODELS,'test_used_for_selection':False,
        'config_sha256':sha256(ROOT/'configs/experiment.json'),
        'split_sha256':sha256(OUT/'split_assignments.csv'),
        'cv_score_hashes':{f'{name}/cv{k}':sha256(OUT/f'cv{k}'/name/'metrics.json')
            for name in cfg['models'] for k in range(1,4)}}
    selection_file = OUT/'model_selection.json'
    if selection_file.exists():
        if json.loads(selection_file.read_text(encoding='utf-8')) != selection:
            raise RuntimeError('Frozen pre-test model selection changed.')
    else:
        save_json(selection_file,selection)
    print(f'Frozen selection from CV: {chosen}. Test labels did not determine this selection.',flush=True)
    if args.cv_only:
        return
    # Ten prespecified systems share exactly the same test set; selection is already frozen.
    for name in cfg['models']:
        run_fit(name,'test',train,test,x,y,g,f,metadata,cfg,cfg['seed'])


if __name__ == '__main__':
    main()
