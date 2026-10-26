"""Replay all forty saved protocol-2 models, not merely their stored scores."""
import json
import subprocess
import sys
import joblib
import numpy as np
import torch
from .common import ROOT, load_data, save_json, seed_all, sha256
from .protocol import OUT, load_split
from .models import make_model
from .experiment import make_loader, predict_loader


def main():
    seed_all(42)
    tests = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    cfg = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    data_audit = json.loads((ROOT/'outputs/data_audit.json').read_text(encoding='utf-8'))
    for name,digest in data_audit['array_hashes'].items():
        assert sha256(ROOT/'data/processed'/name) == digest, f'Dataset changed: {name}'
    x,y,g,f,metadata = load_data()
    train,test,folds = load_split(y,g,metadata)
    rows = []
    for name in cfg['models']:
        for stage,indices in [(f'cv{k}',v) for k,(_,v) in enumerate(folds,1)]+[('test',test)]:
            model_dir = ROOT/'models/holdout'/stage/name
            expected = np.load(OUT/stage/name/'predictions.npz')
            np.testing.assert_array_equal(indices,expected['indices'])
            scores = json.loads((OUT/stage/name/'metrics.json').read_text(encoding='utf-8'))
            model_file = model_dir/('model.pt' if name in ('cnn','fusion') else 'model.joblib')
            assert sha256(model_file) == scores['model_sha256']
            if name in ('cnn','fusion'):
                state = torch.load(model_file,map_location='cpu',weights_only=True)
                model = make_model(name);model.load_state_dict(state['state_dict'])
                with np.load(model_dir/'normalization.npz') as saved:
                    norm = {k:saved[k] for k in saved.files}
                prediction = predict_loader(model,make_loader(x,f,y,indices,norm,256,False,42))
            else:
                saved = joblib.load(model_file)
                prediction = saved['model'].predict(f[indices][:,saved['feature_columns']])
            np.testing.assert_array_equal(prediction,expected['y_pred'],err_msg=f'{stage}/{name} replay differs')
            rows.append({'model':name,'stage':stage,'windows':len(indices),'exact_prediction_match':True})
        print(f'Verified three CV models and final test model: {name}',flush=True)
    paper_assets = json.loads((ROOT/'paper/asset_manifest.json').read_text(encoding='utf-8'))
    for path,digest in paper_assets['sources'].items():
        assert sha256(ROOT/path) == digest
    for name,digest in paper_assets['figures'].items():
        assert sha256(OUT/'figures'/name) == digest
    save_json(OUT/'verification.json',{'protocol_version':2,'unit_tests_passed':True,
        'unit_test_log':tests.stderr,'saved_models_verified':len(rows),'model_replays':rows,
        'processed_arrays_match_original_audit':True,'paper_sources_and_figures_verified':True,
        'source_hashes':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted((ROOT/'src/pamap2').glob('*.py'))},
        'config_sha256':sha256(ROOT/'configs/experiment.json'),
        'split_sha256':sha256(OUT/'split_assignments.csv'),'paper_pdf_sha256':sha256(ROOT/'paper/main.pdf')})


if __name__ == '__main__':
    main()
