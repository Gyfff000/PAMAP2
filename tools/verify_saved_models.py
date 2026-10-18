"""Replay all saved outer models and require exact agreement with stored predictions."""
from pathlib import Path
import json
import subprocess
import sys
import joblib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pamap2.common import ROOT, SUBJECTS, load_data, save_json, seed_all, sha256
from pamap2.models import make_model
from pamap2.experiment import make_loader, predict_loader

def main():
    seed_all(42)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:
        raise RuntimeError(test.stdout+test.stderr)
    x,y,g,f,meta=load_data()
    cfg=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    rows=[]
    for name in cfg['models']:
        for subject in SUBJECTS:
            directory=ROOT/f'models/folds/{name}_seed42/{subject}'
            expected=np.load(ROOT/f'outputs/folds/{name}_seed42/{subject}/predictions.npz')
            indices=expected['indices']
            if name in ('cnn','fusion'):
                saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
                model=make_model(name);model.load_state_dict(saved['state_dict'])
                with np.load(directory/'normalization.npz') as data:
                    norm={k:data[k] for k in data.files}
                pred=predict_loader(model,make_loader(x,f,y,indices,norm,256,False,42))
            else:
                saved=joblib.load(directory/'model.joblib')
                pred=saved['model'].predict(f[indices][:,saved['feature_columns']])
            np.testing.assert_array_equal(pred,expected['y_pred'],err_msg=f'{name}/{subject} replay differs')
            rows.append({'model':name,'test_subject':subject,'windows':len(indices),'exact_prediction_match':True})
        print(f'Verified all 8 saved models: {name}',flush=True)
    save_json(ROOT/'outputs/verification.json',{'unit_tests_passed':True,'unit_test_log':test.stderr,
        'saved_outer_models_verified':len(rows),'model_replays':rows,
        'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted((ROOT/'src/pamap2').glob('*.py'))},
        'config_sha256':sha256(ROOT/'configs/experiment.json'),'paper_pdf_sha256':sha256(ROOT/'paper/main.pdf')})

if __name__=='__main__':
    main()
