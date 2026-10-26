"""Export the CV-selected 70%-trained model and fixed test examples, without refitting."""
import json
import shutil
import joblib
import numpy as np
import torch
from .common import ROOT, LABEL_NAMES, load_data, save_json, write_csv, sha256
from .protocol import OUT, load_split
from .experiment import make_loader, predict_loader
from .models import make_model


def main():
    selection = json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    name = selection['selected_model']
    x,y,g,f,metadata = load_data()
    train,test,_ = load_split(y,g,metadata)
    source = ROOT/'models/holdout/test'/name
    # The final artifact is the exact independently evaluated 70%-trained model.
    with np.load(OUT/'test'/name/'predictions.npz') as p:
        np.testing.assert_array_equal(p['indices'],test)
        lookup = dict(zip(p['indices'].tolist(),p['y_pred'].tolist()))
    selected = []
    for k in range(12):
        available = test[y[test]==k]
        selected.extend(available[np.minimum((np.array([.25,.75])*len(available)).astype(int),len(available)-1)])
    selected = np.array(selected,dtype=np.int64)
    for purpose in ['final','demo']:
        directory = ROOT/'artifacts'/purpose
        directory.mkdir(exist_ok=True)
        # Remove only superseded model formats inside these two named artifact folders.
        for filename in ['model.pt','model.joblib','normalization.npz']:
            target = directory/filename
            if target.exists() and not (source/filename).exists():
                target.unlink()
            if (source/filename).exists():
                shutil.copy2(source/filename,target)
        if purpose == 'demo':
            np.savez_compressed(directory/'sample_windows.npz',windows=np.asarray(x[selected]),
                features=f[selected],y_true=y[selected],window_indices=selected,subject=g[selected],
                expected_prediction=np.array([lookup[int(i)] for i in selected]))
        manifest = {'protocol_version':2,'model':name,'training_windows':len(train),
            'test_windows':len(test),'subject_disjoint':False,'group_disjoint':True,
            'selection_rule':selection['selection_metric'],'test_used_for_selection':False,
            'description':'Same 70%-trained model as final test; no all-data refit. Demo uses two fixed temporal-index quantiles per test class, without correctness filtering.',
            'samples':len(selected) if purpose=='demo' else None,'label_names':LABEL_NAMES,
            'source_dataset':'PAMAP2, UCI DOI:10.24432/C5NW2H, CC BY 4.0',
            'config_sha256':sha256(ROOT/'configs/experiment.json'),
            'split_sha256':sha256(OUT/'split_assignments.csv'),
            'files':{p.name:sha256(p) for p in directory.iterdir() if p.name!='manifest.json'}}
        save_json(directory/'manifest.json',manifest)
    print(f'Exported the CV-selected {name} model trained on 5,187 windows; test data not refitted.',flush=True)


def predict_demo():
    from .prepare import features as extract_features
    directory = ROOT/'artifacts/demo'
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        if sha256(directory/name) != digest:
            raise RuntimeError(f'Artifact integrity mismatch: {name}')
    with np.load(directory/'sample_windows.npz') as data:
        x,f,y,indices,subjects,expected = [data[k] for k in
            ['windows','features','y_true','window_indices','subject','expected_prediction']]
    np.testing.assert_allclose(extract_features(x),f,rtol=1e-6,atol=1e-6)
    if manifest['model'] in ('cnn','fusion'):
        state = torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
        model = make_model(manifest['model']);model.load_state_dict(state['state_dict'])
        with np.load(directory/'normalization.npz') as data:
            norm = {k:data[k] for k in data.files}
        prediction = predict_loader(model,make_loader(x,f,y,np.arange(len(y)),norm,256,False,42))
    else:
        saved = joblib.load(directory/'model.joblib')
        prediction = saved['model'].predict(f[:,saved['feature_columns']])
    np.testing.assert_array_equal(prediction,expected)
    print(f'模型：{manifest["model"]}；训练集 5187，独立片段测试集 2223（7∶3）')
    print('演示窗口的录制片段未参与训练；同一人的其他片段可能参与训练。')
    rows = []
    for i in range(len(y)):
        print(f'window {indices[i]:4d} | subject {subjects[i]} | true: {LABEL_NAMES[y[i]]:<18} | predicted: {LABEL_NAMES[prediction[i]]}')
        rows.append({'window_index':int(indices[i]),'subject':int(subjects[i]),'true_activity':LABEL_NAMES[y[i]],
            'predicted_activity':LABEL_NAMES[prediction[i]],'correct':bool(y[i]==prediction[i])})
    write_csv(OUT/'demo_predictions.csv',rows)
    print(f'演示正确 {int((y==prediction).sum())}/{len(y)}；正式成绩见完整 2223 个测试窗口的结果。')


if __name__ == '__main__':
    main()
