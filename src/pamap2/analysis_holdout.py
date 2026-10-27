"""Audit frozen partitions and recompute CV/test tables and figures."""
import csv
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from .common import ROOT, LABEL_NAMES, SUBJECTS, save_json, write_csv, sha256, load_data
from .protocol import OUT, load_split
from .experiment import metrics
from .holdout import MAIN_MODELS, read_result

PRETTY = {'logistic':'Logistic regression','rf_all':'Random forest','cnn':'Early-fusion CNN',
    'fusion':'Sensor + feature fusion','rf_wrist':'Wrist only','rf_chest':'Chest only',
    'rf_ankle':'Ankle only','rf_no_wrist':'Without wrist','rf_no_chest':'Without chest','rf_no_ankle':'Without ankle'}


def ensure_predictions():
    archive = ROOT/'artifacts/holdout_predictions.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                relative = entry.filename
                if '..' in relative.split('/') or not relative.startswith('outputs/holdout/') or not relative.endswith('/predictions.npz'):
                    raise ValueError('Unexpected archive entry.')
                target = ROOT/relative
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(entry))


def main():
    ensure_predictions()
    cfg = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    selection = json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    _,y,g,_,metadata = load_data()
    train,test,folds = load_split(y,g,metadata)
    cache,summary,audits,all_rows = {},[],[],[]
    for name in cfg['models']:
        cv = []
        for stage,fit,indices in [(f'cv{k}',a,b) for k,(a,b) in enumerate(folds,1)]+[('test',train,test)]:
            directory = OUT/stage/name
            stored = read_result(stage,name)
            with np.load(directory/'predictions.npz') as saved:
                p = {k:saved[k] for k in saved.files}
            np.testing.assert_array_equal(p['indices'],indices)
            np.testing.assert_array_equal(p['y_true'],y[indices])
            np.testing.assert_array_equal(p['subject'],g[indices])
            np.testing.assert_array_equal(p['segment'],metadata[indices,0])
            assert set(stored['train_segments']) == set(metadata[fit,0].astype(int))
            assert not set(stored['train_segments']) & set(p['segment'])
            assert stored['config_sha256'] == sha256(ROOT/'configs/experiment.json')
            assert stored['split_sha256'] == sha256(OUT/'split_assignments.csv')
            assert stored['predictions_sha256'] == sha256(directory/'predictions.npz')
            recomputed = metrics(p['y_true'],p['y_pred'])
            for key,value in recomputed.items():
                assert abs(stored[key]-value)<1e-12
            audits.append({'model':name,'stage':stage,'windows':len(indices),'scores_recomputed':True,
                'segment_isolation':True,'predictions_sha256':sha256(directory/'predictions.npz')})
            if stage != 'test':
                assert selection['cv_score_hashes'][f'{name}/{stage}'] == sha256(directory/'metrics.json')
                assert not set(indices) & set(test)
                cv.append(recomputed['macro_f1'])
            else:
                cache[name] = p
                summary.append({'model':name,'train_windows':len(train),'test_windows':len(test),
                    'cv_mean_f1':float(np.mean(cv)),'cv_std_f1':float(np.std(cv,ddof=1)),
                    **recomputed,'selected_by_cv':name==selection['selected_model']})
                if name in MAIN_MODELS:
                    for j,index in enumerate(indices):
                        all_rows.append({'model':name,'window_index':int(index),'subject':int(g[index]),
                            'segment':int(metadata[index,0]),'true_activity':LABEL_NAMES[y[index]],
                            'predicted_activity':LABEL_NAMES[p['y_pred'][j]]})
    chosen = max((r for r in summary if r['model'] in MAIN_MODELS),key=lambda r:r['cv_mean_f1'])['model']
    assert chosen == selection['selected_model'] and not selection['test_used_for_selection']
    write_csv(OUT/'model_comparison.csv',summary)
    save_json(OUT/'model_summary.json',summary)
    write_csv(OUT/'test_predictions.csv',all_rows)
    class_scores = {n:f1_score(p['y_true'],p['y_pred'],labels=range(12),average=None,zero_division=0)
                    for n,p in cache.items()}
    write_csv(OUT/'class_f1.csv',[{'activity':LABEL_NAMES[k],'windows':int((y[test]==k).sum()),
        **{n:float(class_scores[n][k]) for n in cfg['models']}} for k in range(12)])
    subject_rows = []
    for s in SUBJECTS:
        take = g[test]==s
        if not take.any():
            continue
        subject_rows.append({'subject':s,'test_windows':int(take.sum()),
            **{n:float(f1_score(p['y_true'][take],p['y_pred'][take],labels=np.unique(p['y_true'][take]),
                                average='macro',zero_division=0)) for n,p in cache.items()}})
    write_csv(OUT/'subject_f1.csv',subject_rows)
    save_json(OUT/'result_audit.json',{'protocol_version':2,'train_windows':len(train),'test_windows':len(test),
        'exact_70_30':3*len(train)==7*len(test),'cv_folds':3,
        'cv_validation_covers_training_exactly_once':True,'test_absent_from_all_cv':True,
        'group_unit':'recording segment','subject_disjoint':False,'model_fits_audited':len(audits),
        'selected_model':chosen,'selection_uses_cv_only':True,'checks':audits,
        'config_sha256':sha256(ROOT/'configs/experiment.json'),
        'split_sha256':sha256(OUT/'split_assignments.csv')})
    figure_dir = OUT/'figures'
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    def finish(fig,name):
        fig.savefig(figure_dir/f'{name}.pdf');fig.savefig(figure_dir/f'{name}.png',dpi=220);plt.close(fig)
    lookup = {r['model']:r for r in summary}
    fig,ax = plt.subplots(figsize=(8.4,4))
    xx = np.arange(4)
    ax.bar(xx-.18,[lookup[n]['cv_mean_f1'] for n in MAIN_MODELS],width=.34,
        yerr=[lookup[n]['cv_std_f1'] for n in MAIN_MODELS],capsize=4,label='Training 3-fold CV mean +/- SD',color='#b77340')
    ax.bar(xx+.18,[lookup[n]['macro_f1'] for n in MAIN_MODELS],width=.34,label='Fixed 30% test',color='#245b85')
    ax.set_xticks(xx,[PRETTY[n] for n in MAIN_MODELS],fontsize=9);ax.set_ylabel('12-class macro F1')
    ax.set_ylim(0,1.13);ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();finish(fig,'model_comparison')
    ablations = ['rf_wrist','rf_chest','rf_ankle','rf_no_wrist','rf_no_chest','rf_no_ankle','rf_all']
    fig,ax = plt.subplots(figsize=(8.5,4.2))
    ax.plot(range(7),[lookup[n]['cv_mean_f1'] for n in ablations],marker='s',label='Training CV mean',color='#b77340')
    ax.plot(range(7),[lookup[n]['macro_f1'] for n in ablations],marker='o',label='Fixed test',color='#245b85')
    ax.set_xticks(range(7),[PRETTY[n].replace(' ','\n',1) for n in ablations],fontsize=9)
    ax.set_ylim(0,1.04);ax.set_ylabel('12-class macro F1');ax.legend(frameon=False)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();finish(fig,'sensor_ablation')
    fig,axes = plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for ax,name in zip(axes,['rf_all','fusion']):
        p = cache[name];cm = confusion_matrix(p['y_true'],p['y_pred'],labels=range(12))
        normalized = cm/np.maximum(cm.sum(1,keepdims=True),1)
        im = ax.imshow(normalized*100,cmap='Blues',vmin=0,vmax=100)
        for i in range(12):
            for j in range(12):
                if normalized[i,j]>=.02:
                    ax.text(j,i,f'{100*normalized[i,j]:.0f}',ha='center',va='center',fontsize=14,
                        color='white' if normalized[i,j]>.55 else '#24384a')
        ax.set_xticks(range(12),range(1,13));ax.set_yticks(range(12),range(1,13))
        ax.tick_params(labelsize=14)
        ax.set_xlabel('Predicted class',fontsize=15);ax.set_ylabel('True class',fontsize=15)
        ax.set_title(PRETTY[name],fontsize=16)
    colorbar=fig.colorbar(im,ax=axes,shrink=.8)
    colorbar.set_label('Within-class recall (%)',fontsize=14);colorbar.ax.tick_params(labelsize=13)
    finish(fig,'confusion_matrices')
    fig,ax = plt.subplots(figsize=(8.5,4.2))
    for name in MAIN_MODELS:
        ax.plot([r['subject'] for r in subject_rows],[r[name] for r in subject_rows],marker='o',label=PRETTY[name])
    ax.set_xticks([r['subject'] for r in subject_rows]);ax.set_ylim(0,1.04)
    ax.set_xlabel('Subject');ax.set_ylabel('Test F1, supported labels')
    ax.legend(frameon=False,ncol=2,fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();finish(fig,'subject_generalization')
    with (ROOT/'outputs/class_subject_counts.csv').open(encoding='utf-8-sig') as f:
        counts = list(csv.DictReader(f))
    matrix = np.array([[int(r['windows']) for r in counts if int(r['subject'])==s] for s in SUBJECTS])
    fig,ax = plt.subplots(figsize=(9,4));im = ax.imshow(matrix,aspect='auto',cmap='YlGnBu')
    for i in range(8):
        for j in range(12):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=10,
                color='white' if matrix[i,j]>matrix.max()*.55 else '#24384a')
    ax.set_yticks(range(8),SUBJECTS);ax.set_xticks(range(12),range(1,13));ax.set_xlabel('Activity class')
    ax.set_ylabel('Subject');fig.colorbar(im,ax=ax,label='All processed windows');fig.tight_layout();finish(fig,'data_coverage')
    archive = ROOT/'artifacts/holdout_predictions.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.glob('*/*/predictions.npz')):
            z.write(path,path.relative_to(ROOT).as_posix())
    print(f'Audited {len(audits)} fits. Exact 5187/2223 split; selection from training CV: {chosen}.',flush=True)


if __name__ == '__main__':
    main()
