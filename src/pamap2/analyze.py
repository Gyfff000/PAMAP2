"""Recompute every reported number from outer-fold predictions; never refit a model."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from .common import ROOT, SUBJECTS, LABEL_NAMES, save_json, write_csv, sha256, ensure_predictions

PRETTY={'logistic':'Logistic regression','rf_all':'Random forest','cnn':'Early-fusion CNN',
        'fusion':'Sensor + feature fusion','rf_wrist':'Wrist only','rf_chest':'Chest only',
        'rf_ankle':'Ankle only','rf_no_wrist':'Without wrist','rf_no_chest':'Without chest','rf_no_ankle':'Without ankle'}

def load_predictions(name,seed=42):
    result=[]
    for subject in SUBJECTS:
        path=ROOT/f'outputs/folds/{name}_seed{seed}/{subject}/predictions.npz'
        if not path.exists():
            raise FileNotFoundError(f'Incomplete experiment: {path}')
        with np.load(path) as p:
            result.append({k:p[k] for k in p.files})
    return result

def paired_subject_comparison(a,b):
    """Pair by participant, with supported-label macro-F1, never independent windows."""
    delta=np.array(a)-np.array(b)
    rng=np.random.default_rng(20261018)
    boot=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
    observed=abs(delta.mean())
    signs=np.array(list(itertools.product([-1,1],repeat=len(delta))))
    null=np.abs((signs*delta).mean(1))
    return {'differences':delta.tolist(),'mean_difference':float(delta.mean()),
            'bootstrap_percentile_95ci':np.quantile(boot,[.025,.975]).tolist(),
            'exact_sign_flip_two_sided_p':float((null>=observed-1e-12).mean()),
            'n_subjects':len(delta),'note':'Exploratory comparison; n=8, percentile CI descriptive, no multiple-comparison correction.'}

def main():
    ensure_predictions()
    cfg=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    summary=json.loads((ROOT/'outputs/model_summary_seed42.json').read_text(encoding='utf-8'))
    if {r['model'] for r in summary}!=set(cfg['models']):
        raise RuntimeError('Run all prespecified model configurations before analysis.')
    folder=ROOT/'outputs/figures'
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    cache={name:load_predictions(name) for name in cfg['models']}
    expected_y=np.load(ROOT/'data/processed/labels.npy')
    expected_g=np.load(ROOT/'data/processed/subjects.npy')
    audit=[]
    supported={}
    class_scores={}
    counts=None
    prediction_rows=[]
    for name,folds in cache.items():
        for subject,fold in zip(SUBJECTS,folds):
            np.testing.assert_array_equal(fold['indices'],np.flatnonzero(expected_g==subject))
            np.testing.assert_array_equal(fold['y_true'],expected_y[fold['indices']])
            np.testing.assert_array_equal(fold['subject'],expected_g[fold['indices']])
            stored_fold=json.loads((ROOT/f'outputs/folds/{name}_seed42/{subject}/metrics.json').read_text(encoding='utf-8'))
            if subject in stored_fold['train_subjects']:
                raise AssertionError('Test participant appears in model training provenance.')
        all_indices=np.concatenate([f['indices'] for f in folds])
        if len(np.unique(all_indices))!=len(all_indices) or set(all_indices)!=set(range(len(all_indices))):
            raise AssertionError('Every processed window must be tested exactly once.')
        y=np.concatenate([f['y_true'] for f in folds]);pred=np.concatenate([f['y_pred'] for f in folds])
        recomputed=f1_score(y,pred,labels=range(12),average='macro',zero_division=0)
        stored=next(r for r in summary if r['model']==name)
        assert abs(stored['macro_f1']-recomputed)<1e-12
        assert abs(stored['accuracy']-accuracy_score(y,pred))<1e-12
        supported[name]=[float(f1_score(f['y_true'],f['y_pred'],labels=np.unique(f['y_true']),average='macro',zero_division=0)) for f in folds]
        stored['supported_subject_mean_f1']=float(np.mean(supported[name]))
        stored['supported_subject_std_f1']=float(np.std(supported[name],ddof=1))
        class_scores[name]=f1_score(y,pred,labels=range(12),average=None,zero_division=0).tolist()
        audit.append({'model':name,'tested_windows':len(y),'exactly_once':True,
                      'recomputed_macro_f1':float(recomputed),'prediction_hashes':[
                          sha256(ROOT/f'outputs/folds/{name}_seed42/{s}/predictions.npz') for s in SUBJECTS]})
        counts=np.bincount(y,minlength=12)
        if name in ('logistic','rf_all','cnn','fusion'):
            for f in folds:
                for i in range(len(f['indices'])):
                    prediction_rows.append({'model':name,'window_index':int(f['indices'][i]),
                        'subject':int(f['subject'][i]),'true_activity':LABEL_NAMES[int(f['y_true'][i])],
                        'predicted_activity':LABEL_NAMES[int(f['y_pred'][i])],
                        'segment_id':int(f['segment'][i]),'start_seconds':float(f['start_seconds'][i]),
                        'end_seconds':float(f['end_seconds'][i])})
    write_csv(ROOT/'outputs/outer_predictions.csv',prediction_rows)
    write_csv(ROOT/'outputs/model_comparison.csv',summary)
    write_csv(ROOT/'outputs/class_f1.csv',[
        {'activity':label,'windows':int(counts[k]),**{n:class_scores[n][k] for n in class_scores}}
        for k,label in enumerate(LABEL_NAMES)])
    subject_rows=[{'subject':s,**{n:supported[n][i] for n in supported}} for i,s in enumerate(SUBJECTS)]
    write_csv(ROOT/'outputs/supported_subject_f1.csv',subject_rows)
    comparisons={f'{a}_minus_{b}':paired_subject_comparison(supported[a],supported[b])
                 for a,b in [('fusion','rf_all'),('fusion','cnn'),('rf_all','rf_wrist'),
                             ('rf_all','rf_chest'),('rf_all','rf_ankle')]}
    save_json(ROOT/'outputs/statistical_comparisons.json',comparisons)
    save_json(ROOT/'outputs/result_audit.json',{'checks':audit,'all_predictions_verified':True,
              'config_sha256':sha256(ROOT/'configs/experiment.json'),
              'dataset_audit_sha256':sha256(ROOT/'outputs/data_audit.json')})

    def finish(fig,name):
        fig.savefig(folder/f'{name}.pdf')
        fig.savefig(folder/f'{name}.png',dpi=300)
        plt.close(fig)

    main_names=['logistic','rf_all','cnn','fusion']
    fig,ax=plt.subplots(figsize=(8.2,4))
    x=np.arange(4)
    ax.bar(x-.18,[next(r['macro_f1'] for r in summary if r['model']==n) for n in main_names],
           width=.34,color='#245b85',label='Pooled 12-class macro F1')
    ax.bar(x+.18,[np.mean(supported[n]) for n in main_names],width=.34,
           yerr=[np.std(supported[n],ddof=1) for n in main_names],capsize=4,color='#b77340',label='Subject mean F1 (supported labels)')
    ax.set_xticks(x,[PRETTY[n] for n in main_names],fontsize=9)
    ax.set_ylabel('Macro F1');ax.set_ylim(0,1.12);ax.legend(frameon=False,fontsize=9)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();finish(fig,'model_comparison')

    fig,axes=plt.subplots(2,1,figsize=(7.4,10.3),layout='constrained')
    for ax,name in zip(axes,['rf_all','fusion']):
        folds=cache[name];y=np.concatenate([f['y_true'] for f in folds]);pr=np.concatenate([f['y_pred'] for f in folds])
        cm=confusion_matrix(y,pr,labels=range(12));norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        for i in range(12):
            for j in range(12):
                if norm[i,j]>=.02:
                    ax.text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',fontsize=11,color='white' if norm[i,j]>.55 else '#24384a')
        ax.set_xticks(range(12),[str(i+1) for i in range(12)])
        ax.set_yticks(range(12),[str(i+1) for i in range(12)])
        ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(PRETTY[name])
    fig.colorbar(im,ax=axes,shrink=.75,label='Within-class recall');finish(fig,'confusion_matrices')

    ablations=['rf_wrist','rf_chest','rf_ankle','rf_no_wrist','rf_no_chest','rf_no_ankle','rf_all']
    fig,ax=plt.subplots(figsize=(8.5,4.2))
    for i,s in enumerate(SUBJECTS):
        ax.plot(range(7),[supported[n][i] for n in ablations],color='#8898a6',alpha=.38,lw=.8)
    ax.plot(range(7),[np.mean(supported[n]) for n in ablations],color='#245b85',marker='o',lw=2.5,label='Subject mean')
    ax.set_xticks(range(7),[PRETTY[n].replace(' ','\n',1) for n in ablations],fontsize=9)
    ax.set_ylim(0,1.03);ax.set_ylabel('Subject macro F1 (supported labels)');ax.legend(frameon=False)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();finish(fig,'sensor_ablation')

    fig,ax=plt.subplots(figsize=(9,4.6))
    for n,c in zip(main_names,['#5b6874','#245b85','#a26f50','#499277']):
        ax.plot(range(12),class_scores[n],marker='o',label=PRETTY[n],color=c,lw=1.5,ms=3)
    ax.set_xticks(range(12),[str(i+1) for i in range(12)]);ax.set_ylim(0,1.04)
    ax.set_xlabel('Activity class (mapping in the paper)');ax.set_ylabel('Pooled class F1')
    ax.legend(frameon=False,ncol=2,fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();finish(fig,'class_f1')

    fig,ax=plt.subplots(figsize=(8.5,4.2))
    for n in main_names:
        ax.plot(SUBJECTS,supported[n],marker='o',label=PRETTY[n],lw=1.4,ms=3)
    ax.set_xticks(SUBJECTS);ax.set_ylim(0,1.03);ax.set_xlabel('Held-out subject')
    ax.set_ylabel('Macro F1 (supported labels)');ax.legend(frameon=False,ncol=2,fontsize=9)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();finish(fig,'subject_generalization')

    with (ROOT/'outputs/class_subject_counts.csv').open(encoding='utf-8-sig') as f:
        cc=list(csv.DictReader(f))
    matrix=np.array([[int(r['windows']) for r in cc if int(r['subject'])==s] for s in SUBJECTS])
    fig,ax=plt.subplots(figsize=(9,4))
    im=ax.imshow(matrix,aspect='auto',cmap='YlGnBu')
    for i in range(8):
        for j in range(12):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=10,color='white' if matrix[i,j]>matrix.max()*.55 else '#24384a')
    ax.set_yticks(range(8),SUBJECTS);ax.set_xticks(range(12),range(1,13))
    ax.set_xlabel('Activity class');ax.set_ylabel('Subject');fig.colorbar(im,ax=ax,label='Windows')
    fig.tight_layout();finish(fig,'data_coverage')
    print('All predictions audited; tables, figures and participant-level comparisons saved.',flush=True)

if __name__=='__main__':
    main()
