"""Generate paper tables and factual result descriptions from audited outputs."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'src'))
from pamap2.common import ROOT, SUBJECTS, LABEL_ZH, save_json, sha256
from pamap2.models import make_model

ZH={'logistic':'逻辑回归','rf_all':'随机森林','cnn':'早期融合 CNN','fusion':'部位与统计融合',
    'rf_wrist':'仅手腕','rf_chest':'仅胸部','rf_ankle':'仅脚踝','rf_no_wrist':'移除手腕','rf_no_chest':'移除胸部','rf_no_ankle':'移除脚踝'}
OUT=ROOT/'paper/generated'

def read_csv(name):
    with (ROOT/'outputs'/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def table(filename,columns,headers,rows):
    text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(str(c) for c in row)+' \\\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (OUT/filename).write_text(text,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((ROOT/'outputs/data_audit.json').read_text(encoding='utf-8'))
    env=json.loads((ROOT/'outputs/environment.json').read_text(encoding='utf-8'))
    rows=read_csv('model_comparison.csv'); results={r['model']:r for r in rows}
    class_rows=read_csv('class_f1.csv'); subject_rows=read_csv('supported_subject_f1.csv')
    comparison=json.loads((ROOT/'outputs/statistical_comparisons.json').read_text(encoding='utf-8'))
    best=max(['logistic','rf_all','cnn','fusion'],key=lambda n:float(results[n]['macro_f1']))
    macros={'LRFone':f'{float(results["logistic"]["macro_f1"]):.4f}',
            'RFFone':f'{float(results["rf_all"]["macro_f1"]):.4f}',
            'CNNFone':f'{float(results["cnn"]["macro_f1"]):.4f}',
            'FusionFone':f'{float(results["fusion"]["macro_f1"]):.4f}',
            'BestName':ZH[best],'BestFone':f'{float(results[best]["macro_f1"]):.4f}',
            'RawRows':str(sum(r['raw_rows'] for r in audit['subjects'] if r['primary_included'])),
            'CNNParams':str(sum(p.numel() for p in make_model('cnn').parameters())),
            'FusionParams':str(sum(p.numel() for p in make_model('fusion').parameters())),
            'PythonVersion':env['python'].split()[0],'TorchVersion':env['torch'].replace('_',r'\_'),
            'SklearnVersion':env['scikit_learn']}
    (OUT/'results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    counts=read_csv('class_subject_counts.csv')
    table('data_table.tex','rrrrr',['受试者','原始记录数','活动数','窗口数','缺失率（\\%）'],
          [[r['subject'],r['raw_rows'],sum(int(c['windows'])>0 for c in counts if int(c['subject'])==r['subject']),
            r['windows'],f'{100*r["missing_sensor_fraction"]:.3f}'] for r in audit['subjects'] if r['primary_included']])
    def metric_row(name):
        r=results[name]
        return [ZH[name],f'{float(r["accuracy"]):.4f}',f'{float(r["macro_f1"]):.4f}',
                f'{float(r["supported_subject_mean_f1"]):.4f} $\\pm$ {float(r["supported_subject_std_f1"]):.4f}']
    table('main_table.tex','lrrr',['模型','Accuracy','汇总 Macro-F1','个人 F1 均值 $\\pm$ SD'],
          [metric_row(n) for n in ['logistic','rf_all','cnn','fusion']])
    table('ablation_table.tex','lrrr',['输入部位','Accuracy','汇总 Macro-F1','个人 F1 均值 $\\pm$ SD'],
          [metric_row(n) for n in ['rf_wrist','rf_chest','rf_ankle','rf_no_wrist','rf_no_chest','rf_no_ankle','rf_all']])
    table('class_table.tex','rlrrrr',['编号','活动','窗口数','逻辑回归','随机森林','融合模型'],
          [[i+1,LABEL_ZH[i],r['windows'],f'{float(r["logistic"]):.4f}',f'{float(r["rf_all"]):.4f}',f'{float(r["fusion"]):.4f}'] for i,r in enumerate(class_rows)])
    epoch_rows=[]
    for s in SUBJECTS:
        a=json.loads((ROOT/f'outputs/folds/cnn_seed42/{s}/metrics.json').read_text(encoding='utf-8'))
        b=json.loads((ROOT/f'outputs/folds/fusion_seed42/{s}/metrics.json').read_text(encoding='utf-8'))
        epoch_rows.append([s,a['inner_validation_subject'],a['selected_epochs'],b['selected_epochs']])
    table('epoch_table.tex','rrrr',['外层测试人','内层验证人','CNN 轮数','融合模型轮数'],epoch_rows)
    def f(n,key):return float(results[n][key])
    diff=(f('fusion','macro_f1')-f('rf_all','macro_f1'))*100
    paired=comparison['fusion_minus_rf_all'];lo,hi=np.array(paired['bootstrap_percentile_95ci'])*100
    text=(f'四种主模型中，{ZH[best]}的汇总 Macro-F1 最高（{f(best,"macro_f1"):.4f}），'
          f'Accuracy 为 {f(best,"accuracy"):.4f}。融合模型相对随机森林的汇总 F1 差值为 {diff:+.2f} 个百分点。'
          f'表\\ref{{tab:main}}同时展示个人均值与离散程度，用于区分总体窗口表现与个体迁移波动。\n\n'
          f'按个人实际出现类别 F1 进行配对，融合减随机森林的均值为 {paired["mean_difference"]*100:+.2f} 个百分点，'
          f'受试者重采样百分位区间为 [{lo:.2f}, {hi:.2f}] 个百分点，探索性符号翻转双侧 $p={paired["exact_sign_flip_two_sided_p"]:.4f}$。'
          '这些统计只基于八个配对受试者，不能把区间或单个最好分数当作大样本人群中的保证。\n\n'
          '融合与早期 CNN 的输入表示和容量均不同；即使融合成绩较高，也不能只据此证明多分支结构本身有效。'
          '现有实验没有单独删除融合模型的统计分支或保持等参数量，因而只能比较完整系统。')
    (OUT/'main_discussion.tex').write_text(text,encoding='utf-8')
    best_single=max(['rf_wrist','rf_chest','rf_ankle'],key=lambda n:f(n,'macro_f1'))
    removed=min(['rf_no_wrist','rf_no_chest','rf_no_ankle'],key=lambda n:f(n,'macro_f1'))
    text=(f'单部位中，{ZH[best_single]}的汇总 Macro-F1 最高，为 {f(best_single,"macro_f1"):.4f}。'
          f'完整三部位随机森林的成绩为 {f("rf_all","macro_f1"):.4f}。'
          f'三种移除配置中，{ZH[removed]}的汇总分数最低，为 {f(removed,"macro_f1"):.4f}。'
          '图\\ref{fig:ablation}显示同一配置对不同受试者的影响，能够避免只依赖汇总均值解释部位信息。\n\n'
          '部位包含的姿态、身体运动和局部动作信息具有不同分布；增加部位既可能补充信号，也可能引入个体差异。'
          '这里的结果支持对实际传感器组合进行验证，而不支持“传感器越多必然越好”的无条件结论。')
    (OUT/'ablation_discussion.tex').write_text(text,encoding='utf-8')
    worst=sorted(range(12),key=lambda i:float(class_rows[i]['fusion']))[:3]
    improvements=sorted(range(12),key=lambda i:float(class_rows[i]['fusion'])-float(class_rows[i]['rf_all']))
    text=('融合模型中 F1 较低的三个类别为'+ '、'.join(f'{LABEL_ZH[i]}（{float(class_rows[i]["fusion"]):.4f}）' for i in worst)+'。'
          f'与随机森林相比，类别 F1 差值最小的活动为{LABEL_ZH[improvements[0]]}，最大的为{LABEL_ZH[improvements[-1]]}。'
          '由于各活动的训练覆盖、样本量及动作特征不同，汇总成绩不能替代逐类别分析。'
          '混淆矩阵按真实类别逐行归一化，便于查看类别内部的误判分布；它不同于按预测类别归一化的精确率。')
    (OUT/'class_discussion.tex').write_text(text,encoding='utf-8')
    worst_subject=min(subject_rows,key=lambda r:float(r['fusion']))
    best_subject=max(subject_rows,key=lambda r:float(r['fusion']))
    text=(f'融合模型在受试者 {worst_subject["subject"]} 上的实际出现类别 F1 最低，为 {float(worst_subject["fusion"]):.4f}；'
          f'在受试者 {best_subject["subject"]} 上最高，为 {float(best_subject["fusion"]):.4f}。'
          '这些差异同时受到个体运动分布与录制类别覆盖影响，不能仅按一个人缺少的类别数解释，'
          '也不能由少量受试者直接推断年龄、性别或具体生理因素造成差异。')
    (OUT/'subject_discussion.tex').write_text(text,encoding='utf-8')
    save_json(ROOT/'paper/asset_manifest.json',{'macros':macros,
              'sources':{n:sha256(ROOT/'outputs'/n) for n in ['data_audit.json','environment.json','model_comparison.csv','class_f1.csv','supported_subject_f1.csv','statistical_comparisons.json','result_audit.json']},
              'figures':{p.name:sha256(p) for p in (ROOT/'outputs/figures').glob('*.pdf')}})
    print('Paper tables and factual result descriptions generated from audited predictions.')

if __name__=='__main__':
    main()
