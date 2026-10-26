"""Build the paper's exact numeric tables from audited protocol-2 outputs."""
import csv
import json
import numpy as np
from .common import ROOT, LABEL_ZH, save_json, sha256
from .protocol import OUT
from .models import make_model

ZH = {'logistic':'逻辑回归','rf_all':'随机森林','cnn':'早期融合 CNN','fusion':'部位与统计融合',
    'rf_wrist':'仅手腕','rf_chest':'仅胸部','rf_ankle':'仅脚踝','rf_no_wrist':'移除手腕',
    'rf_no_chest':'移除胸部','rf_no_ankle':'移除脚踝'}
GENERATED = ROOT/'paper/generated'


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def table(filename,columns,headers,rows):
    text = '\\begin{tabular}{'+columns+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(str(v) for v in row)+' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (GENERATED/filename).write_text(text,encoding='utf-8')


def main():
    GENERATED.mkdir(exist_ok=True)
    audit = json.loads((OUT/'result_audit.json').read_text(encoding='utf-8'))
    assert audit['exact_70_30'] and audit['test_absent_from_all_cv'] and audit['model_fits_audited']==40
    rows = json.loads((OUT/'model_summary.json').read_text(encoding='utf-8'))
    results = {r['model']:r for r in rows}
    env = json.loads((OUT/'environment.json').read_text(encoding='utf-8'))
    source = json.loads((ROOT/'outputs/data_audit.json').read_text(encoding='utf-8'))
    selected = audit['selected_model']
    macros = {key:f'{results[name]["macro_f1"]:.4f}' for key,name in
        [('LRFone','logistic'),('RFFone','rf_all'),('CNNFone','cnn'),('FusionFone','fusion')]}
    macros.update(BestName=ZH[selected],BestFone=f'{results[selected]["macro_f1"]:.4f}',
        BestCV=f'{results[selected]["cv_mean_f1"]:.4f}',BestAccuracy=f'{results[selected]["accuracy"]:.4f}',
        RawRows=str(sum(r['raw_rows'] for r in source['subjects'] if r['primary_included'])),
        CNNParams=str(sum(p.numel() for p in make_model('cnn').parameters())),
        FusionParams=str(sum(p.numel() for p in make_model('fusion').parameters())),
        PythonVersion=env['python'].split()[0],TorchVersion=env['torch'].replace('_',r'\_'),
        SklearnVersion=env['scikit_learn'])
    (GENERATED/'results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    counts = read_csv(ROOT/'outputs/class_subject_counts.csv')
    table('data_table.tex','rrrrr',['受试者','原始记录数','活动数','窗口数','缺失率（\\%）'],
        [[r['subject'],r['raw_rows'],sum(int(c['windows'])>0 for c in counts if int(c['subject'])==r['subject']),
          r['windows'],f'{100*r["missing_sensor_fraction"]:.3f}'] for r in source['subjects'] if r['primary_included']])
    assignments = read_csv(OUT/'split_assignments.csv')
    def groups(part):
        return len({r['segment'] for r in assignments if (int(r['partition']) in part)})
    table('split_table.tex','lrrr',['数据部分','完整片段数','窗口数','占全部窗口（\\%）'],
        [['训练集',groups([1,2,3]),5187,'70.00'],['测试集',groups([0]),2223,'30.00']]+
        [[f'训练内第 {k} 验证折',groups([k]),1729,'23.33'] for k in range(1,4)])
    split_counts = read_csv(OUT/'split_class_counts.csv')
    table('split_class_table.tex','lrrrrr',['活动','训练集','测试集','验证折 1','验证折 2','验证折 3'],
        [[LABEL_ZH[k]]+[r[n] for n in ['train','test','cv_validation_1','cv_validation_2','cv_validation_3']]
         for k,r in enumerate(split_counts)])
    def result_row(name):
        r = results[name]
        return [ZH[name],f'{r["cv_mean_f1"]:.4f} $\\pm$ {r["cv_std_f1"]:.4f}',
            f'{r["accuracy"]:.4f}',f'{r["macro_f1"]:.4f}']
    header = ['模型','训练 CV F1 均值 $\\pm$ SD','测试 Accuracy','测试 F1']
    table('main_table.tex','lrrr',header,[result_row(n) for n in ['logistic','rf_all','cnn','fusion']])
    ablations = ['rf_wrist','rf_chest','rf_ankle','rf_no_wrist','rf_no_chest','rf_no_ankle','rf_all']
    table('ablation_table.tex','lrrr',['部位输入']+header[1:],[result_row(n) for n in ablations])
    cv = read_csv(OUT/'cv_summary.csv')
    table('cv_table.tex','lrrrr',['模型','验证折 1 F1','验证折 2 F1','验证折 3 F1','均值'],
        [[ZH[r['model']]]+[f'{float(r[n]):.4f}' for n in ['fold1_f1','fold2_f1','fold3_f1','cv_mean_f1']] for r in cv])
    class_rows = read_csv(OUT/'class_f1.csv')
    table('class_table.tex','rlrrrr',['编号','活动','测试窗口数','逻辑回归','随机森林','融合模型'],
        [[k+1,LABEL_ZH[k],r['windows']]+[f'{float(r[n]):.4f}' for n in ['logistic','rf_all','fusion']]
         for k,r in enumerate(class_rows)])
    best_test = max(['logistic','rf_all','cnn','fusion'],key=lambda n:results[n]['macro_f1'])
    text = (f'依据训练集三折验证的平均 Macro-F1，选定{ZH[selected]}作为最终展示模型，'
        f'其验证均值为 {results[selected]["cv_mean_f1"]:.4f}，样本标准差为 {results[selected]["cv_std_f1"]:.4f}。'
        f'在冻结的 2223 个测试窗口上，该模型 Accuracy 为 {results[selected]["accuracy"]:.4f}，'
        f'Macro-F1 为 {results[selected]["macro_f1"]:.4f}。选择记录在本轮测试预测生成前写入，'
        '最终模型没有根据测试排名更换。\n\n'
        f'四个预设主模型中，测试 F1 数值最高的是{ZH[best_test]}（{results[best_test]["macro_f1"]:.4f}）。'
        f'融合模型相对随机森林的测试 F1 差值为 {(results["fusion"]["macro_f1"]-results["rf_all"]["macro_f1"])*100:+.2f} 个百分点。'
        '表中交叉验证标准差描述三个验证折之间的波动，既不是测试分数的置信区间，也不能用来直接断言显著性。'
        '训练 CV 与测试分数的差异受到片段组成和最终训练样本量变化影响。\n\n'
        f'早期 CNN 的验证均值为 {results["cnn"]["cv_mean_f1"]:.4f}，测试 F1 为 {results["cnn"]["macro_f1"]:.4f}，'
        f'融合模型对应为 {results["fusion"]["cv_mean_f1"]:.4f} 与 {results["fusion"]["macro_f1"]:.4f}。'
        '神经网络在当前固定设置下的折间与测试波动较大，说明不能只依据训练损失下降判断泛化质量。'
        '需要额外划分和初始化复验，才能进一步区分训练设置与片段组成的影响。\n\n'
        '融合与早期 CNN 的表示方式和参数量均不同，且未单独删除统计分支或保持等参数量，'
        '因此现有对比评价的是完整系统，不能将差异全部归因于多分支结构。')
    (GENERATED/'main_discussion.tex').write_text(text,encoding='utf-8')
    best_single = max(['rf_wrist','rf_chest','rf_ankle'],key=lambda n:results[n]['macro_f1'])
    worst_removed = min(['rf_no_wrist','rf_no_chest','rf_no_ankle'],key=lambda n:results[n]['macro_f1'])
    text = (f'单部位实验中，{ZH[best_single]}的测试 Macro-F1 最高，为 {results[best_single]["macro_f1"]:.4f}。'
        f'完整三部位随机森林为 {results["rf_all"]["macro_f1"]:.4f}；移除实验中{ZH[worst_removed]}的测试 F1 最低，'
        f'为 {results[worst_removed]["macro_f1"]:.4f}。'
        '各配置使用完全相同的训练、验证和测试窗口；仅改变输入部位的统计特征列。'
        '不同部位同时包含姿态、局部运动和个体习惯等信息，训练 CV 排名与测试排名可能不同，'
        '这些成绩不构成神经网络分支重要性的因果证明。')
    (GENERATED/'ablation_discussion.tex').write_text(text,encoding='utf-8')
    worst = sorted(range(12),key=lambda k:float(class_rows[k]['fusion']))[:3]
    text = ('融合模型测试 F1 较低的三个类别为'+ '、'.join(f'{LABEL_ZH[k]}（{float(class_rows[k]["fusion"]):.4f}）' for k in worst)+
        '。不同类别的片段数量、训练覆盖及动作特征不同，不能仅凭总体成绩解释所有类别。'
        '混淆矩阵按真实类别逐行归一化；矩阵对角线为召回率，与表中的逐类 F1 定义不同。')
    (GENERATED/'class_discussion.tex').write_text(text,encoding='utf-8')
    subject_rows = read_csv(OUT/'subject_f1.csv')
    lo = min(subject_rows,key=lambda r:float(r['fusion']))
    hi = max(subject_rows,key=lambda r:float(r['fusion']))
    text = (f'按测试窗口中的实际出现类别计算，融合模型在受试者 {lo["subject"]} 的测试片段上 F1 最低（{float(lo["fusion"]):.4f}），'
        f'在受试者 {hi["subject"]} 上最高（{float(hi["fusion"]):.4f}）。'
        '部分个体的测试类别较少，应结合活动覆盖理解这些分数。相同受试者的其他片段可能用于训练，'
        '此图是固定测试集内部的描述，不能解释为对全新受试者的迁移性能。')
    (GENERATED/'subject_discussion.tex').write_text(text,encoding='utf-8')
    names = ['model_summary.json','cv_summary.csv','split_assignments.csv','split_manifest.json',
        'split_class_counts.csv','model_selection.json','class_f1.csv','subject_f1.csv','result_audit.json','environment.json']
    save_json(ROOT/'paper/asset_manifest.json',{'protocol_version':2,'macros':macros,
        'sources':{f'outputs/holdout/{n}':sha256(OUT/n) for n in names},
        'figures':{p.name:sha256(p) for p in (OUT/'figures').glob('*.pdf')}})
    print('Protocol-2 tables and text generated from audited CV and holdout predictions.',flush=True)


if __name__ == '__main__':
    main()
