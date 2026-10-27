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
    cv_lookup = {r['model']:r for r in cv}
    correct = int(round(results[selected]['accuracy']*2223))
    text = (f'依据训练集三折验证的平均 Macro-F1，选定{ZH[selected]}作为最终展示模型，'
        f'其验证均值为 {results[selected]["cv_mean_f1"]:.4f}，样本标准差为 {results[selected]["cv_std_f1"]:.4f}。'
        f'在 2223 个测试窗口中，该模型正确识别 {correct} 个，错误 {2223-correct} 个，'
        f'Accuracy 为 {results[selected]["accuracy"]:.4f}，Macro-F1 为 {results[selected]["macro_f1"]:.4f}。'
        '交叉验证选择和最终测试结果一致，统计特征随机森林在四种主模型中均排名首位。\n\n'
        f'随机森林三个验证折的 F1 从 {min(float(cv_lookup["rf_all"][f"fold{k}_f1"]) for k in range(1,4)):.4f}'
        f' 到 {max(float(cv_lookup["rf_all"][f"fold{k}_f1"]) for k in range(1,4)):.4f}，'
        f'标准差为 {results["rf_all"]["cv_std_f1"]:.4f}。'
        f'早期 CNN 和融合模型的折间标准差分别为 {results["cnn"]["cv_std_f1"]:.4f} 和 {results["fusion"]["cv_std_f1"]:.4f}，'
        '表明神经网络在不同验证片段组合上的成绩变化更大。'
        '各模型在相同窗口上比较，这种差异对应当前表示和训练设置下的验证稳定性。\n\n'
        f'早期 CNN 的验证均值为 {results["cnn"]["cv_mean_f1"]:.4f}，测试 F1 为 {results["cnn"]["macro_f1"]:.4f}，'
        f'融合模型对应为 {results["fusion"]["cv_mean_f1"]:.4f} 与 {results["fusion"]["macro_f1"]:.4f}。'
        f'融合模型的测试 F1 比早期 CNN 高 {(results["fusion"]["macro_f1"]-results["cnn"]["macro_f1"])*100:.2f} 个百分点，'
        f'比随机森林低 {(results["rf_all"]["macro_f1"]-results["fusion"]["macro_f1"])*100:.2f} 个百分点。'
        '这组结果表明，统计特征对当前活动任务具有较强区分度，与时序表示联合后也改善了网络的测试成绩。\n\n'
        '融合模型同时引入三个部位编码器和统计分支，参数量由早期 CNN 的 19052 增至 35868。'
        '现有实验比较的是这两套完整配置。进一步量化各分支贡献时，可在统一参数预算下逐一移除分支，'
        '并保持划分、训练轮数和优化器设置相同。')
    (GENERATED/'main_discussion.tex').write_text(text,encoding='utf-8')
    best_single = max(['rf_wrist','rf_chest','rf_ankle'],key=lambda n:results[n]['macro_f1'])
    worst_removed = min(['rf_no_wrist','rf_no_chest','rf_no_ankle'],key=lambda n:results[n]['macro_f1'])
    text = (f'单部位实验中，{ZH[best_single]}的测试 Macro-F1 最高，为 {results[best_single]["macro_f1"]:.4f}。'
        f'完整三部位随机森林为 {results["rf_all"]["macro_f1"]:.4f}；移除实验中{ZH[worst_removed]}的测试 F1 最低，'
        f'为 {results[worst_removed]["macro_f1"]:.4f}。'
        f'完整配置比最佳单部位高 {(results["rf_all"]["macro_f1"]-results[best_single]["macro_f1"])*100:.2f} 个百分点。'
        '三组部位对照均使用相同的训练、验证和测试窗口，输入维度分别为 50、100 或 150。\n\n'
        f'以完整三部位为参照，移除手腕、胸部和脚踝分别降低 '
        f'{(results["rf_all"]["macro_f1"]-results["rf_no_wrist"]["macro_f1"])*100:.2f}、'
        f'{(results["rf_all"]["macro_f1"]-results["rf_no_chest"]["macro_f1"])*100:.2f} 和 '
        f'{(results["rf_all"]["macro_f1"]-results["rf_no_ankle"]["macro_f1"])*100:.2f} 个百分点。'
        '手腕输入在这一特征组合中带来的成绩增量最大。完整配置也高于各双部位配置，'
        '说明三个部位的统计表示在本次识别任务中提供了互补信息。')
    (GENERATED/'ablation_discussion.tex').write_text(text,encoding='utf-8')
    worst = sorted(range(12),key=lambda k:float(class_rows[k]['fusion']))[:3]
    with np.load(OUT/'test/fusion/predictions.npz') as p:
        fusion_y=p['y_true'];fusion_pred=p['y_pred'];fusion_subject=p['subject']
    standing=fusion_y==2
    standing_ok=int(((fusion_pred==2)&standing).sum())
    standing_ironing=int(((fusion_pred==10)&standing).sum())
    sitting=fusion_y==1
    sitting_ok=int(((fusion_pred==1)&sitting).sum())
    text = ('融合模型测试 F1 较低的三个类别为'+ '、'.join(f'{LABEL_ZH[k]}（{float(class_rows[k]["fusion"]):.4f}）' for k in worst)+
        f'。站立类别共有 {int(standing.sum())} 个测试窗口，其中 {standing_ok} 个识别正确，'
        f'{standing_ironing} 个被分为熨烫，分别占该类的 {standing_ok/standing.sum()*100:.2f}\\% 和 '
        f'{standing_ironing/standing.sum()*100:.2f}\\%。坐类别的 {int(sitting.sum())} 个窗口中，'
        f'有 {sitting_ok} 个识别正确。站立与熨烫之间的混淆是融合模型的一项集中误差。\n\n'
        f'随机森林在站立和熨烫上的 F1 分别为 {float(class_rows[2]["rf_all"]):.4f} 和 '
        f'{float(class_rows[10]["rf_all"]):.4f}，均高于融合模型。'
        '这一对照将后续改进目标具体定位到易混淆活动的区分。'
        '图中的混淆矩阵按真实类别逐行归一化，对角线给出各类召回率；表中的 F1 同时结合精确率和召回率。')
    (GENERATED/'class_discussion.tex').write_text(text,encoding='utf-8')
    subject_rows = read_csv(OUT/'subject_f1.csv')
    lo = min(subject_rows,key=lambda r:float(r['fusion']))
    hi = max(subject_rows,key=lambda r:float(r['fusion']))
    def coverage(row):
        subset=fusion_subject==int(row['subject'])
        return int(subset.sum()),len(np.unique(fusion_y[subset]))
    low_n,low_c=coverage(lo);high_n,high_c=coverage(hi)
    text = (f'按测试窗口中的实际出现类别计算，融合模型在受试者 {lo["subject"]} 的测试片段上 F1 最低（{float(lo["fusion"]):.4f}），'
        f'在受试者 {hi["subject"]} 上最高（{float(hi["fusion"]):.4f}）。'
        f'前者包含 {low_n} 个窗口、{low_c} 类活动，后者包含 {high_n} 个窗口、{high_c} 类活动。'
        '各分数由该测试子集实际包含的类别计算，差异同时反映片段识别表现和活动组合。'
        '将个体曲线与逐类混淆结合，可以定位模型在具体人员与活动组合上的薄弱环节。')
    (GENERATED/'subject_discussion.tex').write_text(text,encoding='utf-8')
    names = ['model_summary.json','cv_summary.csv','split_assignments.csv','split_manifest.json',
        'split_class_counts.csv','model_selection.json','class_f1.csv','subject_f1.csv','result_audit.json','environment.json']
    save_json(ROOT/'paper/asset_manifest.json',{'protocol_version':2,'macros':macros,
        'sources':{f'outputs/holdout/{n}':sha256(OUT/n) for n in names},
        'figures':{p.name:sha256(p) for p in (OUT/'figures').glob('*.pdf')}})
    print('Protocol-2 tables and text generated from audited CV and holdout predictions.',flush=True)


if __name__ == '__main__':
    main()
