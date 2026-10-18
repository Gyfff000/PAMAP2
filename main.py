"""Visual Studio start file: show verified saved results; never starts training implicitly."""
from pathlib import Path
import json
import csv
ROOT = Path(__file__).resolve().parent

def main():
    print('PAMAP2 - 跨受试者多传感器活动识别')
    path = ROOT / 'outputs/model_summary_seed42.json'
    if not path.exists():
        print('尚无完整实验结果。按顺序运行 00_download_data.py、01_prepare_data.py、02_run_experiments.py。')
        return
    comparison = ROOT / 'outputs/model_comparison.csv'
    if comparison.exists():
        with comparison.open(encoding='utf-8-sig') as f:
            results=list(csv.DictReader(f))
        for row in results:
            for key in ('macro_f1','accuracy','supported_subject_mean_f1','supported_subject_std_f1'):
                row[key]=float(row[key])
    else:
        print('请运行 03_analyze_results.py 生成核验后的完整结果表。')
        return
    print('\n8 折留一受试者评估（每个人恰好作为测试集一次）')
    print('个人 F1 按各人实际出现的类别计算；汇总 F1 固定为十二类。')
    print(f'{"Model":<18} {"Pooled F1":>12} {"Subject F1 mean±SD":>24} {"Accuracy":>12}')
    for r in results:
        print(f'{r["model"]:<18} {r["macro_f1"]:>12.4f} '
              f'{r["supported_subject_mean_f1"]:.4f} ± {r["supported_subject_std_f1"]:.4f} '
              f'{r["accuracy"]:>18.4f}')
    print('\n报告：paper/main.pdf\n预测演示：05_predict_demo.py\n复现实验：README.md')

if __name__=='__main__':
    main()
