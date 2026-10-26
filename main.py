"""Visual Studio start file: show verified saved results; never starts training implicitly."""
from pathlib import Path
import json
import csv
ROOT = Path(__file__).resolve().parent

def main():
    print('PAMAP2 - 多传感器活动识别：7∶3 划分 + 训练集内三折交叉验证')
    path = ROOT / 'outputs/holdout/model_summary.json'
    if not path.exists():
        print('尚无完整实验结果。按顺序运行 00_download_data.py、01_prepare_data.py、02_run_experiments.py。')
        return
    comparison = ROOT / 'outputs/holdout/model_comparison.csv'
    if comparison.exists():
        with comparison.open(encoding='utf-8-sig') as f:
            results=list(csv.DictReader(f))
        for row in results:
            for key in ('macro_f1','accuracy','cv_mean_f1','cv_std_f1'):
                row[key]=float(row[key])
    else:
        print('请运行 03_analyze_results.py 生成核验后的完整结果表。')
        return
    print('\n训练集 5187（70%），测试集 2223（30%）；同一完整录制片段不跨集合。')
    print('三折交叉验证只用训练集；各折训练 3458，验证 1729。模型由验证均值选择。')
    print('F1 固定十二类；验证 SD 是折间波动。同一人的不同片段可出现在两侧。')
    print(f'{"Model":<18} {"CV F1 mean±SD":>22} {"Test F1":>12} {"Test Accuracy":>15}')
    for r in results:
        print(f'{r["model"]:<18} {r["cv_mean_f1"]:.4f} ± {r["cv_std_f1"]:.4f} '
              f'{r["macro_f1"]:>15.4f} {r["accuracy"]:>15.4f}')
    selection=json.loads((ROOT/'outputs/holdout/model_selection.json').read_text(encoding='utf-8'))
    print(f'\n按训练交叉验证选择的最终模型：{selection["selected_model"]}')
    print('\n报告：paper/main.pdf\n预测演示：05_predict_demo.py\n复现实验：README.md')

if __name__=='__main__':
    main()
