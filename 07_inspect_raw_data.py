"""Read-only original PAMAP2 preview for the course demonstration (standard library only)."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
ACTIVITIES = {
    0: '过渡/其他活动', 1: '躺卧', 2: '坐', 3: '站立', 4: '步行',
    5: '跑步', 6: '骑车', 7: '北欧式行走', 12: '上楼', 13: '下楼',
    16: '吸尘', 17: '熨烫', 24: '跳绳',
}
# Original file column numbers in the presentation are one-based.
SENSORS = [('手腕', [5, 6, 7, 11, 12, 13]),
           ('胸部', [22, 23, 24, 28, 29, 30]),
           ('脚踝', [39, 40, 41, 45, 46, 47])]


def preview(subject=101, count=5):
    directory = ROOT / 'data/raw/Protocol'
    files = sorted(directory.glob('subject*.dat'))
    path = directory / f'subject{subject}.dat'
    if not path.exists():
        raise FileNotFoundError(f'未找到原始文件：{path}。请先运行 00_download_data.py。')
    print('PAMAP2 原始数据预览（只读，不训练模型）')
    print(f'原始数据目录：{directory}')
    print('\n一、原始文件：每个文件对应一名受试者的 Protocol 记录')
    for file in files:
        print(f'  {file.name:<20} {file.stat().st_size / 1_000_000:>8.1f} MB')
    print(f'\n当前文件：{path.name}')
    print('\n二、原始格式：空格分隔的 .dat 文本，每行一个采样时刻，共 54 列')
    for columns, meaning in [('1', '时间戳，单位 s'), ('2', '活动类别 activityID'),
                             ('3', '心率，单位 bpm'), ('4-20', '手腕 IMU，共 17 列'),
                             ('21-37', '胸部 IMU，共 17 列'), ('38-54', '脚踝 IMU，共 17 列')]:
        print(f'  第 {columns:<7} 列：{meaning}')
    print('  每个 IMU：温度、两种量程的三轴加速度、三轴角速度、三轴磁场、四元数姿态。')
    print('  官方说明指出姿态列无效；本实验使用 ±16g 加速度和角速度，共 18 通道。')
    print('  NaN 表示缺失值；activityID=0 表示过渡/其他活动。')
    first_rows, labelled = [], []
    with path.open(encoding='ascii') as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 54:
                raise ValueError(f'第 {line_number} 行列数为 {len(fields)}，预期 54。')
            if len(first_rows) < 2:
                first_rows.append((line_number, fields))
            if int(fields[1]) in ACTIVITIES and int(fields[1]) != 0:
                labelled.append((line_number, fields))
            if len(first_rows) >= 2 and len(labelled) >= count:
                break
    print('\n三、文件开头两条原始记录（前 3 列）')
    print('  行号      时间(s)    activityID    心率(bpm)')
    for number, fields in first_rows:
        print(f'  {number:<9} {fields[0]:<10} {fields[1]:<13} {fields[2]}')
    print('  开头若出现类别 0，是录制中的过渡阶段；后续预处理会排除。')
    print('\n四、首批有效活动记录的传感器值（从原文件选列，未插值、滤波或标准化）')
    print('  加速度单位 m/s²；角速度单位 rad/s。下方列号按原文件从 1 开始编号。')
    if not labelled:
        raise ValueError('此文件没有找到本实验采用的活动标签。')
    for sensor, columns in SENSORS:
        print(f'\n  {sensor}，原始列号：{columns}')
        print(f'  {"time(s)":>9} {"label":>6} {"acc_x":>12} {"acc_y":>12} {"acc_z":>12} '
              f'{"gyro_x":>12} {"gyro_y":>12} {"gyro_z":>12}')
        for number, fields in labelled:
            values = ''.join(f' {fields[c-1]:>12}' for c in columns)
            print(f'  {fields[0]:>9} {fields[1]:>6}{values}')
    activity = int(labelled[0][1][1])
    print(f'\n  第一条有效记录所在原始行：{labelled[0][0]}；标签 {activity} = {ACTIVITIES[activity]}。')
    audit = ROOT / 'outputs/data_audit.json'
    if audit.exists():
        data = json.loads(audit.read_text(encoding='utf-8'))
        item = next((s for s in data['subjects'] if s['subject'] == subject), None)
        if item:
            print('\n五、此前完整文件审计的统计（此次预览不重新扫描整个文件）')
            print(f'  该文件原始记录数：{item["raw_rows"]:,}；本实验处理后窗口数：{item["windows"]:,}。')
            print('  一行原始记录是一个采样时刻；一个模型窗口包含连续 5.12 秒的信号。')
    print(f'\n官方字段说明：{ROOT / "data/raw/documentation/readme.pdf"}')
    print('预览完成；原始文件保持不变。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, default=101)
    parser.add_argument('--rows', type=int, choices=range(1, 21), default=5)
    args = parser.parse_args()
    try:
        preview(args.subject, args.rows)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f'预览失败：{error}', file=sys.stderr)
        sys.exit(1)
