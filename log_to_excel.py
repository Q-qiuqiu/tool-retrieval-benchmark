import os
import re
import pandas as pd

# 设置文件路径
all_result_path = 'log/all_result.txt'
excel_output_path = 'log/all_summary.xlsx'

# 读取文件内容
with open(all_result_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 定义正则表达式模式提取任务信息
task_pattern = r'任务名: (\S+)\s+Accuracy@1: ([\d.]+)%\s+评估结果:\s+总任务个数: (\d+)\s+Recall@10成功的任务个数: (\d+)\s+Recall@10失败但能解决问题的任务个数: (\d+)\s+Recall@10准确率: ([\d.]+)\(\d+/\d+\)\s+补充解决率: ([\d.]+)\(\d+/\d+\)'

# 提取所有任务数据
tasks_data = re.findall(task_pattern, content, re.MULTILINE)

# 准备数据列表
results = []
for task in tasks_data:
    task_name = task[0]
    accuracy = float(task[1])
    total_tasks = int(task[2])
    recall_success = int(task[3])
    recall_solvable = int(task[4])
    recall_accuracy = float(task[5])
    supplementary_rate = float(task[6])
    
    results.append({
        '任务名': task_name,
        'Accuracy@1': accuracy,
        'Recall@10': recall_accuracy,
        '非GT相关解决率': supplementary_rate,
        'Recall成功任务个数': recall_success,
        '非GT相关任务个数': recall_solvable,
        '总任务个数': total_tasks
    })

# 创建DataFrame
df = pd.DataFrame(results)

# 保存为Excel文件
df.to_excel(excel_output_path, index=False, engine='openpyxl')

print(f"Excel文件已成功生成: {excel_output_path}")
print(f"共处理了 {len(results)} 个任务")
print("Excel文件包含以下列:")
for col in df.columns:
    print(f"- {col}")