#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集所有任务的评估结果到一个统一的文件中
"""
import os
import re
import sys
from pathlib import Path

LOG_DIR = Path("log")
OUTPUT_FILE = Path("log/all_result.txt")

# 定义任务列表
def get_tasks():
    """直接在代码中定义任务列表"""
    # 直接定义任务数组，从run_start.sh中提取的任务列表
    tasks = [
        "apibank", "apigen", "appbench", "autotools-food", "autotools-music", "autotools-weather",
        "craft-math-algebra", "craft-tabmwp", "craft-vqa",
        "gorilla-huggingface", "gorilla-pytorch", "gorilla-tensor",
        "gpt4tools", "gta", "metatool", "mnms", "restgpt-spotify", "restgpt-tmdb",
        "reversechain", "rotbench", "t-eval-dialog", "t-eval-step",
        "taskbench-daily", "taskbench-huggingface", "taskbench-multimedia",
        "tool-be-honest", "toolace", "toolalpaca", "toolbench-sam", "toolbench", "toolemu",
        "tooleyes", "toolink", "toollens", "ultratool"
    ]
    return tasks

def extract_accuracy(task):
    """从log_${task}.txt中提取Accuracy@1值"""
    log_file = LOG_DIR / f"log_{task}.txt"
    if not log_file.exists():
        print(f"警告: 未找到文件 {log_file}")
        return "N/A"
    
    try:
        # 使用grep命令直接查找最后一行Accuracy@1
        import subprocess
        result = subprocess.run(
            f"grep 'Accuracy@1:' {log_file} | tail -n 1", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        output = result.stdout.strip()
        if output:
            # 提取数字部分
            match = re.search(r'Accuracy@1:\s*(\d+\.\d+)%', output)
            if match:
                return match.group(1) + "%"
    except Exception as e:
        print(f"提取Accuracy@1时出错 {task}: {e}")
    
    # 如果grep失败，尝试读取文件末尾
    try:
        # 读取文件最后100行
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[-100:]
        
        # 从后往前搜索
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("Accuracy@1:"):
                # 提取数字部分
                match = re.search(r'Accuracy@1:\s*(\d+\.\d+)%', line)
                if match:
                    return match.group(1) + "%"
    except Exception as e:
        print(f"读取文件 {log_file} 时出错: {e}")
    
    return "N/A"

def extract_recall_results(task):
    """从log_recall_${task}.txt中提取评估结果"""
    log_file = LOG_DIR / f"log_recall_{task}.txt"
    if not log_file.exists():
        print(f"警告: 未找到文件 {log_file}")
        return []
    
    results = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            
            # 查找"评估结果:"部分
            eval_match = re.search(r'评估结果:(.*?)(?=日志已保存|$)', content, re.DOTALL)
            if eval_match:
                eval_content = eval_match.group(1).strip()
                # 按行分割并清理
                for line in eval_content.split('\n'):
                    line = line.strip()
                    if line:
                        results.append(line)
    except Exception as e:
        print(f"提取recall结果时出错 {task}: {e}")
    
    return results

def main():
    """主函数"""
    tasks = get_tasks()
    
    if not tasks:
        print("错误: 未找到任何任务")
        sys.exit(1)
    
    print(f"找到 {len(tasks)} 个任务")
    
    # 准备输出内容
    output_lines = []
    output_lines.append("="*60)
    output_lines.append("所有任务评估结果汇总")
    output_lines.append("="*60)
    output_lines.append("")
    
    total_tasks = len(tasks)
    for i, task in enumerate(tasks, 1):
        print(f"处理任务 {i}/{total_tasks}: {task}")
        
        # 提取Accuracy@1
        accuracy = extract_accuracy(task)
        
        # 提取recall结果
        recall_results = extract_recall_results(task)
        
        # 添加到输出
        output_lines.append(f"任务名: {task}")
        output_lines.append(f"Accuracy@1: {accuracy}")
        
        if recall_results:
            output_lines.append("评估结果:")
            output_lines.extend(recall_results)
        else:
            output_lines.append("评估结果: 未找到")
        
        output_lines.append("-"*60)
    
    # 写入输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"\n✅ 所有结果已保存到: {OUTPUT_FILE}")
    print(f"共处理 {total_tasks} 个任务")

if __name__ == "__main__":
    main()