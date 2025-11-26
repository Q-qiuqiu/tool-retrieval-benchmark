#!/bin/bash

# 创建日志目录
mkdir -p log

# 设置参数
TOPK=10
EMBED_MODEL="/data/labshare/Param/ToolRet/e5-base-v2"

# 定义多个任务名
TASKS=(
  "apibank" "apigen" "appbench" "autotools-food" "autotools-music" "autotools-weather"
  "craft-math-algebra" "craft-tabmwp" "craft-vqa"
  "gorilla-huggingface" "gorilla-pytorch" "gorilla-tensor"
  "gpt4tools" "gta" "metatool" "mnms" "restgpt-spotify" "restgpt-tmdb"
  "reversechain" "rotbench" "t-eval-dialog" "t-eval-step"
  "taskbench-daily" "taskbench-huggingface" "taskbench-multimedia"
  "tool-be-honest" "toolace" "toolalpaca" "toolbench-sam" "toolbench" "toolemu"
  "tooleyes" "toolink" "toollens" "ultratool"
)

# 第一步：为每个任务运行run_conversation.py
echo "🚀 开始运行对话脚本..."
MAX_PARALLEL=5
running=0

for TASK in "${TASKS[@]}"; do
    # 检查是否达到最大并行数
    if [ $running -ge $MAX_PARALLEL ]; then
        # 等待一个进程完成
        wait -n
        running=$((running-1))
    fi

    LOG_FILE="log/log_${TASK}.txt"
    echo "📝 启动对话任务：$TASK -> $LOG_FILE"
    nohup python3 run_conversation.py \
        --task "$TASK" \
        --topk $TOPK \
        --embed-model "$EMBED_MODEL" \
        > "$LOG_FILE" 2>&1 &
    
    # 保存进程ID并增加计数器
    CONV_PIDS+=("$!")
    running=$((running+1))
done

# 等待所有剩余的对话任务完成
while [ $running -gt 0 ]; do
    wait -n
    running=$((running-1))
done

# 等待所有对话任务完成
echo "⏳ 等待所有对话任务完成..."
for PID in "${CONV_PIDS[@]}"; do
    wait $PID
done

echo "✅ 所有对话任务已完成！"

# 第二步：为每个任务运行recall_v1v2.py
echo "🚀 开始运行召回评估脚本..."
running=0

for TASK in "${TASKS[@]}"; do
    # 检查是否达到最大并行数
    if [ $running -ge $MAX_PARALLEL ]; then
        # 等待一个进程完成
        wait -n
        running=$((running-1))
    fi
    
    LOG_FILE="log/log_recall_${TASK}.txt"
    echo "📊 启动召回评估任务：$TASK -> $LOG_FILE"
    nohup python3 recall_v1v2.py \
        --task "$TASK" \
        > "$LOG_FILE" 2>&1 &
    
    # 保存进程ID并增加计数器
    RECALL_PIDS+=("$!")
    running=$((running+1))
done

# 等待所有剩余的召回评估任务完成
while [ $running -gt 0 ]; do
    wait -n
    running=$((running-1))
done

# 等待所有召回评估任务完成
echo "⏳ 等待所有召回评估任务完成..."
for PID in "${RECALL_PIDS[@]}"; do
    wait $PID
done

echo "🎉 所有任务已成功完成！"
echo "📂 日志文件保存在 log/ 目录下"