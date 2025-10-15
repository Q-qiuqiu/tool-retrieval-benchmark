#python query_generate.py --task 'tooleyes' --topk 20 --embed-model "/data/labshare/Param/ToolRet/e5-small-v2" --output_file "query_records.txt"
!/usr/bin/env bash

TOPK=20
EMBED_MODEL="/data/labshare/Param/ToolRet/e5-small-v2"
mkdir -p log
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

# 循环启动任务
for TASK in "${TASKS[@]}"; do
    OUTPUT_FILE="log/record_${TASK}.txt"
    LOG_FILE="log/log_${TASK}.txt"
    echo "🚀 启动任务：$TASK -> $OUTPUT_FILE"
    nohup python query_generate.py \
        --task "$TASK" \
        --topk $TOPK \
        --embed-model "$EMBED_MODEL" \
        --output_file "$OUTPUT_FILE" \
        > "$LOG_FILE" 2>&1 &
done

wait
echo "✅ 所有任务已完成！"


#终止任务：pkill -f query_generate.py