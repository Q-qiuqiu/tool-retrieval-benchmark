import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

from toolret.config import _FIRST_STAGE, _TOOL_REPO, _MODEL
from toolret.eval import trec_eval, load_tools, load_queries, eval_retrieval
from toolret.utils import fix_json_error
from toolret.eval import task_split



def _read_openai_config():
    api_key = "sk-IS7bpO9PnKkn0SxQ711cA3110e8a4a8183Ed5215D4F8Dc1b"#sk-IS7bpO9PnKkn0SxQ711cA3110e8a4a8183Ed5215D4F8Dc1b
    base_url ="https://aihubmix.com/v1"#http://localhost:7001/v1/
    model = "DeepSeek-V3" #/data/labshare/Param/gpt-oss/gpt-oss-20b
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")
    if not base_url:
        # 默认 OpenAI 官方，但也兼容其他服务
        base_url = "https://api.openai.com/v1"
    return base_url.rstrip("/"), api_key, model


def _build_prompt(query: Dict[str, Any], tools: List[Dict[str, Any]]) -> str:
    instruction = query.get("instruction", "你是一个工具选择器。根据用户查询选择最相关的工具。只返回JSON，不要返回多余文字。")
    user_query = query.get("query", "")
    candidate_strs = []
    for idx, tool in enumerate(tools):
        # 工具对象字段统一转换为可读文本
        name = tool.get("name") or tool.get("tool_name") or f"tool_{idx}"
        tool_id = tool.get("id")
        documentation = tool.get("documentation") or tool.get("doc") or ""
        candidate_strs.append(f"-name: {name}\n  id: {tool_id}\n  documentation: {documentation}")

    candidates_block = "\n".join(candidate_strs)

    # 仅返回 JSON，结构固定，包含所选工具 id 列表
    system = (
        "你是一个只输出 JSON 的助手。\n"
        "当被要求在候选工具中选择时，只返回如下 JSON：\n"
        "{\"selected_tool_ids\": [\"<id>\"]}\n"
        "不要输出任何附加解释。"
    )
    user = (
        f"指令: {instruction}\n"
        f"问题: {user_query}\n\n"
        f"候选工具(每个工具包含 name/id/documentation)：\n{candidates_block}\n\n"
        f"请基于查询选择最相关的一个工具，且严格返回 JSON：{{\"selected_tool_ids\": [\"id1\"]}}"
    )
    # 直接返回合并字符串，适配简单 chat.completions
    return json.dumps({"system": system, "user": user}, ensure_ascii=False)


def _call_openai(messages: List[Dict[str, str]]):
    import requests

    base_url, api_key, model = _read_openai_config()
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def _parse_selection(content: str) -> List[str]:
    try:
        content_fixed = fix_json_error(content, return_str=False)
        if isinstance(content_fixed, str):
            content_fixed = json.loads(content_fixed)
        selected = content_fixed.get("selected_tool_ids", [])
        return [str(x) for x in selected]
    except Exception:
        # 回退尝试直接解析
        try:
            obj = json.loads(content)
            return [str(x) for x in obj.get("selected_tool_ids", [])]
        except Exception:
            return []


def run(task: str = "all", from_top_k: int = 20, embed_model: str = None) -> Dict[str, Dict[str, float]]:
    # 1) 先运行第一阶段检索（与仓库评测保持一致），得到每个 query 的候选工具集合
    # 为避免重复检索，直接利用 eval_retrieval 的输出文件携带候选（scores 不重要，只要候选ID）
    tasks = task
    output_file = f"retrieval_{task}.json"
    json_log ="run_log.json"
    model_name = embed_model or _MODEL[0]
    # eval_result = eval_retrieval(model_name=model_name,
    #                    tasks=tasks,
    #                    category='all',
    #                    batch_size=4,
    #                    output_file=output_file,
    #                    top_k=max(from_top_k, 1),
    #                    is_inst=True,
    #                    is_print=False)

    # 读取检索结果 {task: {qid: {tool_id: score}}}
    _tasks = task_split(tasks)
    tools_all = load_tools('all')
    retrieval_results = json.load(open(output_file, 'r'))
    tools_map = {item['id']: item for item in tools_all}
    # 初始化累积变量
    all_results: Dict[str, Dict[str, float]] = {}
    all_qrels: Dict[str, Dict[str, int]] = {}
    total_correct = 0
    total_queries = 0
    report = {
    "meta": {
        "task_arg": task,
    },
    "by_task": {}
}
    current_task = "gta"
    if current_task is not None:
    #for current_task in tqdm(_tasks):
        report["by_task"].setdefault(current_task, [])
        task_results = retrieval_results.get(current_task, {})
        print(f"Processing task: {current_task}, found {len(task_results)} queries")
        # 2) 加载 queries 与全量工具映射
        queries = load_queries(current_task)
        # 处理当前任务的查询
        for q in queries:
            qid = q['id']
            # 取出检索候选（按分数降序），仅保留 top_k
            cand_dict = task_results.get(qid, {})
            cand_sorted = sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)[:from_top_k]
            cand_ids = [k for k, _ in cand_sorted]
            cand_tools = [tools_map[int(tid)] if isinstance(tid, int) else tools_map.get(tid) for tid in cand_ids]
            cand_tools = [t for t in cand_tools if t is not None]
            print("retreval:",cand_ids)
            # 没有候选则跳过
            if not cand_tools:
                continue

            # 构造 prompt，调用 LLM（要求只返回 JSON）
            prompt_blob = _build_prompt(q, cand_tools)
            prompt_obj = json.loads(prompt_blob)
            messages = [
                {"role": "system", "content": prompt_obj["system"]},
                {"role": "user", "content": prompt_obj["user"]},
            ]

            content = _call_openai(messages)
            selected_ids = _parse_selection(content)
            print("selected_ids:",selected_ids)
            # 只评估 Top-1（若多选则取第一项），并用于 Accuracy 统计
            top1 = str(selected_ids[0]) if selected_ids else None
            # 评测标签
            labels = json.loads(q['labels']) if isinstance(q['labels'], str) else q['labels']
            gold_first = str(labels[0]['id']) if labels else None
            all_qrels[qid] = {gold_first: 1} if gold_first is not None else {}
            print("gold:",gold_first)

            #写入日志
            entry = {
                "query_id": qid,
                "retrieval": {
                    "candidate_ids": [str(k) for k, _ in cand_sorted],  # 等价于 cand_ids
                },
                "selection": {
                    "selected_tool_ids": [str(x) for x in (selected_ids or [])],
                },
                "gold": {
                    "first": gold_first
                },
                "evaluation": {
                    "hit_top1": (gold_first is not None and top1 == gold_first)
                }
            }
            report["by_task"][current_task].append(entry)


            if top1 is not None:
                all_results[qid] = {str(top1): 1.0}
                total_queries += 1
                # 命中：答案集合中存在且 relevance>0
                if top1 in all_qrels[qid] and all_qrels[qid][top1] > 0:
                    total_correct += 1
            print("all_qrels:",all_qrels)
            print("all_results:",all_results)
    os.makedirs(os.path.dirname(json_log) or ".", exist_ok=True)
    with open(json_log, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    #trec_metrics = trec_eval(qrels=all_qrels, results=all_results, k_values=(1,))
    acc_top1 = round((total_correct / max(total_queries, 1)), 5)
    # 附加一个简单准确率
    #trec_metrics["ACC@1"] = acc_top1
    return acc_top1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM tool selection over first-stage retrieval and evaluate.")
    parser.add_argument("--task", type=str, default="all", help="任务名；例如: toolbench 或 all")
    parser.add_argument("--topk", type=int, default=20, help="从第一阶段候选中取前K个")
    parser.add_argument("--embed-model", type=str, default=None, help="用于第一阶段检索的嵌入模型名（默认取配置的第一个）")
    args = parser.parse_args()

    metrics = run(task=args.task, from_top_k=args.topk, embed_model=args.embed_model)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


