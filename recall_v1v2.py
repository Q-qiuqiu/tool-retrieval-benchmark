# 评估检索召回率并使用LLM判断
import os
import json
import logging
import datetime
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

from toolret.eval import load_tools, load_queries
from toolret.utils import fix_json_error


def _read_openai_config():
    """读取OpenAI配置，复用run_conversation.py中的逻辑"""
    api_key = "sk-3Kechw8XbX31d8YuF921C9E9D495417a866cC812FaFe429e"
    base_url = "https://aihubmix.com/v1"
    model = "gpt-4o"
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")
    return base_url.rstrip("/"), api_key, model


def _call_openai(messages: List[Dict[str, str]]):
    """调用OpenAI API，复用run_conversation.py中的逻辑"""
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
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def build_judge_prompt_single_tool(query: Dict[str, Any], tool: Dict[str, Any]) -> str:
    """构建用于判断单个工具是否能解决查询的prompt"""
    system = (
        "你是一个工具能力评估专家。"
        "请仔细分析用户的查询和提供的单个工具信息，判断该工具是否能够完全解决用户的查询需求。"
        "只返回JSON格式，包含一个字段'solvable'，值为True或False，表示该工具是否可以解决问题。"
        "不要返回任何其他文字或解释。"
    )
    
    user_query = query.get("query", "")
    instruction = query.get("instruction", "")
    tool_id = tool.get("id")
    
    # 从documentation字段中提取工具名称和描述
    doc = tool.get("documentation", {})
    if isinstance(doc, str):
        try:
            doc = json.loads(doc)
        except json.JSONDecodeError:
            doc = {}
    
    name = doc.get("name", "Unknown Tool")
    description = doc.get("description", "")
    params = doc.get("parameters", [])

    user = (
        f"指令: {instruction}\n"
        f"用户查询: {user_query}\n"
        f"工具信息:\n"
        f"  名称: {name}\n"
        f"  ID: {tool_id}\n"
        f"  描述: {description}\n"
        f"  参数: {params}\n"
        f"请判断这个工具是否能够完全解决用户的查询需求。只返回JSON格式：{{\"solvable\": true|false}}"
    )
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def parse_judgment(content: str) -> bool:
    """解析LLM的判断结果"""
    try:
        content_fixed = fix_json_error(content, return_str=False)
        if isinstance(content_fixed, str):
            content_fixed = json.loads(content_fixed)
        return content_fixed.get("solvable", False)
    except Exception:
        # 如果解析失败，尝试直接提取关键词
        if "solvable" in content.lower() and "true" in content.lower():
            return True
        return False


def setup_logger(task: str):
    """设置日志记录器"""
    # 创建日志目录
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"recall_judgment_{task}.jsonl")
    
    return log_file

def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='评估检索召回率并使用LLM判断')
    parser.add_argument('--task', type=str, default='tooleyes', help='任务名称')
    parser.add_argument('--topk', type=int, default=10, help='检索的topk数量')
    args = parser.parse_args()
    
    # 配置参数
    task = args.task
    retrieval_file = f"log/retrieval_{args.task}.json"
    top_k = args.topk
    
    # 设置日志文件
    log_file = setup_logger(task)
    
    # 初始化日志数据结构
    log_data = []
    
    # 加载数据
    print("加载检索结果...")
    retrieval_results = json.load(open(retrieval_file, 'r'))
    task_results = retrieval_results.get(task, {})
    
    print("加载工具和查询数据...")
    tools_all = load_tools('all')
    queries = load_queries(task)
    
    # 创建工具ID到工具信息的映射
    tools_map = {str(item['id']): item for item in tools_all}
    
    # 初始化统计变量
    total_tasks = len(queries)
    recall_success_count = 0
    llm_solvable_count = 0
    
    print(f"开始处理 {total_tasks} 个查询...")
    
    # 处理每个查询
    for q in tqdm(queries):
        qid = q['id']
        
        # 获取检索结果
        if qid not in task_results:
            print(f"警告: 查询 {qid} 没有检索结果，跳过")
            continue
        
        # 获取top-k工具ID
        cand_dict = task_results[qid]
        cand_sorted = sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        cand_ids = [k for k, _ in cand_sorted]
        
        # 获取gold truth工具ID
        labels = json.loads(q['labels']) if isinstance(q['labels'], str) else q['labels']
        gold_ids = [str(x['id']) for x in labels]
        
        # 计算recall@10
        recall_success = any(gold_id in cand_ids for gold_id in gold_ids)
        
        if recall_success:
            recall_success_count += 1
        else:
            # 对于recall失败的查询，获取top-10工具的详细信息
            
            # 创建日志记录
            log_entry = {
                "query_id": qid,
                "query": q.get("query", ""),
                "instruction": q.get("instruction", ""),
                "gold_tools": gold_ids,
                "retrieved_tools": [],
                "any_tool_solvable": False
            }
            
            # 逐个判断每个工具
            has_solvable_tool = False
            for tid, score in cand_sorted[:top_k]:
                tool = tools_map.get(tid)
                if not tool:
                    continue
                # 从documentation字段中提取工具名称和描述
                doc = tool.get("documentation", {})
                if isinstance(doc, str):
                    try:
                        doc = json.loads(doc)
                    except json.JSONDecodeError:
                        doc = {}
                
                name = doc.get("name", "Unknown Tool")
                description = doc.get("description", "")
                params = doc.get("parameters", [])
    
                tool_entry = {
                    "id": tid,
                    "name": name,
                    "description": description,
                    "parameters": params,
                    "score": score,
                    "is_solvable": False
                }
                
                try:
                    # 对每个工具单独调用LLM进行判断
                    messages = build_judge_prompt_single_tool(q, tool)
                    content = _call_openai(messages)
                    is_solvable = parse_judgment(content)
                    
                    # 更新工具判断结果
                    # tool_entry["llm_response"] = content
                    tool_entry["is_solvable"] = is_solvable
                    
                    if is_solvable and not has_solvable_tool:
                        has_solvable_tool = True
                        llm_solvable_count += 1
                except Exception as e:
                    print(f"判断工具 {tid} 时出错: {e}")
                    tool_entry["error"] = str(e)
                    tool_entry["is_solvable"] = False
                
                log_entry["retrieved_tools"].append(tool_entry)
            
            log_entry["any_tool_solvable"] = has_solvable_tool
            log_data.append(log_entry)
    
    # 输出结果
    print("\n评估结果:")
    print(f"总任务个数: {total_tasks}")
    print(f"Recall@10成功的任务个数: {recall_success_count}")
    print(f"Recall@10失败但能解决问题的任务个数: {llm_solvable_count}")
    print(f"Recall@10准确率: {100*recall_success_count / total_tasks:.4f}({recall_success_count}/{total_tasks})")
    print(f"补充解决率: {100*llm_solvable_count / max(1, total_tasks - recall_success_count):.4f}({llm_solvable_count}/{total_tasks - recall_success_count})")  
    # 将所有日志数据一次性写入JSON文件
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"总体解决率: {100*(recall_success_count + llm_solvable_count) / total_tasks:.4f}%")  
    print(f"日志已保存到: {log_file}")


if __name__ == "__main__":
    main()