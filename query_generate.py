import asyncio
import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
import uuid
from datasets import load_dataset
import re

from toolret.config import _FIRST_STAGE, _TOOL_REPO, _MODEL
from toolret.eval import trec_eval, load_tools, load_queries, eval_retrieval
from toolret.utils import fix_json_error
from toolret.eval import task_split
from utils.clogger import _set_logger
_set_logger(
    exp_dir=pathlib.Path("./logs"),
    logging_level_stdout=logging.INFO,
    logging_level=logging.DEBUG,
    file_name="baseline.log",
)

logger = logging.getLogger(__name__)
def _read_openai_config():
    api_key = "empty"#
    base_url ="http://localhost:7001/v1/"#http://localhost:7001/v1/
    model = "/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507" #/data/labshare/Param/gpt-oss/gpt-oss-20b
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")
    if not base_url:
        # 默认 OpenAI 官方，但也兼容其他服务
        base_url = "https://api.openai.com/v1"
    return base_url.rstrip("/"), api_key, model


def _call_openai(messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None):
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
        "tool_choice": {"type": "function", "function": {"name": "route"}}, 
    }
    
    # 添加 tools 参数
    if tools:
        payload["tools"] = tools

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data


import json
import re

def extract_query_simple(response):
    """统一提取 tool_calls / function_call / <tool_assistant> 中的 query 内容"""
    choices = response.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    content = message.get("content") or ""
    def clean_query(raw_query):
        """清洗 query 内容，去掉 <tool_assistant> 标签"""
        if not raw_query:
            return None
        match = re.search(r"query:\s*(.+?)\s*</tool_assistant>", raw_query, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return raw_query.strip()
    # 优先处理 tool_calls
    tool_calls = message.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        last_call = tool_calls[-1]
        func = last_call.get("function", {})
        args_str = func.get("arguments")
        if args_str:
            try:
                args = json.loads(args_str)
                return clean_query(args.get("query"))
            except json.JSONDecodeError:
                pass

    # 兼容旧格式 function_call
    function_call = message.get("function_call")
    if function_call:
        args_str = function_call.get("arguments")
        if args_str:
            try:
                args = json.loads(args_str)
                return clean_query(args.get("query"))
            except json.JSONDecodeError:
                pass

    # 处理 <tool_assistant> 标签格式
    if "<tool_assistant>" in content:
        # 使用 DOTALL 让 . 能匹配换行
        match = re.search(r"query:\s*(.+?)\s*</tool_assistant>", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "No Route"
    # pattern = r'<tool_assistant>\s*query:\s*(.*?)\s*</tool_assistant>'
    
    # try:
    #     # 优先尝试 tool_calls 格式
    #     tool_calls = response['choices'][0]['message'].get('tool_calls')
    #     if tool_calls and len(tool_calls) > 0:
    #         arguments = json.loads(tool_calls[0]['function']['arguments'])
    #         query_with_tags = arguments.get('query', '')
    #         match = re.search(pattern, query_with_tags, re.DOTALL)
    #         if match:
    #             return match.group(1).strip()
    # except Exception:
    #     pass
    # # 没匹配上
    # return "No Route"

def save_query_record(task_name, query_id, original_query, llm_query, output_file="query_records.txt"):
    """
    将查询记录保存到txt文件
    
    Args:
        task_name: 任务名称
        query_id: 查询ID
        original_query: 原始查询
        llm_query: 大模型生成的查询
        output_file: 输出文件名
    """
    try:
        # 格式化记录
        record = f"任务id: {query_id}\n任务: {original_query}\nllm任务: {llm_query}\n"
        #record += "-" * 50 + "\n"  # 分隔线
        
        # 追加写入文件
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(record)
        
        #print(f"Record saved: task={task_name}, query_id={query_id}")
        
    except Exception as e:
        print(f"Error saving record: {e}")


class LoggingMCPClient():
    def process_query(
        self,
        user_query: str,
        current_task:str,
        qid:str,
        output_file: str = "query_records.txt",
        history: Optional[list] = None,
        max_tool_tokens: int = 10000,
    ) -> Tuple[str, List[dict]]:

        messages = [
            {
                "role": "system",
                "content": """\
You are a tool invocation agent. Your only responsibility is to call the route tool.
Notes:

You are not allowed to answer user questions directly.

You must call the route tool to solve user problems.

You are strictly forbidden from outputting natural language explanations, analyses, or thoughts.

If you output natural language instead of invoking the tool, it is considered illegal output.
""",
            }
        ]

        messages.append({"role": "user", "content": user_query}) 
        #response = await session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": 'route',
                    "description": '\n    This is a tool used to find useful tools that can solve user needs       \n    When to use this tool:\n        -When faced with user needs, you (LLM) are unable to solve them on your own and do not have the tools to solve the problem.\n        -When a user proposes a new task and you (LLM) are unsure which specific tool to use to complete it.\n        -When the user\'s request is vague or complex, and feasible tool options need to be explored first.\n        -This is the first step in executing unknown tasks, known as the "discovery" phase, aimed at finding the correct tool.\n    **Parameter Description**\n    Query (string, required): The input query must contain a <tool_assistant> tag with the query, for example: \n        <tool_assistant>\n        query: xxx \n        </tool_assistant>\n    ',
                    "parameters":{
                        'properties': {
                            'query': {
                                'title': 'Query',
                                'type': 'string'
                            }
                        },
                        'required': ['query'],
                        'title': 'routeArguments',
                        'type': 'object'
                    },
                },
            }
        ]

        try:
            response = _call_openai(messages, available_tools)
            llm_query = extract_query_simple(response)
            # if llm_query == "No Route":
            #     print("response",response)
            print(f"response:{response}")
            save_query_record(
                        task_name=current_task,
                        query_id=qid,
                        original_query=user_query,
                        llm_query=llm_query,
                        output_file=output_file
                    )
                
        except Exception as e:
            logger.error(f"Error processing query '{user_query}': {e}")



def run(task: str = "all", from_top_k: int = 20, embed_model: str = None, output_file: str = None) -> Dict[str, Dict[str, float]]:
    # 1) 先运行第一阶段检索（与仓库评测保持一致），得到每个 query 的候选工具集合
    # 为避免重复检索，直接利用 eval_retrieval 的输出文件携带候选（scores 不重要，只要候选ID）
    tasks = task
    # 读取检索结果 {task: {qid: {tool_id: score}}}
    _tasks = task_split(tasks)

    current_task = _tasks
    #if current_task is not None:
    for current_task in tqdm(_tasks):
        #task_results = retrieval_results.get(current_task, {})
        print(f"Processing task: {current_task}")
               #跨阶段分割线
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"数据集：{current_task}------------------------------------------------------\n")
        # 2) 加载 queries 与全量工具映射
        queries = load_queries(current_task)
        # 处理当前任务的查询
        try:
            for q in queries:
                qid = q['id']
                #instruction = q.get("instruction", "你是一个工具选择器。根据用户查询选择最相关的工具。只返回JSON，不要返回多余文字。")
                user_query = q.get("query", "")

                # 调用 LLM
                client = LoggingMCPClient()
                print("current query:",user_query)
                client.process_query(user_query,current_task,qid,output_file,None)
        except Exception as e:
                logger.error(f"Error processing query")

                



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM tool selection over first-stage retrieval and evaluate.")
    parser.add_argument("--task", type=str, default="all", help="任务名；例如: toolbench 或 all")
    parser.add_argument("--topk", type=int, default=20, help="从第一阶段候选中取前K个")
    parser.add_argument("--embed-model", type=str, default=None, help="用于第一阶段检索的嵌入模型名（默认取配置的第一个）")
    parser.add_argument("--output_file", type=str, default="query_records.txt", help="输出文件路径")
    args = parser.parse_args()
    run(task=args.task, from_top_k=args.topk, embed_model=args.embed_model, output_file=args.output_file)