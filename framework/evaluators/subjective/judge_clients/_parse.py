"""健壮的 JSON 解析工具：处理 LLM 常见的输出格式问题。"""

import json
import re


def extract_json(text: str) -> dict:
    """
    从 LLM 输出中提取 JSON 对象。
    处理以下情况：
    - 纯 JSON
    - ```json ... ``` 代码块
    - JSON 前后有多余文字
    """
    text = text.strip()

    # 1. 先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 去掉 markdown 代码块
    md_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 提取第一个 {...} 块
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 输出中解析 JSON，原始内容：\n{text[:500]}")
