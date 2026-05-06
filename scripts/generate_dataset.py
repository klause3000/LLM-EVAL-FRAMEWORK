#!/usr/bin/env python
"""
评测数据集自动生成工具

用法：
  # 按主题生成
  python scripts/generate_dataset.py --topic "Python异常处理" --num 10 --name python_exception

  # 按文档生成
  python scripts/generate_dataset.py --doc path/to/doc.txt --num 10 --name my_doc_qa

  # 指定输出路径
  python scripts/generate_dataset.py --topic "SQL优化" --num 15 --name sql_qa --output config/subjective_datasets/sql_qa.yaml

  # 指定难度偏好
  python scripts/generate_dataset.py --topic "机器学习基础" --num 10 --name ml_basic --difficulty "简单到中等"
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import yaml

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.evaluators.subjective.judge_clients import build_judge_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def _build_topic_prompt(topic: str, num: int, difficulty: str, lang: str) -> str:
    lang_hint = "中文" if lang == "zh" else "英文"
    return f"""你是一个专业的 LLM 评测数据集设计专家。请针对以下主题，生成 {num} 道高质量问答题，用于评测大语言模型的生成质量（将用 BLEU/ROUGE/召回率指标评测）。

主题：{topic}
难度要求：{difficulty}
语言：{lang_hint}

设计要求：
1. 问题要有明确、唯一的参考答案，不能是开放性主观题
2. 参考答案要准确、完整、简洁（50~200字为宜）
3. 覆盖该主题的不同角度和知识点，避免重复
4. tags 从以下类型中选择（可多选）：知识理解、逻辑推理、代码、数学计算、事实问答、概念解释、流程步骤、对比分析
5. case id 格式：{topic[:4].replace(' ', '_')}_001, {topic[:4].replace(' ', '_')}_002, ...

请严格以 JSON 数组格式输出，不要有任何其他内容：
[
  {{
    "id": "xxx_001",
    "question": "问题内容",
    "reference_answer": "参考答案内容",
    "tags": ["标签1", "标签2"]
  }}
]"""


def _build_doc_prompt(doc_text: str, num: int, difficulty: str, dataset_name: str) -> str:
    # 文档过长则截断
    max_doc_len = 3000
    if len(doc_text) > max_doc_len:
        doc_text = doc_text[:max_doc_len] + "\n...[文档已截断]"

    return f"""你是一个专业的 LLM 评测数据集设计专家。请根据以下文档内容，生成 {num} 道高质量问答题，用于评测大语言模型对该文档的理解和生成质量。

文档内容：
---
{doc_text}
---

难度要求：{difficulty}

设计要求：
1. 问题必须基于文档内容，答案可以在文档中找到或合理推导
2. 参考答案要准确、完整，直接来源于文档（50~200字为宜）
3. 覆盖文档的不同部分，避免集中在某一段
4. tags 从以下类型中选择（可多选）：知识理解、逻辑推理、事实问答、概念解释、流程步骤、对比分析、细节提取
5. case id 格式：{dataset_name}_001, {dataset_name}_002, ...

请严格以 JSON 数组格式输出，不要有任何其他内容：
[
  {{
    "id": "xxx_001",
    "question": "问题内容",
    "reference_answer": "参考答案内容",
    "tags": ["标签1", "标签2"]
  }}
]"""


# ---------------------------------------------------------------------------
# JSON 解析（健壮版）
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> list:
    """从模型输出中提取 JSON 数组，兼容 markdown 代码块包裹的情况。"""
    # 去掉 markdown 代码块
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    raw = raw.replace("```", "").strip()

    # 直接尝试解析
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 提取第一个 [...] 块
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从模型输出中提取 JSON 数组，原始输出:\n{raw[:500]}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def generate(
    topic: str | None,
    doc_path: str | None,
    dataset_name: str,
    num: int,
    difficulty: str,
    lang: str,
    output_path: str,
    judge_config: str,
) -> None:
    # 构造 prompt
    if topic:
        prompt = _build_topic_prompt(topic, num, difficulty, lang)
        description = f"自动生成：{topic}（{num}条，难度：{difficulty}）"
        logger.info("按主题生成数据集: topic=%s num=%d", topic, num)
    else:
        doc_text = Path(doc_path).read_text(encoding="utf-8")
        prompt = _build_doc_prompt(doc_text, num, difficulty, dataset_name)
        description = f"基于文档自动生成：{Path(doc_path).name}（{num}条，难度：{difficulty}）"
        logger.info("按文档生成数据集: doc=%s num=%d", doc_path, num)

    # 调用 judge client
    logger.info("正在调用 judge 模型生成数据集，请稍候...")
    client = build_judge_client(judge_config)

    if hasattr(client, "client"):
        # OpenAI-compatible
        resp = client.client.chat.completions.create(
            model=client.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content or ""
    else:
        raise RuntimeError("不支持的 judge client 类型")

    logger.info("模型返回 %d 字符，开始解析...", len(raw))

    # 解析 JSON
    cases_raw = _extract_json_array(raw)
    logger.info("解析到 %d 条 case", len(cases_raw))

    # 校验和标准化
    cases = []
    for i, item in enumerate(cases_raw):
        if not isinstance(item, dict):
            logger.warning("跳过非字典项: %s", item)
            continue
        question = str(item.get("question", "")).strip()
        reference = str(item.get("reference_answer", "")).strip()
        if not question or not reference:
            logger.warning("跳过缺少 question 或 reference_answer 的项: %s", item)
            continue
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        case_id = str(item.get("id", f"{dataset_name}_{i+1:03d}")).strip()
        cases.append({
            "id": case_id,
            "question": question,
            "reference_answer": reference,
            "tags": tags,
        })

    if not cases:
        logger.error("未生成任何有效 case，请检查 judge 模型输出")
        sys.exit(1)

    # 构造 yaml 数据
    dataset = {
        "dataset_name": dataset_name,
        "description": description,
        "cases": cases,
    }

    # 写入文件
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(dataset, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    logger.info("数据集已写入: %s（共 %d 条）", out, len(cases))

    # 预览前3条
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}  共 {len(cases)} 条")
    print(f"输出: {out}")
    print(f"{'='*60}")
    for c in cases[:3]:
        print(f"\n[{c['id']}] tags: {c['tags']}")
        print(f"  Q: {c['question'][:80]}")
        print(f"  A: {c['reference_answer'][:80]}")
    if len(cases) > 3:
        print(f"\n  ... 还有 {len(cases)-3} 条，查看完整文件: {out}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM 评测数据集自动生成工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topic", "-t", help="按主题生成（如：'Python异常处理'）")
    group.add_argument("--doc", "-d", metavar="FILE", help="按文档生成（传入文本文件路径）")

    parser.add_argument("--name", "-n", required=True, help="数据集名称（作为 dataset_name 和文件名）")
    parser.add_argument("--num", type=int, default=10, help="生成 case 数量（默认 10）")
    parser.add_argument("--difficulty", default="简单到中等", help="难度要求（默认：简单到中等）")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="语言（默认 zh）")
    parser.add_argument(
        "--output", "-o",
        help="输出路径（默认 config/subjective_datasets/<name>.yaml）",
    )
    parser.add_argument(
        "--judge-config",
        default="config/judge_config.yaml",
        help="judge 配置文件路径（默认 config/judge_config.yaml）",
    )
    args = parser.parse_args()

    output = args.output or f"config/subjective_datasets/{args.name}.yaml"

    generate(
        topic=args.topic,
        doc_path=args.doc,
        dataset_name=args.name,
        num=args.num,
        difficulty=args.difficulty,
        lang=args.lang,
        output_path=output,
        judge_config=args.judge_config,
    )


if __name__ == "__main__":
    main()
