"""加载主观评测数据集（来自 yaml 文件）。"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_subjective_datasets(
    dataset_dir: str = "config/subjective_datasets",
) -> list[dict]:
    """
    扫描目录下所有 .yaml 文件，每个文件对应一个数据集。
    返回列表，每项格式：
      {
        "dataset_name": str,
        "description": str,
        "cases": [{"id": str, "question": str, "reference_answer": str, "tags": [...]}, ...]
      }
    """
    base = Path(dataset_dir)
    if not base.exists():
        logger.warning("主观数据集目录不存在: %s", dataset_dir)
        return []

    datasets = []
    for yaml_file in sorted(base.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # 基本校验
            if "cases" not in data:
                logger.warning("跳过无效数据集文件（缺少 cases 字段）: %s", yaml_file)
                continue
            datasets.append(data)
            logger.info("已加载数据集: %s（%d 条）", data.get("dataset_name", yaml_file.stem), len(data["cases"]))
        except Exception as e:
            logger.error("加载数据集文件失败 %s: %s", yaml_file, e)

    return datasets
