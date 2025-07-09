# data_ingestion.py
import os
import logging
from typing import List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
_config_cache: Optional[dict] = None


def get_config() -> dict:
    global _config_cache
    if _config_cache is None:
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError("未找到 config.yaml 文件。")
        except yaml.YAMLError as e:
            raise RuntimeError(f"解析 config.yaml 出错: {e}")
    return _config_cache


def load_raw_data(file_name: str) -> Optional[pd.DataFrame]:
    config = get_config()
    raw_dir = config["RAW_DATA_DIR"]
    mapping = config["COLUMN_MAPPING"]
    required_cols = config["REQUIRED_RAW_COLS_FOR_CHECK"]

    file_path = os.path.join(raw_dir, file_name)
    logger.info(f"加载原始文件: {file_path}")
    try:
        df = pd.read_csv(file_path)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"{file_name} 缺少必要原始列: {missing}")
            return None

        # 字段映射（原始名 ➜ 通用名）
        df = df.rename(columns=mapping)

        # 经纬度缩放并显式转换为 float 类型
        if "longitude" in df.columns:
            df["longitude"] = df["longitude"].astype(float) / 1e5
        if "latitude" in df.columns:
            df["latitude"] = df["latitude"].astype(float) / 1e5

        logger.info(f"{file_name} 加载并处理完成，共 {len(df)} 行")
        return df

    except Exception as e:
        logger.error(f"读取 {file_name} 出错: {e}")
        return None


def get_raw_files_list() -> List[str]:
    config = get_config()
    raw_dir = config["RAW_DATA_DIR"]
    if not os.path.isdir(raw_dir):
        logger.error(f"原始数据目录不存在: {raw_dir}")
        return []
    return [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]