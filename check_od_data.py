# check_od_data.py

from __future__ import annotations
import os
import logging
import pandas as pd
import yaml
from typing import Optional, Tuple

# 设置日志
logger = logging.getLogger(__name__)
# 确保日志配置与 main_pipeline.py 中的一致
# 如果作为独立脚本运行，需要在此处进行基本配置
if not logger.handlers:
    # 尝试从 config.yaml 加载日志配置
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg_log = yaml.safe_load(f) # 使用不同的变量名避免与全局cfg冲突
        log_level = getattr(logging, cfg_log.get('LOG_LEVEL', 'INFO').upper())
        log_format = cfg_log.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = cfg_log.get('LOG_FILE', 'pipeline_check.log') # 使用不同的日志文件避免冲突

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)

        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
        logger.info("日志系统初始化完成 (for check_od_data.py)")
    except Exception as e:
        # 如果config.yaml加载失败，使用默认基本配置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.warning(f"无法从 config.yaml 加载日志配置，使用默认配置。错误: {e}")

# ---------- 配置加载 ----------
_config_cache: Optional[dict] = None

def get_config() -> dict:
    global _config_cache
    if _config_cache is None:
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError("config.yaml 未找到，请放到项目根目录")
        except yaml.YAMLError as e:
            raise RuntimeError(f"解析 config.yaml 出错: {e}")
    return _config_cache

cfg = get_config()

# 状态常量 (从config中获取或使用默认值)
LOADED_STATUS = cfg.get("OD_CONFIG", {}).get("LOADED_STATUS", 268435456) # 载客状态
EMPTY_STATUS = cfg.get("OD_CONFIG", {}).get("EMPTY_STATUS", 0)       # 空载状态

# OD对的关键列，用于检查存在性和空值
OD_REQUIRED_COLUMNS = [
    "car_id",
    "timestamp_O_actual", "lon_gcj_O", "lat_gcj_O", "status_O",
    "timestamp_D_actual", "lon_gcj_D", "lat_gcj_D", "status_D",
    "trip_duration_sec", "trip_distance_meter"
]

def check_od_data_integrity(file_path: str) -> bool:
    """
    检查OD对数据的完整性和正确性。
    Args:
        file_path (str): OD数据文件的完整路径 (例如 .csv 或 .parquet)。
    Returns:
        bool: 如果数据通过所有检查则返回True，否则返回False。
    """
    logger.info(f"\n--- 开始检查OD数据文件: {file_path} ---")

    # 1. 文件存在性与可加载性
    if not os.path.exists(file_path):
        logger.error(f"错误: OD数据文件 '{file_path}' 不存在。")
        return False

    try:
        if file_path.lower().endswith('.csv'):
            df_od = pd.read_csv(file_path)
        elif file_path.lower().endswith('.parquet'):
            df_od = pd.read_parquet(file_path)
        else:
            logger.error(f"错误: 不支持的文件格式 '{os.path.splitext(file_path)[1]}'。只支持CSV和Parquet。")
            return False
        logger.info(f"成功加载文件，共 {len(df_od)} 条记录。")
    except Exception as e:
        logger.error(f"错误: 加载文件 '{file_path}' 失败: {e}")
        return False

    # 2. 数据框非空
    if df_od.empty:
        logger.warning("警告: 加载的OD数据为空 DataFrame。没有记录可供检查。")
        return False

    # 3. 关键列存在性
    missing_columns = [col for col in OD_REQUIRED_COLUMNS if col not in df_od.columns]
    if missing_columns:
        logger.error(f"错误: OD数据缺少以下关键列: {missing_columns}")
        return False
    logger.info("所有关键列均存在。")

    # 确保时间戳列是datetime类型
    try:
        df_od['timestamp_O_actual'] = pd.to_datetime(df_od['timestamp_O_actual'], errors='coerce')
        df_od['timestamp_D_actual'] = pd.to_datetime(df_od['timestamp_D_actual'], errors='coerce')
        # 再次处理因为coerce可能产生的NaN
        df_od.dropna(subset=['timestamp_O_actual', 'timestamp_D_actual'], inplace=True)
        if df_od.empty:
            logger.warning("警告: 时间戳转换后OD数据变为空，跳过后续检查。")
            return False
        logger.info("时间戳列已成功转换为datetime类型。")
    except Exception as e:
        logger.error(f"错误: 转换时间戳列时发生异常: {e}")
        return False

    # 4. 数据类型检查 (针对可能非datetime的列)
    initial_count = len(df_od)

    # Convert car_id to string
    df_od['car_id'] = df_od['car_id'].astype(str)

    # Convert numerical columns using pd.to_numeric with errors='coerce'
    numerical_cols_for_coerce = [
        "lon_gcj_O", "lat_gcj_O", "lon_gcj_D", "lat_gcj_D",
        "status_O", "status_D", "trip_duration_sec", "trip_distance_meter"
    ]
    for col in numerical_cols_for_coerce:
        if col in df_od.columns: # 确保列存在
            df_od[col] = pd.to_numeric(df_od[col], errors='coerce')

    # Check for NaNs introduced by coercion and drop rows
    cols_to_check_nan_after_type_conversion = [
        "lon_gcj_O", "lat_gcj_O", "lon_gcj_D", "lat_gcj_D",
        "status_O", "status_D", "trip_duration_sec", "trip_distance_meter"
    ]
    for col in cols_to_check_nan_after_type_conversion:
        if col in df_od.columns and df_od[col].isnull().any():
            nan_count = df_od[col].isnull().sum()
            logger.warning(f"警告: 列 '{col}' 在类型转换后检测到 {nan_count} 个无效值 (NaN)。这些行将被丢弃。")
            df_od.dropna(subset=[col], inplace=True)

    if len(df_od) < initial_count:
        logger.info(f"因类型转换产生的无效值，已丢弃 {initial_count - len(df_od)} 条记录，剩余 {len(df_od)} 条。")
    if df_od.empty:
        logger.warning("警告: 数据类型检查后OD数据变为空，跳过后续检查。")
        return False

    logger.info("关键列的数据类型已检查。")

    # 5. 空值/NaN检查 (对所有关键列进行，排除时间戳在前面已处理)
    for col in OD_REQUIRED_COLUMNS:
        if col in df_od.columns and df_od[col].isnull().any():
            nan_count = df_od[col].isnull().sum()
            logger.warning(f"警告: 列 '{col}' 存在 {nan_count} 个空值 (NaN)。这些行将被丢弃。")
            df_od.dropna(subset=[col], inplace=True)
    if df_od.empty:
        logger.warning("警告: 空值检查后OD数据变为空，跳过后续检查。")
        return False
    logger.info("所有关键列均无空值。")
    initial_count_after_nan_check = len(df_od)

    # 6. 逻辑一致性检查
    # a. 时间顺序
    invalid_time_order = df_od[df_od['timestamp_O_actual'] >= df_od['timestamp_D_actual']]
    if not invalid_time_order.empty:
        logger.warning(f"警告: 发现 {len(invalid_time_order)} 条记录上车时间晚于或等于下车时间。这些行将被丢弃。")
        df_od = df_od[df_od['timestamp_O_actual'] < df_od['timestamp_D_actual']].copy()

    # b. 行程持续时间
    invalid_duration = df_od[df_od['trip_duration_sec'] <= 0]
    if not invalid_duration.empty:
        logger.warning(f"警告: 发现 {len(invalid_duration)} 条记录行程持续时间 <= 0 秒。这些行将被丢弃。")
        df_od = df_od[df_od['trip_duration_sec'] > 0].copy()

    # c. 行程距离
    invalid_distance = df_od[df_od['trip_distance_meter'] <= 0]
    if not invalid_distance.empty:
        logger.warning(f"警告: 发现 {len(invalid_distance)} 条记录行程距离 <= 0 米。这些行将被丢弃。")
        df_od = df_od[df_od['trip_distance_meter'] > 0].copy()

    # d. 状态值
    invalid_o_status = df_od[df_od['status_O'] != LOADED_STATUS]
    if not invalid_o_status.empty:
        logger.warning(f"警告: 发现 {len(invalid_o_status)} 条记录上车状态不是载客状态 ({LOADED_STATUS})。这些行将被丢弃。")
        df_od = df_od[df_od['status_O'] == LOADED_STATUS].copy()

    invalid_d_status = df_od[df_od['status_D'] != EMPTY_STATUS]
    if not invalid_d_status.empty:
        logger.warning(f"警告: 发现 {len(invalid_d_status)} 条记录下车状态不是空载状态 ({EMPTY_STATUS})。这些行将被丢弃。")
        df_od = df_od[df_od['status_D'] == EMPTY_STATUS].copy()

    if len(df_od) < initial_count_after_nan_check:
        logger.info(f"因逻辑不一致已丢弃 {initial_count_after_nan_check - len(df_od)} 条记录。")
    if df_od.empty:
        logger.warning("警告: 逻辑检查后OD数据变为空，请检查数据质量。")
        return False

    logger.info(f"所有逻辑一致性检查通过，剩余 {len(df_od)} 条有效OD记录。")

    # 7. 统计信息
    logger.info("\n--- OD数据统计信息 ---")
    numerical_cols = ['trip_duration_sec', 'trip_distance_meter', 'lon_gcj_O', 'lat_gcj_O', 'lon_gcj_D', 'lat_gcj_D']
    for col in numerical_cols:
        if col in df_od.columns:
            logger.info(f"列 '{col}' 统计:")
            logger.info(f"  Min: {df_od[col].min():.2f}")
            logger.info(f"  Max: {df_od[col].max():.2f}")
            logger.info(f"  Mean: {df_od[col].mean():.2f}")
            logger.info(f"  Median: {df_od[col].median():.2f}")
            logger.info(f"  Std Dev: {df_od[col].std():.2f}")
    
    # 额外统计：唯一车辆ID数量
    if 'car_id' in df_od.columns:
        unique_cars = df_od['car_id'].nunique()
        logger.info(f"唯一车辆ID数量: {unique_cars}")

    logger.info("\n--- OD数据完整性检查完成 ---")
    return True

if __name__ == '__main__':
    # 从 config.yaml 获取文件路径
    config = get_config()
    cleaned_data_dir = config["CLEANED_DATA_DIR"]
    od_data_subdir = config["OD_DATA_SUBDIR"]
    all_week_od_filename = config["ALL_WEEK_OD_FILENAME"]

    # 示例1: 检查合并后的周级OD数据 (CSV)
    all_week_od_path_csv = os.path.join(cleaned_data_dir, all_week_od_filename)
    logger.info(f"尝试检查总周OD数据: {all_week_od_path_csv}")
    check_od_data_integrity(all_week_od_path_csv)

    # 示例2: 检查某个特定日期的OD数据 (Parquet)
    # 假设你有一个日期前缀，例如 '20230101'
    # 你可以修改这里来检查你想要的文件
    # 单日OD文件名为 'YYYYMMDD_od_wh_drop.parquet'
    single_day_prefix = '20230101' # 替换为你要检查的实际日期前缀
    single_day_od_path_parquet = os.path.join(cleaned_data_dir, od_data_subdir, f"{single_day_prefix}_od_wh_drop.parquet")
    logger.info(f"\n尝试检查单日OD数据: {single_day_od_path_parquet}")
    check_od_data_integrity(single_day_od_path_parquet)

    # 你可以遍历 cleaned_data/od_drop/ 目录下的所有parquet文件进行检查
    logger.info("\n--- 开始批量检查 cleaned_data/od_drop/ 目录下的所有OD Parquet文件 ---")
    od_drop_dir = os.path.join(cleaned_data_dir, od_data_subdir)
    if os.path.exists(od_drop_dir):
        od_files = [f for f in os.listdir(od_drop_dir) if f.lower().endswith('_od_wh_drop.parquet')]
        if od_files:
            for od_file in sorted(od_files):
                full_path = os.path.join(od_drop_dir, od_file)
                check_od_data_integrity(full_path)
        else:
            logger.info(f"在目录 {od_drop_dir} 中未找到任何 '_od_wh_drop.parquet' 文件。")
    else:
        logger.warning(f"OD数据目录 '{od_drop_dir}' 不存在，跳过批量检查。")