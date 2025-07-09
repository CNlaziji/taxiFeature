# verify_exported_data.py

import os
import pandas as pd
import logging
import yaml
# 导入数据加载和清洗模块的关键函数
from data_ingestion import load_raw_data
from data_cleaning import clean_single_day_data, get_config 

# --- 1. 日志系统初始化 ---
# 为验证脚本设置独立的日志，方便跟踪验证过程
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_data(day_prefix: str) -> None:
    """
    验证特定日期的数据导出正确性。
    通过重新运行清洗和OD识别流程，并将结果与导出的Parquet文件进行比较。

    Args:
        day_prefix (str): 要验证的数据日期前缀 (例如 'jn0912')。
    """
    logger.info(f"--- 开始验证 {day_prefix} 的导出数据 ---")

    # 获取项目配置
    config = get_config() 

    # 从配置中获取目录信息
    RAW_DATA_DIR = config["RAW_DATA_DIR"]
    CLEANED_DATA_DIR = config["CLEANED_DATA_DIR"]
    VIEW_DATA_SUBDIR = config["VIEW_DATA_SUBDIR"]
    OD_DATA_SUBDIR = config["OD_DATA_SUBDIR"]

    # 假设原始数据文件名为 {day_prefix}.csv
    raw_file_name = f"{day_prefix}.csv" 

    # --- 1. 加载原始数据 ---
    logger.info(f"正在加载原始数据文件: {raw_file_name}...")
    df_raw = load_raw_data(raw_file_name)
    if df_raw is None or df_raw.empty:
        logger.error(f"无法加载或原始数据为空: {raw_file_name}，无法进行验证。")
        return

    # --- 2. 在内存中重新运行数据清洗和OD识别流程，生成“预期结果” ---
    logger.info("在内存中重新运行数据清洗和OD识别流程，生成预期结果 (这可能需要一些时间)...")
    # clean_single_day_data 函数会返回清洗后的 df_view 和 df_od
    # 注意：clean_single_day_data 内部会再次尝试保存文件，这可能会覆盖您已导出的文件。
    # 但对于验证来说，这是可以接受的，因为我们只是想确保其生成的数据一致。
    expected_df_view, expected_df_od = clean_single_day_data(df_raw.copy(), day_prefix)

    # --- 3. 加载之前导出的 Parquet 文件 ---
    logger.info("正在加载之前导出的 Parquet 文件...")
    view_parquet_path = os.path.join(CLEANED_DATA_DIR, VIEW_DATA_SUBDIR, f"{day_prefix}_view.parquet")
    od_parquet_path = os.path.join(CLEANED_DATA_DIR, OD_DATA_SUBDIR, f"{day_prefix}_od_wh_drop.parquet")

    loaded_df_view = None
    loaded_df_od = None

    if os.path.exists(view_parquet_path):
        try:
            loaded_df_view = pd.read_parquet(view_parquet_path)
            logger.info(f"成功加载 {view_parquet_path}，行数: {len(loaded_df_view)}")
        except Exception as e:
            logger.error(f"加载 {view_parquet_path} 失败: {e}")
    else:
        logger.warning(f"文件 {view_parquet_path} 不存在，无法验证 view 数据。")

    if os.path.exists(od_parquet_path):
        try:
            loaded_df_od = pd.read_parquet(od_parquet_path)
            logger.info(f"成功加载 {od_parquet_path}，行数: {len(loaded_df_od)}")
        except Exception as e:
            logger.error(f"加载 {od_parquet_path} 失败: {e}")
    else:
        logger.warning(f"文件 {od_parquet_path} 不存在，无法验证 OD 数据。")

    # --- 4. 比较 DataFrames ---
    logger.info("开始比较预期数据和导出数据...")

    # 比较 df_view (清洗后的轨迹数据)
    if expected_df_view is not None and loaded_df_view is not None:
        if expected_df_view.empty and loaded_df_view.empty:
            logger.info("df_view: 预期和导出数据都为空，视为一致。")
        elif expected_df_view.empty != loaded_df_view.empty:
            logger.error(f"df_view: 预期数据为空({expected_df_view.empty})但导出数据不为空({loaded_df_view.empty})，或反之。")
        else:
            try:
                # 排序以确保行顺序一致性，这是 pd.testing.assert_frame_equal 的最佳实践
                # 使用 ['car_id', 'timestamp'] 作为排序键通常能保证唯一性和稳定性
                sort_cols_view = ['car_id', 'timestamp']
                expected_df_view_sorted = expected_df_view.sort_values(sort_cols_view).reset_index(drop=True)
                loaded_df_view_sorted = loaded_df_view.sort_values(sort_cols_view).reset_index(drop=True)

                # 使用 pd.testing.assert_frame_equal 进行严格比较
                # check_exact=False 允许浮点数有微小差异 (Parquet 存储可能导致)
                # atol 设置绝对容忍度，对于 GPS 坐标通常很小的值即可
                pd.testing.assert_frame_equal(
                    expected_df_view_sorted,
                    loaded_df_view_sorted,
                    check_dtype=True,       # 检查数据类型是否一致
                    check_exact=False,      # 允许浮点数存在微小差异
                    atol=1e-8,              # 绝对容忍度，例如 0.00000001
                    check_names=True        # 检查列名是否一致
                )
                logger.info("df_view: 预期数据与导出数据完全一致 (已考虑浮点精度差异)。")
            except AssertionError as e:
                logger.error(f"df_view: 预期数据与导出数据存在差异！详细信息: {e}")
                # 您可以在这里添加更多诊断信息，例如差异的行数或列名
            except KeyError as e:
                logger.error(f"df_view: 排序键缺失 ({e})，无法进行精确比较。请检查列名。")
            except Exception as e:
                logger.error(f"df_view: 比较过程中发生意外错误: {e}")
    elif expected_df_view is None and loaded_df_view is None:
        logger.warning("df_view: 预期数据和导出文件均不存在。跳过比较。")
    else:
        logger.error("df_view: 预期数据或导出文件状态不一致 (一个存在，另一个不存在)。")


    # 比较 df_od (OD对数据)
    if expected_df_od is not None and loaded_df_od is not None:
        if expected_df_od.empty and loaded_df_od.empty:
            logger.info("df_od: 预期和导出数据都为空，视为一致。")
        elif expected_df_od.empty != loaded_df_od.empty:
            logger.error(f"df_od: 预期数据为空({expected_df_od.empty})但导出数据不为空({loaded_df_od.empty})，或反之。")
        else:
            try:
                # OD 对通常通过 car_id 和 O点时间戳来唯一标识
                sort_cols_od = ['car_id', 'timestamp_O_actual']
                expected_df_od_sorted = expected_df_od.sort_values(sort_cols_od).reset_index(drop=True)
                loaded_df_od_sorted = loaded_df_od.sort_values(sort_cols_od).reset_index(drop=True)

                pd.testing.assert_frame_equal(
                    expected_df_od_sorted,
                    loaded_df_od_sorted,
                    check_dtype=True,
                    check_exact=False,
                    atol=1e-8,
                    check_names=True
                )
                logger.info("df_od: 预期数据与导出数据完全一致 (已考虑浮点精度差异)。")
            except AssertionError as e:
                logger.error(f"df_od: 预期数据与导出数据存在差异！详细信息: {e}")
                # 您可以在这里添加更多诊断信息，例如差异的行数或列名
            except KeyError as e:
                logger.error(f"df_od: 排序键缺失 ({e})，无法进行精确比较。请检查列名。")
            except Exception as e:
                logger.error(f"df_od: 比较过程中发生意外错误: {e}")
    elif expected_df_od is None and loaded_df_od is None:
        logger.warning("df_od: 预期数据和导出文件均不存在。跳过比较。")
    else:
        logger.error("df_od: 预期数据或导出文件状态不一致 (一个存在，另一个不存在)。")

    logger.info(f"--- {day_prefix} 数据验证结束 ---")

if __name__ == "__main__":
    # --- 示例用法 ---
    # 请将 'jn0912' 替换为您要验证的实际日期前缀
    # 确保 raw_data/jn0912.csv 和 cleaned_data/view/jn0912_view.parquet, cleaned_data/od_drop/jn0912_od_wh_drop.parquet 存在
    verify_data('jn0912')

    # 如果有多个日期需要验证，可以这样循环调用：
    # daily_prefixes = ['day1', 'day2', 'day3'] # 替换为您的实际日期列表
    # for prefix in daily_prefixes:
    #     verify_data(prefix)