# main_pipeline.py

import os
import pandas as pd
import logging
import yaml
from data_ingestion import load_raw_data, get_raw_files_list
from data_cleaning import clean_single_day_data, get_config # 导入清洗函数与配置获取函数

# --- 1. 配置加载与初始化 ---
try:
    # 加载配置文件
    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    config = load_config('config.yaml')
    if not isinstance(config, dict):
        raise RuntimeError("配置文件解析失败，预期字典类型")
    # 再次加载配置到模块级别的logger，确保所有模块的日志配置一致

except FileNotFoundError:
    print("错误: config.yaml 文件未找到，请确保它在项目根目录下。")
    exit(1)
except yaml.YAMLError as e:
    print(f"错误: 解析 config.yaml 文件时发生错误: {e}")
    exit(1)

# --- 2. 日志系统初始化 ---
# 获取根 logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, config['LOG_LEVEL']))

# 确保不重复添加 handler，这在模块被多次导入时很重要
# 仅当根 logger 没有处理器时才添加
if not root_logger.handlers:
    # 创建一个 StreamHandler 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config['LOG_FORMAT']))
    root_logger.addHandler(console_handler)

    # 创建一个 FileHandler 输出到文件
    log_file_path = config['LOG_FILE']
    # 确保日志文件目录存在
    os.makedirs(os.path.dirname(log_file_path) or '.', exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(config['LOG_FORMAT']))
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # 获取 main_pipeline 自身的 logger
logger.info("日志系统初始化完成")

# --- 3. 目录与文件配置 ---
RAW_DATA_DIR = config["RAW_DATA_DIR"]
CLEANED_DATA_DIR = config["CLEANED_DATA_DIR"]
ALL_WEEK_OD_FILENAME = config["ALL_WEEK_OD_FILENAME"]

os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
logger.info(f"原始数据目录: {RAW_DATA_DIR}")
logger.info(f"清洗后数据输出目录: {CLEANED_DATA_DIR}")


# --- 4. 数据清洗主管道 ---
def run_data_cleaning_pipeline():
    all_cleaned_od_dfs = []

    # 1. 获取所有原始数据文件列表
    raw_files = get_raw_files_list(RAW_DATA_DIR)
    if not raw_files:
        logger.error(f"在目录 {RAW_DATA_DIR} 中未找到任何原始数据文件。请检查配置或数据是否存在。")
        return

    logger.info(f"找到 {len(raw_files)} 个原始数据文件，开始按天处理...")

    # 2. 循环处理每一天的原始数据
    for raw_file_name in raw_files:
        day_prefix = os.path.splitext(raw_file_name)[0] # 通常文件名就是日期前缀
        logger.info(f"\n--- 开始处理 {day_prefix} 的数据 ---")

        # 加载单日原始数据
        df_raw_day = load_raw_data(raw_file_name)

        if df_raw_day is not None and not df_raw_day.empty:
            # 清洗并识别单日OD对
            # 修正了这里 clean_single_day_data 的调用，传入 df_raw_day 和 day_prefix
            view_df, od_df = clean_single_day_data(df_raw_day, day_prefix)

            if od_df is not None and not od_df.empty:
                # 收集所有日期的 OD 数据，方便后续进行周级分析
                od_df['date_prefix'] = day_prefix # 添加一个日期标识列
                all_cleaned_od_dfs.append(od_df)
            elif od_df is None:
                logger.warning(f"文件 {raw_file_name} 的清洗过程未返回OD数据。")
            else: # od_df is not None and od_df.empty
                logger.info(f"文件 {raw_file_name} 清洗后未生成有效OD数据。")
        else:
            logger.warning(f"跳过处理文件: {raw_file_name}，因为加载失败或数据为空。")

    # 3. 合并所有日期的 OD 数据，用于周级分析
    if all_cleaned_od_dfs:
        df_all_week_od = pd.concat(all_cleaned_od_dfs, ignore_index=True)
        logger.info(f"\n所有 {len(all_cleaned_od_dfs)} 天的 OD 数据已合并，总计 {len(df_all_week_od)} 条记录。")

        # 保存合并后的周级 OD 数据
        all_od_output_path = os.path.join(CLEANED_DATA_DIR, ALL_WEEK_OD_FILENAME)
        os.makedirs(os.path.dirname(all_od_output_path), exist_ok=True) # 确保目录存在
        df_all_week_od.to_csv(all_od_output_path, index=False)
        logger.info(f"合并后的周级OD数据已保存到: {all_od_output_path}")

        # 打印部分结果
        logger.info("\n合并后的周级OD数据 (前5行):")
        logger.info(f"\n{df_all_week_od.head().to_string()}")
    else:
        logger.warning("未生成任何周级OD数据。")


if __name__ == '__main__':
    logger.info("数据清洗管道开始运行...")
    run_data_cleaning_pipeline()
    logger.info("数据清洗管道运行结束。")