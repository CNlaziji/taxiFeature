# od_clustering.py

import os
import pandas as pd
import logging
import yaml
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime, timedelta

# 从data_cleaning.py导入get_config函数，确保配置一致
try:
    from data_cleaning import get_config
except ImportError:
    # 如果data_cleaning.py中没有get_config，或者您想独立加载，则自行定义
    def get_config():
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError("config.yaml 未找到，请放到项目根目录")
        except yaml.YAMLError as e:
            raise RuntimeError(f"解析 config.yaml 出错: {e}")

logger = logging.getLogger(__name__)

def run_od_clustering_by_folder():
    logger.info("开始OD对上客点批量聚类 (通过遍历文件夹)...")

    config = get_config()
    
    # 路径配置
    CLEANED_DATA_DIR = config["CLEANED_DATA_DIR"]
    OD_DATA_SUBDIR = config["OD_DATA_SUBDIR"]
    HOTSPOT_OUTPUT_DIR = config["OD_CLUSTERING"]["HOTSPOT_OUTPUT_DIR"]
    
    # 确保输出目录存在
    os.makedirs(HOTSPOT_OUTPUT_DIR, exist_ok=True)

    # 聚类参数
    EPS = config["OD_CLUSTERING"]["DBSCAN_EPS_DEG"]
    MIN_SAMPLES = config["OD_CLUSTERING"]["DBSCAN_MIN_SAMPLES"]
    TIME_BIN_INTERVAL_HOURS = config["OD_CLUSTERING"]["TIME_BIN_INTERVAL_HOURS"]
    OD_FILE_PREFIX_PATTERN = config["OD_CLUSTERING"]["OD_FILE_PREFIX_PATTERN"]

    # 构建OD数据文件夹路径
    od_data_folder_path = os.path.join(CLEANED_DATA_DIR, OD_DATA_SUBDIR)
    
    if not os.path.isdir(od_data_folder_path):
        logger.error(f"错误: OD数据目录 '{od_data_folder_path}' 不存在。请检查配置或前序步骤。")
        return

    # 获取所有符合模式的parquet文件
    all_od_files = [
        f for f in os.listdir(od_data_folder_path) 
        if f.startswith(OD_FILE_PREFIX_PATTERN) and f.endswith('_od_wh_drop.parquet')
    ]
    all_od_files.sort() # 按文件名排序，通常会按日期顺序处理

    if not all_od_files:
        logger.warning(f"在目录 '{od_data_folder_path}' 中未找到任何符合 '{OD_FILE_PREFIX_PATTERN}*_od_wh_drop.parquet' 模式的OD数据文件。")
        return

    logger.info(f"找到 {len(all_od_files)} 个OD数据文件，开始逐个处理...")

    for od_file_name in all_od_files:
        day_prefix = od_file_name.replace('_od_wh_drop.parquet', '') # 从文件名提取前缀 (例如 'jn0912')
        input_od_file_path = os.path.join(od_data_folder_path, od_file_name)
        
        logger.info(f"\n--- 开始处理文件: {od_file_name} ---")

        logger.info(f"加载OD数据: {input_od_file_path}")
        try:
            df_od = pd.read_parquet(input_od_file_path)
        except Exception as e:
            logger.error(f"读取文件 '{input_od_file_path}' 失败: {e}")
            continue

        if df_od.empty:
            logger.warning(f"文件 '{input_od_file_path}' 为空，无法进行聚类。")
            continue

        logger.info(f"成功加载 {len(df_od)} 条OD记录。")

        # 确保时间戳是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_od['timestamp_O_actual']):
            df_od['timestamp_O_actual'] = pd.to_datetime(df_od['timestamp_O_actual'])

        # 提取小时作为时间分箱依据
        df_od['hour'] = df_od['timestamp_O_actual'].dt.floor(f'{TIME_BIN_INTERVAL_HOURS}H')

        all_hotspots_for_day = []

        # 按小时分箱进行聚类
        unique_hours = sorted(df_od['hour'].unique())
        logger.info(f"将按 {len(unique_hours)} 个时间段 ({TIME_BIN_INTERVAL_HOURS}小时间隔) 进行聚类。")

        for current_hour in unique_hours:
            df_hour = df_od[df_od['hour'] == current_hour].copy()
            
            # 提取上客点经纬度
            # 确保这些列是数值类型，以防万一
            X = df_hour[['lon_gcj_O', 'lat_gcj_O']].astype(float).values 
            
            if len(X) < MIN_SAMPLES:
                logger.info(f"时间段 {current_hour}: 数据点不足 ({len(X)} < {MIN_SAMPLES})，跳过聚类。")
                continue

            logger.info(f"时间段 {current_hour}: 开始DBSCAN聚类，共 {len(X)} 个上客点...")
            
            # 应用DBSCAN
            db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)
            labels = db.labels_

            # 提取聚类结果
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # -1 表示噪声点，我们不将其视为热力点
                    continue

                # 获取当前簇中的点
                cluster_points_mask = (labels == label)
                cluster_points = X[cluster_points_mask]
                
                # 计算簇的中心点和数量
                center_lon = np.mean(cluster_points[:, 0])
                center_lat = np.mean(cluster_points[:, 1])
                count = len(cluster_points)

                # 记录热力点信息
                hotspot_time = current_hour.strftime('%Y-%m-%d %H:00:00') # 格式化时间点
                all_hotspots_for_day.append({
                    'lat': center_lat,
                    'lng': center_lon,
                    'count': count,
                    'time': hotspot_time
                })
            logger.info(f"时间段 {current_hour}: 识别到 {len(unique_labels) - (1 if -1 in unique_labels else 0)} 个热力簇。")

        if all_hotspots_for_day:
            df_hotspots = pd.DataFrame(all_hotspots_for_day)
            
            # 构建输出文件路径，为每天生成一个文件
            output_hotspot_file = os.path.join(HOTSPOT_OUTPUT_DIR, f"{day_prefix}_hear.csv")
            df_hotspots.to_csv(output_hotspot_file, index=False)
            logger.info(f"DBSCAN聚类结果（热力点数据）已保存到: {output_hotspot_file}")
            logger.info("\n热力点数据 (前5行):")
            logger.info(f"\n{df_hotspots.head().to_string()}")
        else:
            logger.warning(f"文件 '{input_od_file_path}' 未识别到任何热力点。请检查DBSCAN参数或数据。")
        
    logger.info("所有指定目录下的OD对上客点聚类完成。")


if __name__ == '__main__':
    # 为了让这个脚本独立运行，也需要初始化日志系统
    root_logger = logging.getLogger()
    if not root_logger.handlers: # 避免重复添加handler
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    run_od_clustering_by_folder()