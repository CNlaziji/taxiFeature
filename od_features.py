# od_features.py

import os
import pandas as pd
import logging
import yaml
from datetime import datetime

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

def run_od_feature_extraction():
    logger.info("开始提取OD数据特征...")

    config = get_config()
    
    # 路径配置
    CLEANED_DATA_DIR = config["CLEANED_DATA_DIR"]
    ALL_WEEK_OD_FILENAME = config["ALL_WEEK_OD_FILENAME"]
    
    # 定义输出目录，可以考虑在config中增加一个feature_output_dir
    FEATURE_OUTPUT_DIR = os.path.join(CLEANED_DATA_DIR, "features") 
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"特征数据输出目录: {FEATURE_OUTPUT_DIR}")

    # 构建输入文件路径
    input_all_week_od_file = os.path.join(CLEANED_DATA_DIR, ALL_WEEK_OD_FILENAME)
    
    if not os.path.exists(input_all_week_od_file):
        logger.error(f"错误: 汇总OD文件 '{input_all_week_od_file}' 不存在。请先运行数据清洗管道。")
        return

    logger.info(f"加载汇总OD数据: {input_all_week_od_file}")
    try:
        # 这里使用 'trip_duration_sec' 和 'trip_distance_meter' 的字段类型应该没有问题
        # 如果有日期字段导致问题，可以调整 parse_dates 参数
        df_all_week_od = pd.read_csv(input_all_week_od_file, parse_dates=['timestamp_O_actual', 'timestamp_D_actual'])
    except Exception as e:
        logger.error(f"读取汇总OD文件 '{input_all_week_od_file}' 失败: {e}")
        return

    if df_all_week_od.empty:
        logger.warning(f"汇总OD文件 '{input_all_week_od_file}' 为空，无法进行特征提取。")
        return

    logger.info(f"成功加载 {len(df_all_week_od)} 条汇总OD记录。")

    # --- 1. 统计乘客打车数量 (15分钟和1小时间隔) ---
    logger.info("开始统计乘客打车数量 (15分钟和1小时间隔)...")
    
    # 将上车时间向下取整到最近的15分钟或小时
    df_all_week_od['timestamp_O_15min_bin'] = df_all_week_od['timestamp_O_actual'].dt.floor('15min')
    df_all_week_od['timestamp_O_1hour_bin'] = df_all_week_od['timestamp_O_actual'].dt.floor('H')

    # 15分钟间隔统计
    time_o_15min = df_all_week_od.groupby('timestamp_O_15min_bin').size().reset_index(name='count')
    time_o_15min.rename(columns={'timestamp_O_15min_bin': 'O_time'}, inplace=True)
    time_o_15min['O_time'] = time_o_15min['O_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_15min_file = os.path.join(FEATURE_OUTPUT_DIR, "JN_TIME_O.csv")
    time_o_15min.to_csv(output_15min_file, index=False)
    logger.info(f"15分钟间隔乘客打车数量已保存到: {output_15min_file}")
    logger.info(f"\nJN_TIME_O.csv (前5行):\n{time_o_15min.head().to_string()}")

    # 1小时间隔统计
    time_o_1hour = df_all_week_od.groupby('timestamp_O_1hour_bin').size().reset_index(name='count')
    time_o_1hour.rename(columns={'timestamp_O_1hour_bin': 'O_time'}, inplace=True)
    time_o_1hour['O_time'] = time_o_1hour['O_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_1hour_file = os.path.join(FEATURE_OUTPUT_DIR, "JN_TIME_O_WEEK.csv")
    time_o_1hour.to_csv(output_1hour_file, index=False)
    logger.info(f"1小时间隔乘客打车数量已保存到: {output_1hour_file}")
    logger.info(f"\nJN_TIME_O_WEEK.csv (前5行):\n{time_o_1hour.head().to_string()}")

    # --- 2. 统计载客出租车的数量 (1分钟间隔) ---
    logger.info("开始统计载客出租车的数量 (1分钟间隔)...")
    
    # 创建一个空的DataFrame来存储每分钟的载客车辆计数
    # 找到所有行程的最小开始时间和最大结束时间
    min_time = df_all_week_od['timestamp_O_actual'].min().floor('min')
    max_time = df_all_week_od['timestamp_D_actual'].max().ceil('min')
    
    # 生成1分钟的时间序列
    time_index = pd.date_range(start=min_time, end=max_time, freq='1min')
    
    # 为每个时间点计算处于载客状态的车辆数量
    # 考虑到数据量可能较大，这里采取一种优化的方法
    # 对于每个OD对，它在 (timestamp_O_actual, timestamp_D_actual] 期间处于载客状态
    
    # 扁平化数据，记录每个车辆在何时开始载客和何时结束载客
    events = []
    for index, row in df_all_week_od.iterrows():
        events.append({'time': row['timestamp_O_actual'], 'type': 'pickup', 'car_id': row['car_id']})
        # 排除行程时长为0的情况，避免时间戳一致导致错误
        if row['timestamp_D_actual'] > row['timestamp_O_actual']:
            events.append({'time': row['timestamp_D_actual'], 'type': 'dropoff', 'car_id': row['car_id']})
            
    df_events = pd.DataFrame(events).sort_values('time').reset_index(drop=True)
    
    # 使用 resample 进行计数 (需要时间序列作为索引)
    # 载客车辆数量 = 当前活动的车辆 - 刚刚结束载客的车辆
    # 这是一个简化版本，更精确的需要用事件计数
    
    # 构造一个稀疏矩阵或更复杂的数据结构来处理高时间分辨率的载客状态
    # 针对载客数量变化图的需求，最直接的方法是：
    # 对于每个1分钟的时间点t，统计所有满足 timestamp_O_actual <= t < timestamp_D_actual 的 OD 对数量
    
    # 这种方法虽然直观，但计算量大。考虑转换为事件流：
    # 每当有上客事件发生，载客数+1；每当下客事件发生，载客数-1。
    # 然后在1分钟间隔上累积这些变化。

    # 更高效的载客数量统计方法：
    # 创建一个包含所有上车和下车事件的列表
    all_times = pd.concat([df_all_week_od['timestamp_O_actual'], df_all_week_od['timestamp_D_actual']]).unique()
    all_times = pd.Series(all_times).sort_values().reset_index(drop=True)

    # 创建一个空的DataFrame，以分钟为索引
    df_carrier_count = pd.DataFrame(index=time_index)
    df_carrier_count['active_carriers'] = 0

    # 遍历每个OD对，将其行程时间段内的计数增加
    # 注意: 这个循环对于大数据集效率不高，但能保证正确性。
    # 对于数百万甚至千万级OD数据，建议考虑更高级别的优化，如用并行处理或数据库操作
    
    # 优化后的方法: 统计区间重叠
    # 首先，把每个OD对表示为一个区间 [start, end)
    # 对所有start和end时间进行排序，这些是事件点。
    # 扫描事件点，计算当前活动的区间数。
    
    active_cars_per_minute = []
    
    # 遍历每分钟的时间点
    for t_minute in time_index:
        # 计算在当前时间点 't_minute' (包括t_minute，不包括 t_minute + 1min) 
        # 处于载客状态的车辆数量
        # 载客车辆的条件是：timestamp_O_actual <= t_minute AND timestamp_D_actual > t_minute
        # 注意这里的 ">" 是为了确保载客状态持续到该分钟结束
        active_count = df_all_week_od[
            (df_all_week_od['timestamp_O_actual'] <= t_minute) & 
            (df_all_week_od['timestamp_D_actual'] > t_minute)
        ]['car_id'].nunique() # 统计不重复的载客车辆ID
        
        active_cars_per_minute.append({
            'TIME': t_minute.strftime('%Y-%m-%d %H:%M:%S'),
            'number': active_count
        })

    df_o_number = pd.DataFrame(active_cars_per_minute)
    output_o_number_file = os.path.join(FEATURE_OUTPUT_DIR, "JN_o_number.csv")
    df_o_number.to_csv(output_o_number_file, index=False)
    logger.info(f"载客出租车数量变化图数据已保存到: {output_o_number_file}")
    logger.info(f"\nJN_o_number.csv (前5行):\n{df_o_number.head().to_string()}")


    # --- 3. 运客路程分析图 (短途、中途、长途) ---
    logger.info("开始进行运客路程分析图统计...")
    
    # 阈值定义 (米)
    SHORT_TRIP_THRESHOLD = 4000  # 4公里
    MIDDLE_TRIP_THRESHOLD = 8000 # 8公里

    # 根据 trip_distance_meter 划分
    def classify_trip_distance(distance_meter):
        if distance_meter <= SHORT_TRIP_THRESHOLD:
            return 'near'
        elif distance_meter <= MIDDLE_TRIP_THRESHOLD:
            return 'middle'
        else:
            return 'far'

    df_all_week_od['trip_category'] = df_all_week_od['trip_distance_meter'].apply(classify_trip_distance)
    
    # 按天统计
    df_all_week_od['day'] = df_all_week_od['timestamp_O_actual'].dt.day # 提取日期中的天数
    
    # 分组统计各类行程数量
    trip_counts = df_all_week_od.groupby('day')['trip_category'].value_counts().unstack(fill_value=0)
    
    # 确保 'near', 'middle', 'far' 列存在，即使某天没有
    for col in ['near', 'middle', 'far']:
        if col not in trip_counts.columns:
            trip_counts[col] = 0
            
    # 重置索引并重命名列
    df_luc = trip_counts[['near', 'middle', 'far']].reset_index()
    output_luc_file = os.path.join(FEATURE_OUTPUT_DIR, "JN_LuC.csv")
    df_luc.to_csv(output_luc_file, index=False)
    logger.info(f"运客路程分析图数据已保存到: {output_luc_file}")
    logger.info(f"\nJN_LuC.csv (前5行):\n{df_luc.head().to_string()}")

    logger.info("所有OD数据特征提取完成。")


if __name__ == '__main__':
    # 为了让这个脚本独立运行，也需要初始化日志系统
    root_logger = logging.getLogger()
    if not root_logger.handlers: # 避免重复添加handler
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    run_od_feature_extraction()