# data_cleaning.py
"""
职责：单日清洗 + OD 识别
整合改进：
1. 使用 CoordinatesConverter 库实现 WGS84 到 GCJ02 坐标转换，移除百度地图API依赖。
2. 配置参数从config.yaml统一管理。
3. 优化日志记录和性能。
4. 针对'left keys must be sorted'错误，修正merge_asof的匹配键和排序逻辑。
5. 增加数据去重和更全面的异常数据过滤（包括时间戳、重复项、地理围栏、速度）。
6. 优化速度计算逻辑，使用向量化的 Haversine 公式提高效率。
"""

from __future__ import annotations

import math
import os
import logging
import numpy as np
from typing import Optional, Tuple
import pandas as pd
import yaml

# 导入 CoordinatesConverter 库
import CoordinatesConverter

# 状态常量
LOADED_STATUS = 268435456  # 载客状态
EMPTY_STATUS = 0  # 空载状态

logger = logging.getLogger(__name__)
_config_cache: Optional[dict] = None


# ---------- 配置加载 ----------
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

# 初始化配置
cfg = get_config()

# ---------- 目录配置 ----------
CLEANED_DATA_DIR = cfg["CLEANED_DATA_DIR"]
VIEW_DATA_DIR = os.path.join(CLEANED_DATA_DIR, cfg["VIEW_DATA_SUBDIR"])
OD_DATA_DIR = os.path.join(CLEANED_DATA_DIR, cfg["OD_DATA_SUBDIR"])
os.makedirs(VIEW_DATA_DIR, exist_ok=True)
os.makedirs(OD_DATA_DIR, exist_ok=True)

# ---------- 过滤阈值 ----------
MIN_TRIP_DURATION = cfg["OD_CONFIG"]["min_trip_duration_sec"]
MIN_TRIP_DISTANCE = cfg["OD_CONFIG"]["min_trip_distance_meter"]
MAX_SPEED_KMH = cfg["DATA_FILTER"]["MAX_SPEED_KMH"]

# 地理围栏配置 (GCJ02坐标系下) - 已适配济南市坐标
GEO_FENCE_MIN_LON = cfg["DATA_FILTER"]["GEO_FENCE"]["MIN_LON"]
GEO_FENCE_MIN_LAT = cfg["DATA_FILTER"]["GEO_FENCE"]["MIN_LAT"]
GEO_FENCE_MAX_LON = cfg["DATA_FILTER"]["GEO_FENCE"]["MAX_LON"]
GEO_FENCE_MAX_LAT = cfg["DATA_FILTER"]["GEO_FENCE"]["MAX_LAT"]


# ---------- 距离和速度计算 ----------
def haversine(lon1, lat1, lon2, lat2) -> float:
    """Haversine公式计算球面距离(米)"""
    R = 6371000  # 地球半径(米)
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# 向量化 Haversine 函数
def haversine_vectorized(lons1, lats1, lons2, lats2) -> np.ndarray:
    """向量化 Haversine 公式计算球面距离(米)"""
    R = 6371000  # 地球半径(米)
    lats1_rad, lats2_rad = np.radians(lats1), np.radians(lats2)
    
    dlat_rad = np.radians(lats2 - lats1)
    dlon_rad = np.radians(lons2 - lons1)

    a = np.sin(dlat_rad / 2.0)**2 + np.cos(lats1_rad) * np.cos(lats2_rad) * np.sin(dlon_rad / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_speed(df: pd.DataFrame) -> pd.DataFrame:
    """计算相邻GPS点之间的速度(km/h)，使用 GCJ02 坐标（向量化优化）。"""
    df_copy = df.copy() # 操作副本
    
    # 确保数据按车辆ID和时间戳排序
    df_copy = df_copy.sort_values(['car_id', 'timestamp']).reset_index(drop=True)
    
    # 获取前一个点的 GCJ02 坐标和时间戳
    df_copy['prev_lat_gcj'] = df_copy.groupby('car_id')['lat_gcj'].shift(1)
    df_copy['prev_lon_gcj'] = df_copy.groupby('car_id')['lon_gcj'].shift(1)
    df_copy['prev_timestamp'] = df_copy.groupby('car_id')['timestamp'].shift(1)

    # 识别需要计算距离和速度的行 (非第一个点且前一个点有效)
    # 通过判断 'prev_lat_gcj' 是否为 NaN 来识别行
    valid_mask = df_copy['prev_lat_gcj'].notna()

    # 向量化计算距离
    if valid_mask.any(): # 只有当有需要计算的行时才执行
        df_copy.loc[valid_mask, 'distance_m'] = haversine_vectorized(
            df_copy.loc[valid_mask, 'prev_lon_gcj'],
            df_copy.loc[valid_mask, 'prev_lat_gcj'],
            df_copy.loc[valid_mask, 'lon_gcj'],
            df_copy.loc[valid_mask, 'lat_gcj']
        )
    else:
        df_copy['distance_m'] = 0.0 # 没有有效点则全为0
    
    df_copy['distance_m'].fillna(0, inplace=True) # 第一个点距离为0

    # 向量化计算时间差
    df_copy['time_diff_sec'] = (df_copy['timestamp'] - df_copy['prev_timestamp']).dt.total_seconds().fillna(0)
    
    # 向量化计算速度(km/h)，避免除零错误
    df_copy['speed_kmh'] = np.where(df_copy['time_diff_sec'] > 0,
                                   (df_copy['distance_m'] / df_copy['time_diff_sec']) * 3.6,
                                   0)
    
    # 清理辅助列
    df_copy.drop(columns=['prev_lat_gcj', 'prev_lon_gcj', 'prev_timestamp', 'distance_m', 'time_diff_sec'], inplace=True)
    
    return df_copy


# ---------- OD 识别 (保持不变，因为它已经在使用GCJ02坐标) ----------
def identify_od_pairs(df: pd.DataFrame, od_cfg: dict) -> Optional[pd.DataFrame]:
    """识别原始数据中的OD对"""
    # 参数解包
    id_col = od_cfg["id_column"]
    t_col = od_cfg["time_column"]
    lon_col = od_cfg["longitude_column"] # 现在将是 'lon_gcj'
    lat_col = od_cfg["latitude_column"]  # 现在将是 'lat_gcj'
    status_col = od_cfg["status_column"]
    min_trip_duration = od_cfg["min_trip_duration_sec"]
    min_trip_distance = od_cfg["min_trip_distance_meter"]

    # 1. 确保数据已按车辆ID和时间排序
    # df 应该已经在 clean_single_day_data 中排序过
    
    # 2. 识别上客点(O点)
    df_on = df[
        (df[status_col].shift(1) == EMPTY_STATUS) &
        (df[status_col] == LOADED_STATUS)
    ].copy()

    # 3. 识别下客点(D点)
    df_off = df[
        (df[status_col].shift(1) == LOADED_STATUS) &
        (df[status_col] == EMPTY_STATUS)
    ].copy()

    logger.info(f"识别到上车点: {len(df_on)} | 下车点: {len(df_off)}")

    # 4. 检查数据有效性
    if df_on.empty or df_off.empty:
        logger.warning("没有足够的上车点或下车点来匹配OD对")
        return None

    # 5. 准备合并数据
    df_on_processed = df_on.rename(columns={
        t_col: f"{t_col}_O_actual",
        lon_col: f"{lon_col}_O",
        lat_col: f"{lat_col}_O",
        status_col: f"{status_col}_O"
    })
    df_on_processed = df_on_processed.sort_values([id_col, f"{t_col}_O_actual"]).reset_index(drop=True)

    df_off_processed = df_off.rename(columns={
        t_col: f"{t_col}_D_actual",
        lon_col: f"{lon_col}_D",
        lat_col: f"{lat_col}_D",
        status_col: f"{status_col}_D"
    })
    df_off_processed = df_off_processed.sort_values([id_col, f"{t_col}_D_actual"]).reset_index(drop=True)

    # 6. 使用 group-wise merge_asof 进行时间向前匹配
    od_list = []
    try:
        unique_car_ids = pd.concat([df_on_processed[id_col], df_off_processed[id_col]]).unique()

        df_on_grouped = df_on_processed.groupby(id_col, group_keys=False)
        df_off_grouped = df_off_processed.groupby(id_col, group_keys=False)

        for car_id in unique_car_ids:
            on_group = df_on_grouped.get_group(car_id) if car_id in df_on_grouped.groups else pd.DataFrame()
            off_group = df_off_grouped.get_group(car_id) if car_id in df_off_grouped.groups else pd.DataFrame()

            if not on_group.empty and not off_group.empty:
                merged_group = pd.merge_asof(
                    on_group,
                    off_group,
                    left_on=f"{t_col}_O_actual",
                    right_on=f"{t_col}_D_actual",
                    by=id_col,
                    direction="forward",
                    suffixes=("_O", "_D")
                )
                od_list.append(merged_group)

        if od_list:
            od = pd.concat(od_list, ignore_index=True)
        else:
            od = pd.DataFrame(columns=[
                f"{t_col}_O_actual", f"{lon_col}_O", f"{lat_col}_O", f"{status_col}_O",
                f"{t_col}_D_actual", f"{lon_col}_D", f"{lat_col}_D", f"{status_col}_D",
                id_col
            ])

    except Exception as e:
        logger.error(f"OD匹配失败: {str(e)}")
        return None

    # 7. 移除未匹配的行
    od.dropna(subset=[f"{lon_col}_D", f"{lat_col}_D"], inplace=True)
    logger.info(f"初步匹配到OD对: {len(od)}")

    if od.empty:
        logger.warning("未匹配到任何有效的OD对")
        return None

    # 8. 计算行程时间和距离
    od["trip_duration_sec"] = (od[f"{t_col}_D_actual"] - od[f"{t_col}_O_actual"]).dt.total_seconds()
    od["trip_distance_meter"] = od.apply(
        lambda row: haversine(
            row[f"{lon_col}_O"], row[f"{lat_col}_O"],
            row[f"{lon_col}_D"], row[f"{lat_col}_D"]
        ),
        axis=1
    )

    # 9. 过滤无效数据
    initial_count = len(od)
    od = od[
        (od["trip_duration_sec"] > 0) &
        (od["trip_distance_meter"] > 0) &
        ((od[f"{lon_col}_O"] != od[f"{lon_col}_D"]) |
         (od[f"{lat_col}_O"] != od[f"{lat_col}_D"]))
    ].copy()
    logger.info(f"基础过滤后剩余OD对: {len(od)} (过滤掉 {initial_count - len(od)})")

    # 10. 应用业务规则过滤
    initial_count = len(od)
    od = od[
        (od["trip_duration_sec"] >= min_trip_duration) &
        (od["trip_distance_meter"] >= min_trip_distance)
    ]
    logger.info(f"最终有效OD对: {len(od)} (过滤掉 {initial_count - len(od)})")

    if od.empty:
        return None

    # 11. 去重处理 (针对OD对的去重)
    od = od.drop_duplicates(
        subset=[id_col, f"{t_col}_O_actual"],
        keep="first"
    ).reset_index(drop=True)

    return od

# ---------- 单日清洗主函数 ----------
def clean_single_day_data(
    df_raw: pd.DataFrame, day_prefix: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    单日数据处理流程:
    1. 列校验 → 2. 时间类型纠正/去重 → 3. 坐标转换 → 4. 异常数据过滤 → 5. OD识别
    """
    config = get_config() # 确保获取最新的配置
    if df_raw.empty:
        logger.warning(f"{day_prefix} 原始数据为空，跳过")
        return None, None

    logger.info(f"开始处理 {day_prefix} 数据，共 {len(df_raw)} 条记录")

    # 1. 时间列处理 (纠正数据类型错误)
    initial_count_ts = len(df_raw)
    logger.info(f"转换前 timestamp 列类型: {df_raw['timestamp'].dtype}")
    logger.info(f"转换前 timestamp 列示例 (前5行):\n{df_raw['timestamp'].head()}")
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai')
    df_raw.dropna(subset=["timestamp"], inplace=True)
    if len(df_raw) < initial_count_ts:
        logger.info(f"时间戳转换后丢弃 {initial_count_ts - len(df_raw)} 条记录，剩余 {len(df_raw)} 条")
    logger.info(f"转换后 timestamp 列类型: {df_raw['timestamp'].dtype}")
    logger.info(f"转换后 timestamp 列示例 (前5行):\n{df_raw['timestamp'].head()}")
    logger.info("时间戳已从UTC转换为北京时间。")

    # 2. 删除完全重复项
    initial_count_dup = len(df_raw)
    df_raw.drop_duplicates(subset=["car_id", "timestamp", "longitude", "latitude", "status"], inplace=True)
    if len(df_raw) < initial_count_dup:
        logger.info(f"删除重复GPS记录 {initial_count_dup - len(df_raw)} 条，剩余 {len(df_raw)} 条")

    # 3. 排序 (在坐标转换前进行，确保数据有序性，便于后续识别状态变化及速度计算)
    df_raw = df_raw.sort_values(["car_id", "timestamp"]).reset_index(drop=True)
    
    # 4. 坐标转换 (WGS84 -> GCJ02)
    logger.info("开始进行WGS84到GCJ02坐标转换 (使用 CoordinatesConverter)...")
    
    # 直接使用 CoordinatesConverter 进行向量化转换
    converted_lons, converted_lats = CoordinatesConverter.wgs84togcj02(
        df_raw["longitude"].values, df_raw["latitude"].values
    )
    
    df_view = df_raw.copy() # 创建副本，将转换后的坐标添加到此副本
    df_view["lon_gcj"] = converted_lons
    df_view["lat_gcj"] = converted_lats
    
    logger.info(f"完成坐标转换，共 {len(df_view)} 条记录。新增 'lon_gcj' 和 'lat_gcj' 列。")

    # 5. 异常数据过滤第一部分：地理围栏过滤 (在速度计算前执行) - 已解除注释
    initial_count_filter = len(df_view)
    df_view = df_view[
        (df_view["lon_gcj"] >= GEO_FENCE_MIN_LON) &
        (df_view["lon_gcj"] <= GEO_FENCE_MAX_LON) &
        (df_view["lat_gcj"] >= GEO_FENCE_MIN_LAT) &
        (df_view["lat_gcj"] <= GEO_FENCE_MAX_LAT)
    ].copy()
    logger.info(f"地理围栏过滤后，剩余 {len(df_view)} 条记录 (过滤掉 {initial_count_filter - len(df_view)})")

    # 6. 计算速度
    logger.info("开始计算速度...")
    df_view = calculate_speed(df_view)
    logger.info("速度计算完成。")
    
    # 诊断：打印速度统计信息
    logger.info(f"速度计算后的统计信息:\n{df_view['speed_kmh'].describe()}")
    
    # 7. 异常数据过滤第二部分：速度过滤
    initial_count_speed_filter = len(df_view)
    df_view = df_view[
        (df_view["speed_kmh"] <= MAX_SPEED_KMH)
    ].copy()

    if len(df_view) < initial_count_speed_filter:
        logger.info(f"速度过滤后，剩余 {len(df_view)} 条记录 (过滤掉 {initial_count_speed_filter - len(df_view)})")

    # 检查数据是否为空，如果为空则直接返回
    if df_view.empty:
        logger.warning(f"{day_prefix} 经过地理围栏和速度过滤后数据为空，跳过OD识别和保存。")
        return None, None
        
    # 8. OD识别
    od_cfg_for_od_identification = config["OD_CONFIG"].copy()
    df_od = identify_od_pairs(
        df_view, 
        od_cfg_for_od_identification
    )
    
    if df_view is not None and not df_view.empty:
        # 更改为 Parquet 格式
        view_output_path = os.path.join(VIEW_DATA_DIR, f"{day_prefix}_view.parquet")
        df_view.to_parquet(view_output_path, index=False) # 使用 to_parquet
        logger.info(f"清洗后的 view 数据已保存到: {view_output_path}")

    if df_od is not None and not df_od.empty:
        # 按照流程图，修改输出文件名为 _od_wh_drop.parquet
        od_output_path = os.path.join(OD_DATA_DIR, f"{day_prefix}_od_wh_drop.parquet")
        df_od.to_parquet(od_output_path, index=False) # 使用 to_parquet
        logger.info(f"识别的 OD 对数据已保存到: {od_output_path}")
    else:
        logger.info(f"{day_prefix} 未找到有效的OD对或OD数据为空，不保存OD文件。")

    return df_view, df_od