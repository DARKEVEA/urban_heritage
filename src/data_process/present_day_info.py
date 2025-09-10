import argparse
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from src.config import key0, key1, key2, gd_key0, gd_key1, tx_key0, tx_key1


# 坐标转换：WGS84 -> GCJ-02 -> BD-09
import math

PI = math.pi
A = 6378245.0
EE = 0.00669342162296594323

query = (
    "银行|领事馆|政府|医院|公安局|寺庙|"
    "住宅|公寓|酒店|"
    "大楼|广场|综合体|中心|"
)

queries_list = [
    "领事馆", "银行", "医院", "写字楼", "寺庙", "住宅", "商场", "酒店"
]

query_test = ["领事馆"]

CONFIG = {
    # 初始 JSON 文件路径
    "JSON_IN": r"E:\CODE\urban_heritage\data\processed\filtered_for_api.json",

    # 循环搜索时 JSON 文件路径
    "OUTPUT_MATCHED": r"E:\CODE\urban_heritage\data\processed\matched.json",
    "OUTPUT_UNMATCHED": r"E:\CODE\urban_heritage\data\processed\unmatched.json",
}


def _out_of_china(lng: float, lat: float) -> bool:
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)

def _transform_lat(lng: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 * math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret

def _transform_lng(lng: float, lat: float) -> float:
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 * math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * PI) + 40.0 * math.sin(lng / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * PI) + 300.0 * math.sin(lng / 30.0 * PI)) * 2.0 / 3.0
    return ret

def wgs84_to_gcj02(lng: float, lat: float) -> Tuple[float, float]:
    if _out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (A / sqrtmagic * math.cos(radlat) * PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat

def gcj02_to_bd09(lng: float, lat: float) -> Tuple[float, float]:
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * PI)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * PI)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lng, bd_lat

def wgs84_to_bd09(lng: float, lat: float) -> Tuple[float, float]:
    lng2, lat2 = wgs84_to_gcj02(lng, lat)
    return gcj02_to_bd09(lng2, lat2)

def load_buildings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("buildings", [])

def save_buildings(path: str, buildings: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"buildings": buildings}, f, ensure_ascii=False, indent=2)

def load_match_json(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("buildings", [])
    return pd.DataFrame(rows)

def filter_no_name(input_path: str, output_path_filtered: str, output_path_other: str):
    # 读取原 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取 buildings 列表
    buildings = data.get("buildings", [])

    # 过滤出 osm_properties 中 name 为空的记录
    filtered = []
    other = []
    for b in buildings:
        osm_props = b.get("osm_properties", {})
        name_value = osm_props.get("name")
        if not name_value or str(name_value).strip() == "":
            filtered.append(b)
        else:
            other.append(b)

    # 保存成新的 JSON 文件
    with open(output_path_filtered, "w", encoding="utf-8") as f:
        json.dump({"buildings": filtered}, f, ensure_ascii=False, indent=2)
    # 也可以保存其他记录
    with open(output_path_other, "w", encoding="utf-8") as f:
        json.dump({"buildings": other}, f, ensure_ascii=False, indent=2)

    print(f"筛选完成，原数据 {len(buildings)} 条，筛选后 {len(filtered)} 条需进一步处理，其他 {len(other)} 条已另外保存。")


def baidu_place_search_nearby(bd_lng: float, bd_lat: float, ak: str, query: str, radius: int = 50, page_size: int = 5) -> Dict[str, Any]:
    url = "https://api.map.baidu.com/place/v2/search"
    params = {
        "query": query,
        "location": f"{bd_lat},{bd_lng}",  # 注意顺序: lat,lng
        "radius": radius,
        "output": "json",
        "ak": ak,
        "page_size": page_size,
        "coord_type": "bd09ll",
        "scope": 2
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def gaode_place_search_nearby(gcj_lng: float, gcj_lat: float, key: str, keywords: str, radius: int = 50, offset: int = 5, page: int = 1) -> Dict[str, Any]:
    """
    高德地图附近搜索 API
    文档: https://restapi.amap.com/v3/place/around
    """
    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key": key,
        "location": f"{gcj_lng},{gcj_lat}",  # 注意顺序 lng,lat
        "keywords": keywords,
        "radius": radius,
        "offset": offset,
        "page": page,
        "extensions": "all",  # 获取更多信息
        "output": "JSON"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def tencent_place_search_nearby(lng: float, lat: float, key: str, keyword: str, radius: int = 50, page_size: int = 10, page_index: int = 1) -> Dict[str, Any]:
    """
    腾讯地图周边搜索 API
    文档：https://lbs.qq.com/service/webService/webServiceGuide/webServicePlace
    """
    url = "https://apis.map.qq.com/ws/place/v1/search"
    params = {
        "keyword": keyword,  # 搜索关键词
        "boundary": f"nearby({lat},{lng},{radius})",  # 注意顺序 lat,lng
        "key": key,
        "page_size": page_size,  # 每页数量
        "page_index": page_index,  # 页码，从1开始
        "orderby": "_distance"  # 按距离排序
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def enrich_status_with_batches(
    json_in: str,
    json_out: str,
    csv_out: str,
    platform_plan: list,
    query: str,
    radius: int,
    sleep_sec: float,
    progress_file: str
):
    # 读取进度
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_index = progress.get("last_index", 0)
    else:
        start_index = 0

    df = load_match_json(json_in)
    if df.empty:
        raise ValueError("输入 JSON 中未发现 buildings 列表")

    total = len(df)
    print(f"总记录数: {total}，从 {start_index} 开始处理")

    results = []
    for plan in platform_plan:
        s, e, platform, key = plan["start"], plan["end"], plan["platform"], plan["key"]

        if e >= total:
            e = total - 1

        if start_index > e:
            print(f"跳过区间 {s}-{e}（已处理）")
            continue

        # 从断点继续
        batch_start = max(s, start_index)

        for i in range(batch_start, e + 1):
            row = df.iloc[i]
            osm_id = row.get("osm_id")
            osm_props = row.get("osm_properties", {})
            lon = osm_props.get("centroid_x")
            lat = osm_props.get("centroid_y")

            print(f"[{platform}] 处理 {i + 1}/{total}  osm_id={osm_id}  坐标=({lon}, {lat})")

            if pd.isna(lon) or pd.isna(lat):
                results.append({**row.to_dict(), "status_sources": {platform: {"ok": False, "reason": "missing coordinates"}}})
                continue

            try:
                if platform == "baidu":
                    bd_lng, bd_lat = wgs84_to_bd09(float(lon), float(lat))
                    data = baidu_place_search_nearby(bd_lng, bd_lat, key, query=query, radius=radius)
                    ok = data.get("status") == 0
                    pois = data.get("results", []) if ok else []
                    top = pois[0] if pois else None
                    status_obj = {
                        "ok": ok,
                        "raw_status": data.get("status"),
                        "raw_message": data.get("message"),
                        "top_poi": top,
                        "poi_count": len(pois),
                        "search_center": {"lng": bd_lng, "lat": bd_lat},
                        "raw_data": data  # 🔹新增，完整保存 API 响应
                    }


                elif platform == "gaode":
                    gcj_lng, gcj_lat = wgs84_to_gcj02(float(lon), float(lat))
                    data = gaode_place_search_nearby(gcj_lng, gcj_lat, key, keywords=query, radius=radius)
                    ok = data.get("status") == "1"
                    pois = data.get("pois", []) if ok else []
                    top = pois[0] if pois else None
                    status_obj = {
                        "ok": ok,
                        "raw_status": data.get("status"),
                        "raw_message": data.get("info"),
                        "top_poi": top,
                        "poi_count": len(pois),
                        "search_center": {"lng": gcj_lng, "lat": gcj_lat},
                        "raw_data": data
                    }
                else:
                    raise ValueError(f"未知平台: {platform}")

            except Exception as e:
                status_obj = {"ok": False, "reason": str(e)}

            row_dict = row.to_dict()
            row_dict["status_sources"] = {platform: status_obj}
            results.append(row_dict)

            # 保存进度
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({"last_index": i}, f)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        # 输出区间 CSV
        flat = []
        for r in results:
            base = {k: v for k, v in r.items() if k not in ("matched_points", "osm_properties", "status_sources")}
            plat = plan["platform"]
            info = r.get("status_sources", {}).get(plat, {})

            # 把整个 info（含所有 POI）保存为 JSON 字符串
            info_json = json.dumps(info, ensure_ascii=False)

            flat.append({
                **base,
                f"{plat}_full_info": info_json  # 全量信息
            })

        pd.DataFrame(flat).to_csv(
            csv_out,
            index=False,
            encoding="utf-8-sig",
            mode='a',
            header=not os.path.exists(csv_out)
        )


    # 最终输出 JSON
    payload = {"buildings": results}
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 清空 results 以免下个区间重复
    results.clear()


def search_with_queries(input_json: str, queries: List[str], output_matched: str, output_unmatched: str,
                        api="baidu", ak_or_key="", start_index=0, end_index=None):
    # 读取所有数据
    all_data = load_buildings(input_json)
    match_done = load_buildings(output_matched) if os.path.exists(output_matched) else []

    if end_index is None:
        end_index = len(all_data)

    # 本次要处理的范围
    batch_data = all_data[start_index:end_index]

    matched_total = match_done
    unmatched = []  # 用来存储最终 unmatched

    for query in queries:
        print(f"\n🔍 当前关键词: {query}，待搜索数量: {len(batch_data)}")

        matched = []
        unmatched_batch = []

        for b in batch_data:
            lng = b["osm_properties"]["centroid_x"]
            lat = b["osm_properties"]["centroid_y"]

            if api == "baidu":
                result = baidu_place_search_nearby(lng, lat, ak_or_key, query)
                found = result.get("results")
            elif api == "gaode":
                result = gaode_place_search_nearby(lng, lat, ak_or_key, query)
                found = result.get("pois")
            elif api == "tencent":
                result = tencent_place_search_nearby(lng, lat, ak_or_key, query)
                found = result.get("data")
            else:
                raise ValueError(f"未知的 API 平台: {api}")

            if found:
                matched.append(b)
                matched.append(result)
            else:
                unmatched_batch.append(b)

        matched_total.extend(matched)

        # 这一轮 unmatched 直接替换 batch_data，用于下个 query
        batch_data = unmatched_batch

        print(f"✅ 本轮匹配到 {len(matched)} 条，剩余 {len(unmatched_batch)} 条")

        if not unmatched_batch:
            break

    # 最终 unmatched = 批次内未匹配 + 批次外数据
    unmatched = all_data[:start_index] + all_data[end_index:] + batch_data

    save_buildings(output_matched, matched_total)
    save_buildings(output_unmatched, unmatched)
    save_buildings(input_json, unmatched)

    print(f"\n🎯 总匹配: {len(matched_total)} 条，最终未匹配: {len(unmatched)} 条")



def main():
    search_with_queries(
        input_json=CONFIG["JSON_IN"],
        queries=queries_list,
        output_matched=CONFIG["OUTPUT_MATCHED"],
        output_unmatched=CONFIG["OUTPUT_UNMATCHED"],
        api="baidu",
        ak_or_key=key1,
        start_index=0,
        end_index=15
    )



if __name__ == "__main__":
    main()