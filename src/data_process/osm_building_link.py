import argparse
import os
from typing import Optional, List

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import json

try:
    from shapely import make_valid  # shapely>=2.0
    HAS_MAKE_VALID = True
except Exception:
    HAS_MAKE_VALID = False

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def safe_make_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if HAS_MAKE_VALID:
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    else:
        # classic fix for many invalid polygon issues
        gdf["geometry"] = gdf.buffer(0)
    return gdf


def load_historical_points(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    # 仅保留点要素
    gdf = gdf[gdf.geometry.type.isin(["Point"])]
    gdf = gdf.dropna(subset=["geometry"]).copy()
    if gdf.empty:
        raise ValueError("历史点数据为空或缺少点要素。")
    return gdf


# 兼容不同 osmnx 版本的多边形要素抓取
def osm_features_from_polygon(poly, tags):
    # 优先使用 v2 API
    if hasattr(ox, "features_from_polygon"):
        return ox.features_from_polygon(poly, tags=tags)
    # 回退到 v1 API
    if hasattr(ox, "geometries_from_polygon"):
        return ox.geometries_from_polygon(poly, tags=tags)
    # 进一步回退到子模块导入
    try:
        from osmnx import features as ox_features
        return ox_features.features_from_polygon(poly, tags=tags)
    except Exception:
        try:
            from osmnx import geometries as ox_geometries
            return ox_geometries.geometries_from_polygon(poly, tags=tags)
        except Exception as e:
            raise AttributeError("未找到合适的 OSMnx 抓取函数（features_from_polygon / geometries_from_polygon）。请升级 osmnx。") from e


def fetch_osm_buildings(
    place_name: str = "Shanghai, China",
    overpass_endpoint: Optional[str] = None,
    building_tags: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    # OSMnx 基本设置
    ox.settings.use_cache = True
    ox.settings.timeout = 180
    if overpass_endpoint:
        ox.settings.overpass_endpoint = overpass_endpoint

    # 获取行政边界
    boundary = ox.geocode_to_gdf(place_name)
    if boundary.empty:
        raise ValueError(f"未能获取行政边界: {place_name}")

    # 统一到 WGS84，提取单一 polygon
    boundary = boundary.to_crs(epsg=4326)
    geom = boundary.unary_union
    if geom.geom_type in ["MultiPolygon", "GeometryCollection"]:
        # 用 union 统一成一个面
        geom = unary_union([g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))])

    # 抓取建筑要素
    tags = building_tags or {"building": True}
    bld = osm_features_from_polygon(geom, tags=tags)

    # 仅保留面几何
    bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    bld = bld.dropna(subset=["geometry"])
    if bld.empty:
        raise ValueError("在指定区域内未抓取到建筑面要素。")

    # 净化字段
    bld = bld.reset_index()
    # OSMnx 通常包含 'osmid'，若无则尝试使用 'id' 或索引
    if "osmid" in bld.columns:
        bld["osm_id"] = bld["osmid"]
    elif "id" in bld.columns:
        bld["osm_id"] = bld["id"]
    else:
        bld["osm_id"] = bld.index

    # 保留全部OSM属性列，确保后续JSON导出尽可能完整
    # 修复无效几何
    bld = safe_make_valid(bld)
    # 删除空/无效几何
    bld = bld[~bld.geometry.is_empty & bld.geometry.is_valid].copy()

    return bld


# 新增：基于历史点数据 bounds（可选米级缓冲）构造抓取 polygon 并拉取 OSM 建筑
def _polygon_from_points_bounds(points_gdf: gpd.GeoDataFrame, buffer_m: float = 0.0) -> Polygon:
    if points_gdf.crs is None:
        raise ValueError("历史点数据缺少 CRS，无法根据 bounds 确定抓取范围。")

    # 若不是投影坐标，则投影到米制坐标以便使用米级缓冲
    if not points_gdf.crs.is_projected:
        pts_metric = ox.projection.project_gdf(points_gdf)
    else:
        pts_metric = points_gdf

    minx, miny, maxx, maxy = pts_metric.total_bounds
    if buffer_m and buffer_m > 0:
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m

    rect_metric = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    rect_4326 = gpd.GeoSeries([rect_metric], crs=pts_metric.crs).to_crs(epsg=4326).iloc[0]
    return rect_4326


def fetch_osm_buildings_by_bounds(
    points_gdf: gpd.GeoDataFrame,
    overpass_endpoint: Optional[str] = None,
    building_tags: Optional[dict] = None,
    bounds_buffer_m: float = 200.0,
) -> gpd.GeoDataFrame:
    # OSMnx 基本设置
    ox.settings.use_cache = True
    ox.settings.timeout = 180
    if overpass_endpoint:
        ox.settings.overpass_endpoint = overpass_endpoint

    poly_4326 = _polygon_from_points_bounds(points_gdf, buffer_m=bounds_buffer_m)

    tags = building_tags or {"building": True}
    bld = osm_features_from_polygon(poly_4326, tags=tags)

    # 仅保留面几何
    bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    bld = bld.dropna(subset=["geometry"])
    if bld.empty:
        raise ValueError("在 points bounds 指定范围内未抓取到建筑面要素。")

    # 净化字段
    bld = bld.reset_index()
    if "osmid" in bld.columns:
        bld["osm_id"] = bld["osmid"]
    elif "id" in bld.columns:
        bld["osm_id"] = bld["id"]
    else:
        bld["osm_id"] = bld.index

    # 保留全部OSM属性列，确保后续JSON导出尽可能完整
    bld = safe_make_valid(bld)
    bld = bld[~bld.geometry.is_empty & bld.geometry.is_valid].copy()

    return bld


def project_pair_to_metric(points: gpd.GeoDataFrame, polys: gpd.GeoDataFrame):
    # 使用 OSMnx 的投影助手以保证两者投影一致
    pts_proj = ox.projection.project_gdf(points)
    polys_proj = ox.projection.project_gdf(polys)
    # 若投影不同，再强制统一为多边形投影
    if pts_proj.crs != polys_proj.crs:
        pts_proj = pts_proj.to_crs(polys_proj.crs)
    return pts_proj, polys_proj


def match_points_to_buildings(
    points_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    buffer_meters: float = 5.0,
    keep_building_attrs: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    keep_building_attrs = keep_building_attrs or ["osm_id", "building", "name", "addr:housenumber", "addr:street", "building:levels"]

    pts_proj, bld_proj = project_pair_to_metric(points_gdf, buildings_gdf)

    # 第一阶段：严格包含 within
    left_cols = [c for c in pts_proj.columns if c != "geometry"]
    right_cols = [c for c in keep_building_attrs if c in bld_proj.columns] + ["geometry"]
    bld_sub = bld_proj[right_cols].copy()

    within_join = gpd.sjoin(pts_proj, bld_sub, how="left", predicate="within")
    # 检查 osm_id 列是否存在，如果不存在则添加空值列
    if "osm_id" not in within_join.columns:
        within_join["osm_id"] = None
    within_join["match_method"] = within_join["osm_id"].apply(lambda x: "within" if pd.notna(x) else None)

    # 找出未匹配的点
    unmatched = within_join[within_join["match_method"].isna()].copy()
    matched = within_join[within_join["match_method"].notna()].copy()

    # 第二阶段：最近邻，限制最大距离
    if not unmatched.empty and buffer_meters and buffer_meters > 0:
        nn = gpd.sjoin_nearest(
            unmatched.drop(columns=[c for c in unmatched.columns if c.endswith("_left") or c.endswith("_right")], errors="ignore"),
            bld_sub,
            how="left",
            max_distance=buffer_meters,
            distance_col="dist_m",
        )
        # 检查 osm_id 列是否存在，如果不存在则添加空值列
        if "osm_id" not in nn.columns:
            nn["osm_id"] = None
        nn["match_method"] = nn["osm_id"].apply(lambda x: f"nearest<={buffer_meters}m" if pd.notna(x) else "unmatched")
        # 合并 matched + nn
        common_cols = list(set(matched.columns).intersection(set(nn.columns)))
        result = pd.concat([matched[common_cols], nn[common_cols]], ignore_index=True)
    else:
        result = within_join.copy()
        result.loc[result["match_method"].isna(), "match_method"] = "unmatched"

    # 输出前恢复原始 CRS
    result = result.set_geometry("geometry")
    result = result.set_crs(pts_proj.crs)
    result = result.to_crs(points_gdf.crs)

    # 调整列顺序：原点字段 + 匹配信息 + 建筑字段
    point_cols_order = [c for c in points_gdf.columns if c != "geometry"]
    bld_cols_order = [c for c in keep_building_attrs if c in result.columns]
    final_cols = point_cols_order + ["match_method"] + bld_cols_order + ["geometry"]
    final_cols = [c for c in final_cols if c in result.columns]
    result = result[final_cols].copy()

    return result


def save_gpkg(gdf: gpd.GeoDataFrame, path: str, layer: str = "data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gdf.to_file(path, layer=layer, driver="GPKG")


def _compute_focus_bounds_web(matches_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame, buffer_m: float = 100.0):
    """基于匹配结果计算可视化聚焦范围（Web Mercator坐标系的bounds）。
    - 优先使用 match_method != 'unmatched' 的点的范围
    - 若无匹配，则退化为全部点范围；再无则用建筑范围
    - 最终返回 (minx, miny, maxx, maxy) in EPSG:3857，并按 buffer_m 扩展
    """
    try:
        matches_web = matches_gdf.to_crs(epsg=3857)
    except Exception:
        matches_web = matches_gdf
    try:
        buildings_web = buildings_gdf.to_crs(epsg=3857)
    except Exception:
        buildings_web = buildings_gdf

    focus_bounds = None
    if "match_method" in matches_web.columns:
        matched_pts = matches_web[matches_web["match_method"] != "unmatched"]
        if not matched_pts.empty:
            focus_bounds = matched_pts.total_bounds
    if focus_bounds is None or any(pd.isna(focus_bounds)):
        if not matches_web.empty:
            focus_bounds = matches_web.total_bounds
        else:
            focus_bounds = buildings_web.total_bounds

    minx, miny, maxx, maxy = focus_bounds
    if buffer_m and buffer_m > 0:
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m
    return (minx, miny, maxx, maxy)


def visualize_osm_buildings(buildings_gdf: gpd.GeoDataFrame, output_path: str, focus_bounds_web: Optional[tuple] = None):
    """可视化OSM建筑轮廓分布；若提供 focus_bounds_web，则按其范围裁切显示（仍绘制所有建筑）。"""
    fig, ax = plt.subplots(figsize=(15, 12))
    buildings_web = buildings_gdf.to_crs(epsg=3857)

    # 绘制全部建筑
    buildings_web.plot(ax=ax, facecolor='lightblue', edgecolor='darkblue', 
                      alpha=0.7, linewidth=0.5)

    # 设定聚焦范围
    if focus_bounds_web is not None:
        minx, miny, maxx, maxy = focus_bounds_web
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_title(f'OSM建筑轮廓分布\n总计: {len(buildings_gdf)} 个建筑', fontsize=16, pad=20)
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OSM建筑轮廓可视化已保存到: {output_path}")


def visualize_point_building_matches(matches_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame, output_path: str, focus_bounds_web: Optional[tuple] = None):
    """可视化历史点与建筑的匹配结果：高亮当前建筑，并按匹配类型着色。
    - 建筑高亮：within=绿色，nearest=橙色；其余为灰色背景
    - 点可选（当前按你的设定已关闭）
    - 若提供 focus_bounds_web，则仅调整显示范围，不删除任何未匹配建筑
    """
    fig, ax = plt.subplots(figsize=(15, 12))

    matches_web = matches_gdf.to_crs(epsg=3857)
    buildings_web = buildings_gdf.to_crs(epsg=3857)

    # 计算每个建筑的匹配类别
    def normalize_method(m: str) -> str:
        if pd.isna(m):
            return None
        if m == "within":
            return "within"
        if "nearest" in str(m):
            return "nearest"
        return "other"

    matched_rows = matches_gdf.copy()
    if "osm_id" not in matched_rows.columns:
        matched_rows["osm_id"] = None
    matched_rows["method_norm"] = matched_rows["match_method"].apply(normalize_method)
    matched_rows = matched_rows[pd.notna(matched_rows["osm_id"]) & pd.notna(matched_rows["method_norm"])].copy()

    priority_map = {"within": 2, "nearest": 1, "other": 0}
    if not matched_rows.empty and "osm_id" in buildings_gdf.columns:
        agg = (
            matched_rows.groupby("osm_id")["method_norm"]
            .agg(lambda s: sorted(s, key=lambda x: priority_map.get(x, 0), reverse=True)[0])
            .reset_index()
            .rename(columns={"method_norm": "building_match"})
        )
        buildings_anno = buildings_gdf.merge(agg, on="osm_id", how="left")
    else:
        buildings_anno = buildings_gdf.copy()
        buildings_anno["building_match"] = None

    buildings_web_bg = buildings_anno.to_crs(epsg=3857)
    buildings_web_bg.plot(ax=ax, facecolor="#d9d9d9", edgecolor="#bdbdbd", linewidth=0.3, alpha=0.4)

    within_bld = buildings_web_bg[buildings_web_bg["building_match"] == "within"]
    if not within_bld.empty:
        within_bld.plot(ax=ax, facecolor="#a6dba0", edgecolor="#1b9e77", linewidth=0.8, alpha=0.75, label="建筑: 严格包含")

    nearest_bld = buildings_web_bg[buildings_web_bg["building_match"] == "nearest"]
    if not nearest_bld.empty:
        nearest_bld.plot(ax=ax, facecolor="#fdb863", edgecolor="#e66101", linewidth=0.8, alpha=0.75, label="建筑: 最近邻")

    # 设定聚焦范围
    if focus_bounds_web is not None:
        minx, miny, maxx, maxy = focus_bounds_web
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # 绘制未匹配的点
    matches_other = matches_web[matches_web["match_method"].astype(str).str.contains("other", na=False)]
    if not matches_other.empty:
        matches_other.plot(
            ax=ax, color="black", marker="x", markersize=2, alpha=0.9, label="other"
        )

    # 统计信息
    total_points = len(matches_gdf)
    matched_within = (matches_gdf["match_method"] == "within").sum() if "match_method" in matches_gdf.columns else 0
    matched_nearest = matches_gdf["match_method"].astype(str).str.contains("nearest", na=False).sum() if "match_method" in matches_gdf.columns else 0
    unmatched = (matches_gdf["match_method"] == "unmatched").sum() if "match_method" in matches_gdf.columns else 0

    ax.set_title(
        f"历史建筑点与OSM建筑匹配（建筑高亮）\n"
        f"总点数: {total_points}, 严格包含: {matched_within}, 最近邻: {matched_nearest}, 未匹配: {unmatched}",
        fontsize=16,
        pad=18,
    )

    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"点-建筑匹配（建筑高亮）可视化已保存到: {output_path}")


def _clean_for_json(obj):
    """递归清理对象使可序列化，同时去除所有值为None的键或元素，并修复常见字符串乱码"""

    # 先把 pandas 的 NaN 识别为 None
    if pd.isna(obj):
        return None

    # 处理时间类型转字符串
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)

    # 基础类型：int, float, bool, None
    elif isinstance(obj, (int, float, bool, type(None))):
        return obj

    # 字符串类型：尝试修复 UTF-8 被误解码的情况
    elif isinstance(obj, str):
        try:
            # encode成latin1再decode utf-8
            return obj.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return obj  # 修复失败就保持原始字符串

    # 处理字典
    elif isinstance(obj, dict):
        cleaned_dict = {}
        for k, v in obj.items():
            cleaned_v = _clean_for_json(v)
            if cleaned_v is not None:
                cleaned_dict[k] = cleaned_v
        return cleaned_dict

    # 处理列表或元组
    elif isinstance(obj, (list, tuple)):
        cleaned_list = [_clean_for_json(item) for item in obj]
        return [item for item in cleaned_list if item is not None]

    # 处理 numpy 标量
    elif hasattr(obj, 'item'):
        return obj.item()

    # 其他类型全部转字符串
    else:
        return str(obj)


def export_match_details_to_json(matches_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame, json_path: str):
    """导出匹配建筑的详细信息到JSON。
    - 合并点与建筑的属性，尽量保留OSM的全部列
    - 对于同一建筑被多个点匹配的情况，聚合匹配点列表
    结构：{
        "buildings": [
            {
                "osm_id": <id>,
                "match_types": ["within", "nearest", ...],
                "num_matched_points": <int>,
                "matched_points": [ { 原点字段..., "match_method": str, "dist_m": float? }, ...],
                "osm_properties": { 全量OSM属性（不含geometry） }
            }, ...
        ]
    }
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # 仅保留有建筑ID的匹配（within 或 nearest 匹配），并转为普通DataFrame以便处理
    matches_df = matches_gdf.copy()
    if "osm_id" not in matches_df.columns:
        matches_df["osm_id"] = None
    matches_df = matches_df[pd.notna(matches_df["osm_id"])].copy()

    bld_df = buildings_gdf.copy()
    if "geometry" in bld_df.columns:
        bld_df["centroid_x"] = bld_df.geometry.centroid.x
        bld_df["centroid_y"] = bld_df.geometry.centroid.y

    # 准备建筑属性表（去掉几何）
    bld_df = bld_df.drop(columns=["geometry"], errors="ignore").copy()
    # 确保 osm_id 存在
    if "osm_id" not in bld_df.columns:
        # 若不存在就尝试使用索引
        bld_df = bld_df.reset_index().rename(columns={"index": "osm_id"})

    # 将点的属性（去geometry）准备好
    point_attr_cols = [c for c in matches_df.columns if c not in ("geometry",)]
    matches_df_flat = matches_df[point_attr_cols].copy()

    # 聚合每个建筑对应的匹配点信息与匹配类型集合
    building_groups = []
    for osm_id, group in matches_df_flat.groupby("osm_id"):
        match_types = sorted(group["match_method"].dropna().astype(str).unique().tolist())
        # 清理匹配点数据，确保JSON可序列化
        matched_points_raw = group.drop(columns=[col for col in group.columns if col in bld_df.columns], errors="ignore").to_dict(orient="records")
        matched_points = [_clean_for_json(point) for point in matched_points_raw]
        
        building_groups.append({
            "osm_id": _clean_for_json(osm_id),
            "match_types": match_types,
            "num_matched_points": len(group),
            "matched_points": matched_points,
        })

    # 将OSM属性合并进去
    groups_df = pd.DataFrame(building_groups)
    if not groups_df.empty:
        details_df = groups_df.merge(bld_df, on="osm_id", how="left", suffixes=("", ""))
    else:
        details_df = pd.DataFrame(columns=["osm_id"])  # 空结果

    # 转为最终JSON结构
    buildings_json = []
    osm_cols = [c for c in details_df.columns if c not in ("osm_id", "match_types", "num_matched_points", "matched_points")]
    for _, row in details_df.iterrows():
        # 清理OSM属性
        osm_props_raw = {k: row.get(k) for k in osm_cols}
        osm_props = _clean_for_json(osm_props_raw)
        
        buildings_json.append({
            "osm_id": _clean_for_json(row.get("osm_id")),
            "match_types": row.get("match_types", []),
            "num_matched_points": int(row.get("num_matched_points", 0)) if pd.notna(row.get("num_matched_points")) else 0,
            "matched_points": row.get("matched_points", []),
            "osm_properties": osm_props,
        })

    payload = {"buildings": buildings_json}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"匹配建筑详细信息已导出到: {json_path}")


def run(
    hist_points_path: str,
    out_buildings_path: str,
    out_matches_path: str,
    place_name: str = "Shanghai, China",
    overpass_endpoint: Optional[str] = None,
    buffer_m: float = 5.0,
    fetch_mode: str = "bounds",
    bounds_buffer_m: float = 200.0,
    create_visualizations: bool = True,
    out_json_path: Optional[str] = None,
):
    hist_pts = load_historical_points(hist_points_path)

    if fetch_mode == "bounds":
        buildings = fetch_osm_buildings_by_bounds(
            hist_pts,
            overpass_endpoint=overpass_endpoint,
            bounds_buffer_m=bounds_buffer_m,
        )
    else:
        buildings = fetch_osm_buildings(place_name=place_name, overpass_endpoint=overpass_endpoint)

    save_gpkg(buildings.to_crs(epsg=4326), out_buildings_path, layer="osm_buildings")

    matches = match_points_to_buildings(hist_pts, buildings, buffer_meters=buffer_m)
    save_gpkg(matches, out_matches_path, layer="point_building_matches")

    if create_visualizations:
        vis_dir = os.path.dirname(out_buildings_path)
        focus_bounds_web = _compute_focus_bounds_web(matches, buildings, buffer_m=100.0)

        buildings_vis_path = os.path.join(vis_dir, "osm_buildings_visualization.png")
        visualize_osm_buildings(buildings, buildings_vis_path, focus_bounds_web=focus_bounds_web)

        matches_vis_path = os.path.join(vis_dir, "major_building_matches_visualization.png")
        visualize_point_building_matches(matches, buildings, matches_vis_path, focus_bounds_web=focus_bounds_web)

    if out_json_path:
        export_match_details_to_json(matches, buildings, out_json_path)


def main():
    parser = argparse.ArgumentParser(description="关联历史建筑点与当前 OSM 建筑轮廓（上海）")
    parser.add_argument("--hist-points", type=str, default=r"E:\\CODE\\urban_heritage\\data\\processed\\Buildings_aligned.gpkg", help="历史点数据 GPKG 路径（当前坐标系例如 EPSG:32651）")
    parser.add_argument("--place", type=str, default="Shanghai, China", help="行政边界名称（当 fetch-mode=place 时使用）")
    parser.add_argument("--out-buildings", type=str, default=r"E:\\CODE\\urban_heritage\\data\\processed\\osm_buildings_shanghai.gpkg", help="输出：OSM 建筑轮廓 GPKG")
    parser.add_argument("--out-matches", type=str, default=r"E:\\CODE\\urban_heritage\\data\\processed\\historic_major_bulidings_to_osm_buildings.gpkg", help="输出：点-建筑匹配结果 GPKG")
    parser.add_argument("--buffer-m", type=float, default=5.0, help="最近邻最大匹配距离（米）")
    parser.add_argument("--overpass-endpoint", type=str, default=None, help="可选：自定义 Overpass 端点")
    parser.add_argument("--fetch-mode", type=str, choices=["bounds", "place"], default="bounds", help="建筑抓取方式：bounds=按历史点 bounds（默认），place=按行政区名")
    parser.add_argument("--bounds-buffer-m", type=float, default=200.0, help="在历史点 bounds 上的米级缓冲，用于扩大抓取范围")
    parser.add_argument("--no-visualization", action="store_true", help="跳过生成可视化图表")
    parser.add_argument("--out-json", type=str, default=r"E:\\CODE\\urban_heritage\\data\\processed\\major_building_match_details.json", help="导出匹配建筑详细信息的JSON路径")
    args = parser.parse_args()

    run(
        hist_points_path=args.hist_points,
        out_buildings_path=args.out_buildings,
        out_matches_path=args.out_matches,
        place_name=args.place,
        overpass_endpoint=args.overpass_endpoint,
        buffer_m=args.buffer_m,
        fetch_mode=args.fetch_mode,
        bounds_buffer_m=args.bounds_buffer_m,
        create_visualizations=not args.no_visualization,
        out_json_path=args.out_json,
    )


if __name__ == "__main__":
    main()