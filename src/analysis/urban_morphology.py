import geopandas as gpd
import momepy
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.colors as mcolors
import numpy as np
import contextily as ctx
import networkx as nx
from momepy import meshedness
from shapely.geometry import box
from src.config import PROCESSED_DATA_DIR, MORPHOLOGY_DIR

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def run_urban_morphology_analysis():
    """运行城市形态分析"""
    # 输入和输出目录
    input_dir = PROCESSED_DATA_DIR
    output_dir = MORPHOLOGY_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("开始城市形态分析...")

    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"目录不存在: {input_dir}")
        return

    # 读取建筑数据（点数据）
    buildings_file = os.path.join(input_dir, "Buildings_aligned.gpkg")
    address_file = os.path.join(input_dir, "AddressCM_aligned.gpkg")
    buildings = gpd.read_file(buildings_file)
    addresses_building = gpd.read_file(address_file)
    print(f"读取主要建筑点数据: {len(buildings)} 个主要建筑点")
    print(f"读取建筑点地址数据: {len(addresses_building)} 个地址")

    # 读取街区数据
    blocks_file_1939 = os.path.join(input_dir, "GEOCODER_1939_blocks.gpkg")
    blocks_file_1946 = os.path.join(input_dir, "GEOCODER_1946_blocks.gpkg")
    if os.path.exists(blocks_file_1939):
        blocks_1939 = gpd.read_file(blocks_file_1939)
        print(f"读取生成1939的街区数据: {len(blocks_1939)} 个街区")
    else:
        # 如果没有生成的街区数据，则使用GEOCODER_1939数据
        blocks_file_1939 = os.path.join(input_dir, "GEOCODER_1939_aligned.gpkg")
        if not os.path.exists(blocks_file_1939):
            print(f"文件不存在: {blocks_file_1939}")
            return
        blocks_1939 = gpd.read_file(blocks_file_1939)
        print(f"读取1939街区数据: {len(blocks_1939)} 个街区")
    if os.path.exists(blocks_file_1946):
        blocks_1946 = gpd.read_file(blocks_file_1946)
        print(f"读取生成1946的街区数据: {len(blocks_1946)} 个街区")
    else:
        # 如果没有生成的街区数据，则使用GEOCODER_1946数据
        blocks_file_1946 = os.path.join(input_dir, "GEOCODER_1946_aligned.gpkg")
        if not os.path.exists(blocks_file_1946):
            print(f"文件不存在: {blocks_file_1946}")
            return
        blocks_1946 = gpd.read_file(blocks_file_1946)
        print(f"读取1946街区数据: {len(blocks_1946)} 个街区")

    # 读取道路网络数据（从GEOCODER读取）
    road_file_1939 = os.path.join(input_dir, "GEOCODER_1939_aligned.gpkg")
    if not os.path.exists(road_file_1939):
        print(f"文件不存在: {road_file_1939}")
        return
    roads_1939 = gpd.read_file(road_file_1939)
    print(f"读取1939道路网络数据: {len(roads_1939)} 条道路")

    road_file_1946 = os.path.join(input_dir, "GEOCODER_1946_aligned.gpkg")
    if not os.path.exists(road_file_1946):
        print(f"文件不存在: {road_file_1946}")
        return
    roads_1946 = gpd.read_file(road_file_1946)
    print(f"读取1946道路网络数据: {len(roads_1946)} 条道路")

    # 读取水系网络数据
    water_file = os.path.join(input_dir, "WaterNetwork_aligned.gpkg")
    if not os.path.exists(water_file):
        print(f"文件不存在: {water_file}")
        return
    water = gpd.read_file(water_file)
    print(f"读取水系网络数据: {len(water)} 条水系")

    # 分别对1939和1946年的数据进行分析
    analyze_year_data(blocks_1939, roads_1939, buildings, addresses_building, water, "1939", output_dir)
    analyze_year_data(blocks_1946, roads_1946, buildings, addresses_building, water, "1946", output_dir)

    print(f"\n城市形态分析完成，结果保存在 {output_dir} 目录")

def analyze_year_data(blocks, roads, buildings, add, water, year, output_dir):
    """对特定年份的数据进行城市形态分析"""
    print(f"\n开始 {year} 年的城市形态分析...")

    # 1. 街区分析
    print(f"\n开始 {year} 年街区分析...")

    # 计算街区面积和周长
    blocks['area'] = blocks.geometry.area
    blocks['perimeter'] = blocks.geometry.length

    # 计算街区形状指数
    blocks['fractal_dimension'] = momepy.FractalDimension(blocks)
    blocks['convexity'] = momepy.Convexity(blocks)

    # 计算街区紧凑度
    blocks['compactness'] = blocks['area'] / blocks.geometry.envelope.area

    # 2. 计算建筑点密度
    print(f"\n计算 {year} 年建筑点密度...")

    # 将建筑点分配到街区
    spatial_join = gpd.sjoin(buildings, blocks, how="inner", predicate="within")
    address_spatial_join =  gpd.sjoin(add, blocks, how="inner", predicate="within")
    blocks_with_points = spatial_join.groupby('index_right').size().reset_index(name='building_count')
    blocks_with_addresses = address_spatial_join.groupby('index_right').size().reset_index(name='address_count')

    # 合并回街区数据
    blocks = blocks.merge(blocks_with_points, left_index=True, right_on='index_right', how='left')
    blocks = blocks.merge(blocks_with_addresses, left_index=True, right_on='index_right', how='left')
    blocks['building_count'] = blocks['building_count'].fillna(0)
    blocks['address_count'] = blocks['address_count'].fillna(0)

    # 计算建筑点密度
    blocks['building_density'] = blocks['building_count'] / blocks['area']
    blocks['address_density'] = blocks['address_count'] / blocks['area']

    # 3. 道路网络分析
    print(f"\n开始 {year} 年道路网络分析...")

    # 创建网络
    try:
        # 转换为线网络
        network = momepy.gdf_to_nx(roads)

        # 计算网络指标
        network = momepy.node_degree(network)
        network = momepy.mean_node_dist(network)
        network = momepy.cds_length(network)
        mesh_rate = momepy.meshedness(network, radius=None)

        # 转换回GeoDataFrame进行可视化
        nodes, edges = momepy.nx_to_gdf(network)
        edges['length'] = edges.geometry.length

        print(f"{year} 年网络分析完成: {len(nodes)} 个节点, {len(edges)} 条边")
    except Exception as e:
        print(f"{year} 年道路网络分析错误: {str(e)}")
        nodes = None
        edges = None

    # 4. 可视化分析结果
    print(f"\n创建 {year} 年可视化图...")

    # 4.1 建筑点分布可视化
    fig, ax = plt.subplots(figsize=(12, 10))
    buildings.plot(color='red', markersize=0.5, ax=ax)
    blocks.boundary.plot(color='black', linewidth=0.5, ax=ax)
    plt.title(f"{year} 年建筑点分布")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{year}_building_points_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 10))
    add.plot(color='red', markersize=0.5, ax=ax)
    blocks.boundary.plot(color='black', linewidth=0.5, ax=ax)
    plt.title(f"{year} 年建筑点分布")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{year}_address_points_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4.2 街区建筑点密度可视化
    fig, ax = plt.subplots(figsize=(12, 10))
    blocks.plot(column='building_density', cmap='Reds', legend=True, ax=ax)
    buildings.plot(ax=ax, color='black', markersize=0.5, alpha=0.5)
    plt.title(f"{year} 年街区建筑点密度")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{year}_block_building_density.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 10))
    blocks.plot(column='address_density', cmap='Reds', legend=True, ax=ax)
    plt.title(f"{year} 年街区建筑点密度")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{year}_block_address_density.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4.3 街区形态指数可视化
    fig, ax = plt.subplots(figsize=(12, 10))
    blocks.plot(column='compactness', cmap='plasma', legend=True, ax=ax)
    plt.title(f"{year} 年街区紧凑度")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{year}_block_compactness.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4.4 道路网络可视化
    if nodes is not None and edges is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        edges.plot(ax=ax, color='grey', linewidth=0.5, zorder=1)
        nodes.plot(ax=ax, column='degree', cmap='viridis', markersize=3, legend=True , zorder=2)
        plt.title(f"{year} 年道路网络连通性分析")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{year}_road_network_connectivity.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 5. 综合形态特征分析
    print(f"\n生成 {year} 年综合形态特征报告...")

    # 5.1 建筑点统计信息
    building_stats = {
        "建筑点数量": len(buildings),
        "建筑点X坐标均值": buildings.geometry.x.mean(),
        "建筑点Y坐标均值": buildings.geometry.y.mean(),
        "建筑点X坐标标准差": buildings.geometry.x.std(),
        "建筑点Y坐标标准差": buildings.geometry.y.std()
    }

    # 5.2 街区统计信息
    block_stats = {
        "街区数量": len(blocks),
        "平均街区面积": blocks['area'].mean(),
        "最大街区面积": blocks['area'].max(),
        "最小街区面积": blocks['area'].min(),
        "平均紧凑度": blocks['compactness'].mean() if 'compactness' in blocks.columns else "未计算",
        "平均建筑点密度": blocks['building_density'].mean() if 'building_density' in blocks.columns else "未计算",
        "平均地址密度": blocks['address_density'].mean() if 'address_density' in blocks.columns else "未计算"
    }
    # 5.3 道路网络统计信息
    if nodes is not None and edges is not None:
        network_stats = {
            "节点数量": len(nodes),
            "边数量": len(edges),
            "平均节点度": nodes['degree'].mean(),
            "网格化指数": mesh_rate if mesh_rate else "未计算",
            "平均边长度": edges['length'].mean() if 'length' in edges.columns else "未计算"
        }
    else:
        network_stats = {"状态": "网络分析未完成"}

    # 打印统计信息
    print(f"\n{year} 年建筑点统计信息:")
    for key, value in building_stats.items():
        print(f"  {key}: {value}")

    print(f"\n{year} 年街区统计信息:")
    for key, value in block_stats.items():
        print(f"  {key}: {value}")

    print(f"\n{year} 年道路网络统计信息:")
    for key, value in network_stats.items():
        print(f"  {key}: {value}")

    # 保存分析结果
    blocks.to_file(os.path.join(output_dir, f"{year}_blocks_analyzed.gpkg"), driver="GPKG")
    if nodes is not None and edges is not None:
        nodes.to_file(os.path.join(output_dir, f"{year}_network_nodes.gpkg"), driver="GPKG")
        edges.to_file(os.path.join(output_dir, f"{year}_network_edges.gpkg"), driver="GPKG")

    # 返回分析结果，以便后续比较
    return blocks, nodes, edges if nodes is not None and edges is not None else (blocks, None, None)


def plot_block_metric(blocks, column, cmap, title, output_path, points=None, point_color='black'):
    """
    绘制街区指标分布图，颜色条与地图等高

    参数:
        blocks (GeoDataFrame): 街区数据
        column (str): 指标列名（如 'building_density'）
        cmap (str): 颜色映射，如 'Reds', 'plasma'
        title (str): 图标题
        output_path (str): 图保存路径
        points (GeoDataFrame, 可选): 叠加点数据（如建筑点、地址点）
        point_color (str): 点的颜色，默认黑色
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制街区
    blocks.plot(column=column, cmap=cmap, ax=ax)

    # 构造颜色映射器（ScalarMappable）
    norm = mcolors.Normalize(
        vmin=blocks[column].min(),
        vmax=blocks[column].max()
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # 避免警告

    # 添加颜色条，保证与图等高
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_aspect(30)  # 调整纵横比，使色条接近图高

    # 如果有点数据，则叠加
    if points is not None:
        points.plot(ax=ax, color=point_color, markersize=0.5, alpha=0.5)

    # 设置标题
    plt.title(title)
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_urban_morphology_analysis()