import geopandas as gpd
import momepy
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import contextily as ctx
import networkx as nx
from shapely.geometry import LineString, Point
from src.config import PROCESSED_DATA_DIR, SPACE_SYNTAX_DIR

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def run_space_syntax_analysis():
    """运行空间句法分析"""
    # 输入和输出目录
    input_dir = PROCESSED_DATA_DIR
    output_dir = SPACE_SYNTAX_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始空间句法分析...")
    
    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"目录不存在: {input_dir}")
        return
    
    # 读取道路网络数据 - 使用GEOCODER数据作为公路路网
    road_file = os.path.join(input_dir, "GEOCODER_1946_aligned.gpkg")
    if not os.path.exists(road_file):
        print(f"文件不存在: {road_file}")
        return
    roads = gpd.read_file(road_file)
    print(f"读取公路路网数据: {len(roads)} 条道路")
    
    # 读取建筑数据作为参考
    buildings_file = os.path.join(input_dir, "Buildings_aligned.gpkg")
    if not os.path.exists(buildings_file):
        print(f"文件不存在: {buildings_file}")
        return
    buildings = gpd.read_file(buildings_file)
    print(f"读取建筑点数据: {len(buildings)} 个建筑点")
    
    # 1. 准备道路网络
    print("\n准备道路网络...")
    
    # 确保道路网络是连通的
    try:
        # 转换为网络图
        G = momepy.gdf_to_nx(roads)
        print(f"创建网络: {len(G.nodes)} 个节点, {len(G.edges)} 条边")
        
        # 获取最大连通子图
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"最大连通子图: {len(G.nodes)} 个节点, {len(G.edges)} 条边")
        
        # 转回GeoDataFrame
        nodes, edges = momepy.nx_to_gdf(G)
        
    except Exception as e:
        print(f"网络处理错误: {str(e)}")
        # 如果出错，尝试直接使用原始数据
        try:
            G = momepy.gdf_to_nx(roads)
            nodes, edges = momepy.nx_to_gdf(G)
        except:
            print("无法创建网络，退出分析")
            return
    
    # 2. 计算基本网络指标
    print("\n计算基本网络指标...")
    
    # 计算节点度
    G = momepy.node_degree(G)
    
    # 计算边长度
    for u, v, data in G.edges(data=True):
        if 'length' not in data:
            try:
                # 尝试直接从边数据中获取几何信息
                if 'geometry' in data and hasattr(data['geometry'], 'length'):
                    data['length'] = data['geometry'].length
                else:
                    # 从节点坐标计算距离
                    if 'x' in G.nodes[u] and 'y' in G.nodes[u] and 'x' in G.nodes[v] and 'y' in G.nodes[v]:
                        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
                        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
                        data['length'] = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                    else:
                        # 如果没有坐标信息，设置默认长度
                        data['length'] = 1.0
                        print(f"警告: 节点 {u}-{v} 没有坐标信息，设置默认长度为1.0")
            except Exception as e:
                print(f"计算边 {u}-{v} 长度时出错: {str(e)}")
                data['length'] = 1.0
    
    # 计算网络密度
    G.graph['meshedness'] = nx.density(G)
    
    # 计算节点之间的平均距离
    G = momepy.mean_node_dist(G)
    
    # 3. 空间句法分析
    print("\n进行空间句法分析...")
    
    # 3.1 计算连通性（Connectivity）
    # 连通性就是节点度
    for node, data in G.nodes(data=True):
        data['connectivity'] = data['degree']
    
    # 3.2 计算控制值（Control）
    # 控制值计算：节点的度数除以其所有邻居的度数之和
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbors_degree_sum = sum(G.nodes[neighbor]['degree'] for neighbor in neighbors)
            if neighbors_degree_sum > 0:
                G.nodes[node]['control'] = G.nodes[node]['degree'] / neighbors_degree_sum
            else:
                G.nodes[node]['control'] = 0
        else:
            G.nodes[node]['control'] = 0
    
    # 3.3 计算局部整合度（Local Integration）
    try:
        # 使用closeness_centrality计算整合度
        local_closeness = nx.closeness_centrality(G, distance='length')
        nx.set_node_attributes(G, local_closeness, 'local_integration')
        print("计算局部整合度成功")
    except Exception as e:
        print(f"计算局部整合度错误: {str(e)}")
    
    # 3.4 计算全局整合度（Global Integration）
    try:
        # 全局整合度也使用closeness_centrality
        global_closeness = nx.closeness_centrality(G, distance='length')
        nx.set_node_attributes(G, global_closeness, 'global_integration')
        print("计算全局整合度成功")
    except Exception as e:
        print(f"计算全局整合度错误: {str(e)}")
    
    # 3.5 计算选择值（Choice）
    try:
        # 选择值使用betweenness_centrality
        betweenness = nx.betweenness_centrality(G, weight='length')
        nx.set_node_attributes(G, betweenness, 'choice')
        print("计算选择值成功")
    except Exception as e:
        print(f"计算选择值错误: {str(e)}")
    
    # 3.6 计算可达性（Reach）
    try:
        # 可达性：计算500米内可达的节点数量
        reach_500 = {}
        for node in G.nodes():
            reach_nodes = nx.single_source_dijkstra_path_length(G, node, cutoff=500, weight='length')
            reach_500[node] = len(reach_nodes) - 1  # 减去自身
        nx.set_node_attributes(G, reach_500, 'reach_500')
        print("计算可达性成功")
    except Exception as e:
        print(f"计算可达性错误: {str(e)}")
    
    # 3.7 计算直线度（Straightness）
    try:
        # 直线度：欧氏距离与网络距离的比值
        straightness = {}
        for source in G.nodes():
            if 'x' in G.nodes[source] and 'y' in G.nodes[source]:
                source_x, source_y = G.nodes[source]['x'], G.nodes[source]['y']
                path_lengths = nx.single_source_dijkstra_path_length(G, source, weight='length')
                straight_sum = 0
                count = 0
                for target, path_length in path_lengths.items():
                    if source != target and 'x' in G.nodes[target] and 'y' in G.nodes[target]:
                        target_x, target_y = G.nodes[target]['x'], G.nodes[target]['y']
                        euclidean_dist = ((source_x - target_x)**2 + (source_y - target_y)**2)**0.5
                        if path_length > 0:
                            straight_sum += euclidean_dist / path_length
                            count += 1
                if count > 0:
                    straightness[source] = straight_sum / count
                else:
                    straightness[source] = 0
            else:
                straightness[source] = 0
        nx.set_node_attributes(G, straightness, 'straightness')
        print("计算直线度成功")
    except Exception as e:
        print(f"计算直线度错误: {str(e)}")
    
    # 转换回GeoDataFrame进行可视化
    nodes, edges = momepy.nx_to_gdf(G)
    
    # 4. 可视化分析结果
    print("\n创建可视化图...")
    
    # 4.1 连通性可视化
    if 'connectivity' in nodes.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        edges.plot(ax=ax, color='grey', linewidth=0.5, alpha=0.5)
        nodes.plot(ax=ax, column='connectivity', cmap='viridis', markersize=5, legend=True)
        buildings.plot(ax=ax, color='lightgrey', markersize=0.5, alpha=0.3)
        plt.title("公路网络连通性分析")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "connectivity_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("连通性可视化完成")
    
    # 4.2 局部整合度可视化
    if 'local_integration' in nodes.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        edges.plot(ax=ax, color='grey', linewidth=0.5, alpha=0.5)
        nodes.plot(ax=ax, column='local_integration', cmap='hot', markersize=5, legend=True)
        buildings.plot(ax=ax, color='lightgrey', markersize=0.5, alpha=0.3)
        plt.title("局部整合度分析")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "local_integration_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("局部整合度可视化完成")
    
    # 4.3 全局整合度可视化
    if 'global_integration' in nodes.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        edges.plot(ax=ax, color='grey', linewidth=0.5, alpha=0.5)
        nodes.plot(ax=ax, column='global_integration', cmap='hot', markersize=5, legend=True)
        buildings.plot(ax=ax, color='lightgrey', markersize=0.5, alpha=0.3)
        plt.title("全局整合度分析")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "global_integration_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("全局整合度可视化完成")
    
    # 4.4 选择值可视化
    if 'choice' in nodes.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        edges.plot(ax=ax, color='grey', linewidth=0.5, alpha=0.5)
        nodes.plot(ax=ax, column='choice', cmap='plasma', markersize=5, legend=True)
        buildings.plot(ax=ax, color='lightgrey', markersize=0.5, alpha=0.3)
        plt.title("选择值分析")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "choice_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("选择值可视化完成")
    
    # 4.5 可达性可视化
    if 'reach_500' in nodes.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        edges.plot(ax=ax, color='grey', linewidth=0.5, alpha=0.5)
        nodes.plot(ax=ax, column='reach_500', cmap='YlOrRd', markersize=5, legend=True)
        buildings.plot(ax=ax, color='lightgrey', markersize=0.5, alpha=0.3)
        plt.title("500米可达性分析")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reach_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("可达性可视化完成")
    
    # 4.6 综合分析可视化 - 整合度与选择值对比
    if 'global_integration' in nodes.columns and 'choice' in nodes.columns:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        
        # 整合度
        edges.plot(ax=axs[0], color='grey', linewidth=0.5, alpha=0.5)
        nodes.plot(ax=axs[0], column='global_integration', cmap='hot', markersize=5, legend=True)
        buildings.plot(ax=axs[0], color='lightgrey', markersize=0.5, alpha=0.3)
        axs[0].set_title("全局整合度")
        
        # 选择值
        edges.plot(ax=axs[1], color='grey', linewidth=0.5, alpha=0.5)
        nodes.plot(ax=axs[1], column='choice', cmap='plasma', markersize=5, legend=True)
        buildings.plot(ax=axs[1], color='lightgrey', markersize=0.5, alpha=0.3)
        axs[1].set_title("选择值")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "integration_choice_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("整合度与选择值对比可视化完成")
    
    # 5. 保存分析结果
    print("\n保存分析结果...")
    nodes.to_file(os.path.join(output_dir, "syntax_nodes.gpkg"), driver="GPKG")
    edges.to_file(os.path.join(output_dir, "syntax_edges.gpkg"), driver="GPKG")
    
    # 6. 生成统计报告
    print("\n生成统计报告...")
    
    # 6.1 基本统计信息
    basic_stats = {
        "节点数量": len(nodes),
        "边数量": len(edges),
        "平均节点度": nodes['degree'].mean(),
        "最大节点度": nodes['degree'].max(),
        "网络密度": G.graph['meshedness'] if 'meshedness' in G.graph else "未计算"
    }
    
    # 6.2 空间句法指标统计
    syntax_stats = {}
    for metric in ['connectivity', 'control', 'local_integration', 'global_integration', 'choice', 'reach_500', 'straightness']:
        if metric in nodes.columns:
            syntax_stats[f"{metric}_平均值"] = nodes[metric].mean()
            syntax_stats[f"{metric}_最大值"] = nodes[metric].max()
            syntax_stats[f"{metric}_最小值"] = nodes[metric].min()
            syntax_stats[f"{metric}_标准差"] = nodes[metric].std()
    
    # 打印统计信息
    print("\n基本网络统计信息:")
    for key, value in basic_stats.items():
        print(f"  {key}: {value}")
    
    print("\n空间句法指标统计:")
    for key, value in syntax_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n空间句法分析完成，结果保存在 {output_dir} 目录")

if __name__ == "__main__":
    run_space_syntax_analysis() 