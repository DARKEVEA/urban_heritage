import geopandas as gpd
import momepy
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import contextily as ctx
import pandas as pd
import networkx as nx
from shapely.geometry import box
from matplotlib.colors import LinearSegmentedColormap
from src.config import PROCESSED_DATA_DIR, HISTORICAL_COMPARISON_DIR

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def run_historical_comparison():
    """运行历史城市形态对比分析"""
    # 输入和输出目录
    input_dir = PROCESSED_DATA_DIR
    output_dir = HISTORICAL_COMPARISON_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始历史城市形态对比分析...")
    
    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"目录不存在: {input_dir}")
        return
    
    # 定义不同时期的数据
    time_periods = {
        "1939年": {
            "地址": os.path.join(input_dir, "GEOCODER_1939_aligned.gpkg")
        },
        "1946年": {
            "地址": os.path.join(input_dir, "GEOCODER_1946_aligned.gpkg")
        },
        "1949年": {
            "建筑": os.path.join(input_dir, "Buildings_aligned.gpkg"),
            "街区": os.path.join(input_dir, "GEOCODER_1946_aligned.gpkg")
        }

    }
    
    # 读取各时期数据
    data = {}
    for period, datasets in time_periods.items():
        data[period] = {}
        for key, file in datasets.items():
            if not os.path.exists(file):
                print(f"文件不存在: {file}")
                continue
            try:
                gdf = gpd.read_file(file)
                # 确保所有数据都是EPSG:32651投影坐标系
                if gdf.crs != "EPSG:32651":
                    gdf = gdf.to_crs("EPSG:32651")
                data[period][key] = gdf
                print(f"读取 {period} {key} 数据: {len(gdf)} 条记录")
            except Exception as e:
                print(f"读取错误 ({period} {key}): {str(e)}")
    
    # 1. 地址点密度分析
    print("\n进行地址点密度分析...")
    
    # 创建分析网格
    # 获取所有数据的总边界
    bounds = None
    for period, datasets in data.items():
        for key, gdf in datasets.items():
            if bounds is None:
                bounds = gdf.total_bounds
            else:
                bounds = np.array([
                    min(bounds[0], gdf.total_bounds[0]),
                    min(bounds[1], gdf.total_bounds[1]),
                    max(bounds[2], gdf.total_bounds[2]),
                    max(bounds[3], gdf.total_bounds[3])
                ])
    
    # 创建网格 - 使用投影坐标系下的500米
    cell_size = 500  # 500米（投影坐标系单位为米）
    x_min, y_min, x_max, y_max = bounds
    
    # 限制网格数量，避免创建过多单元格导致性能问题
    max_cells = 50000  # 设置最大单元格数量限制
    estimated_cells = ((x_max - x_min) // cell_size + 1) * ((y_max - y_min) // cell_size + 1)
    
    if estimated_cells > max_cells:
        print(f"警告: 预计网格单元数量 ({estimated_cells}) 超过限制 ({max_cells})，增加单元格大小")
        # 动态调整单元格大小
        cell_size = max(cell_size, min(
            (x_max - x_min) / np.sqrt(max_cells / 2),
            (y_max - y_min) / np.sqrt(max_cells / 2)
        ))
        print(f"调整后的单元格大小: {cell_size}米")
    
    # 创建网格单元
    grid_cells = []
    for x0 in np.arange(x_min, x_max, cell_size):
        for y0 in np.arange(y_min, y_max, cell_size):
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append(box(x0, y0, x1, y1))
    
    print(f"创建网格: {len(grid_cells)} 个单元格")
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32651")
    
    # 计算每个网格单元中的地址点数量
    address_density = {}
    for period in ["1939年", "1946年"]:
        if period in data and "地址" in data[period]:
            print(f"计算 {period} 地址密度...")
            addresses = data[period]["地址"]
            
            # 使用空间索引加速空间连接操作
            joined = gpd.sjoin(grid, addresses, how="left", predicate="contains")
            counts = joined.groupby(level=0).size()
            grid[f'address_count_{period}'] = counts
            
            # 计算密度 - 使用投影坐标系下的平方公里
            cell_area_sq_km = (cell_size * cell_size) / 1_000_000  # 平方米转平方公里
            grid[f'address_density_{period}'] = counts / cell_area_sq_km
            address_density[period] = grid[f'address_density_{period}'].fillna(0)
            print(f"计算 {period} 地址密度完成")
    
    # 2. 建筑点密度分析
    print("\n进行建筑点密度分析...")
    
    if "1949年" in data and "建筑" in data["1949年"] and "街区" in data["1949年"]:
        buildings = data["1949年"]["建筑"]
        blocks = data["1949年"]["街区"]
        
        # 计算街区面积 - 在投影坐标系下计算，单位为平方米
        blocks['area'] = blocks.geometry.area
        
        # 将建筑点分配到街区
        spatial_join = gpd.sjoin(buildings, blocks, how="inner", predicate="within")
        blocks_with_points = spatial_join.groupby('index_right').size().reset_index(name='building_count')
        
        # 合并回街区数据
        blocks = blocks.merge(blocks_with_points, left_index=True, right_on='index_right', how='left')
        blocks['building_count'] = blocks['building_count'].fillna(0)
        
        # 计算建筑点密度 - 每平方公里的建筑数量
        blocks['building_density'] = blocks['building_count'] / (blocks['area'] / 10_000)  # 平方米转平方百米
        
        # 更新数据
        data["1949年"]["街区"] = blocks
        
        print("计算1949年建筑点密度完成")
    
    # 3. 道路网络分析
    print("\n进行道路网络分析...")
    
    network_stats = {}
    for period in ["1949年", "2008年"]:
        if period in data and "铁路" in data[period]:
            try:
                railway = data[period]["铁路"]
                
                # 转换为网络
                G = momepy.gdf_to_nx(railway)
                
                # 获取最大连通子图
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
                
                # 计算网络指标
                G = momepy.node_degree(G)
                
                # 计算边长度
                for u, v, data in G.edges(data=True):
                    if 'length' not in data:
                        coords_u = G.nodes[u]['geometry'].coords[0]
                        coords_v = G.nodes[v]['geometry'].coords[0]
                        data['length'] = ((coords_u[0] - coords_v[0])**2 + (coords_u[1] - coords_v[1])**2)**0.5
                
                # 计算网络密度
                G.graph['meshedness'] = nx.density(G)
                
                # 计算连通性
                for node, data in G.nodes(data=True):
                    data['connectivity'] = data['degree']
                
                # 计算整合度
                try:
                    closeness = nx.closeness_centrality(G, distance='length')
                    nx.set_node_attributes(G, closeness, 'global_integration')
                except:
                    print(f"计算整合度错误 ({period})")
                
                # 转换回GeoDataFrame
                nodes, edges = momepy.nx_to_gdf(G)
                
                # 保存网络统计信息
                network_stats[period] = {
                    "节点数量": len(nodes),
                    "边数量": len(edges),
                    "平均节点度": nodes['degree'].mean(),
                    "网格化指数": G.graph['meshedness'] if 'meshedness' in G.graph else None,
                    "平均边长度": edges['length'].mean() if 'length' in edges.columns else None,
                    "平均连通性": nodes['connectivity'].mean() if 'connectivity' in nodes.columns else None,
                    "平均整合度": nodes['global_integration'].mean() if 'global_integration' in nodes.columns else None
                }
                
                # 保存处理后的网络数据
                data[period]["网络节点"] = nodes
                data[period]["网络边"] = edges
                
                print(f"计算 {period} 道路网络分析完成")
            except Exception as e:
                print(f"道路网络分析错误 ({period}): {str(e)}")
    
    # 4. 可视化分析结果
    print("\n创建可视化图...")
    
    # 4.1 地址点密度对比可视化
    if len(address_density) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        
        # 创建颜色映射
        colors = [(1, 1, 1), (1, 0.8, 0), (1, 0, 0)]  # 从白色到黄色到红色
        cmap = LinearSegmentedColormap.from_list('density_cmap', colors)
        
        # 找到最大密度值，用于统一颜色比例
        max_density = max(
            address_density["1939年"].max() if "1939年" in address_density else 0,
            address_density["1946年"].max() if "1946年" in address_density else 0
        )
        
        # 1939年地址密度
        if "1939年" in address_density:
            grid.plot(column=f'address_density_1939年', cmap=cmap, vmin=0, vmax=max_density, 
                     legend=True, ax=axs[0], alpha=0.7)
            axs[0].set_title("1939年地址点密度 (每平方百米)")
        
        # 1946年地址密度
        if "1946年" in address_density:
            grid.plot(column=f'address_density_1946年', cmap=cmap, vmin=0, vmax=max_density, 
                     legend=True, ax=axs[1], alpha=0.7)
            axs[1].set_title("1946年地址点密度 (每平方百米)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "address_density_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("地址点密度对比可视化完成")
    
    # 4.2 建筑点密度可视化
    if "1949年" in data and "街区" in data["1949年"] and "building_density" in data["1949年"]["街区"].columns:
        blocks = data["1949年"]["街区"]
        buildings = data["1949年"]["建筑"]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        blocks.plot(column='building_density', cmap='Reds', legend=True, ax=ax)
        buildings.plot(ax=ax, color='black', markersize=0.5, alpha=0.3)
        # ctx.add_basemap(ax, crs=blocks.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        plt.title("1949年街区建筑点密度 (每平方公里)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "building_density_1949.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("建筑点密度可视化完成")
    
    # 4.3 道路网络对比可视化
    if "1949年" in data and "网络节点" in data["1949年"] and "2008年" in data and "网络节点" in data["2008年"]:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        
        # 1949年道路网络
        data["1949年"]["网络边"].plot(ax=axs[0], color='grey', linewidth=1)
        data["1949年"]["网络节点"].plot(ax=axs[0], column='connectivity', cmap='viridis', markersize=5, legend=True)
        # ctx.add_basemap(axs[0], crs=data["1949年"]["网络节点"].crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        axs[0].set_title("1949年道路网络连通性")
        
        # 2008年道路网络
        data["2008年"]["网络边"].plot(ax=axs[1], color='grey', linewidth=1)
        data["2008年"]["网络节点"].plot(ax=axs[1], column='connectivity', cmap='viridis', markersize=5, legend=True)
        # ctx.add_basemap(axs[1], crs=data["2008年"]["网络节点"].crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        axs[1].set_title("2008年道路网络连通性")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "road_network_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("道路网络对比可视化完成")
    
    # 4.4 整合度对比可视化
    if "1949年" in data and "网络节点" in data["1949年"] and "2008年" in data and "网络节点" in data["2008年"]:
        if 'global_integration' in data["1949年"]["网络节点"].columns and 'global_integration' in data["2008年"]["网络节点"].columns:
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            
            # 1949年整合度
            data["1949年"]["网络边"].plot(ax=axs[0], color='grey', linewidth=1)
            data["1949年"]["网络节点"].plot(ax=axs[0], column='global_integration', cmap='hot', markersize=5, legend=True)
            # ctx.add_basemap(axs[0], crs=data["1949年"]["网络节点"].crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            axs[0].set_title("1949年道路网络整合度")
            
            # 2008年整合度
            data["2008年"]["网络边"].plot(ax=axs[1], color='grey', linewidth=1)
            data["2008年"]["网络节点"].plot(ax=axs[1], column='global_integration', cmap='hot', markersize=5, legend=True)
            # ctx.add_basemap(axs[1], crs=data["2008年"]["网络节点"].crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            axs[1].set_title("2008年道路网络整合度")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "integration_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("整合度对比可视化完成")
    
    # 5. 保存分析结果
    print("\n保存分析结果...")
    
    # 保存网格数据
    if 'grid' in locals():
        grid.to_file(os.path.join(output_dir, "address_density_grid.gpkg"), driver="GPKG")
    
    # 保存网络统计数据
    if network_stats:
        network_stats_df = pd.DataFrame(network_stats).T
        network_stats_df.to_csv(os.path.join(output_dir, "network_stats_comparison.csv"))
        print("网络统计数据已保存")
    
    # 6. 生成统计报告
    print("\n生成统计报告...")
    
    # 6.1 地址点统计
    address_stats = {}
    for period in ["1939年", "1946年"]:
        if period in data and "地址" in data[period]:
            address_stats[period] = {
                "地址点数量": len(data[period]["地址"]),
                "平均密度": address_density[period].mean() if period in address_density else "未计算"
            }
    
    # 6.2 网络变化统计
    if len(network_stats) >= 2:
        network_change = {}
        for metric in ["节点数量", "边数量", "平均节点度", "网格化指数", "平均边长度", "平均连通性", "平均整合度"]:
            if "1949年" in network_stats and "2008年" in network_stats:
                if metric in network_stats["1949年"] and metric in network_stats["2008年"]:
                    if network_stats["1949年"][metric] is not None and network_stats["2008年"][metric] is not None:
                        change = network_stats["2008年"][metric] - network_stats["1949年"][metric]
                        change_percent = change / network_stats["1949年"][metric] * 100 if network_stats["1949年"][metric] != 0 else float('inf')
                        network_change[metric] = {
                            "1949年值": network_stats["1949年"][metric],
                            "2008年值": network_stats["2008年"][metric],
                            "变化量": change,
                            "变化百分比": f"{change_percent:.2f}%"
                        }
    
    # 打印统计信息
    print("\n地址点统计信息:")
    for period, stats in address_stats.items():
        print(f"  {period}:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    print("\n网络统计信息:")
    for period, stats in network_stats.items():
        print(f"  {period}:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    if 'network_change' in locals() and network_change:
        print("\n网络变化统计:")
        for metric, changes in network_change.items():
            print(f"  {metric}:")
            for key, value in changes.items():
                print(f"    {key}: {value}")
    
    print(f"\n历史城市形态对比分析完成，结果保存在 {output_dir} 目录")

if __name__ == "__main__":
    run_historical_comparison() 