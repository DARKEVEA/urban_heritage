import geopandas as gpd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib
from shapely.ops import polygonize
from shapely.geometry import LineString, MultiLineString
from src.config import PROCESSED_DATA_DIR

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def convert_multilinestring_to_linestring(input_file):
    """
    将MultiLineString转换为LineString并覆盖保存
    """
    print("开始将MultiLineString转换为LineString...")
    
    # 读取线数据
    lines_gdf = gpd.read_file(input_file)
    print(f"读取线数据: {len(lines_gdf)} 条线")
    
    # 检查几何类型
    geom_types = lines_gdf.geometry.geom_type.unique()
    print(f"几何类型: {geom_types}")
    
    # 如果存在MultiLineString，则转换为LineString
    if 'MultiLineString' in geom_types:
        print("发现MultiLineString，开始转换...")
        new_geometries = []
        
        for geom in lines_gdf.geometry:
            if geom.geom_type == 'MultiLineString':
                # 将MultiLineString拆分为多个LineString
                for line in geom.geoms:
                    new_geometries.append(line)
            else:
                new_geometries.append(geom)
        
        # 创建新的GeoDataFrame
        new_gdf = gpd.GeoDataFrame(geometry=new_geometries, crs=lines_gdf.crs)
        
        # 保存覆盖原文件
        new_gdf.to_file(input_file, driver="GPKG")
        print(f"已将MultiLineString转换为LineString并保存到: {input_file}")
        return new_gdf
    else:
        print("数据中不包含MultiLineString，无需转换")
        return lines_gdf

def create_blocks_from_lines(year="1946"):
    """
    从线数据创建街区多边形
    此函数处理 GEOCODER_{year}_aligned.gpkg 文件，将LineString转换为多边形，并计算面积
    
    参数:
        year (str): 数据年份，默认为"1946"
    """
    print(f"开始从{year}年线数据创建街区多边形...")
    
    # 输入和输出目录
    input_dir = PROCESSED_DATA_DIR
    output_file = os.path.join(input_dir, f"GEOCODER_{year}_blocks.gpkg")
    
    # 读取对齐后的线数据
    input_file = os.path.join(input_dir, f"GEOCODER_{year}_aligned.gpkg")
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return
    
    # 首先将MultiLineString转换为LineString
    lines_gdf = convert_multilinestring_to_linestring(input_file)
    
    # 将所有线合并为一个MultiLineString对象
    all_lines = []
    for geom in lines_gdf.geometry:
        if geom.geom_type == 'LineString':
            all_lines.append(geom)
        elif geom.geom_type == 'MultiLineString':
            all_lines.extend(list(geom.geoms))
    
    merged_lines = MultiLineString(all_lines)
    print(f"合并后的线数量: {len(all_lines)}")
    
    # 使用polygonize从线创建多边形
    print("从线创建多边形...")
    polygons = list(polygonize(merged_lines))
    print(f"创建的多边形数量: {len(polygons)}")
    
    if len(polygons) == 0:
        print("警告: 未能创建任何多边形，可能需要检查线数据的连通性")
        
        # 尝试使用networkx创建图并识别环
        print("尝试使用网络图方法识别环...")
        G = nx.Graph()
        
        # 为每条线添加边
        for i, line in enumerate(all_lines):
            coords = list(line.coords)
            for j in range(len(coords) - 1):
                G.add_edge(coords[j], coords[j + 1])
        
        # 寻找环
        cycles = list(nx.cycle_basis(G))
        print(f"找到 {len(cycles)} 个环")
        
        # 从环创建多边形
        for cycle in cycles:
            # 确保环闭合
            if cycle[0] != cycle[-1]:
                cycle.append(cycle[0])
            polygon = LineString(cycle).convex_hull
            polygons.append(polygon)
        
        print(f"创建的多边形数量: {len(polygons)}")
    
    # 创建GeoDataFrame
    blocks_gdf = gpd.GeoDataFrame(geometry=polygons, crs=lines_gdf.crs)
    
    # 计算面积和周长
    blocks_gdf['area'] = blocks_gdf.geometry.area
    blocks_gdf['perimeter'] = blocks_gdf.geometry.length
    
    # 保存结果
    blocks_gdf.to_file(output_file, driver="GPKG")
    print(f"街区多边形已保存到: {output_file}")
    
    # 可视化结果
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制原始线
    lines_gdf.plot(ax=ax, color='gray', linewidth=0.5)
    
    # 绘制生成的多边形
    blocks_gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')
    
    plt.title(f"从{year}年线数据生成的街区多边形")
    
    # 保存可视化结果
    vis_dir = os.path.join(PROCESSED_DATA_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f"generated_blocks_visualization_{year}.png"), bbox_inches='tight', dpi=300)
    print(f"可视化图已保存到 {os.path.join(vis_dir, f'generated_blocks_visualization_{year}.png')}")
    
    return blocks_gdf

if __name__ == "__main__":
    create_blocks_from_lines() 