import geopandas as gpd
import fiona
import os
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import box
import numpy as np
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 目标坐标系 - 使用WGS84 / UTM Zone 51N (EPSG:32651)，这是一个投影坐标系，适合计算面积和距离
target_crs = "EPSG:32651"

# 输入和输出目录
input_dir = RAW_DATA_DIR
output_dir = PROCESSED_DATA_DIR
os.makedirs(output_dir, exist_ok=True)

# 要处理的文件列表
files_to_process = [
    "AddressCM.shp",
    "Buildings.shp", 
    "Railway_1949.shp",
    "WaterNetwork.shp",
    "GEOCODER_1946.shp",
    "GEOCODER_1939.shp",
    "FC_270blocks.shp",
    # "Railway_2008.gpkg"
]

# 存储所有处理后的数据，用于后续可视化
processed_data = {}

# 处理每个文件
for file in files_to_process:
    try:
        print(f"\n处理文件: {file}")
        file_path = os.path.join(input_dir, file)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"  文件不存在: {file_path}")
            continue
        
        # 读取数据
        if file.endswith('.gpkg'):
            layers = fiona.listlayers(file_path)
            for layer in layers:
                print(f"  处理图层: {layer}")
                gdf = gpd.read_file(file_path, layer=layer)
                
                # 检查并设置坐标系
                if gdf.crs is None:
                    print(f"  警告: {file} 没有坐标系信息，设置为EPSG:32651")
                    gdf.crs = "EPSG:32651"
                
                # 转换到目标坐标系
                gdf = gdf.to_crs(target_crs)
                
                # 保存处理后的数据
                output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_{layer}.gpkg")
                gdf.to_file(output_file, driver="GPKG")
                print(f"  已保存到: {output_file}")
                
                # 存储用于可视化
                processed_data[f"{os.path.splitext(file)[0]}_{layer}"] = gdf
        else:
            gdf = gpd.read_file(file_path)
            
            # 检查并设置坐标系
            if gdf.crs is None:
                print(f"  警告: {file} 没有坐标系信息，设置为EPSG:32651")
                gdf.crs = "EPSG:32651"
            
            # 转换到目标坐标系
            gdf = gdf.to_crs(target_crs)
            
            # 保存处理后的数据
            output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_aligned.gpkg")
            gdf.to_file(output_file, driver="GPKG")
            print(f"  已保存到: {output_file}")
            
            # 存储用于可视化
            processed_data[os.path.splitext(file)[0]] = gdf
            
    except Exception as e:
        print(f"  处理错误: {str(e)}")

print("\n所有文件处理完成")

# 可视化所有数据，检查对齐情况
print("\n创建可视化图...")

# 创建一个足够大的图
fig, ax = plt.subplots(figsize=(15, 15))

# 为不同类型的数据使用不同颜色
colors = plt.cm.tab10(np.linspace(0, 1, len(processed_data)))
color_dict = dict(zip(processed_data.keys(), colors))

# 绘制每个数据集 - 所有数据已经在处理过程中转换为EPSG:32651
for name, gdf in processed_data.items():
    try:
        # 确保使用投影坐标系
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
            
        # 根据几何类型选择不同的绘图方式
        if all(geom.geom_type == 'Point' for geom in gdf.geometry):
            gdf.plot(ax=ax, color=color_dict[name], markersize=2, alpha=0.7, label=name)
        elif all(geom.geom_type == 'LineString' for geom in gdf.geometry):
            gdf.plot(ax=ax, color=color_dict[name], linewidth=1, alpha=0.7, label=name)
        else:
            gdf.plot(ax=ax, color=color_dict[name], alpha=0.5, label=name)
    except Exception as e:
        print(f"  绘图错误 ({name}): {str(e)}")

# 添加图例和标题
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("所有数据集对齐后的可视化 (EPSG:32651 投影)")
plt.axis("equal")

# 保存图像
vis_dir = os.path.join(output_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)
plt.savefig(os.path.join(vis_dir, "aligned_data_visualization.png"), bbox_inches='tight', dpi=300)
print(f"可视化图已保存到 {os.path.join(vis_dir, 'aligned_data_visualization.png')}")

if __name__ == "__main__":
    print("坐标系对齐完成") 