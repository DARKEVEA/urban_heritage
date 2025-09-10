import geopandas as gpd
import os
import glob
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def check_crs(file_path):
    """检查文件的坐标系统"""
    try:
        gdf = gpd.read_file(file_path)
        print(f"文件: {file_path}")
        print(f"  坐标系: {gdf.crs}")
        print(f"  几何类型: {gdf.geometry.geom_type.unique()}")
        print(f"  边界范围: {gdf.total_bounds}")
        print(f"  记录数: {len(gdf)}")
        print()
        return gdf.crs
    except Exception as e:
        print(f"文件: {file_path}")
        print(f"  错误: {str(e)}")
        print()
        return None

def main():
    # 检查data/raw目录中的所有地理数据文件
    data_dir = PROCESSED_DATA_DIR
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"目录不存在: {data_dir}")
        return
    
    # 查找所有地理数据文件
    shp_files = glob.glob(os.path.join(data_dir, "*.shp"))
    gpkg_files = glob.glob(os.path.join(data_dir, "*.gpkg"))
    geojson_files = glob.glob(os.path.join(data_dir, "*.geojson"))
    
    all_files = shp_files + gpkg_files + geojson_files
    
    if not all_files:
        print(f"在 {data_dir} 中没有找到地理数据文件")
        return
    
    print(f"找到 {len(all_files)} 个地理数据文件")
    print("="*50)
    
    for file in all_files:
        check_crs(file)
    
    print("坐标系检查完成")

if __name__ == "__main__":
    main() 