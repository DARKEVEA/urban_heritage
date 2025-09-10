import json
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from pathlib import Path


# 配置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =============================
# 1. 数据加载
# =============================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all(base_path):
    files = {i: load_json(base_path / f"{i}.json") for i in [1, 2, 3, 4]}
    return files


# =============================
# 2. 覆盖率统计
# =============================
def coverage_stats(files):
    total = len(files[1]["buildings"])
    osm = len(files[3]["buildings"])
    poi = len(files[2]["buildings"])
    none = len(files[4]["buildings"])

    df = pd.DataFrame({
        "类别": ["总数", "OSM匹配", "POI匹配", "无结果"],
        "数量": [total, osm, poi, none]
    })
    print(df)

    plt.figure(figsize=(6, 4))
    sns.barplot(x="类别", y="数量", data=df, palette="Set2")
    plt.title("覆盖率统计")
    plt.show()
    return df


# =============================
# 3. 功能对照 (Sankey数据预处理)
# =============================
def sankey_data(files):
    links = []
    for b in files[3]["buildings"]:
        # 历史功能
        if "matched_points" in b and b["matched_points"]:
            hist_func = (
                    b["matched_points"][0].get("TYP03")
                    or b["matched_points"][0].get("TYP02")
                    or b["matched_points"][0].get("TYP01")
                    or "Unknown"
            )
            hist_name = b["matched_points"][0].get("CHINESE", "")

        else:
            hist_func = "Unknown"
            hist_name = ""

        modern_func = b.get("osm_properties", {}).get("amenity") \
                      or b.get("osm_properties", {}).get("shop") \
                      or b.get("osm_properties", {}).get("tourism") \
                      or b.get("osm_properties", {}).get("diplomatic") \
                      or b.get("osm_properties", {}).get("historic") \
                      or b.get("osm_properties", {}).get("office") \
                      or b.get("osm_properties", {}).get("building") \
                      or "Unknown"
        modern_name = b.get("osm_properties", {}).get("name", "")

        links.append({
            "历史功能": f"{hist_func} ({hist_name})" if hist_name else hist_func,
            "现代功能": f"{modern_func} ({modern_name})" if modern_name else modern_func
        })

    # links = []
    for b in files[2]["buildings"]:
        # =============== 历史功能 =================
        if "matched_points" in b and b["matched_points"]:
            hist_func = (
                    b["matched_points"][0].get("TYP03")
                    or b["matched_points"][0].get("TYP02")
                    or b["matched_points"][0].get("TYP01")
                    or "Unknown"
            )
            hist_name = b["matched_points"][0].get("CHINESE", "")
        else:
            hist_func = "Unknown"
            hist_name = ""

        # =============== 现代功能 =================
        modern_func = "Unknown"
        modern_name = ""

        if "results" in b and isinstance(b["results"], list) and b["results"]:
            # 选取最近的POI
            best = min(
                b["results"],
                key=lambda r: r.get("detail_info", {}).get("distance", float("inf"))
            )
            modern_func = (
                    best.get("detail_info", {}).get("classified_poi_tag")
                    or best.get("detail_info", {}).get("tag")
                    or "Unknown"
            )
            modern_name = best.get("name", "")

        # 组装结果
        links.append({
            "历史功能": f"{hist_func} ({hist_name})" if hist_name else hist_func,
            "现代功能": f"{modern_func} ({modern_name})" if modern_name else modern_func
        })

        links = clean_unknown_pairs(links)

    return pd.DataFrame(links)

def clean_unknown_pairs(links):
    cleaned = []
    skip = False

    for i in range(len(links)):
        if skip:
            skip = False
            continue

        h = links[i]["历史功能"]
        m = links[i]["现代功能"]

        # 检查是否满足配对模式
        if (m.startswith("Unknown") and
            i + 1 < len(links) and links[i+1]["历史功能"] == "Unknown"):
            # 合并两行
            merged = {
                "历史功能": h,
                "现代功能": links[i+1]["现代功能"]
            }
            cleaned.append(merged)
            skip = True  # 跳过下一行
        else:
            cleaned.append(links[i])

    return cleaned


# =============================
# 4. 空间可视化
# =============================
def make_map(files, save_path="map.html"):
    m = folium.Map(location=[31.23, 121.48], zoom_start=14)
    colors = {1: "blue", 2: "orange", 3: "green", 4: "gray"}

    for idx, data in files.items():
        for b in data["buildings"]:
            # OSM数据
            if "osm_properties" in b:
                lat = b["osm_properties"].get("centroid_y")
                lon = b["osm_properties"].get("centroid_x")
            # 百度POI数据
            elif "results" in b:
                lat = b["results"][0]["location"]["lat"]
                lon = b["results"][0]["location"]["lng"]
            else:
                continue
            name = b.get("osm_properties", {}).get("name") or b.get("results", [{}])[0].get("name", "未命名")
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=colors[idx],
                fill=True,
                fill_color=colors[idx],
                popup=name
            ).add_to(m)
    m.save(save_path)
    print(f"✅ 地图已保存 {save_path}")


# =============================
# 主函数
# =============================
def main():
    base_path = Path(r"E:/CODE/urban_heritage/data/processed/filter_results")
    files = load_all(base_path)

    # 覆盖率
    coverage_stats(files)

    # 功能对照
    df = sankey_data(files)
    df.to_csv(base_path / "sankey_data.csv", index=False, encoding="utf-8-sig")
    print(df.head())

    # 地图
    make_map(files, save_path=base_path / "map.html")


if __name__ == "__main__":
    main()
