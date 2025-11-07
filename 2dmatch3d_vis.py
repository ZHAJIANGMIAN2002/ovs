"""
2D图像到3D点云的匹配和可视化脚本
输入：2D图像路径
输出：对应的3D场景的scene.ply文件
"""
# python 2dmatch3d_vis.py /path/to/image.jpg /path/to/output.ply
import torch
import open3d as o3d
import numpy as np
import os
import sys
from glob import glob

def find_matching_region(image_path, data_root_3d, data_root_2d):
    """
    根据2D图像路径找到对应的3D region文件
    
    Args:
        image_path: 2D图像路径
        data_root_3d: 3D数据根目录
        data_root_2d: 2D数据根目录
    
    Returns:
        region_file_path: 匹配的region文件路径
    """
    # 从图像路径提取scene名称
    # 例如: /path/to/matterport_2d/8194nk5LbLH/color/image.jpg -> 8194nk5LbLH
    path_parts = image_path.split(os.sep)
    scene_name = None
    for i, part in enumerate(path_parts):
        if 'matterport_2d' in part or part == 'matterport_2d':
            if i + 1 < len(path_parts):
                scene_name = path_parts[i + 1]
                break
    
    if scene_name is None:
        # 尝试从路径中直接提取scene名称（假设格式为 .../scene_name/color/...）
        parts = image_path.split(os.sep)
        for i, part in enumerate(parts):
            if part == 'color' and i > 0:
                scene_name = parts[i - 1]
                break
    
    if scene_name is None:
        raise ValueError(f"无法从路径中提取scene名称: {image_path}")
    
    print(f"提取的scene名称: {scene_name}")
    
    # 确定split（train/val/test）
    split = None
    if '/val/' in image_path or '/val/' in data_root_3d:
        split = 'val'
    elif '/train/' in image_path or '/train/' in data_root_3d:
        split = 'train'
    elif '/test/' in image_path or '/test/' in data_root_3d:
        split = 'test'
    else:
        # 默认检查val目录
        split = 'val'
    
    data_root_3d_split = os.path.join(data_root_3d, split)
    
    # 加载该图像的相机pose
    pose_path = image_path.replace('color', 'pose').replace('.jpg', '.txt')
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"找不到pose文件: {pose_path}")
    
    pose = np.loadtxt(pose_path)
    cam_loc = pose[:3, -1]  # 相机位置
    print(f"相机位置: [{cam_loc[0]:.2f}, {cam_loc[1]:.2f}, {cam_loc[2]:.2f}]")
    
    # 查找所有region文件
    region_files = sorted(glob(os.path.join(data_root_3d_split, f"{scene_name}_region*.pth")))
    if len(region_files) == 0:
        raise FileNotFoundError(f"找不到region文件: {os.path.join(data_root_3d_split, f'{scene_name}_region*.pth')}")
    
    print(f"找到 {len(region_files)} 个region文件")
    
    matching_regions = []
    
    for region_file in region_files:
        # 加载3D点云数据
        data = torch.load(region_file)
        coords = data[0]  # 点云坐标
        
        # 计算region的bounding box
        bbox_l = coords.min(axis=0)
        bbox_h = coords.max(axis=0)
        
        region_name = os.path.basename(region_file).replace('.pth', '')
        
        # 检查相机位置是否在bounding box内
        in_box = (cam_loc[0] > bbox_l[0]) & (cam_loc[0] < bbox_h[0]) & \
                 (cam_loc[1] > bbox_l[1]) & (cam_loc[1] < bbox_h[1]) & \
                 (cam_loc[2] > bbox_l[2]) & (cam_loc[2] < bbox_h[2])
        
        if in_box:
            matching_regions.append(region_file)
            print(f"  ✓ 匹配: {region_name}")
    
    if matching_regions:
        # 如果有多个匹配，选择第一个
        region_file = matching_regions[0]
        print(f"\n使用region文件: {os.path.basename(region_file)}")
        return region_file
    else:
        print(f"\n未找到完全匹配的region，查找最近的region...")
        # 如果没有完全匹配，找最近的region
        min_dist = float('inf')
        closest_region = None
        for region_file in region_files:
            data = torch.load(region_file)
            coords = data[0]
            centroid = (coords.min(axis=0) + coords.max(axis=0)) / 2
            dist = np.linalg.norm(cam_loc - centroid)
            if dist < min_dist:
                min_dist = dist
                closest_region = region_file
        print(f"  最近的region: {os.path.basename(closest_region)} (距离: {min_dist:.2f})")
        return closest_region


def load_and_process_pointcloud(region_file):
    """
    加载并处理3D点云数据
    
    Args:
        region_file: region文件路径
    
    Returns:
        pcd: Open3D点云对象
    """
    print(f"\n加载3D点云: {os.path.basename(region_file)}")
    
    # 加载数据
    data = torch.load(region_file)
    
    # 获取 3D 坐标（data[0] 是坐标数据）
    coords = data[0]  # 直接使用 NumPy 数组
    
    # 如果有颜色信息，data[1] 是颜色
    if len(data) > 1:
        colors = data[1]  # 直接使用 NumPy 数组
        print(f"原始颜色范围: min={colors.min():.3f}, max={colors.max():.3f}")
        
        # 处理颜色值：如果值大于1，可能是0-255范围，需要除以255
        # 如果值有负数，可能是归一化到[-1,1]范围，需要映射到[0,1]
        if colors.max() > 1:
            colors = colors / 255.0
        elif colors.min() < 0:
            # 将 [-1, 1] 范围映射到 [0, 1]
            colors = (colors + 1) / 2.0
        
        # 确保颜色值在 [0, 1] 范围内（Open3D要求颜色值在[0,1]）
        colors = np.clip(colors, 0, 1)
        print(f"处理后的颜色范围: min={colors.min():.3f}, max={colors.max():.3f}")
    else:
        colors = np.ones_like(coords) * 0.7  # 如果没有颜色，设置为灰色
        print("未找到颜色信息，使用灰色")
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 验证点云是否包含颜色
    print(f"点云包含 {len(pcd.points)} 个点")
    print(f"点云包含颜色: {pcd.has_colors()}")
    
    return pcd


def save_pointcloud(pcd, output_path):
    """
    保存点云为PLY文件
    
    Args:
        pcd: Open3D点云对象
        output_path: 输出文件路径
    """
    success = o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    if success:
        print(f"\n✓ 成功保存点云到: {output_path}")
        # 验证保存的文件
        pcd_loaded = o3d.io.read_point_cloud(output_path)
        print(f"  验证: 加载的点云包含 {len(pcd_loaded.points)} 个点")
        print(f"  验证: 包含颜色信息: {pcd_loaded.has_colors()}")
        if pcd_loaded.has_colors():
            print(f"  ✓ PLY文件包含颜色信息，可以在MeshLab中正确渲染")
    else:
        raise RuntimeError(f"保存点云失败: {output_path}")


def main():
    """主函数"""
    # 默认数据路径
    default_data_root_3d = "/root/zjm/ovs/openscene/data/matterport_3d"
    default_data_root_2d = "/root/zjm/ovs/openscene/data/matterport_2d"
    default_output_path = "/root/zjm/ovs/scene.ply"
    
    # 从命令行参数获取输入图像路径
    if len(sys.argv) < 2:
        print("用法: python 2dmatch3d_vis.py <2d_image_path> [output_ply_path]")
        print("示例: python 2dmatch3d_vis.py /path/to/image.jpg")
        print(f"示例: python 2dmatch3d_vis.py /path/to/image.jpg /path/to/output.ply")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 可选的输出路径
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = default_output_path
    
    # 可选的数据根目录
    if len(sys.argv) >= 4:
        data_root_3d = sys.argv[3]
    else:
        data_root_3d = default_data_root_3d
    
    if len(sys.argv) >= 5:
        data_root_2d = sys.argv[4]
    else:
        data_root_2d = default_data_root_2d
    
    print("=" * 60)
    print("2D图像到3D点云匹配和可视化")
    print("=" * 60)
    print(f"输入图像: {image_path}")
    print(f"输出PLY: {output_path}")
    print(f"3D数据目录: {data_root_3d}")
    print(f"2D数据目录: {data_root_2d}")
    print("=" * 60)
    
    try:
        # 步骤1: 找到匹配的region文件
        region_file = find_matching_region(image_path, data_root_3d, data_root_2d)
        
        # 步骤2: 加载和处理点云
        pcd = load_and_process_pointcloud(region_file)
        
        # 步骤3: 保存为PLY文件
        save_pointcloud(pcd, output_path)
        
        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

