#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将CFPU输入文件夹中的所有patch保存为SVG图片

该脚本会遍历output/cfpu_input文件夹中的所有模型，读取对应的OBJ模型和patch数据，
然后绘制网格和patch（红点标出），并保存为SVG格式的图片到output/figs/目录中。
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath("."))


def load_obj_model(obj_path):
    """
    加载OBJ模型文件
    
    Args:
        obj_path: OBJ文件路径
        
    Returns:
        mesh: pyvista网格对象
    """
    try:
        mesh = pv.read(obj_path)
        return mesh
    except Exception as e:
        print(f"加载OBJ模型失败: {e}")
        return None


def load_patch_data(cfpu_dir):
    """
    加载CFPU输入数据
    
    Args:
        cfpu_dir: CFPU输入数据目录
        
    Returns:
        dict: 包含patches数据的字典
    """
    try:
        patches_path = cfpu_dir / 'patches.txt'
        if not patches_path.exists():
            return None
            
        patches = np.loadtxt(patches_path)
        return {
            'patches': patches
        }
    except Exception as e:
        print(f"加载patch数据失败: {e}")
        return None


def find_matching_obj_model(model_name, obj_models_dir):
    """
    根据模型名称查找对应的OBJ模型文件
    
    Args:
        model_name: 模型名称（如'Cube_surface_cellnormals_cfpu_input'）
        obj_models_dir: OBJ模型目录
        
    Returns:
        Path: OBJ模型文件路径
    """
    # 从模型名称中提取基本名称（去掉_surface_cellnormals_cfpu_input等后缀）
    base_name = model_name.replace('_surface_cellnormals_cfpu_input', '')
    
    # 构建精确匹配的文件名模式
    exact_pattern = f"{base_name}.obj"
    
    # 在obj_model目录中递归搜索匹配的OBJ文件
    for root, dirs, files in os.walk(obj_models_dir):
        for file in files:
            # 优先尝试精确匹配
            if file.lower() == exact_pattern.lower():
                return Path(root) / file
    
    # 如果精确匹配失败，尝试包含匹配（但排除包含其他模型名称的情况）
    for root, dirs, files in os.walk(obj_models_dir):
        for file in files:
            if file.lower().endswith('.obj') and base_name.lower() in file.lower():
                # 检查是否包含其他模型名称（避免Ring匹配到TruncatedRing）
                file_base = file.lower().replace('.obj', '')
                # 如果文件名正好等于base_name，或者文件名以base_name开头，则认为是正确匹配
                if file_base == base_name.lower() or file_base.startswith(base_name.lower() + '_'):
                    return Path(root) / file
    
    return None


def create_visualization(mesh, patches_data, model_name):
    """
    创建网格和patch的可视化
    
    Args:
        mesh: 网格对象
        patches_data: patch数据
        model_name: 模型名称
        
    Returns:
        plotter: pyvista绘图器对象
    """
    # 创建绘图器（离屏渲染，用于保存图片）
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    
    # 设置背景色为白色
    plotter.set_background('white')
    
    # 添加网格模型（灰色，半透明，带边缘）
    mesh_actor = plotter.add_mesh(
        mesh,
        color='lightgray',
        specular=0.1,
        smooth_shading=True,
        opacity=0.7,
        show_edges=True,
        edge_color='gray',
        name='mesh'
    )
    
    # 添加patch点（红色）
    if patches_data and 'patches' in patches_data:
        patches = patches_data['patches']
        if len(patches) > 0:
            patch_cloud = pv.PolyData(patches)
            plotter.add_mesh(
                patch_cloud,
                color='red',
                render_points_as_spheres=True,
                point_size=10,
                name='patches'
            )
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 计算模型的边界框
    bounds = mesh.bounds
    
    # 计算模型中心点
    center = [(bounds[1] + bounds[0]) / 2, 
              (bounds[3] + bounds[2]) / 2, 
              (bounds[5] + bounds[4]) / 2]
    
    # 计算模型的最大尺寸
    size_x = bounds[1] - bounds[0]
    size_y = bounds[3] - bounds[2]
    size_z = bounds[5] - bounds[4]
    max_size = max(size_x, size_y, size_z)
    
    # 更激进的相机设置，进一步减少留白
    # 使用更小的相机距离，让模型占据更多画面
    camera_distance = max_size * 0.8  # 从1.5减少到0.8
    
    # 设置等轴测视角
    plotter.camera_position = 'iso'
    
    # 调整相机到更近的位置，使用更紧凑的布局
    plotter.camera.position = [
        center[0] + camera_distance,
        center[1] + camera_distance, 
        center[2] + camera_distance
    ]
    
    # 设置焦点在模型中心
    plotter.camera.focal_point = center
    
    # 设置更宽的视角，让模型占据更多画面
    plotter.camera.view_angle = 70  # 从60增加到70
    
    # 强制渲染一次，确保相机设置生效
    plotter.render()
    
    # 获取当前视图的边界，进一步调整相机
    view_bounds = plotter.renderer.ComputeVisiblePropBounds()
    if view_bounds:
        # 如果视图边界可用，进一步优化相机位置
        visible_size = max(view_bounds[1] - view_bounds[0], 
                          view_bounds[3] - view_bounds[2], 
                          view_bounds[5] - view_bounds[4])
        
        # 如果可见区域比模型小很多，说明有大量留白，进一步调整
        if visible_size < max_size * 0.7:
            # 进一步缩小相机距离
            camera_distance = max_size * 0.6
            plotter.camera.position = [
                center[0] + camera_distance,
                center[1] + camera_distance, 
                center[2] + camera_distance
            ]
            plotter.camera.view_angle = 75  # 更宽的视角
    
    return plotter


def save_svg_plot(plotter, output_path):
    """
    保存绘图为SVG格式
    
    Args:
        plotter: pyvista绘图器对象
        output_path: 输出文件路径
    """
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为SVG格式
        plotter.save_graphic(output_path)
        print(f"SVG图片已保存: {output_path}")
        
    except Exception as e:
        print(f"保存SVG图片失败: {e}")
        # 如果SVG保存失败，尝试保存为PNG
        png_path = output_path.with_suffix('.png')
        plotter.screenshot(png_path)
        print(f"PNG图片已保存: {png_path}")


def process_single_model(cfpu_model_dir, obj_models_dir, output_figs_dir):
    """
    处理单个模型的patch可视化
    
    Args:
        cfpu_model_dir: CFPU模型数据目录
        obj_models_dir: OBJ模型目录
        output_figs_dir: 输出图片目录
        
    Returns:
        bool: 处理是否成功
    """
    model_name = cfpu_model_dir.name
    print(f"处理模型: {model_name}")
    
    # 查找对应的OBJ模型文件
    obj_path = find_matching_obj_model(model_name, obj_models_dir)
    if not obj_path:
        print(f"  警告: 未找到对应的OBJ模型文件")
        return False
    
    print(f"  找到OBJ模型: {obj_path}")
    
    # 加载OBJ模型
    mesh = load_obj_model(obj_path)
    if not mesh:
        print(f"  错误: 无法加载OBJ模型")
        return False
    
    # 加载patch数据
    patches_data = load_patch_data(cfpu_model_dir)
    if not patches_data:
        print(f"  警告: 无法加载patch数据")
        # 仍然可以可视化网格，只是没有patch点
    
    # 创建可视化
    plotter = create_visualization(mesh, patches_data, model_name)
    
    # 生成输出文件名
    output_filename = f"{model_name}_patches.svg"
    output_path = output_figs_dir / output_filename
    
    # 保存SVG图片
    save_svg_plot(plotter, output_path)
    
    # 关闭绘图器
    plotter.close()
    
    return True


def main():
    """
    主函数：批量处理所有模型的patch可视化
    """
    # 设置路径
    project_root = Path(__file__).resolve().parents[2]
    cfpu_input_dir = project_root / 'output' / 'cfpu_input'
    obj_models_dir = project_root / 'model' / 'obj_model'
    output_figs_dir = project_root / 'output' / 'figs'
    
    print("开始批量处理patch可视化...")
    print(f"CFPU输入目录: {cfpu_input_dir}")
    print(f"OBJ模型目录: {obj_models_dir}")
    print(f"输出图片目录: {output_figs_dir}")
    
    # 检查目录是否存在
    if not cfpu_input_dir.exists():
        print(f"错误: CFPU输入目录不存在: {cfpu_input_dir}")
        return 1
    
    if not obj_models_dir.exists():
        print(f"错误: OBJ模型目录不存在: {obj_models_dir}")
        return 1
    
    # 确保输出目录存在
    output_figs_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有CFPU模型目录
    cfpu_model_dirs = [d for d in cfpu_input_dir.iterdir() if d.is_dir()]
    
    if not cfpu_model_dirs:
        print("警告: 未找到任何CFPU模型目录")
        return 0
    
    print(f"找到 {len(cfpu_model_dirs)} 个CFPU模型目录")
    
    # 处理每个模型
    success_count = 0
    for cfpu_model_dir in cfpu_model_dirs:
        try:
            if process_single_model(cfpu_model_dir, obj_models_dir, output_figs_dir):
                success_count += 1
        except Exception as e:
            print(f"处理模型 {cfpu_model_dir.name} 时出错: {e}")
        print()  # 空行分隔
    
    print(f"处理完成！成功处理 {success_count}/{len(cfpu_model_dirs)} 个模型")
    print(f"SVG图片保存在: {output_figs_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())