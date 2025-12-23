#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式展示CFPU输入文件夹中的patch可视化

该脚本提供了一个交互式界面，让用户可以选择CFPU输入文件夹中的模型，
然后查看对应的网格模型和patch点的3D可视化效果。
用户可以使用鼠标进行旋转、缩放和平移操作，以从不同角度观察模型。
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(".."))


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


def create_interactive_visualization(mesh, patches_data, model_name):
    """
    创建交互式可视化
    
    Args:
        mesh: 网格对象
        patches_data: patch数据
        model_name: 模型名称
        
    Returns:
        plotter: pyvista绘图器对象
    """
    # 创建交互式绘图器（关闭off_screen渲染）
    plotter = pv.Plotter(window_size=[1024, 768])
    
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
                point_size=8,
                name='patches'
            )
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 添加标题
    plotter.add_title(f"模型: {model_name}")
    
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
    
    # 设置相机位置
    camera_distance = max_size * 1.5
    
    # 设置等轴测视角
    plotter.camera_position = 'iso'
    
    # 调整相机到合适的位置
    plotter.camera.position = [
        center[0] + camera_distance,
        center[1] + camera_distance, 
        center[2] + camera_distance
    ]
    
    # 设置焦点在模型中心
    plotter.camera.focal_point = center
    
    # 设置合适的视角
    plotter.camera.view_angle = 60
    
    # 强制渲染一次，确保相机设置生效
    plotter.render()
    
    return plotter


def get_available_models(cfpu_input_dir):
    """
    获取可用的CFPU模型列表
    
    Args:
        cfpu_input_dir: CFPU输入目录
        
    Returns:
        list: 可用模型目录列表
    """
    return [d for d in cfpu_input_dir.iterdir() if d.is_dir()]


def show_model_selection_menu(available_models):
    """
    显示模型选择菜单
    
    Args:
        available_models: 可用模型目录列表
        
    Returns:
        int: 用户选择的模型索引，或-1表示退出
    """
    print("\n" + "="*60)
    print("CFPU模型可视化 - 模型选择菜单")
    print("="*60)
    
    for i, model_dir in enumerate(available_models):
        print(f"{i+1}. {model_dir.name}")
    
    print(f"0. 退出")
    print("="*60)
    
    while True:
        try:
            choice = input("请选择要查看的模型 (0-{}): ".format(len(available_models)))
            choice = int(choice)
            
            if choice == 0:
                return -1
            elif 1 <= choice <= len(available_models):
                return choice - 1  # 转换为0-based索引
            else:
                print(f"无效选择，请输入0到{len(available_models)}之间的数字")
        except ValueError:
            print("无效输入，请输入数字")


def main():
    """
    主函数：交互式展示CFPU模型的patch可视化
    """
    # 设置路径
    project_root = Path(__file__).resolve().parents[1]
    cfpu_input_dir = project_root / 'output' / 'cfpu_input'
    obj_models_dir = project_root / 'model' / 'obj_model'
    
    print("CFPU模型Patch可视化工具")
    print(f"CFPU输入目录: {cfpu_input_dir}")
    print(f"OBJ模型目录: {obj_models_dir}")
    
    # 检查目录是否存在
    if not cfpu_input_dir.exists():
        print(f"错误: CFPU输入目录不存在: {cfpu_input_dir}")
        return 1
    
    if not obj_models_dir.exists():
        print(f"错误: OBJ模型目录不存在: {obj_models_dir}")
        return 1
    
    # 获取所有CFPU模型目录
    available_models = get_available_models(cfpu_input_dir)
    
    if not available_models:
        print("警告: 未找到任何CFPU模型目录")
        return 0
    
    print(f"找到 {len(available_models)} 个CFPU模型")
    
    while True:
        # 显示模型选择菜单
        selected_idx = show_model_selection_menu(available_models)
        
        if selected_idx == -1:
            print("退出程序")
            break
        
        # 获取用户选择的模型目录
        selected_model_dir = available_models[selected_idx]
        model_name = selected_model_dir.name
        
        print(f"\n加载模型: {model_name}")
        
        # 查找对应的OBJ模型文件
        obj_path = find_matching_obj_model(model_name, obj_models_dir)
        if not obj_path:
            print(f"警告: 未找到对应的OBJ模型文件")
            continue
        
        print(f"找到OBJ模型: {obj_path}")
        
        # 加载OBJ模型
        mesh = load_obj_model(obj_path)
        if not mesh:
            print(f"错误: 无法加载OBJ模型")
            continue
        
        # 加载patch数据
        patches_data = load_patch_data(selected_model_dir)
        if not patches_data:
            print(f"警告: 无法加载patch数据")
            # 仍然可以可视化网格，只是没有patch点
        
        # 创建交互式可视化
        print("创建交互式可视化...")
        print("使用鼠标可以进行旋转、缩放和平移操作")
        print("按'q'键退出当前可视化")
        
        plotter = create_interactive_visualization(mesh, patches_data, model_name)
        
        # 启动交互式可视化
        plotter.show()
        
        # 关闭绘图器
        plotter.close()


if __name__ == "__main__":
    exit(main())
