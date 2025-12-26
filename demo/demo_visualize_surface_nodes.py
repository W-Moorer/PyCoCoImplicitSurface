#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化表面节点和Patch点

该脚本用于可视化CFPU输入数据中的表面节点和Patch点，
显示所有节点坐标并将Patch点用红色高亮标记。
"""

# =============================================================================
# 全局配置变量
# 可以直接在此修改默认参数，无需使用命令行参数
# =============================================================================
DEFAULT_CFPU_DIR = r'output\cfpu_input\LinkedGear_surface_cellnormals_cfpu_input'  # 直接指定CFPU输入数据目录（空字符串表示使用菜单选择）
DEFAULT_MODEL_NAME = 'LinkedGear'  # 直接指定模型名称（空字符串表示使用菜单选择）
DEFAULT_SHOW_MESH = False  # 是否默认显示原始OBJ网格
DEFAULT_RECON_DIR = 'output/LinkedGear_recon_GPU'  # 直接指定重建结果目录（空字符串表示自动查找）
DEFAULT_ISOSURFACE = 0.0  # 等值面值，默认为0.0
DEFAULT_OPACITY = 0.7  # 重建曲面不透明度
# =============================================================================

import numpy as np
import pyvista as pv
from pathlib import Path
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(".."))


def load_node_data(cfpu_dir):
    """
    加载表面节点数据
    
    Args:
        cfpu_dir: CFPU输入数据目录
        
    Returns:
        np.ndarray: 节点坐标数组 (N x 3)
    """
    try:
        nodes_path = cfpu_dir / 'nodes.txt'
        if not nodes_path.exists():
            print(f"错误：节点文件不存在于 {nodes_path}")
            return None
            
        nodes = np.loadtxt(nodes_path)
        print(f"成功加载 {len(nodes)} 个节点")
        return nodes
    except Exception as e:
        print(f"加载节点数据失败: {e}")
        return None


def load_patch_data(cfpu_dir):
    """
    加载Patch点数据
    
    Args:
        cfpu_dir: CFPU输入数据目录
        
    Returns:
        np.ndarray: Patch点坐标数组 (M x 3)
    """
    try:
        patches_path = cfpu_dir / 'patches.txt'
        if not patches_path.exists():
            print(f"错误：Patch文件不存在于 {patches_path}")
            return None
            
        patches = np.loadtxt(patches_path)
        print(f"成功加载 {len(patches)} 个Patch点")
        return patches
    except Exception as e:
        print(f"加载Patch数据失败: {e}")
        return None


def load_recon_data(recon_dir):
    """
    加载重建结果数据
    
    Args:
        recon_dir: 重建结果目录
        
    Returns:
        tuple: (potential, X, Y, Z) 或 None
    """
    try:
        potential_path = recon_dir / 'potential.npy'
        grid_path = recon_dir / 'grid.npz'
        
        if not potential_path.exists() or not grid_path.exists():
            print(f"错误：重建结果文件不存在于 {recon_dir}")
            return None
            
        # 加载重建数据
        potential = np.load(potential_path)
        grid_data = np.load(grid_path)
        X = grid_data['X']
        Y = grid_data['Y']
        Z = grid_data['Z']
        
        print(f"成功加载重建结果！")
        print(f"  潜在场形状: {potential.shape}")
        print(f"  网格形状: {X.shape}")
        
        return potential, X, Y, Z
    except Exception as e:
        print(f"加载重建结果失败: {e}")
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


def create_visualization(nodes, patches, mesh=None, model_name="", recon_data=None, isosurface=0.0, opacity=0.7):
    """
    创建交互式可视化
    
    Args:
        nodes: 节点坐标数组
        patches: Patch点坐标数组
        mesh: 可选的网格对象
        model_name: 模型名称
        recon_data: 重建结果数据 (potential, X, Y, Z)
        isosurface: 等值面值
        opacity: 重建曲面不透明度
        
    Returns:
        plotter: pyvista绘图器对象
    """
    # 创建交互式绘图器
    plotter = pv.Plotter(window_size=[1200, 900])
    
    # 设置背景色为白色
    plotter.set_background('white')
    
    # 移除所有文字元素（标题、提示等）
    
    # 添加重建后的等值面（如果提供了重建数据）
    if recon_data:
        potential, X, Y, Z = recon_data
        
        # 创建结构化网格并提取等值面
        sg = pv.StructuredGrid(X, Y, Z)
        sg['potential'] = potential.ravel(order='F')
        recon_surface = sg.contour(isosurfaces=[isosurface])
        
        # 添加等值面
        plotter.add_mesh(
            recon_surface, 
            color='lightblue', 
            specular=0.3, 
            smooth_shading=True, 
            opacity=opacity
        )
    
    # 添加原始网格（如果提供）
    if mesh:
        plotter.add_mesh(
            mesh,
            color='lightgray',
            specular=0.1,
            smooth_shading=True,
            opacity=0.5,
            show_edges=True,
            edge_color='gray',
            name='mesh'
        )
    
    # 添加所有节点（蓝色小点）
    if nodes is not None:
        node_cloud = pv.PolyData(nodes)
        plotter.add_mesh(
            node_cloud,
            color='blue',
            render_points_as_spheres=True,
            point_size=4,
            name='nodes'
        )
    
    # 添加Patch点（红色大点）
    if patches is not None:
        patch_cloud = pv.PolyData(patches)
        plotter.add_mesh(
            patch_cloud,
            color='red',
            render_points_as_spheres=True,
            point_size=8,
            name='patches'
        )
    
    # 设置初始视角为等轴测视图
    plotter.view_isometric()
    
    # 设置相机位置
    if nodes is not None:
        # 计算节点的边界框
        bounds = np.array([
            [nodes[:, 0].min(), nodes[:, 0].max()],
            [nodes[:, 1].min(), nodes[:, 1].max()],
            [nodes[:, 2].min(), nodes[:, 2].max()]
        ])
        
        # 计算中心点
        center = bounds.mean(axis=1)
        
        # 计算最大尺寸
        max_size = (bounds[:, 1] - bounds[:, 0]).max()
        
        # 设置相机位置
        camera_distance = max_size * 2
        plotter.camera.position = [
            center[0] + camera_distance,
            center[1] + camera_distance,
            center[2] + camera_distance
        ]
        plotter.camera.focal_point = center
    
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
    print("表面节点和Patch点可视化 - 模型选择菜单")
    print("="*60)
    
    for i, model_dir in enumerate(available_models):
        print(f"{i+1}. {model_dir.name}")
    
    print(f"0. 退出")
    print("="*60)
    
    while True:
        try:
            choice = input(f"请选择要查看的模型 (0-{len(available_models)}): ")
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
    主函数：可视化表面节点和Patch点
    """
    # 解析命令行参数
    ap = argparse.ArgumentParser(description='表面节点和Patch点可视化工具')
    ap.add_argument('--cfpu_dir', type=str, default=DEFAULT_CFPU_DIR,
                   help='直接指定CFPU输入数据目录（跳过菜单选择）')
    ap.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                   help='直接指定模型名称（用于查找CFPU输入目录）')
    ap.add_argument('--show_mesh', action='store_true',
                   help='显示原始OBJ网格')
    ap.add_argument('--recon_dir', type=str, default=DEFAULT_RECON_DIR,
                   help='直接指定重建结果目录（跳过自动查找）')
    ap.add_argument('--isosurface', type=float, default=DEFAULT_ISOSURFACE,
                   help='等值面值')
    ap.add_argument('--opacity', type=float, default=DEFAULT_OPACITY,
                   help='重建曲面不透明度')
    args = ap.parse_args()
    
    # 使用命令行参数覆盖全局配置（如果提供）
    cfpu_dir_arg = args.cfpu_dir
    model_name_arg = args.model_name
    show_mesh_arg = args.show_mesh
    recon_dir_arg = args.recon_dir
    isosurface_arg = args.isosurface
    opacity_arg = args.opacity
    
    # 设置路径
    project_root = Path(__file__).resolve().parents[1]
    cfpu_input_dir = project_root / 'output' / 'cfpu_input'
    obj_models_dir = project_root / 'model' / 'obj_model'
    
    print("表面节点和Patch点可视化工具")
    print(f"CFPU输入目录: {cfpu_input_dir}")
    if args.show_mesh:
        print(f"OBJ模型目录: {obj_models_dir}")
    
    # 检查CFPU输入目录是否存在
    if not cfpu_input_dir.exists():
        print(f"错误: CFPU输入目录不存在: {cfpu_input_dir}")
        return 1
    
    # 确定要使用的CFPU模型目录
    selected_model_dir = None
    
    if cfpu_dir_arg:
        # 直接使用命令行指定的目录
        selected_model_dir = Path(cfpu_dir_arg)
        if not selected_model_dir.exists():
            print(f"错误: 指定的CFPU目录不存在: {selected_model_dir}")
            return 1
    elif model_name_arg:
        # 根据模型名称查找目录
        model_dir_path = cfpu_input_dir / model_name_arg
        if model_dir_path.exists():
            selected_model_dir = model_dir_path
        else:
            print(f"错误: 未找到模型目录: {model_dir_path}")
            return 1
    else:
        # 获取所有CFPU模型目录
        available_models = get_available_models(cfpu_input_dir)
        
        if not available_models:
            print("警告: 未找到任何CFPU模型目录")
            return 0
        
        print(f"找到 {len(available_models)} 个CFPU模型")
        
        # 显示模型选择菜单
        selected_idx = show_model_selection_menu(available_models)
        
        if selected_idx == -1:
            print("退出程序")
            return 0
        
        selected_model_dir = available_models[selected_idx]
    
    model_name = selected_model_dir.name
    print(f"\n加载模型: {model_name}")
    
    # 加载节点数据
    nodes = load_node_data(selected_model_dir)
    if nodes is None:
        print("无法继续：没有节点数据")
        return 1
    
    # 加载Patch数据
    patches = load_patch_data(selected_model_dir)
    if patches is None:
        print("警告：没有Patch数据，将只显示节点")
    
    # 加载OBJ模型（如果需要）
    mesh = None
    if show_mesh_arg:
        obj_path = find_matching_obj_model(model_name, obj_models_dir)
        if obj_path:
            print(f"找到OBJ模型: {obj_path}")
            mesh = load_obj_model(obj_path)
            if not mesh:
                print(f"警告: 无法加载OBJ模型")
        else:
            print(f"警告: 未找到对应的OBJ模型文件")
    
    # 加载重建结果
    recon_data = None
    recon_dir = None
    
    if recon_dir_arg:
        # 直接使用命令行指定的重建结果目录
        recon_dir = Path(recon_dir_arg)
        if not recon_dir.exists():
            print(f"警告: 指定的重建结果目录不存在: {recon_dir}")
        else:
            recon_data = load_recon_data(recon_dir)
    else:
        # 自动查找重建结果目录
        # 从模型名称中提取基本名称
        base_name = model_name.replace('_surface_cellnormals_cfpu_input', '')
        
        # 可能的重建结果目录名称
        potential_recon_dirs = [
            project_root / 'output' / f'{base_name}_recon_GPU',
            project_root / 'output' / f'{base_name}_recon_CPU'
        ]
        
        for dir_path in potential_recon_dirs:
            if dir_path.exists():
                recon_dir = dir_path
                break
        
        if recon_dir:
            print(f"自动找到重建结果目录: {recon_dir}")
            recon_data = load_recon_data(recon_dir)
        else:
            print(f"警告: 未找到重建结果目录")
            print(f"已尝试的目录: {[str(d) for d in potential_recon_dirs]}")
            print("将只显示节点和Patch点")
    
    # 创建可视化
    print("创建交互式可视化...")
    plotter = create_visualization(nodes, patches, mesh, model_name, recon_data, isosurface_arg, opacity_arg)
    
    # 启动交互式可视化
    plotter.show()
    
    # 关闭绘图器
    plotter.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
