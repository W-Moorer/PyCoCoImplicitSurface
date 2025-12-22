#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFPU 补丁交互可视化示例 - 点击补丁节点可视化其半径、nodes和normals
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import argparse


def main():
    """
    演示如何点击Patch节点，可视化该patch对应的半径以及半径内用于CFRBF拟合计算的node和node对应的normal
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfpu_dir', type=str, 
                   default='output\cfpu_input\zuheti_surface_cellnormals_cfpu_input',
                   help='CFPU输入数据目录')
    ap.add_argument('--recon_dir', type=str, 
                   default='output\zuheti_surface_cellnormals_cfpu_recon',
                   help='CFPU重建结果输出目录')
    args = ap.parse_args()
    
    # 设置工作目录 - 脚本在demo文件夹，需要上一级目录作为根目录
    root = Path(__file__).resolve().parents[1]
    cfpu_dir = root / args.cfpu_dir
    recon_dir = root / args.recon_dir
    
    print(f"读取CFPU数据文件...")
    
    # 读取CFPU输入数据
    nodes_path = cfpu_dir / 'nodes.txt'
    normals_path = cfpu_dir / 'normals.txt'
    patches_path = cfpu_dir / 'patches.txt'
    radii_path = cfpu_dir / 'radii.txt'
    
    # 读取重建结果
    potential_path = recon_dir / 'potential.npy'
    grid_path = recon_dir / 'grid.npz'
    
    # 检查文件是否存在
    required_files = [nodes_path, normals_path, patches_path, radii_path, potential_path, grid_path]
    for file_path in required_files:
        if not file_path.exists():
            print(f"错误：文件不存在于 {file_path}")
            print("请先运行demo_cfpu_recon.py生成CFPU数据和重建结果")
            return 1
    
    # 加载数据
    nodes = np.loadtxt(nodes_path)
    normals = np.loadtxt(normals_path)
    patches = np.loadtxt(patches_path)
    radii = np.loadtxt(radii_path)
    
    potential = np.load(potential_path)
    grid_data = np.load(grid_path)
    X = grid_data['X']
    Y = grid_data['Y']
    Z = grid_data['Z']
    
    print(f"读取完成！")
    print(f"节点数量: {nodes.shape[0]}")
    print(f"法向量数量: {normals.shape[0]}")
    print(f"补丁数量: {patches.shape[0]}")
    print(f"半径数量: {radii.shape[0]}")
    print(f"潜在场形状: {potential.shape}")
    print(f"网格形状: {X.shape}")
    
    # 使用pyvista进行可视化
    print(f"创建可视化...")
    
    # 创建结构化网格
    sg = pv.StructuredGrid(X, Y, Z)
    sg['potential'] = potential.ravel(order='F')
    
    # 提取等值面
    iso = sg.contour(isosurfaces=[0.0])
    
    # 创建可视化窗口 - 只显示一个窗口
    plotter = pv.Plotter()
    
    # 添加等值面
    plotter.add_mesh(iso, color='lightgray', specular=0.1, smooth_shading=True, opacity=0.85, name='isosurface')
    
    # 添加补丁点
    # 先创建补丁点云对象
    patch_cloud = pv.PolyData(patches)
    patch_cloud['radii'] = radii
    
    # 初始颜色为红色
    patch_cloud['colors'] = np.zeros(len(patches))
    
    # 添加补丁点到可视化窗口
    plotter.add_mesh(
        patch_cloud, 
        scalars='colors',
        render_points_as_spheres=True, 
        point_size=12, 
        cmap='coolwarm',  # 使用颜色映射代替固定颜色
        name='patches',
        show_scalar_bar=False
    )
    
    # 初始化高亮对象
    highlighted_patch = None
    patch_radius_actor = None
    nodes_in_patch = None
    normals_in_patch = None
    
    # 创建节点和法向量的PolyData对象
    nodes_cloud = pv.PolyData(nodes)
    nodes_cloud['normals'] = normals
    
    # 添加底部提示
    plotter.add_text("左键点击补丁节点查看详细信息，按 'c' 键清除高亮", position='lower_left')
    
    # 添加坐标轴
    plotter.add_axes()
    
    # 添加标题
    plotter.add_title("CFPU 补丁交互可视化演示")
    
    def clear_highlight():
        """
        清除所有高亮显示
        """
        nonlocal highlighted_patch, patch_radius_actor, nodes_in_patch, normals_in_patch
        
        # 重置补丁颜色
        patch_cloud['colors'] = np.zeros(len(patches))
        
        # 移除所有高亮相关的actors
        actors_to_remove = []
        
        # 移除半径球体
        if patch_radius_actor is not None:
            actors_to_remove.append(patch_radius_actor)
            patch_radius_actor = None
        
        # 移除节点
        if nodes_in_patch is not None:
            actors_to_remove.append(nodes_in_patch)
            nodes_in_patch = None
        
        # 移除法向量
        if normals_in_patch is not None:
            actors_to_remove.append(normals_in_patch)
            normals_in_patch = None
        
        # 执行移除操作
        for actor in actors_to_remove:
            try:
                plotter.remove_actor(actor)
            except Exception as e:
                print(f"移除actor时出错: {e}")
        
        highlighted_patch = None
        # 重新渲染整个场景
        plotter.render()
        print("已清除所有高亮")
    
    def show_patch_details(patch_id):
        """
        显示补丁的详细信息
        """
        nonlocal highlighted_patch, patch_radius_actor, nodes_in_patch, normals_in_patch
        
        # 清除之前的高亮
        clear_highlight()
        
        # 高亮当前补丁
        patch_cloud['colors'] = np.zeros(len(patches))
        patch_cloud['colors'][patch_id] = 1.0
        highlighted_patch = patch_id
        
        # 重新添加补丁点云以更新颜色
        plotter.remove_actor('patches')
        plotter.add_mesh(
            patch_cloud, 
            scalars='colors',
            render_points_as_spheres=True, 
            point_size=12, 
            cmap='coolwarm',
            name='patches',
            show_scalar_bar=False
        )
        
        # 获取补丁信息
        patch_center = patches[patch_id]
        patch_radius = radii[patch_id]
        
        print(f"\n=== 补丁 {patch_id} 详细信息 ===")
        print(f"中心坐标: {patch_center}")
        print(f"半径: {patch_radius:.6f}")
        
        try:
            # 可视化补丁半径 - 创建球体
            print(f"创建补丁半径球体: 中心={patch_center}, 半径={patch_radius}")
            sphere = pv.Sphere(radius=patch_radius, center=patch_center, theta_resolution=20, phi_resolution=20)
            patch_radius_actor = plotter.add_mesh(
                sphere, 
                color='red', 
                opacity=0.3, 
                name=f'patch_radius_{patch_id}',
                show_edges=True, 
                edge_color='red',
                ambient=0.5
            )
            
            # 找到半径内的节点
            # 计算所有节点到补丁中心的距离
            distances = np.linalg.norm(nodes - patch_center, axis=1)
            in_radius_indices = np.where(distances <= patch_radius)[0]
            
            print(f"半径内节点索引数量: {len(in_radius_indices)}")
            
            # 收集所有需要显示的节点和法向量
            # 支持一个节点有多个法向量的情况
            all_nodes = []
            all_normals = []
            
            for i in in_radius_indices:
                node = nodes[i]
                normal = normals[i]
                
                # 添加当前节点和法向量
                all_nodes.append(node)
                all_normals.append(normal)
                
                # 检查是否存在其他具有相同坐标的节点（即一个节点有多个法向量）
                # 这里使用一个小的容差来比较节点坐标
                same_nodes = np.where(np.linalg.norm(nodes - node, axis=1) < 1e-10)[0]
                
                for j in same_nodes:
                    if j != i:  # 跳过自身
                        # 添加相同节点的不同法向量
                        all_nodes.append(nodes[j])
                        all_normals.append(normals[j])
            
            # 转换为numpy数组
            all_nodes = np.array(all_nodes)
            all_normals = np.array(all_normals)
            
            print(f"半径内节点-法向量对数量: {len(all_nodes)}")
            
            # 显示半径内的节点
            if len(all_nodes) > 0:
                # 创建节点点云
                inside_cloud = pv.PolyData(all_nodes)
                nodes_in_patch = plotter.add_mesh(
                    inside_cloud, 
                    color='blue', 
                    render_points_as_spheres=True, 
                    point_size=10, 
                    name=f'nodes_in_patch_{patch_id}',
                    ambient=0.5
                )
                
                # 显示法向量
                # 缩放法向量以便更好地可视化
                scaled_normals = all_normals * 0.0001  # 按照用户要求使用0.0001的缩放因子
                normals_in_patch = plotter.add_arrows(
                    all_nodes, 
                    scaled_normals, 
                    color='green', 
                    line_width=3, 
                    name=f'normals_in_patch_{patch_id}',
                    ambient=0.5
                )
        except Exception as e:
            print(f"可视化补丁细节时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 刷新渲染
        plotter.render()
    
    # PyVista 0.46.4版本的callback直接接收picked point IDs的numpy数组
    def patch_picking_callback(picked_point):
        """
        补丁拾取回调函数
        注意：PyVista 0.46.4中，callback收到的是拾取点的三维坐标，而不是点的索引ID
        """
        try:
            print(f"收到拾取事件，拾取点坐标: {picked_point}")
            print(f"拾取点类型: {type(picked_point)}")
            
            # 检查是否为三维坐标
            if len(picked_point) == 3:
                # 计算该坐标与所有补丁点的距离
                distances = np.linalg.norm(patches - picked_point, axis=1)
                
                # 找到距离最近的补丁点索引
                patch_id = np.argmin(distances)
                min_distance = distances[patch_id]
                
                print(f"最近的补丁ID: {patch_id}, 距离: {min_distance}")
                
                # 只有当距离小于一定阈值时才视为有效拾取
                if min_distance < 0.01:  # 阈值可以根据实际情况调整
                    show_patch_details(patch_id)
                else:
                    print(f"拾取点距离所有补丁点太远，距离: {min_distance}")
            else:
                print(f"收到无效的拾取数据: {picked_point}")
        except Exception as e:
            print(f"处理拾取点时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 启用点拾取，针对补丁点云
    # 移除之前的回调，使用更简单直接的方式
    plotter.enable_point_picking(
        callback=patch_picking_callback,
        show_message=True,
        picker='point',
        show_point=False,
        color='yellow',
        point_size=15,
        tolerance=0.05,  # 增大公差，提高拾取成功率
        # 使用不同的点拾取器配置
        style='surface',
        reset_camera=False
    )
    
    # 添加键盘事件
    plotter.add_key_event('c', clear_highlight)
    
    # 显示可视化窗口
    plotter.show()
    
    return 0


if __name__ == "__main__":
    exit(main())
