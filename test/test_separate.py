import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_plane_mesh(normal, origin, range_val=1.0, grid_density=2):
    """
    生成经过 origin 点，法向为 normal 的平面网格数据 (X, Y, Z)
    """
    normal = normalize(normal)
    
    # 1. 寻找平面上的两个正交基向量 u, v
    # 防止 normal 与参考向量平行导致叉乘为0
    if np.abs(normal[2]) < 0.9:
        ref_vec = np.array([0, 0, 1])
    else:
        ref_vec = np.array([0, 1, 0])
        
    u = normalize(np.cross(normal, ref_vec))
    v = normalize(np.cross(normal, u))
    
    # 2. 生成网格
    # 生成 4 个角点即可，plot_surface 会自动填充，减少计算量，视觉更清爽
    grid_u = np.linspace(-range_val, range_val, grid_density)
    grid_v = np.linspace(-range_val, range_val, grid_density)
    U, V = np.meshgrid(grid_u, grid_v)
    
    # 3. 计算坐标 P = Origin + U*u + V*v
    X = origin[0] + U * u[0] + V * v[0]
    Y = origin[1] + U * u[1] + V * v[1]
    Z = origin[2] + U * u[2] + V * v[2]
    
    return X, Y, Z

def calculate_offset_logic(p_origin, n_target, n_others, offset_distance=0.5):
    """
    计算逻辑：将其他法向量的合力，投影到当前法向量的切平面上，取反方向偏移
    """
    n1 = normalize(n_target)
    n_sum = np.sum(n_others, axis=0)
    
    # 投影: v_proj = v - (v . n) * n
    dist = np.dot(n_sum, n1)
    v_proj = n_sum - dist * n1
    
    # 取反方向
    if np.linalg.norm(v_proj) > 1e-6:
        v_offset_dir = normalize(-v_proj)
    else:
        v_offset_dir = np.array([0, 0, 0])
        
    p_offset = p_origin + v_offset_dir * offset_distance
    return p_offset, v_offset_dir

def visualize_all_planes():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. 设置场景数据 ---
    origin = np.array([0.0, 0.0, 0.0])
    
    # 模拟立方体角点的三个法向 (分别对应 X, Y, Z 轴)
    # 你可以修改这些值来测试非正交情况
    n_list = [
        normalize(np.array([1.0, 1.0, 0.0])), # Red 组
        normalize(np.array([0.0, 1.0, 0.25])), # Green 组
        normalize(np.array([0.5, 0.0, 1.0]))  # Blue 组
    ]
    
    colors = ['red', 'green', 'blue']
    labels = ['Normal X', 'Normal Y', 'Normal Z']
    
    # --- 2. 绘制原点 ---
    ax.scatter(*origin, color='k', s=200, label='Singular Vertex (Origin)', zorder=10)

    # --- 3. 循环处理每一组 (法向 + 平面 + 偏移点) ---
    for i in range(3):
        target_n = n_list[i]
        color = colors[i]
        
        # 找出"其他"法向量
        others = [n for j, n in enumerate(n_list) if j != i]
        
        # A. 计算偏移
        p_offset, v_dir = calculate_offset_logic(origin, target_n, others, offset_distance=0.5)
        
        # B. 绘制法向量 (实线箭头)
        ax.quiver(*origin, *target_n, color=color, length=1.2, arrow_length_ratio=0.1, linewidth=2)
        
        # C. 绘制切平面 (半透明面)
        # 这里的 range_val 决定面画多大
        PX, PY, PZ = get_plane_mesh(target_n, origin, range_val=1.0)
        ax.plot_surface(PX, PY, PZ, color=color, alpha=0.15, edgecolor=color, linewidth=0.5)
        
        # D. 绘制偏移轨迹 (虚线)
        ax.plot([origin[0], p_offset[0]], 
                [origin[1], p_offset[1]], 
                [origin[2], p_offset[2]], 
                color=color, linestyle='--', linewidth=1.5)
        
        # E. 绘制最终分裂点 (同色星星)
        ax.scatter(*p_offset, color=color, marker='*', s=300, edgecolor='k', label=f'Constraint {i+1}', zorder=20)
        
        print(f"[{labels[i]}] Offset Vector: {np.round(v_dir, 3)} -> Point: {np.round(p_offset, 3)}")

    # --- 4. 图形设置 ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置视口范围
    limit = 1.2
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    # 设置等轴视图
    ax.set_aspect('equal')
    
    ax.set_title("Full Visualization: 3-Way Vertex Splitting via Tangent Projection")
    
    # 视角调整 (方便观察立体关系)
    ax.view_init(elev=25, azim=45)
    
    # 手动图例
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([], [], color='k', marker='o', linestyle='None', label='Original Vertex'),
        mlines.Line2D([], [], color='red', marker='*', linestyle='-', label='Group 1 (Plane+Point)'),
        mlines.Line2D([], [], color='green', marker='*', linestyle='-', label='Group 2 (Plane+Point)'),
        mlines.Line2D([], [], color='blue', marker='*', linestyle='-', label='Group 3 (Plane+Point)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.show()

if __name__ == "__main__":
    visualize_all_planes()