import numpy as np
import pickle
import pyvista as pv
import sys
from bspline_fitting import fit_spline_points

# 定义文件路径
vtp_file = r'input\leftGear_surface_cellnormals_segmented.vtp'
sharp_edges_file = r'input\leftGear_surface_cellnormals_segmented_sharp_edges.pkl'

class GearSharpEdgeVisualizer:
    def __init__(self):
        # 读取模型和数据
        self.mesh = pv.read(vtp_file)
        
        # 读取尖锐边数据
        with open(sharp_edges_file, 'rb') as f:
            self.sharp_edges_data = pickle.load(f)
        
        self.sharp_edges = self.sharp_edges_data['sharp_edges']
        self.num_edges = len(self.sharp_edges)
        
        # 当前显示的边索引
        self.current_edge_idx = 0
        
        # 创建可视化窗口
        self.plotter = pv.Plotter()
        
        # 初始化可视化
        self._initialize_visualization()
        
        # 显示窗口
        self.plotter.show()
    
    def _initialize_visualization(self):
        # 添加齿轮模型
        self.plotter.add_mesh(self.mesh, color='lightgray', opacity=0.7, show_edges=True)
        
        # 添加尖锐边
        self.edge_actor = self.plotter.add_mesh(
            pv.PolyData(), 
            color='red', 
            line_width=3,
            name='current_edge'
        )
        
        # 添加边编号文本
        self.text_actor = self.plotter.add_text(
            f"尖锐边编号: {self.current_edge_idx + 1}/{self.num_edges}",
            position='lower_left',
            color='white',
            font_size=12,
            name='edge_number'
        )
        
        # 添加使用说明
        self.plotter.add_text(
            "左右键切换尖锐边 | ESC退出",
            position='upper_right',
            color='white',
            font_size=10,
            name='instructions'
        )
        
        # 设置回调
        self.plotter.add_key_event('Left', self._previous_edge)
        self.plotter.add_key_event('Right', self._next_edge)
        
        # 显示初始边
        self._update_edge_display()
    
    def _get_edge_points(self, edge_idx):
        """
        获取指定边的点坐标
        """
        edge = self.sharp_edges[edge_idx]
        point1_idx = edge['point1_idx']
        point2_idx = edge['point2_idx']
        
        # 获取点坐标
        point1 = self.mesh.points[point1_idx]
        point2 = self.mesh.points[point2_idx]
        
        # 创建点数组
        edge_points = np.array([point1, point2])
        
        return edge_points
    
    def _fit_edge_spline(self, edge_points):
        """
        使用B-spline拟合边缘
        """
        _, fine_curve = fit_spline_points(edge_points, smooth_factor=0, num_samples=100)
        return fine_curve
    
    def _update_edge_display(self):
        """
        更新当前显示的边
        """
        # 获取当前边的点
        edge_points = self._get_edge_points(self.current_edge_idx)
        
        # 拟合B-spline曲线
        spline_points = self._fit_edge_spline(edge_points)
        
        # 创建线几何
        line = pv.PolyData(spline_points)
        line.lines = np.array([len(spline_points)] + list(range(len(spline_points))))
        
        # 更新边显示
        self.plotter.update_mesh(line, name='current_edge')
        
        # 更新文本
        self.plotter.add_text(
            f"尖锐边编号: {self.current_edge_idx + 1}/{self.num_edges}",
            position='lower_left',
            color='white',
            font_size=12,
            name='edge_number',
            replace=True
        )
    
    def _previous_edge(self, event=None):
        """
        显示前一条边
        """
        self.current_edge_idx = (self.current_edge_idx - 1) % self.num_edges
        self._update_edge_display()
    
    def _next_edge(self, event=None):
        """
        显示后一条边
        """
        self.current_edge_idx = (self.current_edge_idx + 1) % self.num_edges
        self._update_edge_display()

if __name__ == "__main__":
    print("=== 齿轮尖锐边可视化 ===")
    print(f"尖锐边数量: {len(pickle.load(open(sharp_edges_file, 'rb'))['sharp_edges'])}")
    print("使用左右键切换尖锐边")
    print("按ESC退出")
    print()
    
    # 启动可视化
    visualizer = GearSharpEdgeVisualizer()