# PyCoCoImplicitSurface

PyCoCoImplicitSurface是一个基于Python的曲面重建库，专注于使用无旋多项式（Curl-Free Polynomial，CFPU）方法进行高质量曲面重建。该库提供了完整的工作流程，包括网格处理、尖锐边缘检测、片段构建和CFPU重建。

## 功能特点

- **网格处理**：支持读取、处理各种格式的网格文件
- **尖锐边缘检测**：自动检测网格中的尖锐边缘
- **尖锐连接点检测**：识别尖锐边缘的连接点
- **边缘片段构建**：将尖锐边缘组织为片段，包含间断点信息
- **CFPU重建**：使用无旋多项式进行高质量曲面重建
- **可视化工具**：提供多种可视化选项，便于结果分析

## 依赖安装

### 基本依赖

```bash
pip install numpy scipy pyvista
```

### 可选依赖

- **CuPy**：用于GPU加速（可选）
- **threadpoolctl**：用于线程控制（可选）

```bash
pip install cupy threadpoolctl
```

## 项目结构

```
PyCoCoImplicitSurface/
├── demo/              # 演示脚本
├── input/             # 输入数据
├── main/              # 主要执行脚本
├── src/               # 核心源码
│   ├── cfpurecon.py   # CFPU重建核心实现
│   └── precompute.py  # 预处理功能（边缘检测、片段构建等）
├── visualize/         # 可视化脚本
├── README.md          # 项目说明文档
└── LICENSE            # 许可证文件
```

## 核心模块

### precompute.py

提供网格预处理功能：
- `read_mesh()`：读取网格文件
- `detect_sharp_edges()`：检测尖锐边缘
- `detect_sharp_junctions_degree()`：检测尖锐连接点
- `build_sharp_segments()`：构建尖锐边缘片段
- `build_cfpu_input()`：构建CFPU重建输入数据

### cfpurecon.py

提供CFPU重建功能：
- `curlfree_poly()`：计算无旋多项式基函数
- `cfpurecon()`：执行CFPU曲面重建
- `configure_patch_radii()`：配置补丁半径

## 使用方法

### 1. 预处理：构建CFPU输入

```bash
python main/main_build_cfpu_input.py --inputs input/your_mesh.vtp --out_root output/your_output
```

### 2. 执行CFPU重建

```bash
python demo/demo_cfpu_recon.py
```

### 3. 可视化结果

```bash
python visualize/visualize_cfpu_input.py --path output/your_output
```

## 示例

### 生成尖锐边缘片段

```bash
python demo/demo_generate_sharp_segments.py
```

### 可视化尖锐边缘和连接点

```bash
python visualize/visualize_sharp_edges_junctions.py --input input/your_mesh.vtp
```

### 齿轮网格示例

```bash
python visualize/visualize_gear_sharp_edges.py
```

## 命令行参数

### main_build_cfpu_input.py

- `--inputs`：输入网格文件路径（多个文件用空格分隔）
- `--out_root`：输出目录根路径
- `--angle_threshold`：尖锐边缘检测角度阈值（默认30.0）
- `--r_small_factor`：小半径因子（默认0.5）
- `--r_large_factor`：大半径因子（默认3.0）
- `--edge_split_threshold`：边缘分割阈值（默认None）
- `--require_step_face_id_diff`：是否需要step_face_id差异
- `--sharp_edges_pkl`：尖锐边缘 pickle 文件路径

### demo_cfpu_recon.py

该脚本使用预配置的输入和输出路径，直接运行即可。

## 输入数据格式

支持多种网格格式，包括：
- .vtp（Visualization Toolkit PolyData）
- .vtk（Visualization Toolkit）
- .obj（Wavefront OBJ）
- .stl（Standard Tessellation Language）

## 输出结果

- CFPU输入数据（nodes.txt, normals.txt, patches.txt, radii.txt等）
- 尖锐边缘信息（sharp_edges.pkl）
- 尖锐边缘片段（sharp_segments_debug.json）
- 重建后的潜在场（potential.npy）
- 可视化结果

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过GitHub Issues提交。

---

**PyCoCoImplicitSurface** - 高质量曲面重建库