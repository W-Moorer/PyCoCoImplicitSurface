# Todo
### **Section 5. Experiments and Results (实验大纲)**

#### **5.1. Experimental Setup (实验设置)**

* **硬件环境**：明确列出 CPU (e.g., Intel i9) 和 GPU (e.g., NVIDIA RTX 3090) 的具体型号，以及内存大小。这对于证明 `cfpurecon.py` 中 GPU 加速和 Shared Memory 的有效性至关重要。
* **数据集**：
* **合成数据集 (Synthetic CAD Models)**：使用具有已知解析解的 CAD 模型（如你的 `leftGear` 齿轮模型、立方体、圆柱组合体），用于精确计算误差。
* **真实扫描数据 (Real-world Scans)**：使用 Stanford Bunny 或 Armadillo 等标准模型，测试对噪声的鲁棒性。
* **大规模数据 (Massive Point Clouds)**：准备一个百万级点云数据，专门用于测试 Shared Memory 和 GPU 的吞吐量。


* **评价指标 (Metrics)**：
* **几何精度**：Hausdorff Distance (豪斯多夫距离), Mean Squared Error (MSE), Normal Deviation (法向偏差)。
* **性能指标**：Runtime (秒), Peak Memory Usage (GB), Speedup Ratio (加速比)。



---

#### **5.2. Reconstruction Quality & Feature Preservation (重建质量与特征保持)**

* **目的**：证明你的“预处理策略（尖锐边分离）”比原始 CFPU 和其他经典方法更能恢复 CAD 模型的棱角。
* **对比对象**：
1. Baseline (Original CFPU [Drake et al. 2022]): 使用原始 k-NN 半径策略，无尖锐点分离。
2. Screened Poisson Reconstruction (SPR): 工业界最常用的基准。
3. **Ours (Proposed Method)**: 启用 `detect_sharp_edges` 和法向分离。


* **核心展示内容**：
* **齿轮/机械件对比图**：展示 `leftGear` 的齿顶和齿根。Baseline 可能会把齿尖磨圆（Over-smoothing），而你的方法应该呈现锐利的边缘。
* **误差热力图 (Error Heatmap)**：将重建曲面与原始 CAD 模型对齐，用颜色编码误差。重点展示在尖锐边缘处，你的误差显著低于 Baseline。


* **代码支撑**：引用 `precompute.py` 中 `detect_sharp_edges` 和 `build_cfpu_input` 里的法向量偏移逻辑。

---

#### **5.3. Performance Analysis (性能分析)**

* **目的**：这是你最大的工程贡献，必须用硬数据说话。
* **实验 A: 加速比分析 (Acceleration Study)**
* 绘制柱状图：对比 **Matlab (Original)** vs **Python CPU (Multiprocess)** vs **Python GPU (CuPy)**。
* 展示随着 Grid Size 增加（128 -> 256 -> 512），GPU 版本的耗时增长远低于 CPU 版本。
* **代码支撑**：引用 `cfpurecon.py` 中 `GPU_AVAILABLE` 分支与 CPU 分支的运行时间差异。


* **实验 B: 大规模数据扩展性 (Scalability & Memory)**
* 测试不同点云规模（10k, 100k, 1M, 5M）。
* 证明使用 `SharedMemory` 后，内存占用是线性的，而传统多进程（Pickle序列化）在大数据下会发生 OOM (Out of Memory)。
* **代码支撑**：引用 `cfpurecon.py` 中的 `_init_proc` 和 `SharedMemory` 实现。



---

#### **5.4. Robustness to Non-Uniform Sampling (拓扑自适应性测试)**

* **目的**：证明你的“拓扑自适应半径 + 兜底策略”解决了“空洞”问题。
* **实验设计**：
* 对一个标准模型进行**非均匀降采样**（例如，在平坦区域稀疏采样，在曲率大区域密集采样）。
* **Baseline 结果**：由于使用统一的 k-NN 半径，稀疏区域会出现破洞或断裂。
* **Ours 结果**：展示你的算法自动扩大了稀疏区域的 Patch 半径，成功闭合了曲面。


* **代码支撑**：引用 `precompute.py` 中 `build_cfpu_input` 函数里的“兜底 1：不足 n_min 就扩半径”和“兜底 2：漏点补覆盖”逻辑。

---

#### **5.5. Numerical Stability & Regularization (数值稳定性)**

* **目的**：证明 Schur 补和正则化参数的作用。
* **实验设计**：
* **Schur 补效率**：对比开启 `nrmlschur=1` 与直接求解大矩阵的时间差，特别是在高阶（Order=2）基函数下。
* **正则化效果**：在含噪数据上，对比 `nrmlreg=1` (启用正则化) 与 `nrmlreg=0` (无正则化) 的重建结果。无正则化可能出现虚假片状物（Spurious sheets），而你的方法更光滑。


* **代码支撑**：引用 `cfpurecon.py` 中关于 `nrmlschur` 和 `nrmllambda` 的求解逻辑。

---

### **针对 CAD 期刊的写作建议 (Writing Tips)**

1. **强调 "Engineering-Ready"**：在描述实验时，不要只说“速度快”，要强调你的方法使得“在普通工作站上并在数秒内重建高精度 CAD 模型成为可能”。
2. **视觉呈现 (Visualization)**：
* CAD 的审稿人非常挑剔**拓扑结构**。你的网格线（Wireframe）图必须清晰，证明在尖锐边处网格没有乱，而是紧贴特征线。
* 使用 `main_4_view_result.py` 生成标准四视图，展示模型的整体一致性。


3. **诚实讨论局限性**：在实验章节末尾，简要提一下当前方法的局限（例如：极度稀疏的数据可能仍需人工干预参数），这会增加文章的可信度。

这个大纲将你的代码逻辑（预处理、GPU计算、数值优化）一一对应到了具体的实验证明上，逻辑闭环非常紧密，非常适合 CAD 期刊的口味。