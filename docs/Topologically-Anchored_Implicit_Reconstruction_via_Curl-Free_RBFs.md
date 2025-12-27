# Topologically-Anchored Implicit Reconstruction via Curl-Free RBFs  
# 基于无旋径向基函数与拓扑图谱锚定的隐式曲面重建理论（补全版 v3）

> 本文档在 v2 的基础上补足了：  
> 1) 零水平集如何与输入几何对齐（不仅拟合梯度，还约束函数值）；  
> 2) PHS / curl-free RBF 的可解性（多项式增广与侧条件、线性系统结构）；  
> 3) 锚定约束在“硬约束/软约束”下的可解性处理；  
> 4) PU 权重的光滑性保证；  
> 5) 全局符号/法向方向一致化与“并集”语义的严格约定。

---

## 0. 记号与约定（Notation & Conventions）

- 空间维度：$x\in\mathbb{R}^3$。  
- 目标曲面（最终输出）：$\mathcal{S}=\{x\mid F(x)=0\}$。  
- **符号约定**：曲面外侧 $F(x)>0$，内侧 $F(x)<0$，并且外法向满足 $\nabla F(x)$ 指向外侧。  
- 输入点法数据：$D=\{(x_i, n_i)\}_{i=1}^N$，其中 $n_i$ 需尽可能一致朝向外侧（见 0.2）。  

### 0.1 尺度与“近似有符号距离”目标

为避免仅靠 $\nabla s\approx n$ 导致零水平集漂移，我们显式引入函数值约束，使 $s$ 在采样点附近近似有符号距离（SDF-like）：

- 选定局部步长 $h>0$（典型取采样平均间距的 $0.5\sim 2$ 倍）。  
- 定义偏移点：
  $$x_i^+ = x_i + h n_i,\qquad x_i^- = x_i - h n_i.$$

目标直觉：$s(x_i)=0$，$s(x_i^+)\approx +h$，$s(x_i^-)\approx -h$，从而同时锁定零集位置与尺度。

### 0.2 法向一致化（Orientation）

若输入法向并非全局一致，需要在图谱上做一致化（工程上可用网格 BFS/最小生成树传播，或在 region 邻接边上最小化翻转代价）。本文假设得到一致朝向的 $n_i$，否则需要将“法向拟合项”改为无向约束（例如拟合法向轴而非方向），这会显著改变问题形式。

---

## 1. 理论基础：无旋向量场与标量势

核心假设：隐式函数可由某个标量势场 $s(x)$ 表示，并且在曲面上 $\nabla s$ 与法向一致，即 $n(x) \parallel \nabla s(x)$。因此，我们用 **无旋（curl-free）核**来拟合梯度场并恢复势场。

本文采用 **Polyharmonic Splines (PHS)** 作为基础径向核（例如 $\phi(r)=r^3$ 或 $\phi(r)=r^4 \log r$），以避免传统 RBF 的形状参数选择问题。

### 1.1 矩阵值无旋核（curl-free kernel）

令 $r=\lVert x-y\rVert$，基础径向核为 $\phi(r)$。定义矩阵值无旋核 $\Phi:\mathbb{R}^3\times\mathbb{R}^3\to\mathbb{R}^{3\times 3}$：

$$
\Phi(x,y) = -\nabla_x \nabla_x^{T}\, \phi(\lVert x-y\rVert).
$$

对任意 $c\in\mathbb{R}^3$，向量场 $x\mapsto \Phi(x,y)c$ 为无旋场（可表示为某个标量势的梯度），适用于“从法向恢复势”的建模。

### 1.2 标量势的解析表示（与法向插值的一致性）

令势场 $s(x)$ 取如下形式（与 1.1 一致的标准构造）：

$$
s(x) = -\sum_{j=1}^{M} \nabla \phi(\lVert x-x_j\rVert)^{T} c_j + \sum_{k=1}^{L} b_k\, p_k(x),
$$

则其梯度为：

$$
\nabla s(x) = \sum_{j=1}^{M} \Phi(x,x_j)\, c_j + \sum_{k=1}^{L} b_k\, \nabla p_k(x).
$$

其中：$c_j\in\mathbb{R}^3$，$p_k$ 为多项式基（典型取 1 与一次项），$b_k$ 为其系数。

---

## 2. 几何拓扑图谱抽象（Geometric Graph Abstraction）

将几何抽象为拓扑图谱：

$$
G = (\mathcal{R}, \mathcal{E}, \mathcal{J}).
$$

1. **区域（Regions）$\mathcal{R}$**：光滑曲面片集合。每个区域 $r$ 拥有点法数据 $D_r=\{(x_i,n_i)\}$。  
2. **棱边（Crease Edges）$\mathcal{E}$**：区域 2-way 邻接关系 $(i,j)$，其公共边界曲线为 $\Gamma_{ij}$。  
3. **结点（Junctions）$\mathcal{J}$**：$k$-way 汇交点（$k\ge 3$）。对结点 $v$，其关联区域集合记为 $J_v\subset\mathcal{R}$。

---

## 3. 局部势场拟合：良定化、侧条件与锚定（Local Fitting）

对每个区域 $r\in\mathcal{R}$，求解势场 $s_r(x)$。为了使零水平集与数据对齐，必须同时约束 **梯度** 与 **函数值**。

### 3.1 统一的加权最小二乘目标（含函数值约束）

对区域 $r$ 的样本 $(x_i,n_i)\in D_r$，定义偏移点 $x_i^{\pm}=x_i\pm h n_i$。推荐目标为：

$$
\begin{aligned}
E(s_r)=
&\; w_g\sum_{(x_i,n_i)\in D_r} \lVert \nabla s_r(x_i) - n_i \rVert^2
+ w_0\sum_{(x_i,n_i)\in D_r} \bigl(s_r(x_i)\bigr)^2\\
&+ w_{\pm}\sum_{(x_i,n_i)\in D_r}\Bigl(\bigl(s_r(x_i^+)-h\bigr)^2 + \bigl(s_r(x_i^-)+h\bigr)^2\Bigr)
+ \lambda\, \lVert s_r \rVert_{\mathcal{N}_\Phi}^2.
\end{aligned}
$$

- $w_g,w_0,w_{\pm}$ 控制梯度/零集/尺度约束的权重；  
- $\lambda$ 为平滑正则；  
- 该形式避免了“只拟合梯度但零集漂移”的不完备性。

> 可选简化：若你只想做 SDF-like 拟合，可将 $w_g=0$，仅使用 $s(x_i)=0,\; s(x_i^{\pm})=\pm h$ + 正则。  
> 若法向质量极高，也可将 $w_{\pm}=0$，但必须保留至少一类函数值约束（例如 $s(x_i)=0$ 或外侧点 $s(x_{out})>0$），否则零集仍可能漂移。

### 3.2 PHS 可解性：多项式增广与侧条件（Side Conditions）

PHS 核是“条件正定”的。为保证线性系统可解且唯一，需要：

- 选择多项式空间 $\Pi_{m-1}$（例如 $m=2$ 对应常数+一次项），其基为 $\{p_k\}_{k=1}^{L}$；  
- 对系数施加侧条件（典型形式）：
  $$\sum_{j=1}^{M} p_k(x_j)\, c_j = 0\quad (k=1,\dots,L).$$

其作用是消除核的零空间并保证系统非奇异（配合“采样点对 $\Pi_{m-1}$ 的 unisolvent 条件”）。

### 3.3 线性系统结构（可直接实现）

将所有约束（梯度约束与函数值约束）写成线性方程组：

- 对任意点 $x$，$s_r(x)$ 与 $\nabla s_r(x)$ 都是对未知量 $\{c_j\},\{b_k\}$ 的线性函数；  
- 因此 3.1 的最小二乘可写为正规方程或直接用加权 QR/LSQR 求解。

一个典型的“块结构”写法如下（示意）：

$$
\min_{c,b}\; \lVert W(Ac+Pb-d)\rVert^2 + \lambda\, c^T K c
\quad\text{s.t.}\quad P^T c=0,
$$

其中：  
- $A$ 由核与其导数组成（对应 $s$ 与 $\nabla s$ 的采样）；  
- $P$ 由多项式基或其梯度采样组成；  
- $d$ 为目标值拼接（含 $0,\pm h,n_i$ 等）；  
- $K$ 为正则相关矩阵（对 PHS 可写成等价的“弯曲能”离散形式，工程上可直接并入 Tikhonov）。  

若采用拉格朗日乘子处理侧条件，可写成 KKT 系统（示意）：

$$
\begin{bmatrix}
A^T W^T W A + \lambda K & A^T W^T W P & P \\
P^T W^T W A & P^T W^T W P & 0 \\
P^T & 0 & 0
\end{bmatrix}
\begin{bmatrix}
c\\ b\\ \mu
\end{bmatrix}
=
\begin{bmatrix}
A^T W^T W d\\
P^T W^T W d\\
0
\end{bmatrix}.
$$

---

## 4. 拓扑锚定：硬约束与软约束的闭合（Anchoring Closure）

锚定不仅用于“锁死特征”，也用于固定各区域势场的常数自由度并增强跨区域一致性。

### 4.1 棱锚定（2-way）

对每条棱边 $\Gamma_{ij}$ 上的采样点集合 $A_{ij}$，施加：

$$
\forall x\in A_{ij},\quad s_i(x)=0,\quad s_j(x)=0.
$$

### 4.2 角点锚定（k-way, k≥3）

对每个结点 $v\in\mathcal{J}$ 及其关联区域 $r\in J_v$：

$$
\forall r\in J_v,\quad s_r(v)=0.
$$

### 4.3 约束冲突与可解性（必须说明的逻辑闭合点）

由于噪声存在，锚定与法向/函数值拟合可能冲突。两种标准处理：

**(A) 硬约束（KKT）**：将锚定作为等式约束加入系统（与侧条件类似加乘子）。优点是特征精确锁定；缺点是噪声过大时可能数值病态。

**(B) 软约束（Penalty）**：在能量中加入强罚项：

$$
E_{anchor}(s)=\eta\sum_{x\in A_{ij}}\bigl(s_i(x)^2+s_j(x)^2\bigr)+
\eta\sum_{v\in\mathcal{J}}\sum_{r\in J_v}s_r(v)^2,
$$

并将其并入 3.1 的目标。优点是稳健；缺点是锚定精度受 $\eta$ 影响。

> 建议：默认使用软约束，$\eta$ 取 $w_0$ 的 $10\sim 10^3$ 倍（取决于噪声与尺度），以兼顾稳健与特征精度。

---

## 5. 全局融合：候选函数、soft-min 语义与 PU 光滑性（Global Fusion）

### 5.1 广义 soft-min（用于“并集”语义）

定义：

$$
\mathcal{M}_\varepsilon(\{s_i\}_{i=1}^{k})
= -\varepsilon \log\left(\sum_{i=1}^{k} \exp\left(-\frac{s_i(x)}{\varepsilon}\right)\right)
+ \varepsilon \log(k).
$$

当 $\varepsilon\to 0^+$ 时，$\mathcal{M}_\varepsilon$ 近似 $\min_i s_i$。在本文的符号约定下（外侧为正），$\min$ 对应多隐式函数的“并集”近似（soft union）。

### 5.2 候选函数族（Candidates）

我们把候选索引记为 $\alpha$，它可以是：

- 区域候选：$\alpha=r\in\mathcal{R}$，$F_r(x)=s_r(x)$；  
- 棱候选：$\alpha=(i,j)\in\mathcal{E}$，$F_{ij}(x)=\mathcal{M}_\varepsilon(\{s_i(x),s_j(x)\})$；  
- 角点候选：$\alpha=v\in\mathcal{J}$，$F_v(x)=\mathcal{M}_\varepsilon(\{s_r(x)\mid r\in J_v\})$。

### 5.3 PU 权重：用光滑紧支撑函数保证可微性

为了避免“距离函数不可微”导致的权重不光滑，推荐使用紧支撑光滑 bump（或 Wendland）构造权重。对每个候选 $\alpha$ 定义中心 $x_\alpha$ 与半径 $\rho_\alpha$（可取自 patch/radius 或特征邻域尺度），并定义：

$$
w_\alpha(x)=\psi\left(\frac{\lVert x-x_\alpha\rVert}{\rho_\alpha}\right),
\qquad
\tilde{w}_\alpha(x)=\frac{w_\alpha(x)}{\sum_{\beta} w_\beta(x)}.
$$

其中 $\psi:[0,\infty)\to[0,\infty)$ 可选为 $C^\infty$ bump：

$$
\psi(t)=
\begin{cases}
\exp\left(-\frac{1}{1-t^2}\right), & 0\le t<1,\\
0, & t\ge 1.
\end{cases}
$$

这样 $w_\alpha$ 与 $\tilde{w}_\alpha$ 在其定义域内光滑（只要分母不为 0；工程上通过“覆盖性”保证）。

### 5.4 最终全局隐式函数

$$
F(x)=\sum_{\alpha}\tilde{w}_\alpha(x)\,F_\alpha(x).
$$

**光滑性结论（可直接引用）**：  
若各 $s_r\in C^k$，$\psi\in C^k$，且 $\varepsilon>0$，则 $\mathcal{M}_\varepsilon\in C^\infty$，从而 $F\in C^k$（在 $\sum_\beta w_\beta(x)>0$ 的覆盖域内）。

---

## 6. 参数与数值建议（Practical Closure）

- $h$：取局部点间距的 $0.5\sim 2$ 倍。  
- $\varepsilon$：取 $0.5\sim 2$ 个局部间距（越小越接近硬 $\min$，但数值更尖锐）。  
- $\lambda$：与噪声成正比增大；可在“重建误差 vs 光滑度”之间调节。  
- 权重：默认 $w_0$ 与 $w_{\pm}$ 至少有一个非零；若法向可信则增大 $w_g$。  
- 覆盖性：保证任意 $x$ 至少落在若干候选 support 内，使 $\sum_\beta w_\beta(x)>0$。

---

## 7. 完备性总结（What is now closed）

通过新增 0、3.1、3.2–3.3、4.3、5.3 等节，本文档现在在数学逻辑上闭合了以下关键点：

1) **零水平集对齐**：用 $s(x_i)=0$ 与偏移点 $s(x_i^{\pm})=\pm h$（可选）锁定零集与尺度；  
2) **PHS 可解性**：明确多项式增广与侧条件、给出可实现的线性系统结构；  
3) **锚定可解性**：说明硬/软约束并处理冲突与噪声；  
4) **PU 光滑性**：用 $C^\infty$ bump 权重避免距离函数不可微；  
5) **并集语义与符号约定**：明确 $F>0$ 为外侧，soft-min 近似 $\min$ 对应并集近似。

