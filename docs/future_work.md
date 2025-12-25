# 基于 SDF/隐式曲面的刚体接触仿真：从候选点到“无接触点”能量法（含障碍势与光滑/非光滑曲面对比）

> 本文系统整理我们对话中的全部要点：  
> 1) 两个 SDF 如何表征接触与计算接触响应；  
> 2) 刚体隐式时间积分下，如何做“不依赖候选接触点”的接触；  
> 3) 体积分（volume）与曲面积分（surface）的方案 B 详细推导与实现；  
> 4) 障碍势能与 IPC（Incremental Potential Contact）关系；  
> 5) 光滑与非光滑隐式曲面用于接触的异同与数值影响。

---

## 0. 设定与记号

### 0.1 刚体状态
两刚体 $A,B$ 的位姿：
- $T_A=(R_A,t_A)$，$T_B=(R_B,t_B)$，其中 $R\in SO(3)$，$t\in\mathbb{R}^3$。
- 速度：线速度/角速度分别为 $(v_A,\omega_A)$、$(v_B,\omega_B)$。

世界点 $x\in\mathbb{R}^3$ 转到刚体局部坐标：
$$
x_A = R_A^\top (x-t_A),\qquad x_B = R_B^\top (x-t_B).
$$

### 0.2 SDF / 隐式函数
局部 SDF：
- $\phi_A(x_A)$、$\phi_B(x_B)$。

约定：
- $\phi=0$ 为表面，$\phi<0$ 为内部，$\phi>0$ 为外部。

梯度与法向：
- 若 $\phi$ 为真距离场（signed distance），则表面附近常有 $\|\nabla\phi\|\approx 1$。
- 单位法向 $n=\nabla\phi/\|\nabla\phi\|$。

体素网格 SDF：
- $\phi(\cdot)$ 用三线性插值查询；
- 梯度可用中心差分（实践中建议预计算梯度网格后再插值）。

解析隐式：
- 直接提供 $\phi$ 与 $\nabla\phi$（通常更平滑、更利于隐式求解）。

---

## 1. （有候选点时）用 SDF 表征接触几何量与响应：基础流程回顾

> 对应对话起点：已有两个 SDF，如何在动力学中仿真接触、表征接触并计算响应。

### 1.1 接触检测（工程常用）
**单边表面采样 + 另一边 SDF 查询**
1. 在 $A$ 的表面离线/在线采样点 $\{x_A^{(i)}\}$，满足 $\phi_A(x_A^{(i)})=0$。
2. 变换到世界：$x^{(i)}=R_Ax_A^{(i)}+t_A$。
3. 查询 $B$ 的 SDF：$d^{(i)}=\phi_B\!\big(R_B^\top(x^{(i)}-t_B)\big)$。  
   若 $d^{(i)}<0$，则该点落入 $B$ 内部，判定发生穿透并生成接触项。

### 1.2 接触几何量（每个接触点）
- 穿透深度：
$$
\delta = \max(0,-d).
$$

- 接触法向（常用 $B$ 的梯度）：
$$
n = \frac{\nabla_x \phi_B(x_B)}{\|\nabla_x \phi_B(x_B)\|},\qquad x_B=R_B^\top(x-t_B).
$$
注意局部到世界梯度变换：若已得 $\nabla_{x_B}\phi_B$，则 $\nabla_x\phi_B = R_B \nabla_{x_B}\phi_B$。

- 接触点 $p$：常取 $p=x$（若 $x$ 本就是 $A$ 表面点），或将 $x$ 沿法向投影到 $B$ 的零水平集再组合。

### 1.3 三类主流响应模型（概览）
1) **罚函数/柔顺接触（force-based）**
$$
f_n=\max(0,k\delta - c v_n),\qquad F_n=f_n n.
$$
摩擦可用正则化库仑：
$$
F_t=-\mu f_n\frac{v_t}{\|v_t\|+\varepsilon}.
$$

2) **冲量法（impulse-based）**  
对碰撞瞬间求冲量，使法向速度满足不穿透/恢复系数条件，并更新动量。

3) **约束/互补（LCP/QP/SOCP）**  
以不等式约束与互补条件表示接触：
$$
0\le \lambda_n \perp g(q)\ge 0.
$$
多接触堆叠与静摩擦一般更稳健，但实现复杂度更高。

---

## 2. “不用候选接触点”的两条路线概览（刚体 + 隐式）

你提出：“刚体、隐式积分、不用候选接触点的方案”。我们给出两条路线：

### 2.1 路线 A：最近点作为内层变量的非线性约束（不采样，但会求出接触点）
定义接触间隙为在 $A$ 表面上使 $B$ 的 SDF 最小的值：
$$
g(q)=\min_{x:\,\phi_A(T_A^{-1}x)=0}\ \phi_B(T_B^{-1}x).
$$
通过内层优化/KKT 求得 $x^\*$，它自然是接触点/最近点；无需预先“候选点列表”，但仍会在求解过程中显式得到接触点。

### 2.2 路线 B：体积分/曲面积分的连续接触势能（真正不需要接触点）
用能量定义接触：接触不是离散点，而是空间中的连续“压力场/势能密度”。  
接触力/矩由能量对位姿的负梯度给出，并可直接用于隐式积分牛顿迭代。

> 下文详细展开路线 B。

---

## 3. 方案 B（体积分）：重叠体积的连续惩罚能量（无接触点）

### 3.1 能量定义（对称、无接触点）
令 $s=-\phi$，则内部 $s>0$。选一个仅在正半轴有值的软函数 $\psi(s)\ge 0$。

常见选择：
- 硬拐点：$\psi(s)=\tfrac12\max(0,s)^2$，$\psi'(s)=\max(0,s)$；
- 更平滑（隐式更友好）：用 softplus/smooth-ReLU 近似 $\max$。

体积分能量定义为：
$$
E(T_A,T_B)=k\int_{\mathbb{R}^3}\psi(-\phi_A(x_A))\ \psi(-\phi_B(x_B))\ dx.
$$
直觉：只有当两者同时“在内部”（发生重叠）时，被积项才显著，从而能量推动它们分离。

### 3.2 数值离散（窄带/交叠 AABB）
在世界空间采样点 $x_i$（均匀体素中心或稀疏体素哈希点），体积权重 $\Delta V$：
$$
E \approx k\sum_i \psi(s_{A,i})\psi(s_{B,i})\Delta V,
$$
其中
$$
s_{A,i}=-\phi_A\!\big(R_A^\top(x_i-t_A)\big),\qquad
s_{B,i}=-\phi_B\!\big(R_B^\top(x_i-t_B)\big).
$$

加速要点：
- 仅遍历 $A,B$ 世界 AABB 的交集区域，并向外扩展窄带宽度 $w$；
- 若 $\psi(s_{A,i})=0$ 或 $\psi(s_{B,i})=0$，可直接跳过（贡献为零）。

### 3.3 合力与合矩：逐点累加（关键公式）
定义世界梯度（体素：插值/差分得到；解析：直接算）：
$$
g_{A,i}=\nabla_x \phi_A(x_i),\qquad g_{B,i}=\nabla_x \phi_B(x_i).
$$
若在局部算梯度，则
$$
g_{A,i}=R_A\nabla_{x_A}\phi_A(x_{A,i}),\qquad g_{B,i}=R_B\nabla_{x_B}\phi_B(x_{B,i}).
$$

定义系数：
$$
\alpha_{A,i} = k\ \psi(s_{B,i})\ \psi'(s_{A,i}),\qquad
\alpha_{B,i} = k\ \psi(s_{A,i})\ \psi'(s_{B,i}).
$$

则对刚体 $A$ 的力与力矩贡献：
$$
F_A \mathrel{+}= -\alpha_{A,i}\ g_{A,i}\ \Delta V,
$$
$$
\tau_A \mathrel{+}= -\alpha_{A,i}\ \big(r_{A,i}\times g_{A,i}\big)\ \Delta V,\qquad r_{A,i}=x_i-t_A.
$$

对刚体 $B$ 同理：
$$
F_B \mathrel{+}= -\alpha_{B,i}\ g_{B,i}\ \Delta V,
$$
$$
\tau_B \mathrel{+}= -\alpha_{B,i}\ \big(r_{B,i}\times g_{B,i}\big)\ \Delta V,\qquad r_{B,i}=x_i-t_B.
$$

> 这组公式体现了“无接触点”的本质：接触通过体积上的势能密度产生连续推离场，再积分为合力合矩。

### 3.4 摩擦（无接触点框架下的常见正则化）
严格库仑摩擦依赖接触切空间与互补约束。体积分方案常用连续正则化（黏性或饱和黏性）摩擦：

相对速度（积分点处）：
$$
v_A(x_i)=v_A+\omega_A\times(x_i-t_A),\qquad
v_B(x_i)=v_B+\omega_B\times(x_i-t_B),
$$
$$
v_{\mathrm{rel}}=v_A(x_i)-v_B(x_i).
$$

局部法向（例如用 $B$ 的梯度）：
$$
n_i=\frac{g_{B,i}}{\|g_{B,i}\|+\epsilon},\qquad
v_t=v_{\mathrm{rel}}-(n_i^\top v_{\mathrm{rel}})n_i.
$$

一种黏性摩擦密度：
$$
f_{t,i}=-\eta\ \psi(s_{A,i})\psi(s_{B,i})\ v_t.
$$
然后按与法向同样方式积累到合力/合矩（对 $A$ 加 $f_{t,i}\Delta V$，对 $B$ 加反号）。

---

## 4. 隐式时间积分中如何使用体积分接触力（Newton 与稳定线性化）

以 Backward Euler 为代表：
$$
v_{k+1}=v_k+\Delta t\,M^{-1}\left(f_{\mathrm{ext}}(q_{k+1})+f_c(q_{k+1})\right),
$$
$$
q_{k+1}=\mathrm{Integrate}(q_k,v_{k+1},\Delta t).
$$
其中 $f_c(q)$ 由体积分累加得到的 $(F_A,\tau_A,F_B,\tau_B)$ 组成。

### 4.1 切线刚度/雅可比的实践策略：冻结梯度（Gauss–Newton 主项）
体素 SDF 的二阶信息通常噪声大；解析隐式虽可得二阶，但实现代价高且可能导致更复杂的非线性。实践中常用：
- **冻结法向/冻结梯度**：忽略 $\nabla\phi$ 的变化（忽略 Hessian），保留由 $\psi'(s)$ 主导的“法向刚度”；
- **局部线性化间隙**：把 $g(x)$ 近似成 $g(x)\approx n^\top(x-x_0)$，从而刚度主项近似为 $n n^\top$（有利于稳定与收敛）。

配合阻尼牛顿/线搜索通常显著提升鲁棒性。

---

## 5. 解析隐式曲面下的“曲面积分接触”（更干净的无接触点版本）

当几何以解析隐式曲面表达，且希望接触发生在曲面上，可将体积分替换为曲面积分（在 $A$ 表面上积分）。

### 5.1 曲面上的间隙场
在 $A$ 的局部表面
$$
\Gamma_A=\{X\in\mathbb{R}^3\mid \phi_A(X)=0\},
$$
世界点映射：
$$
x(X)=R_AX+t_A.
$$
定义对 $B$ 的间隙：
$$
g(X)=\phi_B\!\left(R_B^\top(x(X)-t_B)\right).
$$
穿透深度：
$$
\delta(X)=\max(0,-g(X)).
$$

### 5.2 罚函数型曲面积分能量与牵引
罚函数能量：
$$
E = \int_{\Gamma_A}\tfrac12 k\,\delta(X)^2\,dA.
$$
压力：
$$
p(X)=k\,\delta(X).
$$
牵引方向可取 $B$ 的法向（由 $\nabla\phi_B$）：
$$
n_B(x)=\frac{\nabla \phi_B(x_B)}{\|\nabla \phi_B(x_B)\|},\qquad x_B=R_B^\top(x-t_B).
$$
面力密度：
$$
f(X)=p(X)\,n_B(x(X)).
$$
合力合矩（关于 $t_A$）：
$$
F_A=\int_{\Gamma_A} f(X)\,dA,\qquad
\tau_A=\int_{\Gamma_A} (x(X)-t_A)\times f(X)\,dA.
$$
对 $B$ 的反作用为 $F_B=-F_A$，力矩按一致参考点计算后取反（务必保持参考点定义一致）。

### 5.3 曲面积分的数值实现
1) 若有参数化 $X(u,v)$：
- 面元 $dA=\|X_u\times X_v\|\,du\,dv$，对 $(u,v)$ 做二维求积；
- 穿透边界附近可自适应细分。

2) 若仅有隐式 $\phi_A(X)=0$ 无显式参数化：
- 离线生成“曲面求积点” $\{X_i,w_i\}$（用于数值积分，不是接触检测候选点）；
- 在线 $x_i=R_AX_i+t_A$，评估 $g_i,\delta_i,n_i$ 并累加：
$$
F_A\approx \sum_i w_i\,p_i\,n_i,\qquad
\tau_A\approx \sum_i w_i\,(x_i-t_A)\times(p_i n_i),
$$
其中 $p_i=k\delta_i$。

---

## 6. 障碍势能与 IPC 的关系：相似但不自动等同

你问：用障碍势能是不是就是 IPC？

### 6.1 相似之处：范式一致（能量最小化 + 障碍）
可将隐式时间步写成增量势能最小化（inertial + external + contact），并用对间隙 $g$ 的障碍项阻止 $g\to 0^+$ 或 $g\le 0$，例如：
$$
E_{\mathrm{barrier}}=\int_{\Gamma_A} -\mu \log\!\left(\frac{g(X)}{\hat g}\right)\,dA,\qquad g(X)>0.
$$
这种结构与 IPC 体系的核心精神高度一致（variational implicit + barrier contact）。

### 6.2 关键差别：IPC 通常包含“可行性维护”的完整机制
仅写下 $-\log(g)$ 并不能自动获得“严格不穿透”的保证。要接近 IPC 的 intersection-free 行为，通常还需要：
- 在线搜索/步长控制，确保迭代过程中始终保持 $g(X)>0$（否则障碍项不可定义或数值发散）；
- 在更严格的连续时间意义下，常需 CCD/保守推进或足够严密的约束覆盖，避免离散采样漏检导致穿透后才“发现”。

结论：
- **障碍势是 IPC 的关键积木之一**；
- 若缺少可行性维护机制，你得到的是 “IPC-like barrier method”，而非完整 IPC 流程。

---

## 7. 光滑 vs 非光滑隐式曲面用于接触（在本框架下的异同）

在“接触势能 + 隐式求解”框架中，接触力/刚度依赖 $\nabla\phi$（法向）甚至隐含依赖曲率；曲面光滑性直接决定力是否单值与连续。

### 7.1 光滑隐式曲面（至少 $C^1$，最好 $C^2$）
- $\nabla\phi$ 存在且连续，法向 $n$ 单值且连续；
- 接触力场随位姿变化平滑，牛顿迭代更稳定；
- 冻结法向近似（主项 $n n^\top$）更合理。

效果：
- 接触响应连续、抖动小；
- 摩擦切空间稳定，stick/slip 切换更可控。

### 7.2 非光滑隐式曲面（棱/角、CSG min/max、分段函数等）
几何本质：
- 在棱边/角点处 $\nabla\phi$ 可能不存在或跳变；
- 法向可能多值（法向锥/次梯度集合），接触方向不唯一；
- 最近点/支撑面可能不唯一（例如角点对平面）。

数值后果（隐式求解尤甚）：
- 接触力方向突变，残差不光滑，牛顿易震荡；
- active set 频繁切换，线搜索回退增多；
- 摩擦方向随法向跳变，易出现 stick-slip 抖动、锁死或异常滑移。

### 7.3 工程应对策略（通常必须选其一）
1) **几何平滑化（最稳）**
- CSG 的 $\min/\max$ 用 smooth-min/smooth-max（如 log-sum-exp/softmin）；
- 尖角做倒角/圆角；
- 使 $\phi$ 至少 $C^1$，恢复光滑优化性质。

2) **次梯度/广义梯度（理论可行但复杂）**
- 在不可导处选取 Clarke 次梯度；但“如何选”会影响稳定性与可重复性。

3) **半光滑牛顿/Prox/投影类方法**
- 若必须保留尖角，常需使用能处理 kink 的求解器体系，代价是实现与调参显著增加。

---

## 8. 参数、稳定性与实现要点（体素/解析通用）

### 8.1 SDF 质量
力方向依赖 $\nabla\phi$。若 $\phi$ 不是严格距离场，建议至少在窄带内重初始化/归一化，使表面附近满足 $\|\nabla\phi\|\approx 1$，否则法向与深度会系统性偏差。

### 8.2 平滑与带宽
- 用 softplus/smooth-ReLU 代替硬 $\max$ 可显著改善隐式收敛。
- 平滑尺度 $\varepsilon$ 常取 1–2 个体素尺寸或一个合理几何尺度。

### 8.3 刚度 $k$ 与隐式收敛
- $k$ 越大接触越“硬”，但条件数更差、牛顿更难；
- 推荐配合：冻结法向（Gauss–Newton 主项）+ 阻尼牛顿/线搜索；
- 高速运动可考虑子步或自适应步长。

### 8.4 覆盖与漏检（无接触点方案仍需积分覆盖）
- 体积分：采样分辨率需覆盖薄特征与高速运动；
- 曲面积分：求积点密度需覆盖潜在接触区；
- 若追求严格无穿透，需要更严格的可行性维护（barrier + 线搜索 + 可能的 CCD/保守推进思想）。

---

## 9. 体积分方案 B 最小可用伪代码（无接触点）

```cpp
// 输入：phiA, gradPhiA（局部）；phiB, gradPhiB（局部）
// 位姿：RA,tA, RB,tB；参数：k, dV；psi, dpsi

FA=0; TA=0; FB=0; TB=0;

for (x in samples_in_overlap_AABB_expanded) {
    xA = RA.transpose() * (x - tA);
    xB = RB.transpose() * (x - tB);

    dA = phiA(xA);  sA = -dA;
    dB = phiB(xB);  sB = -dB;

    wA  = psi(sA);
    dwA = dpsi(sA);
    wB  = psi(sB);
    dwB = dpsi(sB);

    if (wA==0 || wB==0) continue;

    gA_local = gradPhiA(xA);
    gB_local = gradPhiB(xB);

    gA = RA * gA_local;   // world gradient
    gB = RB * gB_local;

    alphaA = k * wB * dwA;
    alphaB = k * wA * dwB;

    FA += -alphaA * gA * dV;
    TA += -alphaA * cross(x - tA, gA) * dV;

    FB += -alphaB * gB * dV;
    TB += -alphaB * cross(x - tB, gB) * dV;

    // optional regularized friction (viscous-like)
    // vrel = (vA + wA×(x-tA)) - (vB + wB×(x-tB))
    // n = normalize(gB)
    // vt = vrel - dot(n,vrel)*n
    // ft = -eta*wA*wB*vt
    // accumulate ft similarly into (F,T) of A/B with opposite signs
}
