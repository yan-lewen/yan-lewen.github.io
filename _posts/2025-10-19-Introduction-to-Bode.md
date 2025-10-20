---
title: 闭环系统稳定性与Bode图
date: 2025-10-19
categories: [控制理论, 系统分析]
tags: [频域分析, 稳定性, 相位裕度, 伯德图]
math: true
---

## 一. 开篇：一切始于一个简单的共振问题

我的学习之旅始于一个经典问题：为什么给弹簧一个特定频率的外力，它会共振导致幅值很大？

我很快了解到，这是因为外力频率接近了系统的**固有频率** $\omega_n$。但一个困惑随之而来：在伯德图（Bode Plot）上，位移 $X(s)$ 响应外力 $F(s)$ 时，共振峰（最大幅值）对应的相位**不是 $0^\circ$（同相），而是 $-90^\circ$（滞后）**。

**我的第一个"Aha!"时刻：**
我意识到能量的注入看的是**功率 $P = F \cdot v$**（力与速度）。由于速度 $\dot{x}$ 比位移 $x$ **超前 $90^\circ$**，当位移滞后力 $90^\circ$ 时，速度恰好与力**同相（$0^\circ$）**！这正是外力对系统做正功、持续注入能量的完美条件。

这个小小的发现让我意识到，必须将物理直觉、时域行为和频域分析结合起来。于是，我决定在MATLAB中搭建一个主动控制系统，来"驯服"这个弹簧。

## 二. 第一部分：时域实验——当"阻尼"变成了"发动机"

我的目标很简单：给弹簧系统（MSD）增加一个**主动阻尼器**，使其更快地稳定下来。

$$
m\ddot{x} + c\dot{x} + k_p x = F_c
$$

我选择的控制律是**速度负反馈**，这在物理上就是一个理想的阻尼器，它应该**只消耗能量**：

$$
F_c(t) = -k_c \dot{x}(t)
$$

### 2.1 理想情况：无时滞

在代码中，我设置 $k_c = 5$，系统从 $x_0=1$ 处释放。

```matlab
% 理想控制器
Fc_ideal = -kc * v_ideal(i-1);
a_ideal = (Fc_ideal - c*v_ideal(i-1) - kp*x_ideal(i-1)) / m;
v_ideal(i) = v_ideal(i-1) + a_ideal * dt;
x_ideal(i) = x_ideal(i-1) + v_ideal(i-1) * dt;
```

结果如预期所料：系统迅速收敛，振动被完美抑制。

### 2.2 引入"魔鬼"：时间延迟 $\tau$

在现实世界中，传感、计算和执行都需要时间。我引入了一个 $0.4$ 秒的时间延迟 ($\tau = 0.4s$)：

$$
F_c(t) = -k_c \dot{x}(t - \tau)
$$

```matlab
% 有时滞控制器
delay_steps = round(tau/dt);
if i > delay_steps
    Fc_delay = -kc * v_delay(i-delay_steps);
else
    Fc_delay = -kc * v_delay(1);  % 初始时滞期间用初值
end
a_delay = (Fc_delay - c*v_delay(i-1) - kp*x_delay(i-1)) / m;
% ... (积分) ...
```

### 2.3 结果：灾难性的发散

当 $k_c=5, \tau=0.4 \text{ s}$ 时，仿真结果让我大跌眼镜：系统不仅没有收敛，反而**剧烈发散**！

> **Note**: 蓝色曲线为理想收敛，红色曲线为时滞发散

**我的巨大困惑：** 为什么一个本应"消耗能量"的阻尼器，仅仅因为反应慢了 $0.4$ 秒，就变成了向系统"注入能量"的发动机？

**时域的物理分析：**
我意识到，当系统振动时，控制力 $F_c$ 基于的是**过去的速度** $\dot{x}(t-\tau)$。

- 想象一下：当质量块正要向左（$\dot{x}$ 为负）运动时，控制器却在 $0.4$ 秒前（当时 $\dot{x}$ 为正）采样，于是施加了一个**向左（负）**的力。
- **力与速度同向了！**
- 本应消耗能量的阻尼力，在振动周期的关键阶段变成了**做正功**的助推力，不断向系统注入能量，导致振幅越来越大。

## 三. 第二部分：频域解谜——伯德图与"裕度"的审判

时域仿真揭示了"现象"，而频域分析则提供了"病因"。为了在理论上解释发散，我转向了**开环伯德图**分析。

我需要分析的开环传递函数 $L(s)$ 是：

$$
L(s) = C(s) \cdot G(s) = \left( k_c s e^{-\tau s} \right) \cdot \left( \frac{1}{ms^2 + cs + k_p} \right)
$$

**附：我的频域分析代码**

```matlab
% MATLAB Code for Bode Plot Analysis
clear; clc;
m = 1; c = 0.1; kp = 1;
kc = 5; tau = 0.4;
s = tf('s');

% 2. 构造开环传递函数
G_plant = 1 / (m*s^2 + c*s + kp);
L_ideal = G_plant * (kc * s);
L_delay = G_plant * (kc * s * exp(-tau*s)); % 包含时滞的模型

% 3. 绘制伯德图
figure;
bodeplot(L_ideal, L_delay);
legend('无时滞 L_{ideal}', ['有时滞 L_{delay} (\tau=', num2str(tau), 's)']);
grid on;

% 4. 稳定性裕度计算
disp('--- 稳定性裕度分析 (kc = 5) ---');
[Gm_ideal, Pm_ideal, Wcg_ideal, Wcp_ideal] = margin(L_ideal);
disp(['【无时滞系统】: PM = ', num2str(Pm_ideal), ' 度']);

[Gm_delay, Pm_delay, Wcg_delay, Wcp_delay] = margin(L_delay);
disp(['【有时滞系统】: PM = ', num2str(Pm_delay), ' 度']);
disp(['  结论: PM < 0，系统不稳定!']);
```

### 3.1 裕度的物理意义：我的理解

通过分析伯德图，我终于理解了裕度的物理意义：

- **相位裕度 (Phase Margin, PM):** 衡量系统对**时间延迟**的容忍度。它是在**增益穿越频率 $\omega_{gc}$**（增益为1）处，系统距离 $-180^\circ$（完美正反馈）的"安全角度"。
- **增益裕度 (Gain Margin, GM):** 衡量系统对**增益 $k_c$** 的容忍度。它是在**相位穿越频率 $\omega_{pc}$**（相位为-180°）处，系统距离"增益为1"的"安全系数"。

### 3.2 决定性的结果：负相位裕度

运行频域代码，我得到了决定性的诊断结果：

| 系统                 |     相位裕度 (PM)     |          结论           |
| :------------------- | :-------------------: | :---------------------: |
| 无时滞 ($\tau=0$)    | $\approx 89.5^\circ$  | **非常稳定**（接近90°） |
| 有时滞 ($\tau=0.4s$) | $\approx -21.3^\circ$ |      **不稳定！**       |

**分析：**
时滞 $\tau=0.4 \text{ s}$ 引入了巨大的相位滞后（在伯德图上，相位曲线被拉低）。在增益穿越频率 $\omega_{gc}$ 处，总相位已经低于 $-180^\circ$，导致 **PM 为负**。

**PM < 0** 是系统不稳定的明确信号。这在理论上完美解释了我的时域仿真为什么会发散。

## 四. 第三部分：深入探索——两个最后的困惑

我的分析并未止步于此，还有两个问题困扰着我。

### 4.1 困惑 A：为什么增大增益 $k_c$ 也会导致发散？

我在时域仿真中发现，如果 $\tau$ 较小，小的 $k_c$（如 $k_c=1$）系统会收敛，但大的 $k_c$（如 $k_c=10$）又会导致发散。

**伯德图的解答：**

- 增大 $k_c$ $\implies$ 伯德图的**幅频曲线整体上移**。
- 幅频曲线上移 $\implies$ **增益穿越频率 $\omega_{gc}$ 向右移动（频率更高）**。
- 时滞的相位滞后是 $-\omega\tau$ $\implies$ $\omega_{gc}$ 越高，**相位滞后越大**。
- **结论：** 增大 $k_c$ 会进一步压缩本已很小的相位裕度 (PM)，最终使其变为负值，导致系统失稳。

### 4.2 困惑 B：发散频率 $f_{osc}$ 到底是什么？

我仔细测量了时域发散的振荡频率，发现了一个惊人的事实：

- **系统固有频率：** $f_n = \sqrt{k_p/m} / (2\pi) \approx 0.16 \text{ Hz}$

- **时域发散频率：** $f_{osc} \approx 0.685 \text{ Hz}$ (我的观测值)

- **频域 $\omega_{gc}$ (增益穿越)：** $f_{gc} \approx 0.637 \text{ Hz}$

- **频域 $\omega_{pc}$ (相位穿越)：** $f_{pc} \approx 0.826 \text{ Hz}$

  

  系统发散的频率**不是**固有频率 $f_n$！ 它是由**闭环系统的不稳定极点**决定的。在控制理论中，这个实际的振荡频率 $f_{osc}$ 应该恰好位于开环系统的**增益穿越频率 $f_{gc}$** 和**相位穿越频率 $f_{pc}$** 之间。 我的测量结果 $0.637 \text{ Hz} < \mathbf{0.685 \text{ Hz}} < 0.826 \text{ Hz}$ 完美地印证了这一点！

## 五. 时域发散频率的正向推导

### 5.1 问题提出：闭环极点与系统发散

在前文分析中，我们发现系统的实际发散频率 $f_{osc}$ 并非固有频率，而是与开环的两个临界频率 $f_{gc}$ 和 $f_{pc}$ 密切相关。更本质地说，$f_{osc}$ 是闭环系统特征方程的不稳定极点的虚部。

#### 5.1.1 从开环传函到闭环传函

开环传递函数为：
$$
L(s) = C(s) \cdot G(s) = \frac{k_c s e^{-\tau s}}{ms^2 + cs + k_p}
$$

闭环传递函数为：
$$
T(s) = \frac{G(s)}{1 + L(s)} = \frac{1}{ms^2 + cs + k_p + k_c s e^{-\tau s}}
$$

#### 5.1.2 闭环传递函数力学推导
对于带速度反馈和时滞的质量-弹簧系统，其动力学方程为：
$$
m\ddot{x} + c\dot{x} + k_p x = F_c
$$
其中控制力为
$$
F_c(t) = -k_c \dot{x}(t - \tau)
$$

将控制力代入系统方程，得到：
$$
ms^2 X(s) + cs X(s) + k_p X(s) + k_c s e^{-\tau s} X(s) = F_{ext}(s)
$$

闭环传递函数（以外部扰动 $F_{ext}(s)$ 为输入，$X(s)$ 为输出）为：
$$
T(s) = \frac{X(s)}{F_{ext}(s)} = \frac{1}{ms^2 + cs + k_p + k_c s e^{-\tau s}}
$$
整理后，系统的特征方程为：
$$
ms^2 + cs + k_p + k_c s e^{-\tau s} = 0
$$
该方程决定了闭环系统的极点分布和稳定性。



#### 5.1.3 相位裕度与闭环极点的关系

为什么开环分析中的 $\mathbf{PM < 0}$ 必然意味着闭环极点 $\mathbf{s = \sigma + j\omega_{osc}}$ 具有正实部 $\mathbf{\sigma > 0}$，导致系统不稳定？

1. **PM 的定义与物理意义**  
   相位裕度是在增益穿越频率 $\omega_{gc}$ 处，开环传递函数 $L(s)$ 的相位距离 $-180^\circ$ 的"安全角度"：
   $$
   \text{PM} = 180^\circ + \angle L(j\omega_{gc})
   $$
   若 $\text{PM} < 0$，表示系统在该频率处已超过 $-180^\circ$，进入了危险区域。

2. **奈奎斯特判据的简化应用**  
   当 $\text{PM} < 0$，奈奎斯特曲线必然顺时针包围复平面上的 $(-1, 0)$ 点。对于无右半平面开环极点的系统，这意味着闭环系统会出现一个或多个右半平面极点。

3. **极点实部与系统发散**  
   闭环特征方程
   $$
   ms^2 + cs + k_p + k_c s e^{-\tau s} = 0
   $$
   会有至少一个解 $s = \sigma + j\omega_{osc}$ 满足 $\sigma > 0$，即极点位于右半平面。  
   这对应于系统时域响应中出现指数发散 $e^{\sigma t}$，即系统不稳定。

**结论：** 当开环分析得到 $\text{PM} < 0$，根据奈奎斯特判据，闭环系统必然存在实部为正的极点，从而导致时域上的发散。这一频域与时域的紧密对应，是控制系统稳定性分析的核心原理。




### 5.2 理论求解：Pade 近似与数值极点

由于 MATLAB 的 pole() 函数无法直接处理时滞项 $e^{-\tau s}$，我们采用 Pade 近似，将时滞项转化为高阶有理函数，以便数值求解。

#### Pade 近似步骤：
1. 用 $N$ 阶 Pade 近似逼近 $e^{-\tau s}$
2. 构造近似的闭环传递函数 $T(s)$
3. 求解系统极点，筛选出实部为正的不稳定主导极点

### 5.3 MATLAB 推导代码

```matlab
clc;
m = 1; c = 0.1; kp = 1; kc = 5; tau = 0.4;
s = tf('s');
% 1. 构造闭环系统 T(s)
G_loop = (kc * s) / (m*s^2 + c*s + kp);
C_delay = exp(-tau*s);
T_delay = feedback(G_loop * C_delay, 1);
% 2. 使用 N=10 阶 Pade 近似
N_pade = 10;
T_delay_approximated = pade(T_delay, N_pade);
% 3. 求解极点并筛选不稳定极点
poles = pole(T_delay_approximated);
unstable_poles = poles(real(poles) > 0);
% 提取主导极点信息
sigma = real(unstable_poles(1));
omega_osc = abs(imag(unstable_poles(1)));
f_osc = omega_osc / (2*pi);
disp(['主导极点: s = ', num2str(sigma), ' + j', num2str(omega_osc)]);
disp(['发散频率 f_osc = ', num2str(f_osc), ' Hz']);
```

### 5.4 结果与理论验证

| 指标               |             理论推导结果             |            时域仿真观察值            |           吻合度           |
| :----------------- | :----------------------------------: | :----------------------------------: | :------------------------: |
| 不稳定极点 $s$     | $\mathbf{0.4862} + j\mathbf{4.3002}$ |                 N/A                  |            完美            |
| 实部 $\sigma$      |               $0.4862$               |                 N/A                  | $\sigma > 0$，理论确认发散 |
| 振荡频率 $f_{osc}$ |    $\mathbf{0.68439 \text{ Hz}}$     | $\approx \mathbf{0.6854 \text{ Hz}}$ |          极高吻合          |

- 实部 $\sigma = 0.4862$ 决定了振幅以 $e^{0.4862t}$ 的速度指数增长
- 虚部 $\omega_{osc} = 4.3002 \text{ rad/s}$ 决定了振荡周期 $T = 2\pi / 4.3002$
- 数值推导得到的 $f_{osc}$ 与时域仿真观测值高度一致

### 5.5 总结：从现象到本质的闭环推导

本章通过精确的数值方法，完成了时域发散频率的正向推导：
- 时域发散频率 $f_{osc}$ 就是闭环系统不稳定极点的虚部
- 实部决定发散速率，虚部决定振荡频率
- 这条路径实现了从**现象（发散）——开环预测（PM < 0）——闭环精确推导（极点）**的完整理论闭环，深化了对时滞系统稳定性本质的理解

## 六. 附录：时域与频域分析的 MATLAB 代码整理

### 6.1 时域仿真代码（有无时滞控制器对比）

```matlab
clear; close all; clc;

% 参数
m = 1;      % 质量 (kg)
c = 0.1;    % 阻尼系数 (N*s/m)
kp = 1;     % 固有弹簧刚度 (N/m)

kc = 5;     % 速度反馈增益 (N/m)
tau = 0.4;  % 时间延迟 (s)

x0 = 1;       % 初始位置
v0 = 0;       % 初始速度
T_total = 30; % 仿真总时间

% 仿真时间
dt = 0.001;
t = 0:dt:T_total;
N = length(t);

% 初始化状态
x_ideal = zeros(1, N); v_ideal = zeros(1, N);
x_delay = zeros(1, N); v_delay = zeros(1, N);

x_ideal(1) = x0; v_ideal(1) = v0;
x_delay(1) = x0; v_delay(1) = v0;

% 时滞点数
delay_steps = round(tau/dt);

for i = 2:N
    % 无时滞控制器
    Fc_ideal = -kc * v_ideal(i-1);
    a_ideal = (Fc_ideal - c*v_ideal(i-1) - kp*x_ideal(i-1)) / m;
    v_ideal(i) = v_ideal(i-1) + a_ideal * dt;
    x_ideal(i) = x_ideal(i-1) + v_ideal(i-1) * dt;
    
    % 有时滞控制器
    if i > delay_steps
        Fc_delay = -kc * v_delay(i-delay_steps);
    else
        Fc_delay = -kc * v_delay(1);  % 初始时滞期间用初值
    end
    a_delay = (Fc_delay - c*v_delay(i-1) - kp*x_delay(i-1)) / m;
    v_delay(i) = v_delay(i-1) + a_delay * dt;
    x_delay(i) = x_delay(i-1) + v_delay(i-1) * dt;
end

% 绘图
figure;
plot(t, x_ideal, 'b', 'LineWidth', 2); hold on;
plot(t, x_delay, 'r-.', 'LineWidth', 2);
legend('无时滞控制器', '有时滞控制器');
xlabel('时间 (s)');
ylabel('位移 (m)');
title('质量-弹簧系统时域仿真（有无时滞控制器）');
grid on;
```

### 6.2 频域分析代码（Bode图与裕度分析）

```matlab
clc; close all; clear

% 1. 定义系统参数和控制参数
m = 1;      % 质量 (kg)
c = 0.1;    % 阻尼系数 (N*s/m)
kp = 1;     % 固有弹簧刚度 (N/m)

kc = 5;     % 速度反馈增益 (N/m)
tau = 0.4;  % 时间延迟 (s)

% 2. 构造开环传递函数模型 L(s)
s = tf('s'); % 拉普拉斯变量

% 物理系统的传递函数 G_plant(s) = X(s)/F_ext(s)
G_plant = 1 / (m*s^2 + c*s + kp);

% 速度反馈控制器 C_v(s) = Kc * s (无延迟部分)
C_v_ideal = kc * s;

% 2.1. 无时滞开环系统 L_ideal(s)
L_ideal = G_plant * (kc * s);

% 2.2. 有时滞开环系统 L_delay(s)
L_delay = G_plant * (kc * s * exp(-tau*s));

% 3. 绘制伯德图并进行裕度分析
figure('Name','速度反馈开环系统伯德图对比');
h = bodeplot(L_ideal, L_delay);

setoptions(h, 'MagUnits', 'dB', 'PhaseUnits', 'deg', 'Grid', 'on');
title('速度反馈开环系统伯德图对比 (L_{ideal} vs L_{delay})');
legend('无时滞 L_{ideal} (\tau=0)', ['有时滞 L_{delay} (\tau=', num2str(tau), 's)'], 'Location', 'SouthWest');
grid on;

% 4. 稳定性裕度计算
disp('--- 稳定性裕度分析 (速度反馈增益 kc = 5) ---');

% 4.1. 无时滞系统分析
[Gm_ideal, Pm_ideal, Wcg_ideal, Wcp_ideal] = margin(L_ideal);
disp('【无时滞系统 L_ideal】:');
disp(['  相位裕度 (PM): ', num2str(Pm_ideal), ' 度']);
disp(['  增益裕度 (GM): ', num2str(20*log10(Gm_ideal)), ' dB']);
disp(['  结论: 闭环系统稳定 (PM>0)，与仿真结果一致。']);

disp(' ');

% 4.2. 有时滞系统分析
[Gm_delay, Pm_delay, Wcg_delay, Wcp_delay] = margin(L_delay);
disp(['【有时滞系统 L_delay (\tau=', num2str(tau), 's)】:']);
disp(['  相位裕度 (PM): ', num2str(Pm_delay), ' 度']);
disp(['  增益裕度 (GM): ', num2str(20*log10(Gm_delay)), ' dB']);
disp(['  结论: 闭环系统不稳定 (PM < 0)，与您的仿真结果一致。']);
```

### 6.3 Bode与Nyquist图稳定性分析

```matlab
 %% 系统参数定义
clc; close all; clear

m = 1; c = 0.1; kp = 1; kc = 5; tau = 0.4;
s = tf('s');
G_plant = 1 / (m*s^2 + c*s + kp);
L_delay = G_plant * (kc * s * exp(-tau*s));

%% 1. 绘制 Bode 图（含时滞系统）
figure('Name', 'Bode Diagram', 'NumberTitle', 'off');
bode(L_delay, {1e-2, 1e2});  % 频率范围从 0.01 到 100 rad/s
grid on;
title('Bode Diagram of L_{delay}(s) with Time Delay \tau = 0.4s');

%% 2. 绘制奈奎斯特曲线（Nyquist Plot）
figure('Name', 'Nyquist Plot', 'NumberTitle', 'off');
nyquist(L_delay, {1e-2, 1e2});  % 指定频率范围
grid on;
title('Nyquist Plot of L_{delay}(s) with Time Delay \tau = 0.4s');

% 可选：添加单位圆和临界点 (-1, 0) 以评估稳定性
hold on;
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--');  % 单位圆
plot(-1, 0, 'ro', 'MarkerFaceColor', 'r');  % 临界点 (-1, 0)
text(-1.1, 0.1, '(-1, 0)', 'Color', 'red', 'FontSize', 10);
xlabel('Real Axis');
ylabel('Imaginary Axis');
legend('Nyquist Curve', 'Unit Circle', 'Critical Point (-1,0)', 'Location', 'best');
hold off;

%% 3. 稳定性分析
[GM, PM, Wcg, Wcp] = margin(L_delay);
fprintf('=== 稳定性分析结果 ===\n');
fprintf('增益裕度 (Gain Margin): %.2f dB\n',20*log10(GM));
fprintf('相位裕度 (Phase Margin): %.4f 度\n', PM);
fprintf('相位穿越频率 (Wcg): %.4f rad/s\n', Wcg);
fprintf('增益穿越频率 (Wcp): %.4f rad/s\n', Wcp);
```

  

## 七. 总结：学习心得

这次从弹簧共振出发的探索，最终演变成了一场关于控制系统稳定性的深度研究。我学到了：

1. **物理直觉是基础：** 理解"做正功"是理解共振和失稳的钥匙。
2. **时域是现象，频域是病因：** 时域仿真（如发散）是结果，而频域的裕度分析（如 PM < 0）是导致该结果的根本原因。
3. **时间延迟是"相位杀手"：** 延迟 $\tau$ 通过引入 $-\omega\tau$ 的相位滞后，迅速侵蚀了系统的相位裕度 (PM)，这是导致主动控制系统失稳的核心原因。
4. **裕度即"容错率"：** PM 是对延迟的容忍度，GM 是对增益的容忍度。在设计控制系统时，必须留有足够的裕度来应对现实世界的不确定性。
5. **闭环极点揭示本质：** 时域发散频率本质上是闭环系统不稳定极点的虚部，通过 Pade 近似和数值方法可以精确推导，完成从现象到本质的理论闭环。