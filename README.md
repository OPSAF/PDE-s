# TDSE - Time-Dependent Schrödinger Equation Package
# TDSE - 含时薛定谔方程数值求解包

A modular Python package for numerical solutions of the time-dependent Schrödinger equation (TDSE) in 1D and 2D.

一个用于求解一维和二维含时薛定谔方程（TDSE）的模块化 Python 数值计算包。

---

## Project Structure | 项目结构

```
xde/
├── tdse/                      # Main package | 主包目录
│   ├── __init__.py            # Package initialization | 包初始化
│   ├── config.py              # Configuration classes | 配置类
│   ├── potentials.py          # Potentials and wave functions | 势场与波函数
│   ├── solvers.py             # Numerical solvers | 数值求解器
│   └── visualization.py       # Visualization tools | 可视化工具
├── LATEX/                     # LaTeX documentation | LaTeX 文档
│   └── report.tex             # Technical report | 技术报告
├── tdse_outputs/              # Output directory | 输出目录
│   ├── figure*.png            # Visualization plots | 可视化图像
│   ├── figure*.gif            # Animations | 动画
│   └── *.csv                  # Numerical data | 数值数据
├── demo2.py                   # Full experiment suite | 完整实验套件
├── ex_pinn.py                 # PINN (Physics-Informed Neural Networks) | 物理信息神经网络
├── main.py                    # Main entry point | 主入口
├── example.py                 # Usage examples | 使用示例
├── tdse_1d_demo.py            # 1D demonstration | 一维演示
├── run_pinn.bat/.ps1/.sh      # PINN run scripts | PINN 运行脚本
└── README.md                  # This file | 本文件
```

---

## Installation | 安装

No installation required! Just ensure all files are in the same directory and install dependencies:

无需安装！确保所有文件在同一目录，然后安装依赖：

```bash
pip install numpy scipy matplotlib pandas tqdm
```

---

## Quick Start | 快速开始

### Basic Usage | 基础用法

```python
import tdse

# Create configuration | 创建配置
config = tdse.TDSEConfig(quick=True)

# Setup grid and initial state | 设置网格和初始状态
x, dx = tdse.grid(-30.0, 30.0, 384)
psi0 = tdse.gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
v = tdse.potential_free(x)

# Run simulation | 运行模拟
t = np.linspace(0, 10, 200)
dt = t[1] - t[0]
saved_t, psi_hist = tdse.solve("Crank-Nicolson", psi0, v, x, t, dx, dt)

# Visualize | 可视化
viz = tdse.TDSEVisualizer(config)
viz.plot_wavepacket(x, psi_hist[-1], title="Final State")
```

### Using Factories | 使用工厂模式

```python
# Create potentials | 创建势场
factory = tdse.TDSEPotentialFactory()
potential = factory.create("barrier", v0=1.0, a=-0.5, b=0.5)

# Create solvers | 创建求解器
solver_factory = tdse.TDSESolverFactory()
solver = solver_factory.create("CN")
```

---

## Running Experiments | 运行实验

### Full Experiment Suite | 完整实验套件

```bash
python main.py                   # Run all experiments | 运行所有实验
python main.py --quick           # Quick mode with smaller grids | 快速模式
python main.py --no-gif          # Skip GIF animation generation | 跳过动画
python main.py --outdir OUTPUT   # Custom output directory | 自定义输出目录
```

### Selective Execution | 选择性执行

```bash
python main.py --1d-only         # Only 1D experiments | 仅运行一维实验
python main.py --2d-only         # Only 2D experiments | 仅运行二维实验
python main.py --analysis-only   # Only analysis experiments | 仅运行分析实验
```

### PINN Experiments | PINN 实验

```bash
python ex_pinn.py                # Run PINN experiments | 运行 PINN 实验
# Or use scripts | 或使用脚本
./run_pinn.sh                    # Linux/Mac
.\run_pinn.bat                   # Windows
.\run_pinn.ps1                   # PowerShell
```

---

## Package Modules | 包模块

### 1. config - Configuration | 配置模块

- `TDSEConfig`: Main configuration class | 主配置类
- `RunConfig`: Legacy compatibility | 遗留兼容性配置
- `ensure_outdir()`: Create output directory | 创建输出目录
- `setup_plot_style()`: Configure matplotlib | 配置 matplotlib

### 2. potentials - Potentials and Wave Functions | 势场与波函数模块

**Base Classes | 基类:**
- `Potential1D`, `Potential2D`

**Concrete Classes | 具体实现类:**
- `FreePotential` | 自由势场
- `HarmonicPotential` | 谐振子势
- `RectangularBarrier` | 矩形势垒
- `CircularObstacle` | 圆形障碍物
- `Waveguide` | 波导

**Factory | 工厂类:**
- `TDSEPotentialFactory`

**Functions | 函数:**
- `grid()`: Create spatial grid | 创建空间网格
- `gaussian_wavepacket()`: 1D Gaussian wave packet | 一维高斯波包
- `gaussian_wavepacket_2d()`: 2D Gaussian wave packet | 二维高斯波包
- `exact_free_gaussian()`: 1D analytical solution | 一维解析解
- `exact_free_gaussian_2d()`: 2D analytical solution | 二维解析解
- `potential_free()`: Free potential | 自由势场
- `potential_rect_barrier()`: Rectangular barrier | 矩形势垒
- `compute_transmission_reflection()`: T/R coefficients | 透射/反射系数
- `analyze_barrier_scattering()`: Detailed scattering analysis | 详细散射分析
- `probability_mass()`: Compute probability mass | 计算概率质量
- `mass_2d()`: Compute 2D probability mass | 计算二维概率质量
- `l1_l2_linf_error()`: Error norm calculation | 误差范数计算
- `normalize()`: Normalize wave function | 归一化波函数

### 3. solvers - Numerical Methods | 数值方法模块

**Base Classes | 基类:**
- `Solver1D`

**1D Solvers | 一维求解器:**
- `FTCSSolver`: Forward-Time Centered-Space (explicit) | FTCS 显式方法
- `BackwardEulerSolver`: Implicit backward Euler | 向后欧拉隐式方法
- `CrankNicolsonSolver`: Crank-Nicolson (CN) | Crank-Nicolson 方法
- `RKK4Solver`: Runge-Kutta 4th order | 四阶龙格-库塔方法
- `SplitStepFFTSolver`: Split-Step Fourier Transform | 分裂步傅里叶变换

**2D Solvers | 二维求解器:**
- `solve_2d()`: 2D time evolution with method selection
  - `"split-step-fft"`: Split-Step FFT (default) | 分裂步 FFT
  - `"adi"`: Alternating Direction Implicit | 交替方向隐式方法

**Factory | 工厂类:**
- `TDSESolverFactory`

### 4. visualization - Plotting and Animation | 可视化模块

- `TDSEVisualizer`: Unified visualization class | 统一可视化类
- `save_wavepacket_animation()`: 1D wave packet animation | 一维波包动画
- `save_tunneling_animation()`: Tunneling animation | 隧道效应动画
- `save_2d_animation()`: 2D animation | 二维动画

---

## Features | 功能特性

### 1D Capabilities | 一维功能

- Multiple numerical methods (FTCS, Backward Euler, Crank-Nicolson, RK4, Split-Step FFT)
- Convergence and stability analysis
- Quantum tunneling simulations
- Mass conservation verification
- Transmission/reflection calculations
- Performance comparison

### 2D Capabilities | 二维功能

- Split-Step FFT for 2D simulations
- ADI (Alternating Direction Implicit) method
- Free Gaussian wave packet propagation
- Circular obstacle scattering
- Waveguide confinement
- Absorbing boundary layers (PML)
- Grid convergence studies
- Error analysis and visualization

### Analysis Tools | 分析工具

- L1, L2, L∞ error norms
- Convergence order estimation
- Stability analysis and phase diagrams
- Runtime comparison
- Parameter sweeps (obstacle radius, waveguide strength)
- Mass conservation analysis

### Visualization | 可视化

- 1D wave packet plots
- 2D density heatmaps
- Animation generation (GIF format)
- Convergence plots
- Stability maps
- Performance comparison tables

---

## Experiments | 实验列表

| Experiment | Description | Output |
|------------|-------------|--------|
| `experiment_analytic_vs_numerical` | Analytic vs numerical comparison | figure1_analytic_vs_numerical.png |
| `experiment_convergence` | Convergence order analysis | figure2_error_convergence.png |
| `experiment_stability` | Stability phase diagram | figure3_stability_map.png |
| `experiment_performance` | Performance comparison | figure4_performance_table.png |
| `experiment_method_comparison` | Method comparison | method_comparison.png |
| `experiment_tunneling` | Quantum tunneling | figure6_tunneling_simulation.png/gif |
| `experiment_2d_free_propagation` | 2D free propagation | figure7_2d_free_propagation.png |
| `experiment_2d_circular_obstacle` | Circular obstacle scattering | figure8_2d_circle_scatter.png/gif |
| `experiment_2d_waveguide` | Waveguide confinement | figure9_2d_waveguide.png/gif |
| `experiment_circular_obstacle_radius_sweep` | Parameter sweep | circular_obstacle_sweep.csv |
| `experiment_waveguide_strength_sweep` | Waveguide sweep | waveguide_sweep.csv |
| `experiment_1d_conservation_analysis` | Mass conservation | figure_conservation_analysis.png |
| `experiment_runtime_comparison` | Runtime comparison | figure_runtime_comparison.png |
| `experiment_2d_convergence` | 2D convergence study | figure_2d_convergence.png |
| `experiment_2d_error_heatmap` | Error heatmap | figure_2d_error_heatmap.png |

---

## Output Files | 输出文件

### Visualizations | 可视化文件
- `figure1_analytic_vs_numerical.png`: Analytic vs numerical comparison
- `figure2_error_convergence.png`: Convergence plot
- `figure3_stability_map.png`: Stability phase diagram
- `figure4_performance_table.png`: Performance table
- `figure5_wavepacket_animation.gif`: Wave packet animation
- `figure6_tunneling_animation.gif`: Tunneling animation
- `figure7_2d_free_propagation.png`: 2D free propagation
- `figure8_2d_circle_scatter.png`: Circular obstacle scattering
- `figure9_2d_waveguide.png`: Waveguide simulation
- `figure_2d_convergence.png`: 2D convergence
- `figure_2d_error_heatmap.png`: Error heatmap

### Data Files | 数据文件
- `convergence_errors.csv`: Convergence error data
- `convergence_orders.csv`: Convergence orders
- `stability_scan.csv`: Stability scan data
- `performance_table.csv`: Performance metrics
- `runtime_comparison.csv`: Runtime comparison
- `tunneling_RT.csv`: Transmission/reflection data
- `circular_obstacle_sweep.csv`: Obstacle sweep data
- `waveguide_sweep.csv`: Waveguide sweep data
- `2d_convergence_study.csv`: 2D convergence data

---

## PINN Extension | PINN 扩展

The `ex_pinn.py` module implements Physics-Informed Neural Networks for solving TDSE:

- PINN-based TDSE solver
- Training loss visualization
- Error analysis
- Phase portraits
- Runtime comparison with traditional methods

---

## Requirements | 依赖

- numpy >= 1.20
- scipy >= 1.6
- matplotlib >= 3.3
- pandas >= 1.2
- tqdm >= 4.50

---

## Version | 版本

1.0.0

---

## Author | 作者

TDSE Development Team | TDSE 开发团队

---

## License | 许可证

MIT License