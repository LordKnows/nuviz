# NuViz — Terminal-Native ML Training Visualization Tool

> **一行命令，掌握训练全局。不离开终端，完成从实验管理到论文出图的全流程。**

---

## 1. 项目定位

### 1.1 是什么

NuViz 是一个面向 ML 研究者的终端原生训练可视化工具，由两个核心组件组成：

- **`nuviz` (Python 库)** — 轻量级结构化 Logger，嵌入训练脚本，负责记录指标、图片、点云等多模态数据
- **`nuviz` (Rust CLI)** — 终端 TUI 可视化工具，负责实时监控、实验对比、消融分析、论文出图

### 1.2 为什么

| 现有工具 | 痛点 |
|---|---|
| TensorBoard | 启动慢，多实验对比体验差，Google 基本停止维护 |
| Weights & Biases | SaaS 依赖，数据上传云端，网络问题卡住训练，免费版有限制 |
| Aim / ClearML / MLflow | 全部是 Web UI，需要开浏览器，离终端工作流割裂 |
| Rerun | GUI-first，偏 robotics，对 3DGS/NeRF 训练没专门优化 |

**核心洞察：训练时终端本来就开着，为什么还要切到浏览器？**

### 1.3 目标用户

- 3D Vision / NeRF / Gaussian Splatting 研究者（核心圈层）
- 广义 CV / ML 研究者（扩展圈层）
- 偏好终端工作流的 ML 工程师

### 1.4 设计原则

1. **两行代码接入** — Python 库零配置，`import` + `log.step()` 即可
2. **终端原生** — 所有核心功能不离开终端，GUI/Web 仅作可选增强
3. **消融实验优先** — 不做通用 MLOps 平台，专注研究者的消融实验场景
4. **论文友好** — 直接输出 LaTeX 表格、对比拼图、可复现的实验记录

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                   训练脚本 (Python)                    │
│                                                       │
│   from nuviz import Logger                            │
│   log = Logger("exp-001")                             │
│   log.step(i, loss=loss, psnr=psnr)                  │
│   log.image("render", img)                            │
│   log.pointcloud("splats", pts, colors)               │
│                                                       │
└──────────────┬────────────────────────────────────────┘
               │ 写入 JSONL + 二进制文件
               ▼
        ┌──────────────────┐
        │  ~/.nuviz/        │
        │  experiments/     │
        │  ├── metrics.jsonl│
        │  ├── images/      │
        │  └── pointclouds/ │
        └────────┬─────────┘
                 │ inotify / kqueue 文件监听
                 ▼
        ┌──────────────────────────┐
        │      nuviz (Rust CLI)    │
        │                          │
        │  nuviz watch exp-001     │
        │  nuviz compare exp-*     │
        │  nuviz leaderboard       │
        │  nuviz view --stats      │
        │  nuviz table --latex     │
        └──────────────────────────┘
```

> **设计决策：** 使用 inotify/kqueue 文件监听代替 Unix Socket/UDP 实时推送。
> 更简单、无 daemon、跨 SSH 场景天然兼容（共享文件系统即可）。
> 如未来文件监听延迟成为瓶颈，再升级为 socket 方案。

### 2.2 数据存储格式

```
~/.nuviz/
└── experiments/
    └── gaussian_splatting/          # 项目名
        ├── lr1e-4_sh2_20260319/     # 用户命名 + 时间戳 fallback
        │   ├── meta.json            # 实验元信息、环境快照
        │   ├── metrics.jsonl        # 标量指标流（核心）
        │   ├── images/              # 渲染结果图片
        │   │   ├── step_0500_render.png
        │   │   └── step_0500_gt.png
        │   ├── pointclouds/         # PLY 快照
        │   │   └── step_5000.ply
        │   ├── checkpoints.json     # checkpoint 索引
        │   └── config.yaml          # 完整训练配置
        └── lr1e-3_sh3_20260319/
            └── ...
```

> **设计决策：** 实验命名使用用户提供的名称 + 时间戳 fallback，
> 而非"基于超参 diff 自动命名"。自动命名在实践中容易碰撞且可读性差。
> 提供 `--name-from-config` 作为 opt-in 功能。

**`metrics.jsonl` 格式：**

```json
{"step": 100, "timestamp": 1711234567.89, "metrics": {"loss": 0.0523, "psnr": 24.31, "ssim": 0.912, "lr": 1e-4}, "gpu": {"util": 87, "mem_used": 10240, "temp": 72}}
{"step": 200, "timestamp": 1711234590.12, "metrics": {"loss": 0.0481, "psnr": 25.02, "ssim": 0.921, "lr": 1e-4}, "gpu": {"util": 92, "mem_used": 10240, "temp": 73}}
```

### 2.3 通信机制

- **持久化层**：JSONL 文件，内存 buffer 每 N 秒 / 每 M 条 flush 一次
- **实时层**：CLI 端使用 inotify (Linux) / kqueue (macOS) 监听 JSONL 文件变更，实时读取新增行
- **图片/点云**：直接写入文件系统，JSONL 中记录引用路径
- **大文件处理**：JSONL 自动 rotate（按大小或行数），CLI 端支持多文件合并读取

---

## 3. Python 库设计 (`nuviz`)

> **最低 Python 版本：3.10+**（支持 `match`、`X | Y` 类型语法、`dataclasses(slots=True)`）

### 3.1 核心 API

```python
from nuviz import Logger

# ========== 基础用法（两行接入）==========

log = Logger("exp-001", project="gaussian_splatting")

for step in range(30000):
    loss = train_step()
    psnr, ssim, lpips = evaluate()

    # 核心：一行记录所有标量
    log.step(step, loss=loss, psnr=psnr, ssim=ssim, lpips=lpips, lr=optimizer.param_groups[0]['lr'])

    # 记录渲染图片（按需）
    if step % 500 == 0:
        log.image("render", pred_image)          # numpy array 或 torch tensor
        log.image("gt", gt_image)
        log.image("depth", depth_map, cmap="turbo")  # 支持 colormap

    # 记录点云快照（按需）
    if step % 5000 == 0:
        log.pointcloud("gaussians", xyz, colors, opacities)

# ========== Per-Scene 评估 ==========

for scene in ["bicycle", "garden", "stump", "room", "counter"]:
    metrics = evaluate_scene(scene)
    log.scene(scene, psnr=metrics["psnr"], ssim=metrics["ssim"], lpips=metrics["lpips"])

log.finish()
```

> **设计决策：** 移除 `log.camera()` API。相机位姿记录是极小众需求，
> 即使在 3D vision 圈层也不常用于训练监控。如有需求可通过 `log.custom()` 扩展。

### 3.2 消融实验 API

```python
from nuviz import Ablation

# 声明式消融定义
ab = Ablation("3dgs_ablation", base_config="configs/base.yaml")

ab.vary("learning_rate", [1e-3, 1e-4, 1e-5])
ab.vary("sh_degree", [0, 1, 2, 3])
ab.toggle("densification", True, False)
ab.vary("opacity_reset_interval", [3000, 5000])

# 生成实验矩阵（输出配置文件，不负责启动）
configs = ab.generate()   # 返回 list of config dicts
print(f"共 {len(configs)} 组实验")

# 导出为可执行的配置文件
ab.export("configs/ablation/")  # 每组实验一个 YAML 文件
```

> **设计决策：** 移除 `ab.launch()` API（包括 SLURM 和本地多 GPU 调度）。
> 作业编排是一个独立的复杂领域，不属于可视化工具的职责。
> `Ablation` 类仅负责生成配置，用户通过自己的脚本或 SLURM 提交。
> 这也避免了 GPU 分配、错误恢复、队列管理等大量边界情况。

### 3.3 自动环境快照

```python
log = Logger("exp-001", project="my_project", snapshot=True)
# 自动记录到 meta.json:
# - git commit hash + dirty status
# - pip freeze / conda list
# - CUDA 版本、GPU 型号
# - 完整 config / argparse 参数
# - Python 版本、PyTorch 版本
# - 主机名、启动时间
```

### 3.4 异常检测

```python
log = Logger("exp-001", alerts=True)
# MVP 阶段仅检测：
# - loss NaN / Inf
# - loss 突然 spike（偏离移动平均 > 3σ）
```

> **设计决策：** MVP 仅实现 NaN/Inf 和 spike 检测。
> 梯度爆炸监控需要 hook 训练循环（侵入性高），OOM 预警和训练停滞检测
> 的阈值调优容易产生误报。这些功能推迟到用户反馈确认需求后再加。

### 3.5 设计约束

- **零 torch 依赖**：核心库只依赖标准库 + numpy。torch tensor 通过 `.detach().cpu().numpy()` 自动转换，但 `import torch` 仅在需要时动态导入
- **线程安全**：Logger 内部用独立写入线程，`log.step()` 非阻塞，不影响训练速度
- **容错**：日志写入失败不 crash 训练，静默降级 + stderr 警告

---

## 4. Rust CLI 设计 (`nuviz`)

### 4.1 命令体系总览

```
nuviz — Terminal-native ML training visualization

USAGE:
    nuviz <COMMAND>

核心命令:
    watch       实时监控训练过程（TUI Dashboard）
    compare     多实验曲线叠加对比
    leaderboard 实验指标排行榜
    matrix      消融实验矩阵视图
    breakdown   Per-scene 指标分解

可视化命令:
    image       终端内浏览图片（渲染结果、热力图）
    diff        两个实验的逐像素质量对比
    view        查看 3D 点云 / PLY 文件统计信息

论文辅助命令:
    table       生成 LaTeX / Markdown 指标表格
    export      导出数据为 CSV / JSON

实验管理命令:
    ls          列出所有实验
    tag         给实验打标签
    env         查看实验环境快照
    cleanup     清理低价值 checkpoint
    reproduce   输出完整复现命令
```

> **设计决策：**
> - 移除 `nuviz summary`（网络结构摘要）— 用户已有 `torchinfo`，从 Rust 端解析 PyTorch 模型复杂度高、收益低。
> - 移除 `nuviz figure`（论文对比图生成）— 构建图片合成/裁剪/布局引擎偏离核心价值。如有需求，后续通过 Python helper（matplotlib）实现。
> - `nuviz view` 仅提供 `--stats` 模式（统计信息），不含终端内交互式 3D 渲染。

### 4.2 核心命令详细设计

#### `nuviz watch` — 实时 TUI Dashboard

```
nuviz watch exp-001
nuviz watch exp-001 exp-002          # 同时监控多个
nuviz watch --project gaussian_splatting --latest 3   # 最近 3 个实验
```

TUI 布局：

```
┌─ Loss ──────────────────────────┬─ PSNR ─────────────────────────┐
│ 0.08┤                             │ 30┤                    ╭────── │
│     │╲                            │   │              ╭─────╯       │
│ 0.04┤ ╲_____                      │ 25┤        ╭─────╯             │
│     │       ╲___________          │   │  ╭─────╯                   │
│ 0.02┤                   ╲──       │ 20┤──╯                         │
│     ├────┬────┬────┬────┬──       │   ├────┬────┬────┬────┬──      │
│     0   5k  10k  15k  20k        │   0   5k  10k  15k  20k       │
├─ GPU ───────────────────────────┼─ Info ─────────────────────────┤
│ Util: ████████████░░░ 87%        │ Experiment: lr1e-4_sh2         │
│ Mem:  ██████████████░ 10.2/12GB  │ Step: 18,432 / 30,000         │
│ Temp: 72°C                       │ ETA: 23m 14s                   │
│                                  │ Best PSNR: 28.41 @ step 16k   │
│ ⚠ Alert: loss spike @ step 12k  │ Config: lr=1e-4, sh=2          │
└──────────────────────────────────┴────────────────────────────────┘
```

功能：
- 实时曲线绘制（braille 字符）
- GPU 利用率 / 显存 / 温度监控
- 训练 ETA 预估
- 异常事件高亮
- 最新渲染图片预览（Kitty/Sixel 协议）
- 快捷键：`q` 退出, `Tab` 切换面板, `[/]` 缩放时间轴, `i` 查看图片

#### `nuviz compare` — 多实验对比

```
nuviz compare exp-001 exp-002 exp-003
nuviz compare --project 3dgs --tag ablation
nuviz compare exp-* --metric psnr --align step
```

TUI 布局：

```
┌─ PSNR Comparison ──────────────────────────────────────────┐
│ 30┤                                                         │
│   │          ╭── exp-001 (lr=1e-4) ── 28.41               │
│ 28┤    ╭─────╯                                              │
│   │  ╭─╯──── exp-002 (lr=1e-3) ── 27.89                   │
│ 26┤──╯                                                      │
│   │  ╭────── exp-003 (lr=1e-5) ── 26.12                   │
│ 24┤──╯                                                      │
│   ├────────┬────────┬────────┬────────┬──                   │
│   0       5k      10k      15k      20k                    │
├─ Legend ────────────────────────────────────────────────────┤
│ ── exp-001 (best)  ── exp-002  ── exp-003                  │
│ Δ best vs 2nd: +0.52 dB  |  Converge: exp-002 fastest     │
└─────────────────────────────────────────────────────────────┘
```

功能：
- 不同颜色叠加曲线
- 自动标注：最佳值、收敛拐点、显著差异
- 支持 `--align wall_time | step | epoch`
- 交互式：光标移动到某 step 时显示所有实验在该点的值

#### `nuviz leaderboard` — 排行榜

```
nuviz leaderboard --project gaussian_splatting
nuviz leaderboard --sort psnr --top 10
```

输出：

```
 Rank │ Experiment                │ PSNR ↑ │ SSIM ↑ │ LPIPS ↓ │ Train Time │ Status
──────┼───────────────────────────┼────────┼────────┼─────────┼────────────┼────────
  1   │ lr1e-4_sh3_densify-on     │ 28.41  │ 0.934  │ 0.081   │ 2h 14m     │ ✓ done
  2   │ lr1e-3_sh3_densify-on     │ 27.89  │ 0.921  │ 0.093   │ 1h 58m     │ ✓ done
  3   │ lr1e-4_sh2_densify-on     │ 27.52  │ 0.918  │ 0.098   │ 2h 01m     │ ✓ done
  ·   │ ·························  │ ······ │ ······ │ ······· │ ·········· │ ·····
  8   │ lr1e-5_sh0_densify-off    │ 23.11  │ 0.856  │ 0.172   │ 3h 22m     │ ✓ done
──────┼───────────────────────────┼────────┼────────┼─────────┼────────────┼────────
      │ Mean ± Std                │ 26.2±1.8│0.91±.03│0.12±.03│            │
```

功能：
- 按任意指标排序
- 多 seed 运行时自动聚合（均值 ± 标准差）
- `--format markdown | latex | csv` 多格式导出

> **设计决策：** 统计显著性检测（配对 t-test）推迟到 post-launch。
> 需要用户跑多 seed 实验才有意义，MVP 阶段先提供 mean ± std 即可。

#### `nuviz matrix` — 消融矩阵

```
nuviz matrix --project 3dgs_ablation
nuviz matrix --rows lr --cols sh_degree --metric psnr
```

输出：

```
 PSNR ↑       │ sh=0    │ sh=1    │ sh=2    │ sh=3
──────────────┼─────────┼─────────┼─────────┼─────────
 lr=1e-3      │  24.12  │  25.89  │  27.01  │  27.89*
 lr=1e-4      │  24.55  │  26.31  │  27.52  │  28.41*  ← best
 lr=1e-5      │  23.11  │  24.98  │  25.67  │  26.12
──────────────┼─────────┼─────────┼─────────┼─────────
 Δ max-min    │  1.44   │  1.33   │  1.85   │  2.29

 Key findings:
 • sh_degree 影响最大 (Δ=4.3 dB across range)
 • lr=1e-4 在所有 sh_degree 下均为最优
 • densify=on 平均提升 1.2 dB (not shown, use --expand)
```

#### `nuviz breakdown` — Per-Scene 指标分解

```
nuviz breakdown exp-001
nuviz breakdown exp-001 --latex
nuviz breakdown exp-001 exp-002 --diff
```

输出：

```
 exp-001: lr1e-4_sh3_densify-on
 Scene      │ PSNR ↑ │ SSIM ↑ │ LPIPS ↓
────────────┼────────┼────────┼─────────
 bicycle    │ 25.12  │ 0.912  │ 0.098
 garden     │ 27.41  │ 0.945  │ 0.071
 stump      │ 26.88  │ 0.931  │ 0.085
 room       │ 31.24  │ 0.962  │ 0.043
 counter    │ 29.67  │ 0.951  │ 0.052
────────────┼────────┼────────┼─────────
 Mean       │ 28.06  │ 0.940  │ 0.070
```

`--latex` 输出：

```latex
\begin{table}[t]
\centering
\caption{Per-scene quantitative results on Mip-NeRF 360 dataset.}
\label{tab:per_scene}
\begin{tabular}{lccc}
\toprule
Scene & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\
\midrule
Bicycle & 25.12 & 0.912 & 0.098 \\
Garden  & 27.41 & 0.945 & 0.071 \\
...
\bottomrule
\end{tabular}
\end{table}
```

#### `nuviz image` — 终端图片浏览

```
nuviz image exp-001/render/             # 浏览目录下所有图片
nuviz image exp-001 --step 5000         # 查看特定 step 的渲染结果
nuviz image exp-001 --latest            # 最新一张
```

功能：
- Kitty graphics protocol > Sixel > 半块字符 自动 fallback
- 方向键翻页，支持 side-by-side 对比（render vs gt）
- 支持 depth map / normal map 的伪彩色可视化

#### `nuviz diff` — 渲染质量对比

```
nuviz diff exp-001 exp-002 --scene bicycle --step 30000
nuviz diff exp-001 exp-002 --heatmap     # 逐像素误差热力图
```

功能：
- 并排显示两个实验的渲染结果
- 差异热力图（绝对误差 / SSIM map）
- 底部输出 PSNR / SSIM / LPIPS 数值对比

#### `nuviz view` — 3D 点云 / PLY 统计

```
nuviz view model.ply                     # 默认输出统计信息
nuviz view model.ply --stats             # 等同于默认行为
nuviz view model.ply --histogram         # 附加属性分布直方图
```

输出：

```
 model.ply — Gaussian Splatting Point Cloud
─────────────────────────────────────────
 Gaussians:     1,247,832
 Bounding Box:  [-3.2, -1.8, -0.5] → [4.1, 2.3, 3.7]
 SH Degree:     3
 Opacity:       mean=0.72, std=0.18, <0.01: 12.3%
 Scale:         mean=0.0034, max=0.42 (⚠ 23 outliers)
 Memory:        487 MB

 Opacity Distribution:
 [0.0-0.2) ████░░░░░░░░░░░░ 15.2%
 [0.2-0.4) ██████░░░░░░░░░░ 18.7%
 [0.4-0.6) ████████░░░░░░░░ 22.1%
 [0.6-0.8) ██████████░░░░░░ 24.8%
 [0.8-1.0] ████████░░░░░░░░ 19.2%
```

> **设计决策：**
> - 移除终端内交互式 3D 渲染（软光栅 + Kitty/Sixel 帧推送）。终端图形协议的传输开销使 10-15 fps 不现实，且交互体验远不如专业工具。
> - 移除 `--web` WebGPU viewer。构建一个完整的 3DGS web 渲染器是独立项目级别的工作量。推荐用户使用现有开源 viewer（如 antimatter15/splat）。
> - `nuviz view` 聚焦于快速了解 PLY 文件的统计信息，这才是训练监控中最常用的场景。

#### `nuviz table` — LaTeX 表格生成

```
nuviz table --project 3dgs --experiments exp-001,exp-002,exp-003 --latex
nuviz table --project 3dgs --best-per-group --bold-best --latex
```

自动生成论文级别的 LaTeX 表格，最佳值加粗，次佳下划线。

#### `nuviz cleanup` — Checkpoint 清理

```
nuviz cleanup --project gaussian_splatting --keep-top 5
nuviz cleanup --project gaussian_splatting --dry-run   # 仅预览
```

分析所有实验，标记可安全删除的 checkpoint（被更好结果超越的中间实验），预估可回收空间。

#### `nuviz reproduce` — 复现命令生成

```
nuviz reproduce exp-001
```

输出完整复现信息：git checkout 命令、conda 环境、启动命令、config。

---

## 5. 终端图像渲染技术方案

### 5.1 协议优先级

| 优先级 | 协议 | 支持终端 | 色彩能力 | 图片质量 |
|---|---|---|---|---|
| 1 | Kitty graphics protocol | Kitty, WezTerm, Ghostty | 24-bit RGBA | 原生分辨率 |
| 2 | iTerm2 inline images | iTerm2, WezTerm, mintty | 24-bit RGB | 原生分辨率 |
| 3 | Sixel | xterm, mlterm, foot, WezTerm | 通常 256 色 | 较好 |
| 4 | Half-block fallback (▀▄) | 所有终端 | 取决于终端色彩 | 像素化 |

### 5.2 实现策略

- Rust 端使用 `viuer` 库，自动检测终端能力并选择最佳协议
- 图片在 CLI 端缩放到终端窗口尺寸（查询 `$COLUMNS` × `$LINES` + 像素尺寸）
- 大图支持交互式缩放/平移

---

## 6. 技术栈

| 组件 | 技术选型 | 理由 |
|---|---|---|
| Python 库 | Python 3.10+, numpy | 现代语法特性 + 零额外依赖 |
| CLI 核心 | Rust | 性能、单二进制分发、ratatui 生态 |
| TUI 框架 | ratatui | Rust TUI 事实标准 |
| JSONL 解析 | serde_json (Rust) | 高性能流式解析 |
| 文件监听 | notify (Rust) | 跨平台 inotify/kqueue/FSEvents 封装 |
| 终端图片 | viuer | 自动协议检测 + fallback |
| PLY 解析 | ply-rs 或自研 | 3DGS PLY 格式有自定义属性 |
| 曲线绘制 | 自研 (braille chars) | 完全控制布局和交互 |
| 论文出图 | resvg / plotters | 高质量 PDF/PNG 导出（Phase 4 stretch） |
| 分发 | cargo install + PyPI (wrapper) + brew | 多渠道覆盖 |

> **移除项：** WebGPU + wgpu（3D viewer 已砍掉）

---

## 7. 开发路线图

### Phase 1 — MVP：Logger + Watch + Leaderboard（5 周）

**目标：** 能跑通"训练 → 记录 → 终端实时查看 → 排行榜"的完整闭环。

Week 1-3: Python Logger
- [ ] 核心 Logger 类：`step()`, `image()`, `finish()`
- [ ] JSONL 写入 + buffer flush 机制
- [ ] 用户命名 + 时间戳 fallback 实验命名
- [ ] 环境快照（git hash, pip freeze, GPU info）
- [ ] NaN/Inf 检测 + loss spike 检测
- [ ] 单元测试 + 集成测试

Week 3-5: Rust CLI 基础
- [ ] 项目脚手架：clap 命令行解析 + ratatui 初始化
- [ ] `nuviz watch`：inotify 监听 JSONL + 曲线绘制（braille）
- [ ] `nuviz ls`：列出所有实验
- [ ] `nuviz leaderboard`：排行榜表格输出
- [ ] 终端能力检测（为 Phase 3 图片功能做准备）

**里程碑交付：** 可以在自己的 3DGS 训练中使用，替代 print 语句和 TensorBoard。

### Phase 2 — 消融实验 + 论文辅助（4 周）

Week 6-7: 消融实验功能
- [ ] Python `Ablation` 类：`vary()`, `toggle()`, `generate()`, `export()`
- [ ] `nuviz compare`：多实验曲线叠加
- [ ] `nuviz matrix`：消融矩阵视图
- [ ] `nuviz breakdown`：per-scene 指标分解
- [ ] `log.scene()` API
- [ ] 多 seed 聚合（均值 ± 标准差）

Week 8-9: 论文辅助
- [ ] `nuviz table --latex`：LaTeX 表格生成（加粗最佳、下划线次佳）
- [ ] `nuviz table --markdown`：Markdown 表格生成
- [ ] `nuviz export --csv/--json`：数据导出

**里程碑交付：** 完成一次完整的消融实验全流程，从实验定义到论文表格输出。

### Phase 3 — 可视化增强（3 周）

Week 10-11: 图片与对比
- [ ] `nuviz image`：终端图片浏览（Kitty/Sixel/fallback）
- [ ] `nuviz diff`：两实验渲染对比 + 误差热力图
- [ ] `log.pointcloud()` Python 端支持

Week 12: PLY 统计
- [ ] PLY 解析器（支持 3DGS 自定义属性）
- [ ] `nuviz view --stats`：点云统计分析 + 属性分布直方图
- [ ] JSONL rotate 机制（大文件处理）

**里程碑交付：** 完整的可视化工具链，能在终端内查看所有训练产出。

### Phase 4 — 打磨与开源（2 周）

Week 13-14: 发布准备
- [ ] `nuviz cleanup` / `nuviz reproduce` / `nuviz tag`：实验管理
- [ ] GPU 指标自动采集（nvidia-smi 轮询）
- [ ] CI/CD：GitHub Actions + 多平台二进制构建
- [ ] 文档：README + 使用指南 + GIF demo 录制
- [ ] PyPI 发布（Python 库）+ crates.io / brew（CLI）
- [ ] **Stretch:** `nuviz figure` 论文对比图（Python helper, matplotlib）
- [ ] **Stretch:** 统计显著性检测（配对 t-test）

**里程碑交付：** 开源发布，附带高质量 README 和 demo。

---

## 8. 竞品对比总结

| 功能 | TensorBoard | wandb | Aim | Rerun | **NuViz** |
|---|---|---|---|---|---|
| 终端原生 | ✗ | ✗ | ✗ | ✗ | **✓** |
| 两行代码接入 | △ | △ | △ | ✗ | **✓** |
| 离线使用 | ✓ | ✗ | ✓ | ✓ | **✓** |
| 实时监控 | △ | ✓ | ✓ | ✓ | **✓** |
| 消融矩阵 | ✗ | △ | ✗ | ✗ | **✓** |
| Per-scene 分解 | ✗ | ✗ | ✗ | ✗ | **✓** |
| LaTeX 表格输出 | ✗ | ✗ | ✗ | ✗ | **✓** |
| PLY 统计分析 | ✗ | △ | ✗ | ✓ | **✓** |
| 环境快照 | ✗ | ✓ | △ | ✗ | **✓** |
| 零网络依赖 | ✓ | ✗ | ✓ | ✓ | **✓** |

> **变更说明：** 移除了"论文对比图"和"统计显著性"两行（推迟为 stretch goal），
> 移除了"3D 点云查看"（改为"PLY 统计分析"以准确反映功能范围）。

---

## 9. 命名与品牌

**项目名称：`nuviz`**

- **含义：** nu (新/next) + viz (visualization)，"新一代可视化"
- **长度：** 5 个字母，2 个音节
- **辨识度：** `viz` 后缀在开发者工具圈天然关联 visualization（viztracer, vizro, d3-viz）
- **CLI 命令：** `nuviz`
- **Python 包：** `nuviz`（PyPI 可用）
- **Rust crate：** `nuviz`（crates.io 待确认）
- **域名候选：** `nuviz.dev` / `nuviz.io`

---

## 10. 风险与挑战

| 风险 | 影响 | 缓解策略 |
|---|---|---|
| 终端协议碎片化 | 图片功能在部分终端不可用 | 分级 fallback，纯文本模式始终可用 |
| JSONL 大文件性能 | 长训练产生 GB 级日志 | 自动 rotate + 降采样旧数据 |
| 用户迁移成本 | 已有 wandb 用户不愿换 | 定位为互补而非替代（非 MLOps 平台） |
| Rust + Python 双语言维护 | 开发成本高 | Python 库保持极简，核心逻辑在 Rust |
| 3DGS PLY 格式多变 | 不同实现的 PLY 属性不同 | 支持主流实现 + 可扩展属性映射 |

> **移除项：** "wandb 数据导入"不再作为缓解策略，迁移成本通过定位差异化（互补而非替代）来缓解。

---

## Appendix: 设计决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 实时通信 | inotify 文件监听 | 比 Unix socket 简单，无 daemon，SSH 天然兼容 |
| PLY 3D 查看 | 仅统计模式 | 终端交互式 3D 不现实，推荐外部 viewer |
| 作业编排 | 仅生成配置 | SLURM 集成是独立领域，不属于可视化工具职责 |
| 论文对比图 | Phase 4 stretch | 图片合成引擎复杂度高，可用 matplotlib 替代 |
| 网络结构摘要 | 砍掉 | torchinfo 已满足需求，Rust 解析 PyTorch 模型不值得 |
| 最低 Python | 3.10+ | 3.8 已 EOL，3.10 提供实用语法改进 |
| 异常检测 | 仅 NaN/Inf + spike | 梯度/OOM/停滞检测误报率高，等用户反馈再加 |
| 统计显著性 | Phase 4 stretch | 需多 seed 数据，MVP 先提供 mean ± std |
| 实验命名 | 用户命名 + 时间戳 | 自动命名（超参 diff）实践中碰撞且可读性差 |
| wandb 导入 | 不做 | 复杂 ETL，定位为互补工具而非替代品 |

---

*Last updated: 2026-03-19*
*Author: Zhongyuan*
