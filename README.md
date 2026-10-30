# VLN Toolkit

本文档以本地的
VLNVerse `kujiale_0004` 为例，按环境安装、场景层、场景表达层、数据生成层介绍操作。

当前可执行流程：

```text
USD 场景
  └── 场景层：检查场景，保存 scene.json
        ├── 场景表达层：Scene Graph
        └── 场景表达层：Occupancy Map
              └── 数据生成层：采样 → A* → 模板指令 → Episode → 理想轨迹
```

Scene Graph 与 Occupancy Map 可以独立生成。当前 Episode 生成使用 Occupancy Map，
指令由路线几何和初始朝向生成。

## 1. 环境安装

### 1.1. 路径与环境约定

本文命令均在项目根目录执行：

```bash
cd /path_to/vln_toolkit
```


### 1.2. 安装 uv

先检查是否已安装，项目要求 uv >= 0.8：

```bash
uv --version
```

尚未安装时，可使用 [uv 官方安装方式](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后重新打开终端，进入项目根目录。

### 1.3. 安装普通项目环境

```bash
uv sync --locked --extra usd
```

该命令按 `uv.lock` 创建 `.venv`，以 editable 模式安装本项目，并安装运行依赖、
默认 `dev` 依赖组及 `usd` extra：

| 依赖 | 用途 |
|---|---|
| NumPy、PyYAML、Shapely | 地图、配置及几何处理 |
| pytest、Ruff | 测试与代码检查 |
| usd-core | 普通 Python 环境中的 USD 解析 |

验证安装：

```bash
uv run --locked --extra usd python -c \
  'import vln_toolkit, numpy, yaml, shapely; from pxr import Usd; print("Project environment OK")'
```


### 1.4. 安装 Isaac Sim 专用环境

先单独安装 Isaac Sim，并确认本机启动脚本和 GPU 驱动可用：

```bash
test -x /home/pyz/isaac-sim/python.sh
nvidia-smi
```

以下环境本机已创建。首次准备时执行；条件判断用于保留已经存在的环境：

```bash
if [ ! -x workspace/envs/isaac/bin/python ]; then
  uv venv \
    --python /home/pyz/isaac-sim/kit/python/bin/python3 \
    --system-site-packages \
    work_dir/envs/isaac
fi

uv pip install \
  --python work_dir/envs/isaac/bin/python \
  --no-deps -e . 'shapely==2.1.2'
```

验证解释器与依赖：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
  PYTHONEXE="$PWD/workspace/envs/isaac/bin/python" \
  /home/pyz/isaac-sim/python.sh -c \
  'import sys, numpy, yaml, shapely, vln_toolkit; print(sys.executable); print("Isaac environment OK")'
```


## 2. 场景层：接入并检查 USD

输入是本地 USD 或场景描述 JSON，输出是 `Scene` 对象及可保存的 `scene.json`。
场景对象包含 `scene_id`、`usd_path`、`metadata`。

### 2.1. 检查场景并保存描述

```bash
uv run --locked --extra usd python scripts/inspect_scene.py \
  --scene /mnt/data/VLNVerse/VLNVerse_scene/kujiale_0004/start_result_navigation.usd \
  --scene-id kujiale_0004 \
  --output work_dir/kujiale_0004/scene.json
```

本机场景的关键输出为：

```text
Scene ID: kujiale_0004
up_axis: Z
meters_per_unit: 1.0
has_collision: True
has_semantics: True
USD layers: 369
```

`scene.json` 保存源 USD 路径、坐标约定、碰撞与语义检查结果等元数据。
显式设置 `--scene-id` 后，后续地图和 Episode 都使用 `kujiale_0004`。

### 2.2. Python 加载接口

```python
from vln_toolkit.scenes import load_scene

scene = load_scene("work_dir/kujiale_0004/scene.json")
print(scene.scene_id)
print(scene.usd_path)
```

`load_scene()` 也支持直接加载 USD，以及包含 `scene.json` 的目录。
默认通过 OpenUSD 检查文件，读取源坐标系和单位，记录转换到内部 Z-up、米制坐标的矩阵。
原生 Occupancy Builder 目前要求源 Stage 本身采用 Z-up、米单位。

### 2.3. Mesh / GLB 场景适配

TODO

### 2.4. 3DGS 场景适配

TODO

### 2.5. 资产组合与 LLM 场景生成

TODO

## 3. 场景表达层

### 3.1. 从 USD 生成 Scene Graph

```bash
uv run --locked --extra usd python scripts/build_scene_graph.py \
  --scene work_dir/kujiale_0004/scene.json \
  --config configs/vlnverse/kujiale_0004_scene_graph.yaml \
  --output work_dir/kujiale_0004/scene_graph
```

当前配置使用显式 USD 语义，只保留具备有效几何包围盒的已标注实体。
本次数据参考结果是 316 个对象、0 个房间、4064 条有向关系。
该场景本地缺少对应的房间多边形元数据，因此从 USD 提取的这份图没有填入房间归属。

输出：

```text
work_dir/kujiale_0004/scene_graph/
├── scene_graph.json
├── object_dict.json
├── room_dict.json
└── category_dict.json
```

`scene_graph.json` 是内部模型，包含 `objects`、`rooms`、`relations`、`metadata`；
其他三个文件用于与 VLNTube 交换数据。空间关系使用世界 AABB 近似，支持
`near`、`above/below`、`on/below`、`in/contain`。

读取和查询：

```python
from vln_toolkit.representations import SceneGraph

graph = SceneGraph.load("work_dir/kujiale_0004/scene_graph")
sofas = graph.find_objects("sofa")
print(sofas)
if sofas:
    print(graph.relations_for(sofas[0]["instance_id"]))
```

### 3.2. 生成 Occupancy Map

填写配置文件 [kujiale_0004_occupancy.yaml](configs/vlnverse/kujiale_0004_occupancy.yaml)：

```yaml
occupancy:
  resolution: 0.1
  origin: [3.0, 2.0, 0.3]
  lower_bound: [-11.0, -9.5, 0.0]
  upper_bound: [5.0, 5.5, 0.0]
  floor_z: 0.0
  agent_radius: 0.28
  freeze_rigid_bodies: true
```

| 参数 | 含义 |
|---|---|
| `resolution` | 每格 0.1 m × 0.1 m |
| `origin` | 世界坐标中的自由扫描点，当前扫描高度约为 0.3 m |
| `lower_bound` / `upper_bound` | 相对于扫描点的下界、上界偏移 |
| `floor_z` | 三维导航路径使用的地面高度，由人工指定 |
| `agent_radius` | 水平障碍膨胀半径，单位米 |
| `freeze_rigid_bodies` | 建图期间临时禁用刚体动态，保持场景布局 |

当前配置需要人工制定（TODO:自动计算）。新场景需要重新确认地面高度、空闲扫描点、范围和机器人尺寸。
确定世界 XY 范围后，边界偏移计算为：

```text
lower_bound = 世界下界 − origin
upper_bound = 世界上界 − origin

如：
当前世界范围：X ∈ [-8, 8]，Y ∈ [-7.5, 7.5]
lower_bound = [-8−3, -7.5−2, 0] = [-11, -9.5, 0]
upper_bound = [ 8−3,  7.5−2, 0] = [  5,  5.5, 0]
```

上下界 Z 偏移均为 0，表示在配置的扫描高度生成二维地图。

完成配置文件后，使用第 1.4 节的 Isaac Sim 专用环境执行：

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONEXE="$PWD/workspace/envs/isaac/bin/python" \
  /home/pyz/isaac-sim/python.sh scripts/build_map.py \
  --scene work_dir/kujiale_0004/scene.json \
  --config configs/vlnverse/kujiale_0004_occupancy.yaml \
  --output work_dir/kujiale_0004/occupancy
```

成功日志包含：

```text
Saved occupancy map to work_dir/kujiale_0004/occupancy
```

输出 `map.npy`、`metadata.json`、`map.svg`。


打开 `work_dir/kujiale_0004/occupancy/map.svg` 检查范围、墙体、家具和通道。
白色为自由区域，深色为障碍，灰色为未知区域。

```bash
uv run --locked python - <<'PY'
import numpy as np
from vln_toolkit.representations import OccupancyMap

occupancy = OccupancyMap.load("work_dir/kujiale_0004/occupancy")
print("shape:", occupancy.grid.shape)
print("resolution:", occupancy.resolution)
print("origin:", occupancy.origin)
print("floor_z:", occupancy.floor_z)
print("encoding counts:", np.unique(occupancy.grid, return_counts=True))
print("scan XY is free:", occupancy.is_free((3.0, 2.0)))
PY
```

栅格采用 `grid[row, col]`，行增加对应 `+Y`，列增加对应 `+X`：

| 编码 | 含义 |
|---|---|
| `0` | 自由区域 |
| `1` | 占据区域 |
| `255` | 未知区域，不能参与路径规划 |



### 3.3. NavGraph 自动构建

TODO

### 3.4. NavMesh 构建与查询

TODO


## 4. 数据生成层

### 4.1. 生成 NavigationEpisode

```bash
uv run --locked python scripts/generate_episodes.py \
  --map work_dir/kujiale_0004/occupancy \
  --scene-id kujiale_0004 \
  --config configs/generation.yaml \
  --count 5 \
  --seed 42 \
  --output work_dir/kujiale_0004/dataset
```

该命令依次调用已经实现的：

| 步骤 | 方法 | 输出 |
|---|---|---|
| 任务采样 | `RandomTaskSampler.sample()` | 起点、目标点、初始 yaw |
| 参考路径规划 | `AStarPlanner.plan()` | 世界坐标 XYZ 路径点 |
| 几何模板指令 | `TemplateInstructionGenerator.generate()` | 中文或英文指令 |
| Episode 组装 | `build_episode()` | `NavigationEpisode` |
| 保存 | `save_episodes()` | JSONL 和元数据 |

`configs/generation.yaml` 中，`min_distance` / `max_distance` 约束最短可行路径长度，
单位米；`allow_diagonal` 控制对角移动，对角穿越障碍夹角始终禁止；`language` 支持
`zh` / `en`。命令中的 `--count` 和 `--seed` 覆盖配置对应字段。
采样检查起终点自由、可达和距离约束；无法采满指定数量时，批量入口报错。

成功输出 `Saved 5 episodes ...`，产物为：

```text
work_dir/kujiale_0004/dataset/
├── episodes.jsonl
├── metadata.json
└── routes.svg
```

每条 Episode 包含 `episode_id`、`scene_id`、`instruction`、`start_pose`、`goal`、
`reference_path` 等字段。位姿使用米制 XYZ 与 `xyzw` 四元数，内部角度使用弧度。
打开 `routes.svg` 查看前 20 条路线，检查它们与地图障碍的对应关系。

### 4.2. 生成理想运动轨迹

```bash
uv run --locked python scripts/collect_trajectories.py \
  --dataset work_dir/kujiale_0004/dataset \
  --mode kinematic \
  --step-size 0.25 \
  --dt 0.1
```

`KinematicExpert` 沿参考路径插值，`step-size` 是最大插值间距，`dt` 是合成时间戳间隔。
初始朝向来自 Episode，动作采用理想 `move_to`，最后为 `stop`。
这是离线路径回放，当前输出不包含 RGB-D 观测或物理机器人执行数据。

输出：

```text
dataset/trajectories/ep_000000/
├── states.npz     # timestamps、positions、rotations_xyzw
├── actions.npz    # commands：JSON 字符串数组
└── metadata.json  # mode、status、时序约定及回放参数
```

状态/观测 `t` 对应执行动作 `t` 之前的状态，动作 `t` 指向下一状态。
时间戳、位姿、动作数量一致。已实现的 Recorder 还支持将外部提供的 RGB、米制 Depth
数组无损保存为 `rgb.npz` / `depth.npz`，其帧数和时间戳需与状态对应。

### 4.3. 读取 Episode 与轨迹

```bash
uv run --locked python - <<'PY'
from vln_toolkit.data import load_episodes, TrajectoryRecorder

directory = "work_dir/kujiale_0004/dataset"
episodes = load_episodes(directory)
print("episode count:", len(episodes))
episode = episodes[0]
print("instruction:", episode.instruction)
trajectory = TrajectoryRecorder(directory).load(episode.episode_id)
print("mode:", trajectory.mode)
print("state count:", len(trajectory.poses))
print("observation channels:", list(trajectory.observations))
PY
```

### 4.4. Room-to-Room / ObjectGoal / Multi-Waypoint / Cross-Floor 采样

TODO

### 4.5. NavMeshPlanner / NavGraphPlanner

TODO

### 4.6. Scene Graph 语义指令生成

TODO

### 4.7. LLM 指令生成

TODO

### 4.8. LeRobot 数据导出

TODO

## 5. 输出目录汇总

```text
work_dir/kujiale_0004/
├── scene.json
├── scene_graph/
├── scene_graph_prebuilt/       # 可选：导入已有图
├── occupancy/
│   ├── map.npy
│   ├── metadata.json
│   └── map.svg
├── dataset/
│   ├── episodes.jsonl
│   ├── metadata.json
│   ├── routes.svg
│   └── trajectories/
```