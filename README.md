# 自动驾驶模拟器 - 强化学习演示系统

基于 DQN 强化学习算法的自动驾驶车辆模拟器，展示车道保持与超车决策能力。

## 系统架构

```
├── env/                    # Gym 自定义驾驶环境
│   ├── __init__.py         # 环境注册
│   └── highway_env.py     # 多车道高速公路环境
├── train/                  # 训练模块
│   ├── train_dqn.py       # DQN 训练脚本（支持多风格）
│   └── train_all.py       # 一键训练所有风格
├── models/                 # 模型保存目录（训练后生成）
│   ├── aggressive/        # 激进型模型
│   ├── conservative/      # 稳健型模型
│   └── balanced/          # 均衡型模型
├── backend/               # Flask 后端服务
│   └── app.py            # REST API 服务
├── frontend/              # 前端可视化
│   └── index.html        # 网页界面
└── requirements.txt       # Python 依赖
```

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型

训练单个风格：
```bash
# 训练均衡型（默认）
python -m train.train_dqn --style balanced --timesteps 50000

# 训练激进型
python -m train.train_dqn --style aggressive --timesteps 50000

# 训练稳健型
python -m train.train_dqn --style conservative --timesteps 50000
```

一键训练所有风格：
```bash
python -m train.train_all --timesteps 50000
```

> 提示：首次训练建议使用 50000 步，约需 2-5 分钟。如需更好效果可增至 200000 步。

### 3. 启动后端服务

```bash
python -m backend.app
```

服务将在 `http://localhost:5000` 启动。

### 4. 打开前端界面

在浏览器中访问 http://localhost:5000 即可看到可视化界面。

## 使用指南

### 前端界面操作

1. **选择驾驶风格** - 在左侧控制面板选择 aggressive/conservative/balanced
2. **开始模拟** - 点击"开始模拟"按钮，观察车辆自主行驶
3. **实时数据** - 左侧显示累计奖励、速度、超车次数、碰撞次数
4. **奖励曲线** - 右侧实时绘制累计奖励和速度变化曲线
5. **风格对比** - 点击"对比所有风格"可同时比较三种模型表现
6. **回放控制** - 可暂停、调速、重置模拟

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/models` | GET | 获取所有已训练模型列表 |
| `/api/simulate` | POST | 运行一次模拟（参数：style, max_steps） |
| `/api/compare` | POST | 对比多个风格模型表现 |
| `/api/training-log/<style>` | GET | 获取训练过程日志 |

### API 调用示例

```bash
# 获取可用模型
curl http://localhost:5000/api/models

# 运行模拟
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"style": "aggressive", "max_steps": 300}'

# 对比风格
curl -X POST http://localhost:5000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"styles": ["aggressive", "conservative", "balanced"]}'
```

## 五大核心模块说明

### 模块一：环境模块（env/highway_env.py）

- 基于 Gymnasium 的自定义多车道高速公路环境
- 3 条车道，6 辆 NPC 车辆
- 离散动作空间：保持/加速/减速/左变道/右变道
- 观测空间：自车状态 + 最近 4 辆 NPC 的相对位置和速度
- 奖励设计：车道保持奖励 + 速度奖励 + 超车奖励 - 碰撞惩罚 - 变道惩罚

### 模块二：训练模块（train/train_dqn.py）

- 使用 Stable-Baselines3 的 DQN 算法
- MLP 策略网络（128-128 隐藏层）
- 支持三种训练配置对应不同驾驶风格
- 训练过程实时输出 episode 奖励

### 模块三：模型保存模块

- 每种风格独立保存目录
- 保存内容：模型权重(.zip) + 元数据(metadata.json) + 训练日志(training_log.json)
- 元数据记录训练参数、效果等信息

### 模块四：后端服务模块（backend/app.py）

- Flask REST API
- 支持模型加载、推理、状态流返回
- 支持多模型对比
- 集成前端静态文件服务

### 模块五：前端可视化模块（frontend/index.html）

- 纯原生 HTML + Canvas + JavaScript，无需构建工具
- 实时道路动画：自车、NPC 车辆、车道线
- 实时图表：累计奖励曲线、速度变化曲线
- 多风格对比图
- 回放控制：暂停、调速

## 驾驶风格差异

| 风格 | 特点 | 学习率 | 折扣因子 |
|------|------|--------|----------|
| aggressive | 高速激进，频繁超车 | 1e-3 | 0.95 |
| conservative | 谨慎平稳，避免冲突 | 5e-4 | 0.99 |
| balanced | 适中速度，稳步超车 | 7e-4 | 0.97 |

## 注意事项

- 训练需要 PyTorch，首次安装可能较慢
- GPU 不是必需的，CPU 即可完成训练
- 如果训练效果不佳，尝试增加 timesteps 或调整超参数
- 前端界面需要后端服务运行才能正常工作
