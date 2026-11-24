# Mouse Headlift Detection

基于 DeepLabCut 姿态估计与时序分析的小鼠抬头行为识别系统。

## 目录结构

- `mouse_headlift_system/`: 桌面端 GUI 交互系统源码
- `headlift_toolkit/`: 核心行为检测算法与分析工具
- `sample_data/`: 样例关键点数据 (CSV)
- `dlc_config/`: DeepLabCut 训练与配置文件模板
- `remote_infer/`: 远程服务器推理辅助脚本

## 快速上手

### 1. 环境准备

推荐使用 Python 3.9+ 环境，安装运行依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行桌面端 GUI

```bash
cd mouse_headlift_system
python main_gui.py
```

### 3. 命令行快速检测

```bash
cd headlift_toolkit
python headlift.py --csv ../sample_data/1631*.csv
```

## 许可证

本项目基于 Apache-2.0 协议开源。