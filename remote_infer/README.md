# 2166 / 2171 DLC 推理包（服务器用，临时环境）

本机 DLC 环境已坏，此包用于在学校服务器上用**已有 ResNet50 权重**对
`ct 2166 PBS.WMV`、`ct 2171 CNO.WMV` 做关键点推理，产出 CSV 后即可删环境。

## 包内容

```
dlc_infer_2166_2171/
  videos/          两只待分析视频
  project/         config.yaml + dlc-models（训练权重）
  run_infer.py     一键推理脚本
```

约 **795 MB**（主要是模型权重）。

## 服务器步骤（跑完可删环境）

```bash
# 1. 上传整个文件夹（scp/rsync 均可）
#    scp -P <PORT> -r remote_infer <USER>@<SERVER_IP>:~/

# 2. 建临时环境（TensorFlow 版 DLC，与原训练一致）
conda create -n dlc_tmp python=3.9 -y
conda activate dlc_tmp
pip install deeplabcut            # 若服务器有 GPU 可装 deeplabcut[tf]
# 若默认装到 pytorch 引擎且权重是 TF，可显式：
# pip install "deeplabcut[tf]"

# 3. 进入目录推理
cd ~/remote_infer
python run_infer.py

# 4. 把 videos/ 下新生成的 *DLC*.csv 下载回本机
#    scp -P <PORT> "<USER>@<SERVER_IP>:~/remote_infer/videos/*DLC*.csv" ./sample_data/

# 5. 确认 CSV 正常后删除环境
conda deactivate
conda env remove -n dlc_tmp -y
```

## 回传后本机

把 CSV 放到：

`Mouse Main/video/`

命名需包含鼠号与 `DLC`，例如：

- `ct 2166 PBSDLC_resnet50_Mouse_Head_Lift_DetectionApr13shuffle1_15000.csv`
- `ct 2171 CNODLC_...csv`

（若服务器输出名不同，可直接改名对齐 `event_annotate_gui.py` 的查找规则：以鼠号开头、含 `DLC`、`.csv` 结尾。）

然后：

```text
python event_annotate_gui.py --mouse 2166 --annotator 你的名字
```

草稿就会出现，不必再从零手工扫。

## 注意

- 权重是 **TensorFlow** ResNet50 checkpoint（`.meta/.index/.data`）；若服务器装的是纯 PyTorch DLC3，可能需 `pip install "deeplabcut[tf]"` 或换 TF 环境。
- 本包**不含**训练，只做 `analyze_videos` 推理。
- 环境删掉不影响；只要 CSV 回来即可。
