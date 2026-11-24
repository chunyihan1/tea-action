# 论文级实验方案（training/research）

## 边界：这里的东西属于论文，不属于演示系统

| 归属 | 目录 | 说明 |
|---|---|---|
| 论文工件 | `training/research/` | 数据审计、基准、显著性、误差分析、开放集拒识、多种子实验 |
| 演示系统 | `src/`、`pages/`、`app.py` | Streamlit 页面 + WebSocket 服务，**无鉴权、不得接硬件、不得暴露到局域网** |

两条硬规则：

1. `src/`、`pages/`、`app.py` 不得 import 本目录任何模块（当前已满足，可用 `grep -rn "training.research" src pages app.py` 复核）；
2. 评估入口 fail-fast（`benchmark.py` 例外：多模型对比时单模型失败记为 `status=error` 继续，便于横向排表）：权重缺失时 `require_weights()` 直接以退出码 2 中止，**绝不降级到模拟检测**——模拟出的数字会污染论文。

演示系统在缺权重时会切到 `SimDetector` 用合成帧演示，这是页面功能，与论文实验无关。

## 开放集拒识（OSD）实验入口

```powershell
# 1) 看统计（不落盘）
python training\research\make_open_set_split.py --data-root training\datasets\tea_leaf_disease --mode list

# 2) 生成划分（默认 unknown 收集留出类的 train+val+test 全部图像；val 按 5:5 拆 calib/eval）
python training\research\make_open_set_split.py --data-root training\datasets\tea_leaf_disease --splits S1,S5

# 3) 评估（自动识别 known/eval_split/，阈值只在 calib 拟合、指标只在 eval 上报）
python training\research\open_set_eval.py --model research_out\runs\S1_s42\weights\best.pt --split-dir research_out\open_set\S1
```

关键默认值（论文里必须写明）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--unknown-scope` | `all` | unknown 评估集收集留出类在 train/val/test 的全部图像；设 `val` 可复现旧行为 |
| `--unknown-mixed` | `exclude` | 同时含其他类别框的混合图整图排除（避免已知类框被改写成 unknown 而虚高召回）；设 `relabel` 复现旧行为 |
| `--val-split` | `0.5` | known/val 按 seed 确定性拆成 calib 与 eval，两者无交集；设 `0` 关闭 |
| `--calib-list` / `--eval-list` | 自动探测 | 显式指定校准/评估清单，优先于 `known/eval_split/` |

`metrics.json` 会写入 `protocol`（calib/eval/unknown 各多少图、来源是 auto/arg/fallback），便于论文附录复现。

## 实验工具一览

本目录提供可复现的实验工具，产出投稿中文核心（南核 / CSCD）与 SCI 期刊所需的表格、
统计检验与审计证据。所有命令均在**项目根目录**（`tea-action/`）下执行，
输出默认写到 `research_out/`。

```
training/research/
  common.py          共用底座：data.yaml 解析 / 标签读取 / IoU 贪心匹配
  audit_dataset.py   数据质量与近重复泄漏审计（dHash 或 pHash-64，分桶近邻搜索）
  benchmark.py       多模型横向对比（产出表 1 原始 csv）
  make_tables.py     csv -> paper_table.md/.tex / 消融表 / 多种子 mean±std 表
  stats_eval.py      bootstrap 95% CI + McNemar 配对检验 + 多重比较 Holm 校正
  error_analysis.py  FP/FN 裁剪图导出 + 按类别/尺寸/置信度统计
  seed_study.py      多种子重复实验（train+val，输出 mean±std）
  domain_eval.py     跨域评估：多域 P/R/F1 + 95% CI + 相对 in-domain 的下降幅度
  make_splits.py     按组（茶园/批次）切分 train/val/test，输出可复查的 txt/json
  README.md          本文件
```

模型配置在仓库根目录的 `models/` 下（不改 `src/`）：

```
models/tea_yolov8s_p2.yaml   YOLOv8s + P2（stride=4）小目标检测头
```

依赖仅用到项目已有环境：Python 3.11、ultralytics、Pillow、numpy、pandas、pyyaml、
opencv-python（仅 pHash 需要），**不新增任何依赖**，统计量全部手写实现（不使用 scipy）。

---

## 一、六条命令

### 1. 数据审计（先做，决定后面的实验是否可信）

```bat
python training\research\audit_dataset.py
```

常用变体：

```bat
python training\research\audit_dataset.py --data-root training\datasets\tea_leaf_disease --out research_out
python training\research\audit_dataset.py --hash-threshold 6
python training\research\audit_dataset.py --hash phash --hash-threshold 10   :: pHash（需 cv2，缺失时自动回退 dHash）
```

产出：`research_out\audit.json`、`research_out\audit.md`（md 里注明所用哈希类型）。

`audit.md` 里必须抄进论文的数字：

- 各 split 图片数 / 标签数 / 无标签图片 / 孤儿标签；
- 每个类别的实例数（train / valid / test）；
- 图像宽高 min / median / max；
- **泄漏率**：验证集中能在训练集找到近重复（Hamming ≤ 阈值）的图片比例；
- 泄漏最严重的前 20 对文件名与距离。

### 2. 多模型对比（表 1 原始数据）

```bat
python training\research\benchmark.py
```

常用变体：

```bat
python training\research\benchmark.py --models yolov8n.pt,yolov8s.pt,yolov8m.pt,yolo11n.pt,yolo11s.pt
python training\research\benchmark.py --imgsz 512 --device 0 --out research_out\imgsz512
python training\research\benchmark.py --augment --out research_out\tta      :: TTA 消融的另一半
python training\research\benchmark.py --latency-image data\samples\real_tea_01.jpg
```

产出：`benchmark.csv`、`benchmark.md`。单个模型失败（权重下载失败 / 显存不足）只记
`status=error` 并继续下一个；结束打印“完成 N 个模型，成功 S，失败 F”。

### 3. 生成论文表格

```bat
python training\research\make_tables.py
python training\research\make_tables.py --csv research_out\benchmark.csv --out-dir research_out --baseline yolov8n.pt
```

产出：

| 文件 | 触发条件 | 内容 |
| --- | --- | --- |
| `paper_table.md` / `paper_table.tex` | 总是 | 表 1：参数量、mAP@0.5、mAP@0.5:0.95、延迟、相对基线 Δ |
| `paper_table_tta.md` | csv 同时含 `augment=False/True` | TTA 消融表（含 Δ） |
| `paper_table_imgsz.md` | csv 中 `imgsz` 有 ≥2 个取值 | 分辨率消融表 |

注意：`benchmark.py` 每次只写一种 `augment` 取值。做 TTA 消融时用不同的 `--out`
分别跑 `--augment` 关/开，再把两次的 csv 合并成一个（保留表头一行）后交给
`make_tables.py --csv merged.csv`，否则 `paper_table_tta.md` 不会生成。

若 csv 含 `seed` 列（或有"同模型 + 同 imgsz"的多行重复实验），会额外生成
`paper_table_seeds.md`：Model / mAP@0.5 / mAP@0.5:0.95 / P / R 均为 mean ± std
（样本标准差 ddof=1）。分辨率消融（同模型不同 imgsz）不会被误判为重复实验。

### 4. 统计显著性（bootstrap 95% CI + McNemar）

```bat
python training\research\stats_eval.py --model data\models\tea_yolo.pt --data training\datasets\tea_leaf_disease\data.yaml
python training\research\stats_eval.py --model runs\a\best.pt,runs\b\best.pt --bootstrap 2000 --seed 42
python training\research\stats_eval.py --model data\models\tea_yolo.pt --data <yaml> --imgsz 64 --conf 0.01 --device cpu --bootstrap 50
```

产出：`research_out\stats\per_image.csv`（每模型每图一行 TP/FP/FN）、`stats.json`、`stats.md`。
内容：micro P/R/F1 及其 95% bootstrap CI；给两个模型时附 McNemar 2×2 配对表、
连续性校正 χ²、p 值，以及 F1 差值的配对 bootstrap CI。
给 **≥3 个模型**时做所有两两 McNemar，并输出原始 p 与 Holm-Bonferroni 校正后 p
（`--model a.pt,b.pt,c.pt`，校正自实现、不依赖 scipy）；1~2 个模型时输出与旧版一致。
口径：某图"检测正确" = 该图无 FP 且无 FN；TP 要求同类且 IoU ≥ `--iou`。

### 5. 误差分析（FP / FN 裁剪图与统计）

```bat
python training\research\error_analysis.py --model data\models\tea_yolo.pt --data <yaml>
python training\research\error_analysis.py --model runs\best.pt --data <yaml> --max-samples 100
```

产出：`errors.csv`（type/image/class/conf/iou/area_ratio/w/h/size_bin）、
`fp\`、`fn\` 裁剪图（外扩 20%、防越界，每类最多 `--max-samples` 张）、`error_summary.md`
（按类别、COCO 尺寸口径 small/medium/large、置信度分箱统计，并给出"最容易被误判成什么"）。
无检测/无 GT 时只输出空表，不报错。

### 6. 多种子重复实验（mean ± std）

```bat
python training\research\seed_study.py --seeds 1,2,3,4,5 --data <yaml> --dry-run
python training\research\seed_study.py --seeds 1,2,3,4,5 --data <yaml> --epochs 100 --batch 8 --device 0
```

每个 seed 执行 `yolo detect train ... seed=S deterministic=True` 后对 `best.pt` 执行
`yolo detect val`，解析 P/R/mAP50/mAP50-95，产出 `seeds.csv`、`seeds.md`（逐 seed + 汇总
mean ± std 与样本量）。`--dry-run` 只打印命令（每 seed 一行，train && val），完全离线可用。

### 7. 跨域评估（in-domain vs out-of-domain）

```bat
python training\research\domain_eval.py --model data\models\tea_yolo.pt ^
    --data in=training\datasets\tea_leaf_disease\data.yaml ^
    --data bd=data\external\tea_leaf_bd\data.yaml ^
    --data assam=data\external\cs_d_assam\data.yaml
```

`--data` 可重复传，格式 `名称=data.yaml`，**第一个成功的域作为 in-domain 基准**。
每个域输出图片数 / GT 数 / P / R / F1 + 95% bootstrap CI，并自动算出 F1 的绝对与相对下降，
最后给一段“域偏移结论”。单个域缺目录、缺 data.yaml、标签全缺时只跳过该域（表里留一行
`status=error`），不中断其他域。产出 `domain_eval.csv` / `.md` / `.json`。
推理与匹配复用 `stats_eval.py` 的实现，保证与 `stats.md` 的口径一致。

### 8. 按组切分（防数据泄漏）

```bat
python training\research\make_splits.py --images training\datasets\tea_leaf_disease\train\images ^
    --labels training\datasets\tea_leaf_disease\train\labels --ratios 0.7,0.2,0.1 --seed 42
```

组名由 `--group-regex`（默认 `^(.*?)[_-]`，`g1_01.jpg -> g1`）从文件名提取，整组只进一个 split。
产出 `train.txt` / `val.txt` / `test.txt`（绝对路径，可直接写进 data.yaml 的 `train:` / `val:` /
`test:` 字段）、`split_report.md`（各 split 图片数、类别实例数、组名、四条互斥校验）、`splits.json`。
组数 < 3 时自动降级（2 组 → 只切 train/val；1 组 → 只有 train）并在终端与报告里明确提示，
不会静默产生空文件。`--labels` 留空则只切图、不统计类别。

---

## 二、论文级实验协议（逐条执行，逐条留证）

### 2.1 数据划分：按茶园 / 拍摄批次，禁止随机划分

- **做什么**：以“茶园编号 × 拍摄日期 × 设备”为分组键，整组进 train 或 valid 或 test，
  同一批次、同一植株、同一次连拍的所有图片只能落在同一个 split。
- **为什么**：现有 `train/valid` 大概率是随机划分，同一片叶子的连拍被拆到两边，
  指标虚高（`audit.md` 的泄漏率就是证据）。审稿人把“随机划分 + mAP 0.98”视为典型缺陷。
- **产出**：表 2“数据划分与类别分布”——train/valid/test 的图片数、框数、每类实例数；
  同时给出划分键（茶园/批次）清单，说明 test 完全未参与训练与调参。

### 2.2 近重复审计：把泄漏率写进正文

- **做什么**：跑 `audit_dataset.py`，报告泄漏率、近重复对数、前 20 对示例。
- **为什么**：这是判断“0.979 的 mAP50 是否可信”的直接证据；也是解释
  “真实茶园照片漏检”的入口（模型学到的是同源图片的纹理记忆）。
- **产出**：表 3“train–valid 近重复审计”，以及一句结论（工具已自动生成中文结论句）。
  若泄漏率 ≥ 5%，必须重新划分后重训，不能只把审计结果当“免责声明”。

### 2.3 独立跨域测试集：≥200 张自有茶园实拍

- **做什么**：使用手机/相机在自有茶园拍摄 ≥200 张，覆盖不同光照（顺光/逆光/阴天）、
  不同距离、不同病害等级与背景；用同一标注规范标注，**只用于测试**，不参与训练和调参。
  单独建目录（如 `training/datasets/tea_field_test/`），并在 `benchmark.py` 里用
  `--data` 指向该数据集的 yaml、`split="val"` 评估。
- **为什么**：单数据集内部指标无法证明泛化性；跨域测试是南核/CSCD 审稿的硬性期待。
- **产出**：表 4“跨域测试集结果”（各模型 P/R/mAP50/mAP50-95），必须与表 1 并列讨论
  指标下降幅度（ΔmAP 是核心论点）。

### 2.4 基线对比：至少 4 个模型

- **做什么**：`yolov8n / yolov8s / yolov8m / yolo11n / yolo11s`（至少 4 个），
  同一 data.yaml、同一 `imgsz=640`、同一验证集、同一置信度与 IoU 设置。
- **为什么**：没有基线的 mAP 没有信息量；同量级模型对比才能说明“选型合理”。
- **产出**：表 1（`paper_table.md/.tex` 直接可用）+ 参数量/延迟列，
  用于讨论精度-速度权衡（可配图 2 的 P-R 曲线或 mAP-延迟散点）。

### 2.5 消融实验：一次只改一个变量

| 变量 | 取值 | 命令要点 | 产出 |
| --- | --- | --- | --- |
| 输入分辨率 | 320 / 512 / 640 | `--imgsz 320|512|640`，不同 `--out` | `paper_table_imgsz.md` |
| TTA | 关 / 开 | 两次运行，一次加 `--augment` | `paper_table_tta.md` |
| 切片推理 | 关 / 开（如 SAHI 式切图） | 独立脚本，固定其他设置 | 表 5 |
| 置信度阈值 | 0.10~0.50 扫描 | `training\calibrate_conf.py` | 图 4 F1-阈值曲线 |

- **为什么**：审稿人要求“每个结论都有对照”。分辨率影响小目标漏检，TTA 影响召回，
  阈值影响 FP/FN 平衡——三项正好对应现场问题。
- **注意**：每次只改一个变量，其余（数据、seed、imgsz、权重）保持一致；
  消融表的表注要写清“仅改变 X”。

### 2.6 置信度标定

```bat
python training\calibrate_conf.py --model data\models\tea_yolo.pt --imgsz 640
```

- **做什么**：在验证集上扫 `conf`，取 F1 最优阈值；报告 P/R/F1/mAP 随阈值变化。
- **为什么**：现场“低置信度误报”本质是阈值与场景不匹配；论文报告标定过程比直接写
  “conf=0.25”更严谨。
- **产出**：图 4（P/R/F1–conf 曲线）+ 正文一句话给出最优阈值及其 F1。

### 2.7 错误分析：FP / FN 样例与按类别漏检率

- **做什么**：在跨域测试集上导出典型 FP、FN 图片（带预测框与真值框），
  按类别统计漏检率（FN / GT）与误报率（FP / 预测数）。
- **为什么**：mAP 是聚合指标，审稿人更关心“错在哪”。按类别的漏检率能解释
  小目标、遮挡、病斑相似类别（如 brown blight vs leaf rust）的混淆。
- **产出**：图 5（FP/FN 样例 2×3 拼图）、表 5 或正文“按类别漏检率”表。

### 2.8 统计显著性：不能只报单次结果

- **做什么**（二选一，建议都做）：
  1. 至少 5 个随机种子（42/43/44/45/46）重训，报告 mAP 的均值 ± 标准差；
  2. 对验证集/测试集做 bootstrap 重采样（≥1000 次），给出 mAP50-95 的 95% 置信区间。
- **为什么**：单次训练 0.740 与基线 0.735 的差异可能完全在噪声内；
  没有方差/置信区间，任何“优于基线”的结论都不成立。
- **产出**：表 1 增加“mean ± std”列，或正文给出 95% CI；图 6 可画种子间箱线图。

### 2.9 可复现性：固定并记录所有随机源

- 训练脚本固定 `seed=42`（`train_yolo.py` 已设 `SEED`），并开启
  `deterministic=True`（若显存允许）；
- 保存每次训练的 `runs/detect/train*/args.yaml`，论文附录给出关键超参；
- 记录环境：Python 3.11、torch 版本、ultralytics 版本、CUDA/cuDNN、GPU 型号；
- 记录数据版本（数据集下载日期 / commit / 校验和）；
- 所有表格由 `benchmark.csv` 自动生成，不手工改数字。

---

## 三、论文图表清单

### 表

| 编号 | 内容 | 来源 |
| --- | --- | --- |
| 表 1 | 多模型性能对比：Params、P、R、mAP@0.5、mAP@0.5:0.95、延迟、Δ | `paper_table.md/.tex` |
| 表 2 | 数据划分与类别分布（按茶园/批次划分） | `audit.md` 第 1、2 节 |
| 表 3 | train–valid 近重复审计（泄漏率、配对数、示例） | `audit.md` 第 4 节 |
| 表 4 | 独立跨域测试集结果（≥200 张实拍） | 用 `benchmark.py --data <跨域 yaml>` 重跑 |
| 表 5 | 消融实验汇总（imgsz / TTA / 切片 / 阈值） | `paper_table_imgsz.md`、`paper_table_tta.md` + 自建表 |

### 图

| 编号 | 内容 | 说明 |
| --- | --- | --- |
| 图 1 | 数据集示例与类别分布 | 每类 2~3 张典型图 + 柱状图（类别长尾要讨论） |
| 图 2 | 模型结构 / 整体流程 | YOLOv8s 结构 + 数据流，标注输入输出尺寸 |
| 图 3 | 训练曲线（loss、mAP） | `runs/detect/train/results.png`，注明早停策略 |
| 图 4 | 置信度阈值标定曲线 | P/R/F1 vs conf，标出 F1 最优点 |
| 图 5 | 典型 FP / FN 样例 | 跨域测试集，2×3 拼图，附原图与预测 |
| 图 6 | 种子间稳定性 / 置信区间 | 5 个种子的 mAP 箱线图或 bootstrap 分布 |

---

## 四、为什么当前结果“不足以发表”

1. **单数据集**：只在 figshare Tea-Leaf Disease v1.0 上训练与验证，
   没有第二个数据集或自有数据佐证泛化性。
2. **疑似数据泄漏**：验证集与训练集存在近重复（`audit.md` 会给出泄漏率），
   mAP50≈0.979 很可能被同源图片抬高，不能作为论文主结果。
3. **无跨域测试**：真实茶园照片出现漏检与低置信度误报，说明域偏移未评估，
   论文不能只报同分布指标。
4. **无基线对比**：缺少 YOLOv8n/m、YOLO11 等同量级模型的同条件对比，
   无法证明选型与改进的有效性。
5. **无重复实验**：没有多 seed 均值±标准差或置信区间，单次结果的差异可能落在噪声内；
   同时缺少混淆矩阵、按类别漏检率等错误分析。

结论：当前 0.979 / 0.740 只能作为“内部开发指标”，投稿前必须完成第 2.1~2.9 条。

---

## 五、一页 Checklist

### 数据
- [ ] 按茶园/拍摄批次划分 train/valid/test（非随机），划分键写进论文
- [ ] 跑 `audit_dataset.py`，泄漏率 < 5%（否则重新划分重训）
- [ ] 报告各 split 数量、类别分布、无标签图片、孤儿标签
- [ ] 自建 ≥200 张跨域测试集并标注，完全不参与训练与调参

### 实验
- [ ] 基线 ≥4 个模型，同数据、同 imgsz、同验证集（`benchmark.py`）
- [ ] imgsz 消融 320/512/640（`paper_table_imgsz.md`）
- [ ] TTA 消融 关/开（`paper_table_tta.md`）
- [ ] 切片推理消融（如有实现）
- [ ] 置信度阈值标定，报告 F1 最优阈值（`calibrate_conf.py`）
- [ ] 跨域测试集单独报告 P/R/mAP，并与表 1 对比下降幅度
- [ ] 错误分析：FP/FN 样例图 + 按类别漏检率

### 统计与复现
- [ ] ≥5 个 seed 重训，报告 mean ± std；或 bootstrap 95% CI
- [ ] 固定 seed、记录 args.yaml、Python/torch/ultralytics 版本、GPU 型号
- [ ] 所有表格由脚本生成（不手改数字），csv 与 md/tex 一并归档

### 写作
- [ ] 表 1~表 5、图 1~图 6 齐全，编号与正文引用一致
- [ ] 明确写出数据划分方式、泄漏审计结论、跨域指标下降原因
- [ ] 讨论局限：类别不均衡、遮挡/小目标、单区域数据、推理硬件

---

## 六、SCI 级实验规范（投稿前逐条对照）

### 6.1 数据：划分方式与跨域验证

| 要求 | 具体做法 | 本仓库对应手段 |
| --- | --- | --- |
| 划分不随机 | 以"茶园编号 × 拍摄日期 × 设备"为分组键整组进 split，正文写出划分键 | `audit_dataset.py` 报告泄漏率与前 20 对近重复，作为划分是否可信的证据 |
| 自有跨域测试集 ≥200 张 | 另建 `training/datasets/tea_field_test/`，独立标注，只测不训、不调参 | `benchmark.py --data <跨域 yaml> --out research_out\cross_domain` + `stats_eval.py` 给 CI |
| in-domain vs out-of-domain 指标差 | 同一权重分别在同分布 val 与跨域 test 上评估，报告 ΔmAP50 与 ΔmAP50-95 | `benchmark.csv` 两次运行后合并成表 3（跨域表），差值列由 `make_tables.py` 的 Δ 列口径给出 |
| 泄漏可查 | 报告近重复对数、泄漏率、示例文件名对 | `audit.md` 第 4 节；dHash 与 pHash 两种哈希交叉验证（`--hash phash`） |

### 6.2 对比：可复现基线 + 文献数值分开标注

- **可复现基线 ≥4 个**：`yolov8n / yolov8s / yolov8m / yolo11n / yolo11s` 同一 data.yaml、
  同一 `imgsz=640`、同一验证集、同一 `conf/iou`，一次 `benchmark.py` 跑完。
- **文献数值**：从已发表论文抄录的 mAP 必须单独一列或单独一张表，列头标明
  **reported (文献原文)**，本仓库复现的写 **reproduced (本机复现)**；
  **禁止把 reported 与 reproduced 混在同一列比较**——数据集划分、imgsz、后处理都不同，
  混表是审稿人最容易抓的造假嫌疑。
- 表格必须写清：数据集版本、划分、imgsz、是否 TTA、GPU 型号；同一格不得出现两种来源。

### 6.3 统计：方差、置信区间、显著性检验

| 项 | 最低要求 | 命令 / 输出 |
| --- | --- | --- |
| 重复实验 | ≥5 seeds，报告 mean ± std（样本标准差 ddof=1） | `seed_study.py --seeds 1,2,3,4,5` → `seeds.md`；`make_tables.py` → `paper_table_seeds.md` |
| 置信区间 | bootstrap ≥1000 次，报告 95% CI（2.5%/97.5% 分位） | `stats_eval.py --bootstrap 1000` → `stats.md` 的 `P [lo, hi]` 格式 |
| 两模型比较 | McNemar 配对检验（连续性校正 χ²，p=erfc(√(χ²/2))），写清 α=0.05 | `stats_eval.py --model A.pt,B.pt` → 2×2 表 + χ² + p + F1 差值 CI |
| 方法说明 | 正文必须写：检验方法、配对单位（整图）、显著性水平、是否校正 | `stats.md` 已按此格式给出，可直接改写进方法与结果 |

报告格式示例（不要省略 CI）：`F1 = 0.812 [0.789, 0.834]`、`0.740 ± 0.006 (n=5)`、
`χ²=4.17, df=1, p=0.041 < 0.05`。

### 6.4 复杂度：精度之外的必填项

- **参数量 / GFLOPs**：`benchmark.py` 的 `params`、`gflops` 列（数参数 + `model.info()`）。
- **单图延迟**：`benchmark.py --latency-image <图片>`，warmup 3 次 + 计时 20 次取中位数。
- **吞吐（FPS / images·s⁻¹）**：同一 batch 下用 `batch / 平均单批耗时` 计算，
  固定 batch 与精度（FP32/FP16）后再报。
- **必须注明**：GPU 型号、驱动版本、CUDA/cuDNN、torch 与 ultralytics 版本、batch、精度、
  是否 TensorRT；缺一项就会被要求重做（不同硬件上的延迟不可比）。
- 建议同时报 CPU 端侧延迟（`--device cpu`），茶园移动端部署更关心这个。

### 6.5 误差分析：错在哪，比错多少更重要

- **FP / FN 样例**：`error_analysis.py` 导出 `fp\`、`fn\` 裁剪图（外扩 20%），
  文件名含类别与置信度，直接组成图 5 的 2×3 拼图。
- **按类别统计**：各类 FP/FN 计数 + 漏检类别集中在哪些病斑（相似类别混淆）。
- **按尺寸统计**：COCO 口径 small(<32²) / medium(<96²) / large，说明小目标是否为主因。
- **按置信度分箱**：0.25-0.40 / 0.40-0.60 / 0.60+，区分"高置信度误报"（最有害）与
  "低置信度漏检"；FN 一列填"与该真值重叠最多的预测置信度"，无重叠则单独成行。
- **误判来源**：`error_summary.md` 第 4 节给出"最容易被误判成什么"（跨类 IoU > 0.1 的配对）。

### 6.6 可复现：审稿人要能重跑出同一个数

- 训练固定 `seed` + `deterministic=True`（`seed_study.py` 已写入命令）；
- 归档每个 run 的 `args.yaml`（`<out>/runs/seed_S/args.yaml`），附录给出关键超参表；
- 记录环境：Python / torch / ultralytics / CUDA / GPU / 驱动版本（正文"实验环境"一小节）；
- 数据可用性声明：数据来源、许可证、下载日期与校验和；自有跨域数据说明采集设备与标注规范；
- 所有表格由脚本从 csv 生成，不手改数字；csv 与 md/tex 一并作为补充材料提交。

### 6.7 图表清单（编号与正文引用一致）

| 编号 | 内容 | 生成方式 |
| --- | --- | --- |
| 图 1 | 整体流程（数据 → 训练 → 评估 → 部署） | 手绘/绘图工具，标注输入输出尺寸 |
| 图 2 | 数据集分布（类别长尾、尺寸分布、划分） | `audit.md` 第 1~3 节数据 + 柱状图 |
| 图 3 | PR 曲线（各模型同图对比） | `model.val(plots=True)` 的 `PR_curve.png` |
| 图 4 | 混淆矩阵（含背景行/列） | `confusion_matrix_normalized.png` |
| 图 5 | FP / FN 示例（2×3 拼图，含类别与置信度） | `error_analysis.py` 导出的 `fp/`、`fn/` |
| 表 1 | 多模型对比（P/R/mAP/Params/延迟） | `paper_table.md` / `.tex` |
| 表 2 | 消融（imgsz / TTA / 切片 / 阈值） | `paper_table_imgsz.md`、`paper_table_tta.md` |
| 表 3 | 跨域测试（in-domain vs out-of-domain 与 Δ） | `benchmark.py --data <跨域 yaml>` |
| 表 4 | 统计检验（mean±std、95% CI、McNemar p） | `paper_table_seeds.md`、`stats.md` |

### 6.8 会被审稿人直接拒稿的 6 个问题 → 本仓库的解决手段

| # | 拒稿理由 | 本仓库的对应手段 |
| --- | --- | --- |
| 1 | **数据泄漏**：随机划分 + 近重复，mAP 虚高 | `audit_dataset.py`（dHash/pHash）给泄漏率与前 20 对，正文写明按茶园/批次重划分 |
| 2 | **只在单数据集自测**，无跨域/外部验证 | 自建 ≥200 张跨域测试集，`benchmark.py --data <跨域 yaml>` + 表 3 报 ΔmAP |
| 3 | **无基线或基线不公**（同条件不可复现） | `benchmark.py` 固定 data/imgsz/conf/iou 跑 ≥4 个基线；reported 与 reproduced 分列 |
| 4 | **单次结果无方差**，"优于"不成立 | `seed_study.py` 5 seeds 的 mean±std + `stats_eval.py` bootstrap CI |
| 5 | **无显著性检验**，差异落在噪声内 | `stats_eval.py` 的 McNemar（χ²、p、α=0.05）与 F1 差值配对 CI |
| 6 | **误差分析缺失**，只报聚合指标 | `error_analysis.py` 的 FP/FN 裁剪图 + 按类别/尺寸/置信度统计与误判来源 |

---

## 七、SCI 一区（Q1）投稿要求

第六节解决的是"**结果可信**"（划分、泄漏、CI、显著性）；一区还要求"**方法有新意 + 泛化被证明 +
证据链完整**"。以下按审稿人实际会追问的顺序列出，逐条可核对。

### 7.1 创新点必须能一句话讲清，且可验证

一区不会接受"把 YOLOv8s 换成 YOLOv8m / 换数据集 / 调超参"。创新点必须写成
"针对 X 问题，做 Y 改动，在 Z 指标上带来可复现收益"，并配消融与显著性。三个候选方向：

| 方向 | 具体做法 | 预期收益 | 验证方法 | 风险 |
| --- | --- | --- | --- | --- |
| **A. P2 小目标检测头** | `models/tea_yolov8s_p2.yaml`：在 P3/P4/P5 之外增加 P2（stride=4）检测层与 P2→P3 回融 | 小病斑在 640 输入下不被 stride=8 抹掉，AP_small / 小目标召回提升 | 同划分同 imgsz 与 yolov8s 对比；`error_analysis.py` 按尺寸分箱看 FN；只改配置文件即为天然消融 | 收益常只有 1~2 个点，易落在 seed 方差内；实测 10.64 M / 37.0 GFLOPs vs 基线 11.14 M / 28.7 GFLOPs——**参数没涨、算力 +29%**，正文不能写"参数增加" |
| **B. 切片推理 + 置信度标定** | 大图切块推理再融合（SAHI 式）；在独立标定集上扫 conf 取 F1 最优点，报 ECE/Brier | 高分辨率原图整幅缩放会糊掉小病斑，切片提升召回；标定压低"低置信度误报" | 切片开/关消融（同权重）；`calibrate_conf.py` 出 P/R/F1–conf 曲线；报单图延迟随切片数的增长 | 切片边界目标重复计数、延迟×N；标定集与测试集混用会被视为调参泄漏 |
| **C. 跨域自适应 / 伪标签** | 源域模型在无标注外部域生成伪标签，置信度阈值 + 人工抽检后自训练（`pseudo_label.py` 有基础） | 缩小外部域 ΔF1，证明"可落地"而不只是实验室指标 | `domain_eval.py` 报自适应前后各域 F1 与 ΔF1，配合 Holm 校正显著性 | 确认偏误会被放大；必须严格区分"外部域无监督适应"与"外部域参与有监督训练"，后者等于作弊 |

选一个方向**做深**（消融齐全、显著性成立），比三个都浅尝更容易过一区。

### 7.2 对比方法清单（≥5 个，同划分同分辨率）

| 方法 | 家族 | 说明 |
| --- | --- | --- |
| YOLOv8n | YOLOv8 | 轻量基线，`benchmark.py` 默认已含 |
| YOLOv8s | YOLOv8 | **主基线**，改进相对它报告 |
| YOLOv8m | YOLOv8 | 同门更大尺度，证明不是靠堆参数 |
| YOLO11n | YOLO11 | 新一代官方基线 |
| YOLO11s | YOLO11 | 与主基线同量级 |
| RT-DETR（或 YOLOv10n/s） | DETR / 无 NMS | 换范式对照，避免"只和 YOLO 比"的批评 |

硬性要求：

- 全部**同一 data.yaml、同一 split 文件、同一 imgsz=640、同一 conf/iou、同一硬件**；
- 表格必须分两列：`reproduced (本机复现)` 与 `reported (文献原文)`，
  reported 每格标注引用编号 `[3]`，并在参考文献给出数据集/划分/分辨率；
  **reported 与 reproduced 混在一列 = 审稿人直接质疑数据真实性**；
- 命令：`python training\research\benchmark.py --models yolov8n.pt,yolov8s.pt,yolov8m.pt,yolo11n.pt,yolo11s.pt`。

### 7.3 数据要求：in-domain + ≥2 个外部域

| 角色 | 数据集 | 用途 |
| --- | --- | --- |
| in-domain | figshare Tea Leaf Disease v1.0（8 类，本项目主数据） | 训练 + 同分布验证/测试 |
| 外部域 1 | 孟加拉 **TeaLeafDiseaseBD** | **只测不训**，报域偏移 |
| 外部域 2 | 印度 **CS-D Assam** | **只测不训**，报域偏移 |

- 外部域**不得**参与训练、early stopping、阈值选择、模型选择中的任何一步；
  正文用一句话交代："外部域仅用于最终评估"。
- 外部域类别体系不同时，附录给类别映射表，无法映射的类别剔除并说明，不要强行合并。
- 用 `domain_eval.py` 一次报出各域 P/R/F1 + 95% CI + 相对 in-domain 的绝对/相对下降，
  并在正文把 ΔF1 作为泛化性的核心证据（这是 Q1 最常被要求补的实验）。

### 7.4 统计要求

| 项 | 最低要求 | 工具 |
| --- | --- | --- |
| 重复实验 | **≥5 seeds**，mean ± std（ddof=1），正文写 n=5 | `seed_study.py` → `seeds.md` |
| 置信区间 | bootstrap **≥1000** 次，报 95% CI | `stats_eval.py`、`domain_eval.py` |
| 显著性 | 两两 McNemar + **Holm-Bonferroni 校正**，α=0.05，报校正后 p | `stats_eval.py --model a.pt,b.pt,c.pt` |
| 多域/多指标 | 同一批比较必须校正；只挑显著结果汇报视为选择性报告 | 同上 |

正文写法示例：`F1 = 0.812 [0.789, 0.834]`、`0.740 ± 0.006 (n=5)`、
`McNemar χ²=4.17, df=1, p_raw=0.041, p_Holm=0.082（α=0.05，不显著）`。

### 7.5 效率与部署要求

| 指标 | 获取方式 | 必须注明 |
| --- | --- | --- |
| 参数量 | `benchmark.py` 的 `params` 列 | 类别数 nc（改 nc 会改检测头宽度，跨论文比较要说明） |
| GFLOPs | `benchmark.py` 的 `gflops` 列 | 输入分辨率 |
| GPU 单图延迟 | `benchmark.py --latency-image <图>`（warmup 3 + 计时 20 取中位数） | GPU 型号、驱动、CUDA/cuDNN、torch 版本 |
| 吞吐（img/s） | 固定 batch 的 `batch / 平均单批耗时` | batch、精度（FP32/FP16） |
| CPU 单图延迟 | 同上加 `--device cpu` | CPU 型号、线程数 |
| ONNX 导出后延迟 | `model.export(format="onnx")` 后用 onnxruntime 计时 | opset、是否 FP16、provider（CPU/CUDA） |

不同硬件上的延迟不可比；缺任一环境信息，效率表就会被要求重做。
`models/tea_yolov8s_p2.yaml` 的 P2 分支会显著抬高显存与耗时，必须在效率表中体现这一代价。

### 7.6 可复现与开放科学

- 训练固定 `seed` 且 `deterministic=True`，归档每个 run 的 `args.yaml`；
- 记录 Python / torch / ultralytics / CUDA / GPU / 驱动版本（正文"实验环境"小节）；
- **公开 split 文件**：`make_splits.py` 产出的 `train.txt` / `val.txt` / `test.txt` +
  `splits.json` 作为补充材料，审稿人可完全复现划分（这是消除"数据泄漏"质疑最有效的一步）；
- 公开代码与权重：仓库地址 + 权重下载链接 + SHA256；许可证要写清
  （衍生自 ultralytics 的配置/权重受 **AGPL-3.0** 约束，正文与仓库都要声明）；
- 数据可用性声明：来源、许可证、下载日期、校验和；自采数据说明设备、时间、标注规范与标注人。

### 7.7 图表清单（Q1 常见配置）

| 表 | 内容 | 来源 |
| --- | --- | --- |
| 表 1 | 多模型对比（Params / GFLOPs / P / R / mAP / 延迟），含 reported 列 | `paper_table.md/.tex` |
| 表 2 | 数据划分与类别分布（按茶园/批次分组，组数与组名） | `splits.json`、`split_report.md`、`audit.md` |
| 表 3 | 近重复审计（泄漏率、近重复对数、示例） | `audit.md` 第 4 节 |
| 表 4 | 跨域结果（in-domain + 外部域 + ΔF1 + 95% CI） | `domain_eval.md` |
| 表 5 | 消融（P2 / 切片 / TTA / imgsz / 阈值，一次只改一个变量） | `paper_table_imgsz.md`、`paper_table_tta.md` + 自建表 |
| 表 6 | 复杂度与部署（GPU/CPU/ONNX 延迟、吞吐、硬件） | `benchmark.md` + ONNX 计时脚本 |

| 图 | 内容 | 说明 |
| --- | --- | --- |
| 图 1 | 数据集示例与类别/尺寸分布 | 每类 2~3 张典型图 + 长尾柱状图 |
| 图 2 | 方法总体结构 | 必须画出 P2（stride=4）分支与回融路径 |
| 图 3 | 训练曲线（loss、mAP） | `results.png`，注明早停策略 |
| 图 4 | PR 曲线（多模型同图） | 同划分同 imgsz 才有可比性 |
| 图 5 | 混淆矩阵（含背景行列） | 解释相似病斑混淆 |
| 图 6 | FP / FN 典型案例 | `error_analysis.py` 导出裁剪图拼 2×3 |
| 图 7 | 跨域性能下降 + 置信区间 | 条形图带 error bar，直接支撑"泛化性"论点 |

### 7.8 "Q1 最小可投组合"（做什么就够投，缺什么会被拒）

必须全部成立才谈得上送审：

- [ ] **一个能一句话讲清的创新点**（7.1 的 A 或 B 选一个做深），且消融 + Holm 校正后仍显著；
- [ ] **≥5 个对比方法**（含 1 个非 YOLO 家族），reported 与 reproduced 分列并带引用；
- [ ] **按组划分 + 泄漏率 < 5%**，公开 `train/val/test.txt` 与 `splits.json`；
- [ ] **in-domain + ≥2 个外部域**，报 ΔF1 与 95% CI（外部域全程不参与训练/调参）；
- [ ] **≥5 seeds** 的 mean±std + bootstrap 95% CI + Holm 校正后的 p；
- [ ] **效率表**：Params / GFLOPs / GPU 与 CPU 延迟 / 吞吐 / ONNX 延迟，附完整硬件环境；
- [ ] 图 1~7、表 1~6 齐全，编号与正文引用一致；
- [ ] 代码、权重、划分文件、环境版本公开；数据可用性声明与许可证写清。

缺任一项的典型后果：无创新点 → desk reject；无外部域 → 大修后被拒；
无显著性校正 → 结论被质疑；无公开划分 → 被怀疑数据泄漏；无效率表 → 被质疑不可部署。

**结论（必须写进论文的局限段，也必须自己先承认）**：
仅靠当前"单数据集 figshare + 无方法创新 + 单次训练结果"，即使把本目录所有工具都跑一遍，
也**达不到一区门槛**。工具只保证"数字可信、可复现"；一区真正决定成败的是
**方法层面的新意**（P2 / 切片+标定 / 跨域自适应必须有一个被验证有效）、
**独立外部域上的泛化证据**，以及**写作与开源规范**——这三项都不是代码能替代的。

---

## 附：常见问题

- **`audit_dataset.py` 很慢？** 首次要逐张解码图片（1 万张约几分钟），属正常；
  近重复搜索本身是分桶 + 向量化 popcount，远快于 O(n²)。
- **`benchmark.py` 报权重下载失败？** 属预期内，该行记为 `status=error` 并继续；
  离线环境可先手工把 `.pt` 放到工作目录，或只用已下载的权重。
- **`make_tables.py` 没生成 TTA 表？** 检查合并后的 csv 是否同时存在
  `augment=False` 与 `augment=True` 的行。
- **表格里的 Δ 是什么？** 相对基线（默认 `yolov8n.pt`，不存在时取第一个成功模型）
  的 mAP@0.5:0.95 差值，正数表示优于基线。
