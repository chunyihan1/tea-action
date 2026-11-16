# 论文级实验方案（training/research）

本目录提供三件可复现的实验工具，产出投稿中文核心（南核 / CSCD）所需的表格与审计证据。
所有命令均在**项目根目录**（`tea-action/`）下执行，输出默认写到 `research_out/`。

```
training/research/
  audit_dataset.py   数据质量与近重复泄漏审计（dHash-64，分桶近邻搜索）
  benchmark.py       多模型横向对比（产出表 1 原始 csv）
  make_tables.py     benchmark.csv -> paper_table.md / paper_table.tex / 消融表
  README.md          本文件
```

依赖仅用到项目已有环境：Python 3.11、ultralytics、Pillow、numpy、pandas、pyyaml，**不新增任何依赖**。

---

## 一、三条命令

### 1. 数据审计（先做，决定后面的实验是否可信）

```bat
python training\research\audit_dataset.py
```

常用变体：

```bat
python training\research\audit_dataset.py --data-root training\datasets\tea_leaf_disease --out research_out
python training\research\audit_dataset.py --hash-threshold 6
```

产出：`research_out\audit.json`、`research_out\audit.md`。

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

## 附：常见问题

- **`audit_dataset.py` 很慢？** 首次要逐张解码图片（1 万张约几分钟），属正常；
  近重复搜索本身是分桶 + 向量化 popcount，远快于 O(n²)。
- **`benchmark.py` 报权重下载失败？** 属预期内，该行记为 `status=error` 并继续；
  离线环境可先手工把 `.pt` 放到工作目录，或只用已下载的权重。
- **`make_tables.py` 没生成 TTA 表？** 检查合并后的 csv 是否同时存在
  `augment=False` 与 `augment=True` 的行。
- **表格里的 Δ 是什么？** 相对基线（默认 `yolov8n.pt`，不存在时取第一个成功模型）
  的 mAP@0.5:0.95 差值，正数表示优于基线。
