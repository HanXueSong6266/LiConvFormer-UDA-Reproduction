# 🚆 ConvFormer: Train Bogie Bearing Fault Diagnosis with UDA

基于轻量化深度学习模型 `ConvFormer`（`UDTL-master1/models/Liconvformer.py`）进行列车转向架轴承故障诊断，并面向噪声干扰、变负载跨工况场景，构建领域自适应迁移学习（UDA）训练管线。  

本项目复现并深化了开源 LiConvFormer 体系，在此基础上补全/扩展了多损失融合的 UDA 训练流程（支持 `MMD` / `JMMD` / `CORAL` 与 `CDAN/CDA`），以提升跨工况泛化能力。

论文 PDF：项目根目录下的论文文件（含中英文摘要与实验结论）。

---

## ✨ Key Highlights

- 🧩 **ConvFormer 轻量化网络复现与结构化实现**：1D 振动信号建模中，将宽带深度卷积核用于低频全局信息捕获，多层小卷积核提取细粒度特征，并通过 **Broadcast Attention** 融合全局上下文（见 `models/Liconvformer.py`）。
- 🔁 **UDA 多损失融合训练管线（工程可跑通）**：`utils/train_utils_combines.py` 统一管理数据加载、训练/验证/测试、checkpoint 保存与结果导出；训练时同时叠加分类损失、距离对齐损失与（可选）对抗损失，且对距离/对抗权重使用 `Step`/`Cons` 的动态调度。
- 📐 **领域分布与类别级对齐的组合策略**：距离损失支持 `MMD/JMMD/CORAL`，并提供基于 CDAN/CDA 思想的条件对抗域适配实现；在源/目标域特征对齐的同时引入类别条件信息，缓解跨工况域偏移。
- 📊 **实验结论可直接用于面试叙事**：论文摘要给出——在 BJTU 数据库任意信噪比条件下分类准确率可稳定接近 **99.95%**；跨工况迁移中，多损失融合模型平均诊断准确率最高到 **96.1%**，相比单一 `MMD` 或 `CDAN` 提升约 **1 个百分点**。

---

## 📁 Repository Structure

```text
d-bishe-bishe/
├── BJTU/
│   └── 轴承振动数据目录（LA0~LA4 等工况子目录）
├── UDTL-master1/
│   ├── train_advanced.py
│   │   # 跨工况迁移学习训练主入口：循环源/目标工况并记录日志
│   ├── readlog2.py
│   │   # 从训练日志中提取 target_val 准确率并计算统计量
│   ├── datasets/
│   │   ├── BJTU.py
│   │   │   # BJTU 数据加载与源/目标工况划分
│   │   ├── SequenceDatasets.py
│   │   │   # PyTorch Dataset 封装：返回 (seq, label)
│   │   └── sequence_aug.py
│   │       # 归一化与噪声增强等数据增强（如 AddGaussian）
│   ├── models/
│   │   ├── Liconvformer.py
│   │   │   # ConvFormer 轻量化网络实现（Embedding + LFEL + BroadcastAttention）
│   │   ├── AdversarialNet.py
│   │   │   # CDA/DA 对抗域分类器（带 GRL 系数调度）
│   │   └── (其它 backbone / features)
│   ├── loss/
│   │   ├── MMD.py
│   │   ├── JMMD.py
│   │   ├── CORAL.py
│   │   └── LMMD.py（可选）
│   ├── utils/
│   │   └── train_utils_combines.py
│   │       # 训练/验证/测试 + 多损失融合 + 动态权重调度
│   ├── Results/
│   │   # 可视化与实验结果脚本/产物（建议不上传大文件）
│   └── checkpoint/
│       # 模型权重与训练日志（建议不上传大文件）
└── 论文 PDF（根目录）
```

> 说明：上传到 GitHub 时建议忽略 `BJTU/`、`UDTL-master1/checkpoint/`、`UDTL-master1/Results/` 等大文件/产物（已在 `.gitignore` 中配置）。

---

## 🚀 Quick Start

### 1) 环境依赖

推荐依赖（根据代码导入与模块使用）：

- `python>=3.8`
- `torch`, `numpy`, `pandas`, `tqdm`, `scikit-learn`, `scipy`, `Pillow`
- `einops`, `torchsummary`, `torchvision`（部分模型/工具依赖）

示例安装：

```bash
pip install numpy pandas tqdm scikit-learn scipy pillow einops torchsummary
pip install torchvision
# 安装 PyTorch：按你的 CUDA 版本自行选择对应安装命令
```

### 2) 准备数据（BJTU）

`UDTL-master1/datasets/BJTU.py` 的数据期望目录结构（关键点）：

- 工况子目录：`M0_G0_LA0_RA0 ~ M0_G0_LA4_RA0`（对应 5 类标签）
- 每个工况子目录下：`Sample_{k}` 文件夹
- 样本文件命名：`data_leftaxlebox_{subfolder}_{state}.csv`

训练时通过 `--data_dir` 指定 `datasets` 的根路径（例如 `D:\bishe\BJTU`）。

### 3) 运行训练

训练入口脚本：

```bash
python UDTL-master1/train_advanced.py ^
  --model_name Convformer ^
  --data_name BJTU ^
  --data_dir "D:\bishe\BJTU" ^
  --cuda_device "0" ^
  --checkpoint_dir "./UDTL-master1/checkpoint" ^
  --batch_size 64
```

如果你要启用多损失融合（注意：代码中 `argparse` 的 `type=bool` 存在字符串解析陷阱，建议直接“开就写 True，关就不要传”）：

```bash
python UDTL-master1/train_advanced.py ^
  --model_name Convformer ^
  --data_name BJTU ^
  --data_dir "D:\bishe\BJTU" ^
  --cuda_device "0" ^
  --checkpoint_dir "./UDTL-master1/checkpoint" ^
  --batch_size 64 ^
  --distance_metric True ^
  --distance_loss CORAL ^
  --domain_adversarial True ^
  --adversarial_loss CDA
```

训练完成后，`checkpoint/` 下会出现 `best_model.pth`、日志文件以及预测/特征输出（`Prelabel/Truelabel/Feature`）。

### 4) 汇总日志统计

```bash
python UDTL-master1/readlog2.py
```

---

## 🙏 Acknowledgements

- 感谢原始开源项目与作者提供的基础实现（本项目基于开源代码进行学习、复现与深化研究）。
- 声明：本仓库用于学习、复现与科研研究目的；若你在论文/报告/开源发布中使用了相关成果，请遵循相应许可证条款与引用规范。

