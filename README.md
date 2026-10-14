# 航空乘客满意度预测 · Kaggle S6E10

**Built with PriorLabs-TabPFN**（第35/36轮使用许可不同的旧版TabPFN2.0；未使用3.5权重。）
这个项目记录我们从树模型基线、公开原始数据教师、辅助任务与表格神经网络，到融合、排除验证、TabICL 微调的实验过程。

**已取得的最佳公开 ROC-AUC：0.96133。** 最新最佳为第28轮保留原生折覆盖的排序融合，官方提交编号 `57031572`。2026-10-10查分快照为313 / 1354；账号已评分19次，本轮新增1次。第29轮另完成20个新模型训练，未通过预定筛选。本地验证与官方成绩分别记录。

比赛：[Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)

## 先看这份总结

- **[METHODS.md：我们的做法、有效改进、失败实验、资源限制与验证边界](METHODS.md)**
- [docs/SHARING.md：如何分享、下载和协作](docs/SHARING.md)
- [docs/HISTORICAL_README.md：历史运行顺序和逐轮记录](docs/HISTORICAL_README.md)
- [TENFOLD_OPTIMIZATION.md：最佳第十三轮](TENFOLD_OPTIMIZATION.md)
- [ROUND23_25_RESULTS.md：TabICL 实际微调结果及推理一致性检查](ROUND23_25_RESULTS.md)
- **[ROUND28_29_RESULTS.md：最新公榜提升、20个模型重训与验证修订](ROUND28_29_RESULTS.md)**
- [CLOUD_GPU_TRIAL.md：Kaggle实际GPU与完整上下文试验（2026-10-14）](CLOUD_GPU_TRIAL.md)

## 本仓库包含什么

项目自己编写的 Python 建模、评价、审计和 Kaggle 管理脚本，以及方法和实验报告。`analyses/gap_error_diagnostic.py` 提供历史 OOF 错误诊断实现。

比赛数据、原始外部数据、模型权重、预测 CSV、虚拟环境、账号凭证和私人浏览器截图不在仓库中。高级实验依赖前序运行生成的缓存；历史报告中的缓存/截图/逐行预测链接对应原机器记录，不代表这些文件已公开发布。

## 从零运行 CPU 基线

需要 Python 3.11 或以上。以下是 Windows PowerShell 示例：

```powershell
git clone https://github.com/wozhenshiliudada/kaggle-airline-satisfaction-s6e10.git
cd kaggle-airline-satisfaction-s6e10
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_workflow.py
```

`check_workflow.py` 用合成数据实际训练 CatBoost / LightGBM，检查 OOF 覆盖、缺失值、未见类别、ID 对齐以及非法提交拒绝。这个检查的分数不是比赛分数。

准备真实数据：先在 Kaggle 注册并接受比赛规则，然后通过官方登录授权。

```powershell
.\.venv\Scripts\python.exe manage.py login
.\.venv\Scripts\python.exe manage.py download
.\.venv\Scripts\python.exe manage.py local --folds 3 --threads 8
```

最后一条命令运行 CPU 基线并生成 `outputs/`，不会因为该命令自动上传预测。需要提交时，单独执行：

```powershell
.\.venv\Scripts\python.exe manage.py submit --output outputs
.\.venv\Scripts\python.exe manage.py scores
.\.venv\Scripts\python.exe manage.py rank
```

Windows 上如果官方 SDK 出现特定 TLS 连接问题，可以使用随仓库的传输包装器，例如：

```powershell
.\.venv\Scripts\python.exe kaggle_tls13_transport.py competitions list --group entered --search playground-series-s6e10 --format json
```

凭证由 Kaggle 官方工具保存在用户目录，不要写进项目或提交到 Git。

## 继续高级实验

`requirements-tested.txt` 和 `requirements-nn-tested.txt` 保存原机器实际使用的部分版本，包含特定 CUDA 构建信息；它们不是所有平台都能直接安装的统一锁文件。

GPU 与神经网络的历史安装步骤、前序缓存生成及运行顺序见 [历史运行文档](docs/HISTORICAL_README.md)。TabICL 高级试验另需对应依赖和预训练权重；这些未打包。最新最佳仍依赖完整前序模型链和历史预测；仅运行某个高级脚本不能从零复现最佳成绩。第28/29轮脚本和冻结协议用于原工作区的续跑，新工作区应先生成前序缓存。

## 主要文件

| 文件 | 用途 |
|---|---|
| `train.py` / `check_workflow.py` | CPU 基线与合成数据端到端检查 |
| `manage.py` | 官方 Kaggle 登录、下载、运行、提交与查分 |
| `improve.py` / `context_round.py` / `auxiliary_round.py` | 教师、条件特征和辅助特征 |
| `nn_round.py` / `auxiliary_neural_round.py` | RealMLP 神经网络 |
| `round13_tenfold.py` / `evaluate_round13.py` | 最佳十折成员及固定候选比较 |
| `round16_*.py` / `audit_round16_checkpoints.py` | 排除验证与缓存审计 |
| `round18_foundation_*.py` | TabICL 资源与完整开发上下文试验 |
| `round23_finetune.py` / `round24_finetune_features.py` / `round25_finetune_routes.py` | 三组 TabICL 微调 |
| `round26_consistency.py` | 冷加载、种子和缓存一致性检查 |
| `round28_native_fusion.py` / `round29_matched_context.py` | 原生覆盖融合及两条件分支的十折重训 |
| `verify_round28_29.py` / `submit_round28_29.py` | 最终文件重建、官方SDK提交及查分对账 |

## 已知限制

多轮方法选择反复使用历史 OOF；部分成员折数及训练覆盖不同，融合验证不是完整嵌套 CV。新模型内部验证改善没有保证转移到公开榜。详见 [总结中的验证边界](METHODS.md#7-为什么后来很难继续提升)。

首次发布前，97 个 Python 文件通过语法检查，122 个导出源文件与原文件指纹一致，合成数据 CPU 基线的真实训练与提交格式检查通过。此检查没有重新训练全部比赛模型。[发布验证记录](docs/PUBLICATION_VALIDATION.json)

模型、方法和公开资源来源在 [SOURCES.md](SOURCES.md) 中列出。没有导入其他参赛者的预测来作为本项目的模型结果，也没有将第三方库或模型权重打包进仓库。

本轮更新验证了20个真实模型训练任务、候选逐行预测重建和官方提交57031572的评分回执。完整说明见[第28—29轮报告](ROUND28_29_RESULTS.md)，摘要审计见[本轮发布验证记录](docs/ROUND28_29_VALIDATION.json)。

## Kaggle云端大上下文试验

2026-10-14在私有离线Notebook实际分配2×Tesla T4（每张约16GB），使用TabICL v2，未使用TabPFN3.5权重。原始/航程特征版完成559708行训练上下文、139927行验证与四成员预测，耗时约50.5分钟；固定融合未提高同折AUC。教师/辅助特征版经补跑也完整完成，固定融合未改善；另实际完成TabPFN2.0两万行上下文对照，固定融合同样未改善。全部资源与结果见[试验报告](CLOUD_GPU_TRIAL.md)。云端试验分数属于本地外层验证，未据此新增正式比赛提交。

新增cloud-round30/31/32/34/35/36源码用于检查实际执行配方。运行需要自行取得有权使用的竞赛输入、官方TabICL v2权重与离线wheel，并配置自己账号的私有Notebook数据源；教师特征版还依赖历史辅助特征包。收集/评分程序需要原工作区的冻结协议、预测和保护状态，这些私人或逐行产物不在仓库。GPU请求名与实际设备可能不同，应读运行输出的hardware.json。

本轮新增11个第一方Python文件通过AST检查，GPU结果由官方下载并核验索引、权重与预测SHA256；独立秩和AUC重建最大误差1.11e-16。源码公开时统一为LF，保留与原冻结文件的指纹映射；无数据/逐行预测/权重/账号文件。见[云端审计摘要](docs/CLOUD_GPU_VALIDATION.json)。旧版模型权重采用[Prior Labs License1.1](docs/TABPFN2_LICENSE.txt)，代码和权重许可分别处理。
