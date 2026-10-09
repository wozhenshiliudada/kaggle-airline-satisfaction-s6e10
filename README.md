# 航空乘客满意度预测 · Kaggle S6E10

这个项目记录我们从树模型基线、公开原始数据教师、辅助任务与表格神经网络，到融合、排除验证、TabICL 微调的实验过程。

**已取得的最佳公开 ROC-AUC：0.96132。** 最佳为第十三轮十折航程辅助 RealMLP 替换方案，官方提交编号 `56857129`。截至整理时，13 次历史官方提交已评分；本地 OOF、开发验证、历史公榜和未成功的试验分别记录。

比赛：[Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)

## 先看这份总结

- **[METHODS.md：我们的做法、有效改进、失败实验、资源限制与验证边界](METHODS.md)**
- [docs/SHARING.md：如何分享、下载和协作](docs/SHARING.md)
- [docs/HISTORICAL_README.md：历史运行顺序和逐轮记录](docs/HISTORICAL_README.md)
- [TENFOLD_OPTIMIZATION.md：最佳第十三轮](TENFOLD_OPTIMIZATION.md)
- [ROUND23_25_RESULTS.md：TabICL 实际微调结果及推理一致性检查](ROUND23_25_RESULTS.md)

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

GPU 与神经网络的历史安装步骤、前序缓存生成及运行顺序见 [历史运行文档](docs/HISTORICAL_README.md)。TabICL 高级试验另需对应依赖和预训练权重；这些未打包。最佳预测依赖完整的前序模型链，仓库不声称一条命令或仅运行 `round13_tenfold.py` 就能得到 0.96132。

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

## 已知限制

多轮方法选择反复使用历史 OOF；部分成员折数及训练覆盖不同，融合验证不是完整嵌套 CV。新模型内部验证改善没有保证转移到公开榜。详见 [总结中的验证边界](METHODS.md#7-为什么后来很难继续提升)。

首次发布前，97 个 Python 文件通过语法检查，122 个导出源文件与原文件指纹一致，合成数据 CPU 基线的真实训练与提交格式检查通过。此检查没有重新训练全部比赛模型。[发布验证记录](docs/PUBLICATION_VALIDATION.json)

模型、方法和公开资源来源在 [SOURCES.md](SOURCES.md) 中列出。没有导入其他参赛者的预测来作为本项目的模型结果，也没有将第三方库或模型权重打包进仓库。
