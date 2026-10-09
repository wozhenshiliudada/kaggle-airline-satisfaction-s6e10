# 历史运行文档

此文是原实验目录的历史 README 快照，保留逐轮命令和当时结果。当前总览请看 [仓库首页](../README.md) 和 [方法总结](../METHODS.md)。文中的数据、截图、模型和预测缓存未随公开仓库发布。

# 航空乘客满意度预测：Kaggle S6E10

比赛：https://www.kaggle.com/competitions/playground-series-s6e10

预测 `satisfaction` 的概率，评分为 ROC-AUC。截止时间：北京时间 2026 年 11 月 1 日 07:59（以比赛页面为准）。

## 当前状态

账号 `woshiliudada` 已注册，并已在网页核实登录。
已安装 Kaggle CLI 2.2.4，完成 CatBoost/LightGBM 的真实训练流程测试（使用模拟数据），并准备本地/云端训练代码。
比赛规则已接受，官方 SDK 已经通过 OAuth 登录为 `woshiliudada`，正式数据已下载并完成结构检查（训练 699,635 行，测试 299,844 行）。

已完成多轮建模、13 次官方 API 提交和查分。新增排除复验划分，完整重训开发基准，并比较三种原始输入辅助神经网络。本次新增 **0** 条正式提交。最佳公开 AUC **0.96132**，最佳提交 **56857129**；只在公开分数严格提高时更新最佳指针。北京时间 **2026-10-06 10:41** 查询排名 **159/852**，名次为当时快照。开发、复验和全量生产兼容性结果见 [新划分优化记录](../LOCKED_OPTIMIZATION.md)。当前最佳为 [预测 CSV](../outputs-round13/submission.csv)；全部历史文件和最佳指纹保留。

首轮融合公开 AUC 为 0.95830，排名查询时为 453/702。第二轮加入公开原始数据预测、数值频次/尾数、航程统计和折内目标编码，重新训练 LightGBM、CatBoost 和 XGBoost。第三轮使用 Apache-2.0 的 RealMLP 神经网络。第五轮统一五折对照，新增舱位、出行类型、客户类型与服务评分的条件特征，最终组合 OOF AUC 为 **0.96159322**；公开榜从 0.96116 升到 **0.96122**。详见 [结果说明](../RESULTS.md)、[第五轮实验记录](../CONTEXT_OPTIMIZATION.md)、[第六轮实验和正式评分](../BAGGING_OPTIMIZATION.md)、[三个子代理的讨论和实验记录](../DISCUSSION.md)、[公开方法来源](../SOURCES.md)、[官方查询原始结果](../kaggle-results.json) 和 [第五轮网页截图](../evidence/round6-submissions.jpg)。

模拟数据测试结果仅用于验证代码，不是比赛分数。

## 账号准备（在网页完成）

1. 注册 Kaggle，完成邮箱/手机等网站要求的验证。
2. 打开比赛，点击 Join Competition，阅读并接受规则。
3. 推荐由 `manage.py login` 启动官方浏览器授权。在网页确认后，官方 CLI 管理凭证。
4. 若浏览器授权不适用，可在 https://www.kaggle.com/settings/api 创建 API Token，保存到用户目录 `.kaggle/access_token`；或使用旧版 `.kaggle/kaggle.json`。

密码、验证码、API Token 均不要粘贴到聊天或写进项目代码。

## 运行

以下命令在本目录执行；后续可交给 Codex 操作。

```powershell
.\.venv\Scripts\python.exe manage.py doctor
.\.venv\Scripts\python.exe manage.py login
.\.venv\Scripts\python.exe manage.py download
```

首轮入口使用本地 CPU。可自动下载、训练、检查并提交：

```powershell
.\.venv\Scripts\python.exe manage.py local --submit
```

也准备了 Kaggle CPU 云端运行入口，脚本设为私有、关闭 GPU 和外网；该路径本次未实际运行：

```powershell
.\.venv\Scripts\python.exe manage.py cloud --username YOUR_KAGGLE_USERNAME
.\.venv\Scripts\python.exe manage.py status
# 等云端状态为 complete
.\.venv\Scripts\python.exe manage.py collect
.\.venv\Scripts\python.exe manage.py submit
.\.venv\Scripts\python.exe manage.py scores
.\.venv\Scripts\python.exe manage.py rank
```

## 建模与验证

下面的基础流程由 `train.py` 提供。第二轮流程由 `improve.py`、`xgb_round.py` 和 `combine_round.py` 提供，第三轮由 `nn_round.py` 和 `combine_nn.py` 提供；本机 CatBoost/XGBoost/RealMLP 已实际验证使用 NVIDIA GPU。

- 默认 3 折分层交叉验证，固定随机种子，排除乘客 ID。
- CatBoost 和 LightGBM 都使用每折独立验证、早停。
- 保留原始特征，并加入服务评分聚合、缺失数量和延误差等特征。
- 比较 5 个融合权重，以 OOF AUC 选择首轮提交。融合选择与验证共用 OOF 数据，分数存在选择偏差；官方成绩以 Kaggle 为准。
- 输出 `submission.csv`、`metrics.json`、`oof.csv`、`test_predictions.csv` 和 `feature_importance.csv`。
- 提交前检查列名、全部 ID 顺序、行数、概率范围、数据指纹、预测文件指纹；阻止重复提交相同文件。
- 账号累计 13 次提交，均已评分 Complete。本次新增 0 次正式提交；同文件重复调用会跳过上传。最终提交选择和当日剩余额度以 Kaggle 当前页面为准。
- 首轮只使用正式比赛数据；第二轮加入官方数据页引用的公开原始数据。输入结构、真实数据训练和提交格式检查均已完成。

## 复现第二轮

先通过官方 API 下载 `arseniyshutko/binary-aviation-satisfaction-129k` 的 `data.csv` 到 `data-original/`，然后执行：

```powershell
.\.venv\Scripts\python.exe improve.py prepare
.\.venv\Scripts\python.exe improve.py compact --threads 12 --iterations 4000
.\.venv\Scripts\python.exe improve.py route --threads 8 --iterations 3500
.\.venv\Scripts\python.exe improve.py cat --threads 8 --iterations 5000
.\.venv\Scripts\python.exe xgb_round.py
.\.venv\Scripts\python.exe combine_round.py
.\.venv\Scripts\python.exe manage.py submit --output outputs-round2
.\.venv\Scripts\python.exe fetch_results.py
```

输出保存在 `experiments/` 和 `outputs-round2/`，原先的 `outputs/` 保留首轮结果。中断重跑时仅复用配置指纹相同的完整折缓存；若参数或代码变更，应将已有实验目录改名保留，再运行新实验。相同预测文件会跳过重复提交。第二轮排除了与比赛数据特征完全相同的 21 条原始数据，使用 129,859 条原始记录训练辅助模型和生成目标统计。目标编码在外层训练折内再做五折交叉编码，已通过验证标签扰动检查。

官方接入说明：https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md

## 复现第三轮

本机使用 RTX 4060 Laptop 8GB。先完成第二轮的 compact/cat 数据和模型缓存，再通过 PyTorch 官方源安装 GPU 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install pytabkit==1.7.3 --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe route_auc_round.py auc
.\.venv\Scripts\python.exe nn_round.py --folds 3 --ensemble 4
.\.venv\Scripts\python.exe combine_nn.py
.\.venv\Scripts\python.exe manage.py submit --output outputs-round3
.\.venv\Scripts\python.exe fetch_results.py
```

`nn_round.py` 增加数值特征的类别视图和折内交互目标编码，固定训练三 epoch。外层验证标签翻转检查已通过。预测按批处理以控制 GPU 内存。每个实验保存代码、参数和输入指纹，相同配置可逐折恢复。改变 `--folds` 或 `--ensemble` 会使用不同目录，保留旧实验。

RealMLP 单模型三折 OOF 为 0.960948，树模型基线为 0.961137，二者各占 50% 的组合为 0.961425。六个融合权重共用 OOF 选择，存在选择偏差；Kaggle 公开 AUC 0.96107 是独立于本地验证的正式评分。没有使用他人的预测或预训练权重。许可证、来源和实际依赖版本保存在 `third_party/`、`SOURCES.md`、`requirements-nn-tested.txt`。

已完成五折、每折八成员的增强模型，单独 OOF 0.961150，与相同树模型各占 50% 时为 0.961515；正式公开 AUC 为 **0.96116**。本机 GPU 完整训练约 6 分 42 秒。复现命令：

```powershell
.\.venv\Scripts\python.exe nn_round.py --folds 5 --ensemble 8
.\.venv\Scripts\python.exe combine_nn.py --nn experiments\realmlp-5fold-8ens --output outputs-round4
.\.venv\Scripts\python.exe manage.py submit --output outputs-round4
.\.venv\Scripts\python.exe fetch_results.py
```

五折神经网络与三折树模型的 OOF 逐行对齐；报告中的三个评价区块沿用树模型划分，神经网络实际训练为五折，不能视作整个组合的同划分三折验证。两版模型均保留正式预测和配置缓存。

## 第四轮优化

比较原始数据辅助预测、TabM、更多 RealMLP 训练轮数、五折 compact LightGBM、更深的 CatBoost、宽特征 LightGBM 正则参数，以及三种融合尺度。详细失败与有效实验记录见 [OPTIMIZATION.md](../OPTIMIZATION.md)。

最终 logit 融合加 10% 新 LightGBM 的 OOF AUC 为 0.961535，三个评价区块均提高；官方提交 `56832459` 的公开 AUC 仍为 0.96116，与旧最佳持平，没有新的官方排名提升。旧最佳预测和所有实验缓存均保留。

复现最终候选（需要已有第二、三轮模型缓存）：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe route_tuned.py
.\.venv\Scripts\python.exe calibration_round.py
.\.venv\Scripts\python.exe combine_next.py experiments\route-tuned-3fold --base outputs-round5-calibration --output outputs-round5 --description "RealMLP5fold logit plus tuned LightGBM"
.\.venv\Scripts\python.exe manage.py submit --output outputs-round5
.\.venv\Scripts\python.exe fetch_results.py
```

## 第五轮：统一验证与条件特征

三类树模型完整使用相同五折和超参数。条件树模型从 0.96098293 升到 **0.96111771**，五折均提高；原始模型 logit 残差路线五折均下降，未采用。条件神经网络保持原五折八成员参数，单独 OOF 与对照持平，但与旧组合具有互补性。

最终采用 50% 旧组合与 50% 条件树/条件神经网络的等权 logit 组合，OOF 为 **0.96159322**，五个评价区块均提高；额外群体校准没有收益，未采用。官方提交 `56843782` 为 **0.96122**。完整结果、难排序贡献和验证边界见 [CONTEXT_OPTIMIZATION.md](../CONTEXT_OPTIMIZATION.md)。最终文件在 `outputs-round6/`。

本轮提交已成功评分，Windows GBK 读取 CLI Unicode 回执失败曾导致本地命令返回错误；已通过官方 API 恢复该文件的提交记录，将父子进程输出统一为 UTF-8，并核对同文件重复提交会直接跳过。未再次上传。

## 第六轮：独立随机种子与 CatBoost 宽特征

五组新模型完成完整五折：两组新随机种子 RealMLP、CatBoost 对照、条件版和宽条件版。增加神经网络种子没有稳定提高现有组合。宽特征 CatBoost 相对条件对照五折均提高，单模型 OOF 从 0.96093175 升到 **0.96106659**。

以 10% 概率权重加入正式最佳后，实验组合 OOF 为 **0.96160061**，相对基线增加 0.00000740，但四个评价区块提高、一个下降，未通过保守的全区块筛选和留块检查。已单独保存 [实验预测](../outputs-round7-experimental/submission.csv) 和 [指标](../outputs-round7-experimental/metrics.json)。用户随后明确要求先提交一次，官方提交 `56849528` 已 Complete，公开 AUC **0.96121**；正式最佳仍为 `outputs-round6/` 的 **0.96122**。全部模型结果、选择偏差、精确公式和复现命令见 [BAGGING_OPTIMIZATION.md](../BAGGING_OPTIMIZATION.md)。

用户要求加入 `enjinjiang`。浏览器控制工具启动失败，官方 SDK 没有邀请/合并队伍接口，未由本项目发邀请、未核实成员状态。用户后来说明加入的是 Kaggle Groups，比赛 Team 中仍搜不到对方。最新指示“先不管了，你继续想办法优化”已将提交前组队等待暂停；组队待办和此前两次单文件放行历史保留在 `team-request.json`，成员状态仍未确认。同一文件再调用会直接跳过，避免重复上传。

## 最新公开方法复现

本轮检索六份公开 Notebook 与同题 GitHub 方案，独立实现辅助任务重建：80 个 XGBoost 产生 30 列新特征，再做两组匹配五折 RealMLP，共 80 个神经网络成员。新组合 OOF AUC **0.96164276**，相对旧最佳本地 0.96159322 提高 0.00004955，全部五个评价区块提升。候选为 75% 旧最佳 + 25% 新神经网络，见 [详细实验记录](../AUXILIARY_OPTIMIZATION.md) 和 [候选文件](../outputs-round8/submission.csv)。

用户明确要求“先提交先”后，新候选已提交并 Complete，编号 **56850948**，公开 AUC **0.96128**，较原最佳 0.96122 提升 0.00006。北京时间 19:34 官方查询排名 **154/785**，提交前 19:32 为 176/785；本次榜单提升已由官方评分确认。本地 OOF 仍有模型选择偏差。组队前本次单文件授权已使用，队友邀请安排继续保留。

## 最新辅助树模型与航线 TabM

将 30 个辅助特征加入条件 LightGBM，并独立训练航线及辅助特征 TabM。两组均完成完整五折，最终采用替换原条件 LightGBM 后再加入 10% TabM 的组合，OOF **0.96167319**，五个评价区块与留块选择均通过。正式提交 **56851874** 的公开 AUC **0.96129**，成为新最佳。

该轮文件为 [outputs-round9/submission.csv](../outputs-round9/submission.csv)，验证见 [outputs-round9/validation.json](../outputs-round9/validation.json)，实验过程和复现命令见 [ROUND9_OPTIMIZATION.md](../ROUND9_OPTIMIZATION.md)。TabM 概率校正作为额外诊断，未通过组合稳定性筛选，没有提交。

## 发散优化与新最佳

测试辅助不确定性、联合条件神经网络、出行专家三条路线。联合神经网络在首折固定融合中没有改善，停止；另两条路线完成完整五折。最终选择 90% 原最佳与 10% 商务/个人出行专家预测，OOF **0.96168931**，五个区块均提高。留块选择有一次回落，相关选择偏差仍保留在报告中。

正式提交 **56853320**，公开 AUC **0.96131**，比原最佳提高 0.00002。21:31 官方排名 **149/792**；本轮只新增一次提交，回执显示今天剩 5 次。当前最佳文件为 [outputs-round11/submission.csv](../outputs-round11/submission.csv)，实验、复现和验证边界见 [DIVERGENT_OPTIMIZATION.md](../DIVERGENT_OPTIMIZATION.md)，检查见 [outputs-round11/validation.json](../outputs-round11/validation.json)，原始官方结果见 [evidence/round11-after-submit-official-results.json](../evidence/round11-after-submit-official-results.json)。

## 十折训练覆盖率与最新正式评分

第十二轮一致性摘要、舱位专家、先验校正、原始老师接入树和不确定性叠加均没有通过筛选，详见 [失败实验记录](../COHERENCE_OPTIMIZATION.md)。随后保持辅助 RealMLP 的特征及超参数，将外层五折改为十折：每次训练使用 90% 样本，测试预测由 80 个成员平均。

替换旧辅助网络后，组合 OOF 从 **0.96168931 提高到 0.96173256**，五个旧评价区块及留块选择均提高。正式提交 **56857129** 的公开 AUC 为 **0.96132**，高于此前 0.96131；2026-10-06 00:44 排名 **147/807**。只新增一次提交，输入/代码/检查点和全部测试 ID 已核对，旧最佳保留。组合仍包含历史三/五折成员，不能视为全流程统一十折或嵌套验证。详见 [实验与复现](../TENFOLD_OPTIMIZATION.md)、[正式预测](../outputs-round13/submission.csv)、[验证](../outputs-round13/validation.json) 和 [官方原始结果](../evidence/round13-after-submit-official-results.json)。

## GitHub 与相关研究后的最新优化

阅读 TabM、数值嵌入、航空满意度特征筛选研究和多个同题仓库，独立测试批次多样性、航程分箱、排序尺度及十折树覆盖率。前三条路线未通过，停止或保留失败缓存；十折树完成全量评估。正式提交 **56858252**，已 Complete，公开 AUC **0.96131**。没有超过原最佳 0.96132，最佳文件继续保留第十三轮。只新增一次正式提交；上传后同文件重复调用已跳过。

截至 2026-10-06 01:37，最佳 **0.96132**，排名 **148/811**。详见 [实验、来源和复现](../RESEARCH_OPTIMIZATION.md)、[完整十折对照](../analyses/round15-comparison.json)。没有下载别人的预测或预训练权重，没有新增外部数据。


<!-- ROUND16_LOCKED_LATEST_BEGIN -->

## 新划分重训后的最新优化

新增排除复验划分，完整重训开发基准，并比较三种原始输入辅助神经网络。本次新增 **0** 条正式提交。最佳公开 AUC **0.96132**，最佳提交 **56857129**；只在公开分数严格提高时更新最佳指针。北京时间 **2026-10-06 10:41** 查询排名 **159/852**，名次为当时快照。开发、复验和全量生产兼容性结果见 [新划分优化记录](../LOCKED_OPTIMIZATION.md)。

<!-- ROUND16_LOCKED_LATEST_END -->


<!-- ROUND16_FINAL_SNAPSHOT_BEGIN -->

最新核验：北京时间 2026-10-06 10:41，最佳公开 AUC 0.96132，排名 159/852，累计提交 13 次且全部 Complete。本次两轮未新增官方提交，保持第十三轮最佳预测。

<!-- ROUND16_FINAL_SNAPSHOT_END -->
