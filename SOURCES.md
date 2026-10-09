# 优化的公开来源

公开 Notebook 仅用于阅读思路，公开数据和许可明确的软件包用于实际训练。模型在本机重新训练，未下载他人的预测结果，也未执行公开 Notebook 代码。

- 比赛 [官方数据说明](https://www.kaggle.com/competitions/playground-series-s6e10/data) 引用的 [原始数据](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k)，下载时官方 API 显示许可证为 CC0-1.0。
- Дворкин Евгений Владимирович 的 [S6E10 LightGBM CV 5 folds 0.96102](https://www.kaggle.com/code/evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96102)：数值尾数、频次、折内目标编码、航程分组统计，以及原始数据模型预测特征。
- 인공저능연구소 (Artificial Idiocy Lab) 的 [S6E10 Honest OOF Stack](https://www.kaggle.com/code/megayak/s6e10-honest-oof-stack)：紧凑特征组合、固定折分割和对组合模型的独立验证。

实现包含额外的原始数据精确重合排除：任何特征与比赛训练集或测试集完全相同的原始数据行，均不用于原始数据预测模型或目标统计。比赛验证集的标签不进入目标编码，训练行的目标编码使用内部五折。

第三轮使用 [PyTabKit / RealMLP](https://github.com/dholzmueller/pytabkit) 的官方 PyPI 1.7.3 包，许可证为 [Apache-2.0](https://github.com/dholzmueller/pytabkit/blob/v1.7.3/LICENSE.txt)。包来源与 SHA256 保存在 `third_party/pytabkit-source.json`，原始许可证保存在 `third_party/pytabkit-1.7.3-LICENSE.txt`。没有使用预训练模型权重。

神经网络超参数思路参考 [yekenot 的 RealMLP Notebook](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit)，由本项目 `nn_round.py` 独立实现，所有正式预测由本机 GPU 训练生成。GPU 版 Torch 2.9.1+cu130 来自 PyTorch 官方安装源；实际版本保存在 `requirements-nn-tested.txt`。

本场 [规则 §3.6.c](https://www.kaggle.com/competitions/playground-series-s6e10/rules) 要求提交所用开源代码使用 OSI 批准且不限制商业使用的许可证；外部数据和模型还需满足 §2.6 的可获取条件。具体审核结论与实验选择记录在 `DISCUSSION.md`。

第四轮筛选 [TabM 官方项目](https://github.com/yandex-research/tabm) 的参数高效神经网络集成路线，实际调用已安装 Apache-2.0 的 PyTabKit 所带实现；没有新增预训练权重或下载其他参赛者的预测。实验结果见 `OPTIMIZATION.md`。

第八轮阅读了六份最新公开 Notebook 的源码（未下载预测文件、未执行第三方代码）。重点参考 [goodpjw2008 的 More Aux Carriers](https://www.kaggle.com/code/goodpjw2008/s6e10-more-aux-carriers-lb-0-96152)：用其他 20 个原始特征预测 13 个评分和 3 个类别，产生 13 个预计评分、16 个实际取值概率和 1 个总对数概率。由 `auxiliary_round.py` 独立实现，训练使用公开比赛的未标注特征值，辅助五折每行排除自身，完全不加载满意度列。`auxiliary_neural_round.py` 使用匹配五折比较加入前后，附加航线画像和排除重合后的原始数据 XGBoost。

还阅读 [TabPFN + Route Categories](https://www.kaggle.com/code/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160)，其更高分部分使用 TabPFN 3.5；[官方说明](https://github.com/PriorLabs/TabPFN) 将代码许可证和权重许可证分开，3.5 权重为非商业许可，未纳入本轮提交路线。公开最高分之一 [Naji 的分享](https://www.kaggle.com/code/najiama/high-scoring-oof-cv-0-96192-lb-0-96165) 主要加载预测文件，而非提供完整训练实现，本项目没有将其预测用作自己的模型成绩。

[alvis1111 的本场 GitHub 项目](https://github.com/alvis1111/s6e10-airline-solution) 提供精确距离画像、类别占比、十折及原始数据模型思路；其自报公榜 0.96100，低于本项目已核验最佳 0.96122。页面未展示 LICENSE，未复制/执行其源码。文中“最终”不是比赛已经结束的证明。源码审计记录见 `references/round8/source-audit.json`。

第九轮沿用已独立构建的 30 列辅助重建特征，在匹配条件 LightGBM 中测试，并用航线特征重新训练 TabM。2026-10-05 再次阅读 [TabM 官方项目与 Apache-2.0 许可](https://github.com/yandex-research/tabm)，实际模型仍调用已安装的 PyTabKit 1.7.3 所带实现；不需要下载额外代码或预训练权重。与第四轮仅使用普通特征的 TabM 不同，本轮保留航线编码、航线画像、三个原始数据模型和辅助重建特征，实际来源及参数指纹见 `experiments/round9-tabm-auxiliary-5fold/configuration.json`。

第十一轮由本项目独立实现辅助概率不确定性、联合条件网络和按出行类型分开的专家模型。复用本机训练的 80 个辅助任务概率检查点，没有下载额外数据、源码、预测或预训练权重。现有 LightGBM、PyTabKit 与其他依赖许可继续适用。完整两组五折和停止的首折网络见 `DIVERGENT_OPTIMIZATION.md`，最终采用 10% 出行专家与原最佳融合，官方公开 AUC 为 0.96131。

第十二、十三轮继续由本项目独立实现：先测试无标签问卷一致性摘要、舱位专家及既有原始 XGBoost 在树中的接入，再将辅助 RealMLP 外层改为十折。没有新增外部数据、第三方预测、预训练权重或软件依赖；继续使用已核验的公开原始数据、现有 LightGBM/XGBoost 和 Apache-2.0 的 PyTabKit。十折改变训练覆盖率和测试集成成员数，采用与已有开源示例相通的交叉验证思路，未复制没有明确许可的 GitHub 代码。实验结果分别见 `COHERENCE_OPTIMIZATION.md` 和 `TENFOLD_OPTIMIZATION.md`。

第十四、十五轮继续在线阅读 GitHub 同题方案、TabM/数值嵌入论文及航空满意度研究。只独立检验算法思路与现有包配置：TabM 独立批次、航程高分辨率分箱、均匀/正态排序尺度融合，以及保持原树参数的十折覆盖率。前一阶段三个方向未通过完整筛选；后续十折树另行固定规则。没有复制缺少许可的源码，没有下载或使用其他参赛者的预测、外部 TabPFN 结果或新预训练权重。

本次来源的具体取舍、失败、适用边界和复现入口见 [RESEARCH_OPTIMIZATION.md](RESEARCH_OPTIMIZATION.md)，GitHub 文件 blob 指纹见 `evidence/round14-source-audit.json`。同题仓库的自报分数、所谓理论上限和“nested CV”没有在本项目独立验证；航空研究使用不同调查数据与目标，不能用其分数直接比较本比赛。
