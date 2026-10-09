# 同题方法差距审计（本轮只分析）

已保存详细来源、SHA256、0基 Notebook cell 索引和判断边界到 `evidence/gap-peer-methods.json`。没有训练、提交、修改认证、接受新许可、下载权重或第三方预测。

优先检验的缺口是**原始输入 + 辅助重建特征的 RealMLP 载体**。参考 [More Aux Carriers](https://www.kaggle.com/code/goodpjw2008/s6e10-more-aux-carriers-lb-0-96152) 的 cell 12/14/32/33：`static_frame()` 不加入目标编码、画像、频次或原始数据模型预测，`nn_frame` 只加数值类别副本，再以 `te=False` 训练。作者报告两个单体 AUC 为0.960972/0.961020，却给其组合带来0.000047/0.000066。数字尚未由本项目复现，不能当作预计收益。

本项目的 `auxiliary_neural_round.py` all/control 和 round13 十折都经过增强的 `frames()`：三个组合目标编码、六个航线目标编码、航线画像及三个原始模型信号等。因此它们没有测试这条简化输入路线。现有 baseline 五折401.98秒、rich auxiliary 五折462.61秒；新路线耗时只能先估为分钟级，需首折实测。建议在新的完整外层验证中固定 raw-only 对照、raw+13预计值、raw+30全部辅助列三个输入，先保持自己的已验证网络参数，比较组合边际收益。

[TabPFN + Route Categories](https://www.kaggle.com/code/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160) 另有可独立运行的模型源码（cell39/43），但默认仍读作者预测CSV。全56万context使用int8缓存，作者实测17.5GB；本机实际RTX4060 Laptop仅8188MiB，当前空闲6159MiB。50k/200k小context的作者首折AUC为0.958298/0.959963，低于full-context的0.960619，小context不是等效复现。其0.96160是作者整套组合的历史自报成绩，未在本轮核实当前分数。

[PriorLabs当前官方模型文档](https://docs.priorlabs.ai/models) 的3.5支持最高100万行，不能引用旧v2的1万行上限来断言不能用。行数支持和显存需求是两回事。[代码](https://github.com/PriorLabs/TabPFN) 与 [权重许可](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE) 分开；3.5许可明写Kaggle等数据科学竞赛，但还含首次账号/接受许可及professional-purpose条件，需对比赛规则和具体许可继续核对。

更次级的独立模型路线是 raw+aux DeepFM，原文cell36/37报告组合增益仅0.000009，优先级低于前者。[TabICLv2官方项目](https://github.com/soda-inria/tabicl) 提供CPU/磁盘卸载支持较大context，但未在本机确认资源和时间。不要把H100基准速度用于4060。

Nina页面搜索缓存显示v4的0.96159，但当前官方API拒绝读取，根节点浏览器显示页面不存在。因此不能用它证明当前存在可复现的强单模型。

他人的所谓nested CV尚未证实完成每次外层全部基模型重新训练；也不能将上述任何组件单独归因为当前全部榜单差距。
