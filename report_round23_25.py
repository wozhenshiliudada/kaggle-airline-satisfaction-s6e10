"""Verify real checkpoints/predictions and deliver a compact experiment report."""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from round18_joint import ROOT,load,save,sha,idxsha
from round23_finetune import state_sha
from train import read_data

EXPERIMENTS={23:('round23-finetune','round23_finetune.py','原始21列'),
    24:('round24-finetune-features','round24_finetune_features.py','原始列＋教师与辅助特征，共38列'),
    25:('round25-finetune-routes','round25_finetune_routes.py','再加入六组航线目标编码，共44列')}


def main():
    tr,_,_,_,y=read_data(ROOT/'data');y=y.to_numpy()
    oof=pd.read_csv(ROOT/'outputs-round13/oof.csv')
    assert oof.id.equals(tr.id) and np.array_equal(oof.satisfaction,y)
    base=oof.prediction.to_numpy();audit=[];table=[];train_table=[]
    for n,(d,script,name) in EXPERIMENTS.items():
        folder=ROOT/'experiments'/d;p=load(folder/'protocol.json');f=folder/'fold-1'
        s=load(ROOT/'analyses'/d/'fold-1-scores.json');r=load(f/'training.json');manifest=load(f/'split.json')
        assert sha(ROOT/script)==p['script_sha256']
        assert all(sha(ROOT/file)==h for file,h in p['protected_sha256'].items())
        assert all(sha(ROOT/'data'/file)==h for file,h in p['input_sha256'].items())
        assert all(sha(ROOT/file)==h for file,h in p.get('source_sha256',{}).items())
        assert r['successful'] and r['final_weights_changed'] and not r['outer_labels_used_for_training_or_checkpoint_selection']
        ck=torch.load(f/'weights/best.ckpt',map_location='cpu',weights_only=True)
        assert sha(f/'weights/best.ckpt')==r['checkpoint_sha256']
        assert state_sha(ck['state_dict'])==r['final_state_sha256']
        selected=[z for z in r['validation_events'] if z['state_sha256']==r['final_state_sha256']]
        assert len(selected)==1 and abs(selected[0]['auc']-r['best_inner_auc'])<1e-12
        count=r['validation_events'][1]['batch_count'];epochs=r['batch_count']//count
        train_table.append(f"| {n} | {r['validation_events'][0]['auc']:.8f} | {r['best_inner_auc']:.8f} | {selected[0]['batch_count']//count} / {epochs} | {r['seconds']:.1f}秒 |")
        for row in s['results']:
            k=row['context_rows'];mode=row['mode'];path=f/f'{mode}-{k}.npz';resource=load(f/f'{mode}-{k}-resources.json')
            assert resource['successful'] and sha(path)==resource['prediction_sha256']
            with np.load(path) as z:b=z['val_idx'];a=z['train_idx'];pred=z['val_pred']
            assert idxsha(b)==manifest['outer_val_idx_sha256'] and idxsha(a)==manifest['context_idx_sha256'][str(k)]
            assert len(pred)==len(b)==manifest['outer_validation_rows'] and len(a)==k
            assert not np.intersect1d(a,b).size and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
            assert abs(roc_auc_score(y[b],pred)-row['carrier_auc'])<1e-12
            old=roc_auc_score(y[b],base[b]);assert abs(old-row['baseline_auc'])<1e-12
            for blend in row['blends']:
                w=blend['weight'];mix=expit((1-w)*logit(np.clip(base[b],1e-6,1-1e-6))+w*logit(np.clip(pred,1e-6,1-1e-6)))
                assert abs(roc_auc_score(y[b],mix)-blend['auc'])<1e-12
                assert abs(blend['auc']-old-blend['delta'])<1e-12
            best=max(row['blends'],key=lambda z:z['delta'])
            table.append(f"| {n}：{name} | {'微调' if mode=='finetuned' else '预训练'} | {k:,} | {row['carrier_auc']:.8f} | {best['weight']:.0%} | {best['delta']:+.8f} |")
        eligible=any(z['delta']>=p['pilot_continue_delta'] for row in s['results'] if row['mode']=='finetuned' for z in row['blends'])
        assert eligible==s['continue_full_five_folds']
        audit.append(dict(round=n,protocol_sha256=sha(folder/'protocol.json'),score_sha256=sha(ROOT/'analyses'/d/'fold-1-scores.json'),
            checkpoint_sha256=r['checkpoint_sha256'],checkpoint_parameters_verified=True,prediction_metrics_reconstructed=True,
            protected_state_unchanged=True,eligible_pilot=eligible,completed_epochs=epochs,selected_epoch=selected[0]['batch_count']//count))
    independence=load(ROOT/'analyses/round25-finetune-routes/label-independence.json')
    assert independence['successful'] and independence['features_identical_after_label_perturbation']
    assert independence['source_sha256']==sha(ROOT/'round25_finetune_routes.py')
    official=load(ROOT/'evidence/round25-official-snapshot.json');res=official['results']
    assert all(v['successful'] for v in res.values())
    submissions=res['submissions']['data'];comp=res['competition']['data'][0]
    top=res['leaderboard']['data'][0]['score'];best=max(float(z['publicScore']) for z in submissions)
    eligible=[z['round'] for z in audit if z['eligible_pilot']]
    validation=dict(successful=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),experiments=audit,
        protected_best_sha256=sha(ROOT/'outputs-round13/submission.csv'),official_snapshot_sha256=sha(ROOT/'evidence/round25-official-snapshot.json'),
        official_submission_count=len(submissions),best_public_score=best,eligible_pilot_rounds=eligible,
        additional_full_folds_required=bool(eligible),route_label_independence_verified=True,
        new_persistent_disk_offload_files=sum(len(list((ROOT/'experiments'/d).rglob('*.mmap'))) for d,_,_ in EXPERIMENTS.values()))
    save(ROOT/'evidence/round23-25-final-validation.json',validation)
    text=f"""# Round 23–25：TabICL 实际微调与输入信息对照

三组正式微调及对应外层预测已经完成。当前官方最佳为 **{best:.5f}**；官方查询时间 {official['checked_at_utc']}，排名 **{comp['userRank']}/{comp['teamCount']}**，榜首 {top}。提交记录仍为 {len(submissions)} 条。

## 外层单折结果

比较基准是原封不动的 Round13 融合，本折 AUC {s['results'][0]['baseline_auc']:.8f}。包含既有十折辅助网络。以下“融合增量”是预先固定的 5%、10%、20% logit 权重中最大值，属于历史单折筛选结果。

| 输入 | 权重状态 | 预测上下文行数 | 新模型 AUC | 最佳固定权重 | 融合增量 |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

扩展门槛在实验前固定为单折增量 ≥0.000010；扩展后还必须完成全部五折，平均增量 ≥0.000010 且各折为正。符合单折门槛的轮次：{eligible or '无'}。{'仍需完成全五折才能判断是否提交。' if eligible else '三轮均结束为不提交，最佳提交文件保持不变。'}

## 实际训练和权重选择

每组梯度训练 549,708 行，另留 10,000 行做内部验证；内部选择使用固定 10,000 行上下文，外层验证 139,927 行。每个已完成 epoch 都遍历全部梯度训练行，随机分成最多 2,048 行的上下文与查询块。学习率 1e-5，最多四个 epoch。

| 轮次 | 微调前内部 AUC | 选中内部 AUC | 选中 epoch / 已完成 epoch | 训练耗时 |
|---|---:|---:|---:|---:|
{chr(10).join(train_table)}

已核对实际参数摘要，三组权重确实改变；保存的参数与被选中的验证事件一致。额外的 128 行运行探针只检查反向传播与资源，不用于候选选择。扩大的 20 万行上下文仅在 Round23 作为预先固定的对照完成；其耗时明显增加，未带来融合增量。

## 结果解释和验证边界

原始特征版在 5 万上下文微调后单模型 AUC 提升约 0.000388，而在 20 万上下文下降约 0.000536。内部验证的提升没有稳定转移到更大的预测上下文；训练块与预测上下文的差异是一种可能原因，尚未被单独隔离证明。当前新模型单独表现与最佳融合存在差距，固定权重融合不能凭内部验证提高就判断有效。

Round24 的教师预测只来自公开原始数据训练的本地模型，辅助重建只使用特征值。Round25 的航线目标编码只拟合梯度训练部分，梯度行采用五折内部交叉编码；翻转全部内部验证和外层验证标签后，699,635 行的六列编码完全一致。两组外层标签均未用于权重选择或编码拟合。

历史标签与五折划分曾用于研究，这些对照属于适应性筛选，没有新的原封不动留出集。未再次单独计算旧 H1/H2 得分。预测上下文为抽取的固定子集；梯度训练遍历训练部分，不代表预测使用全量训练上下文。显存统计来自 PyTorch，Windows WDDM 可使用共享内存。单折结果不能证明官方榜单会提高。

依赖轮包核对了官方 PyPI SHA256，pip check 通过；已有 Torch、PyTabKit 等训练库版本未变。使用已有 BSD-3-Clause TabICL 权重，没有下载或运行 TabPFN 3.5 权重。

方法来源：[TabICL 官方微调接口](https://github.com/soda-inria/tabicl)、[官方模型及权重许可](https://huggingface.co/jingang/TabICL)。本地完成校验记录：`evidence/round23-25-final-validation.json`；完整配方和预测在 `experiments/round23-finetune`、`experiments/round24-finetune-features`、`experiments/round25-finetune-routes`。
"""
    (ROOT/'ROUND23_25_RESULTS.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(successful=True,report=str(ROOT/'ROUND23_25_RESULTS.md'),eligible_pilot_rounds=eligible)))


if __name__=='__main__':main()
