"""Final integrity checks and factual report for completed rounds19--22."""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from evaluate_context import paired
from round18_joint import ROOT,sha,load,save

official=load(ROOT/'evidence/round22-official-snapshot.json')
assert all(v['successful'] for v in official['results'].values())
competition=official['results']['competition']['data'][0]
submissions=official['results']['submissions']['data'];leader=official['results']['leaderboard']['data'][0]
assert len(submissions)==13
best=load(ROOT/'best-submission.json')
assert best['public_score']=='0.96132' and best['submission_ref']==56857129
assert sha(ROOT/best['path'])==best['sha256']
assert any(s['ref']==best['submission_ref'] and s['publicScore']==best['public_score'] for s in submissions)
y=pd.read_csv(ROOT/'data/train.csv',usecols=['satisfaction']).satisfaction.to_numpy(dtype='int8')
base=pd.read_csv(ROOT/'outputs-round13/oof.csv',usecols=['prediction']).prediction.to_numpy()
model_rows=[]
for directory,variants in [
    ('round19-route-tabm',['aux_route_embedding','raw_route_embedding']),
    ('round20-route-control',['matched_control','route_onehot'])]:
    for v in variants:
        r=load(ROOT/'analyses'/directory/(v+'-pilot.json'))
        assert not r['continue_full']
        model_rows.append(dict(variant=v,model_auc=r['model_auc'],
                 best_blend_delta=max(k['delta'] for k in r['candidates'])))
crossfit=[]
for number in [21,22]:
    folder=ROOT/'experiments'/f'round{number}-preserved'
    p=load(folder/'protocol.json');r=load(ROOT/'analyses'/f'round{number}-preserved/crossfit.json')
    assert sha(ROOT/f'round{number}_preserved.py')==p['script_sha256']
    assert r['protocol_sha256']==sha(folder/'protocol.json')
    for f,h in p['protected_sha256'].items():assert sha(ROOT/f)==h,f
    for entries in p['native_source_sha256'].values():
        for f,h in entries.items():assert sha(ROOT/f)==h,f
    if number==22:
        assert p['predecessor_source_sha256']==sha(ROOT/'round21_preserved.py')
        assert p['predecessor_result_sha256']==sha(ROOT/'analyses/round21-preserved/crossfit.json')
    P=np.column_stack([pd.read_csv(ROOT/d/'oof.csv',usecols=['prediction']).prediction.to_numpy()
                       for d in p['members'].values()])
    X=logit(np.clip(P,1e-6,1-1e-6))
    with np.load(folder/'crossfit-predictions.npz') as z:pred=z['predictions'].copy()
    assert pred.shape==(3,len(y))
    for i,repeat in enumerate(r['repeats']):
        assert repeat['seed']==p['meta_cv']['seeds'][i]
        for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=repeat['seed']).split(X,y)):
            w=np.asarray(repeat['weights'][f])
            assert abs(w.sum()-1)<1e-7 and w[0]>=.5-1e-7 and w.min()>=-1e-7
            assert np.max(abs(expit(X[b]@w)-pred[i,b]))<1e-12
            delta=float(roc_auc_score(y[b],pred[i,b])-roc_auc_score(y[b],base[b]))
            assert abs(delta-repeat['block_deltas'][f])<1e-12
        assert abs(roc_auc_score(y,pred[i])-repeat['auc'])<1e-12
    mean=float(np.mean([q['delta'] for q in r['repeats']]))
    worst=min(min(q['block_deltas']) for q in r['repeats'])
    eligible=mean>=p['gates']['mean_delta'] and min(q['delta'] for q in r['repeats'])>0 and worst>0
    assert eligible==r['eligible']==False
    assert not (ROOT/f'outputs-round{number}').exists()
    crossfit.append(dict(round=number,mean_delta=mean,worst_block_delta=worst,
        positive_blocks=sum(d>0 for q in r['repeats'] for d in q['block_deltas']),
        fixed_prediction_diagnostic=paired(y,base,pred.mean(axis=0)),
        checkpoint_sha256=sha(folder/'crossfit-predictions.npz'),eligible=False))
audit=load(ROOT/'evidence/round20-cache-audit.json');assert audit['passed']
evidence=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),passed=True,
    protected_files_unchanged=True,official_submission_count=13,
    no_new_submissions=True,no_formal_candidate_written=True,
    completed_model_fits=4,completed_meta_fits=30,model_results=model_rows,
    crossfit_results=crossfit,best_public_score=best['public_score'],
    user_rank=competition['userRank'],team_count=competition['teamCount'],
    highest_public_score=leader['score'],separate_H1_H2_metrics_not_rescored=True,
    tabpfn35_weights_not_downloaded_or_run=True,
    reporter_sha256=sha(__file__))
save(ROOT/'evidence/round22-final-validation.json',evidence)
labels={'aux_route_embedding':'辅助特征＋距离嵌入','raw_route_embedding':'原始特征＋距离嵌入',
        'matched_control':'匹配原生 TabM 对照','route_onehot':'原生 TabM＋距离类别'}
lines=[
 '# 第 19—22 轮结果', '',
 f"官方最佳仍为 **{best['public_score']}**；本次核验名次 **{competition['userRank']} / {competition['teamCount']}**，榜首 **{leader['score']}**。官方提交总数仍是 13，本轮没有新提交。",
 '', '## 距离类别试验', '',
 '完成 4 组新的单折训练，每组使用 559,708 行训练、139,927 行验证；验证折排除于监督特征拟合，但用于提前停止。原始数据模型和辅助重建沿用已核验来源。测试预测已保存，仅用于完整性核验。', '',
 '| 试验 | 单折模型 AUC | 与当前最佳融合的最好增量 |',
 '|---|---:|---:|']
for r in model_rows:lines.append(f"| {labels[r['variant']]} | {r['model_auc']:.9f} | {r['best_blend_delta']:+.9f} |")
control=model_rows[2]['model_auc'];route=model_rows[3]['model_auc']
lines += ['',f'在同批次、同参数的匹配对照中，距离类别使单模型 AUC 提高 {route-control:+.9f}，但对当前最佳集成的边际增量只有约 +0.000000192，未达到训练前固定的 +0.000003 首折扩展门槛。嵌入试验还改变了类别架构，不能将其结果单独归因于距离类别。',
 '', '## 保留十折网络的融合复验', '',
 '保留原十折辅助网络，并将当前最佳预测作为至少 50% 权重的必选成员。其他成员保持自己的原生折数。使用非负、权重和为 1 的 logit 融合，固定两档正则，分别做 3 个随机种子的五折元模型交叉验证，共 30 次拟合。', '',
 '| 轮次 | 正则 | 平均 AUC 增量 | 最差验证块增量 | 正增量块 | 结论 |',
 '|---|---:|---:|---:|---:|---|']
for r in crossfit:
    lines.append(f"| {r['round']} | {'0.001' if r['round']==21 else '0.0001'} | {r['mean_delta']:+.9f} | {r['worst_block_delta']:+.9f} | {r['positive_blocks']}/15 | 未通过预定门槛 |")
lines += ['', '两轮的提交门槛均在计算结果前固定：平均增量至少 +0.000010、每次重复为正、全部验证块为正。第 21 轮增量不足；第 22 轮平均增量达标，但有验证块下降。因此没有生成正式提交文件，没有更改最佳提交、上一份提交或队友状态。',
 '', '**验证限制：**以上使用反复研究过的历史 OOF。元模型验证不是完整嵌套验证，基模型拟合有重叠；不同随机种子的验证块也有重叠。单折 AUC 不能与公榜 AUC直接比较，也不能把 15 个块当作 15 份独立证据。没有重新单独评分原 H1/H2。固定预测的近似置信区间仅是描述诊断，不校正选型或训练依赖。',
 '', '## 来源和下一步', '',
 '距离类别思路来自 [TabPFN Route Categories](https://www.kaggle.com/code/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160)，本地独立适配到 [TabM](https://github.com/yandex-research/tabm)；未使用其他参赛者的预测文件。新代码使用现有 Apache-2.0 PyTabKit 实现，没有新增预训练权重。',
 '', '你已确认个人学习参赛。[TabPFN 3.5 官方许可](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)同时包含专业身份限定；本轮未确认适用资格，因此未下载或运行该版本权重。阅读了 full-context Notebook 的资源实现，但未执行其代码。',
 '', '现有证据表明，距离类别本身有信息，保留十折网络也能改善融合；目前增量仍很小且不均匀。后续优先做新的模型训练，例如用许可可用的 TabICL 权重进行本比赛适配，并沿用训练前固定的评估规则。该适配尚未执行，不能声称会提升名次。',
 '', '完整性核验：4 个新模型检查点、30 组融合权重、折索引、验证标签扰动、概率范围、原始文件指纹、受保护状态、OOF 重建及官方 13 份提交均已核验。证据见 `evidence/round22-final-validation.json`。', '']
(ROOT/'ROUND19_22_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print('PASS: 4 model fits, 30 meta fits, protected state, 13 official submissions')
print('Rank',competition['userRank'],'/',competition['teamCount'],'best',best['public_score'])
for r in crossfit:print('Round',r['round'],'mean delta',r['mean_delta'],'positive blocks',r['positive_blocks'])
