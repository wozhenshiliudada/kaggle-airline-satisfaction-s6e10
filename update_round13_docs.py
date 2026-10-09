"""Record the scored round13 outcome without changing model artifacts."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected one documentation match: ' + old[:80])
    return text.replace(old, new, 1)


def main():
    official = load('evidence/round13-after-submit-official-results.json')
    validation = load('outputs-round13/validation.json')
    best = load('best-submission.json')
    metrics = load('outputs-round13/metrics.json')
    if not validation.get('new_official_best') or best['path'] != 'outputs-round13/submission.csv':
        raise ValueError('Documentation expects a verified new official best')
    new = next(row for row in official['submissions'] if row['ref'] == validation['submission_ref'])
    before = load('evidence/round13-before-upload-official-results.json')
    rank = official['competition'][0]
    pre_rank = before['competition'][0]
    local = datetime.fromisoformat(official['checked_at_utc']).astimezone(timezone(timedelta(hours=8)))
    submitted = datetime.fromisoformat(new['date']).replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8)))
    stamp = local.strftime('%Y-%m-%d %H:%M')
    ref, score = new['ref'], new['publicScore']
    count = len(official['submissions'])
    if count != 12 or new['status'] != 'SubmissionStatus.COMPLETE':
        raise ValueError('Unexpected official submission count or status')

    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    readme = replace(readme,
        '已完成多轮建模、十一次官方 API 提交和查分。目前最佳公开 AUC 为 **0.96131**，提交 `56853320` 已 Complete。北京时间 **2026-10-05 21:31** 查询时排名为 **149 / 792 队**。本次提交前 21:28 为 153/792。此前最佳 `56851874` 为 0.96129，其文件作为对照完整保留。',
        f'已完成多轮建模、十二次官方 API 提交和查分。目前最佳公开 AUC 为 **{score}**，提交 `{ref}` 已 Complete。北京时间 **{stamp}** 查询时排名为 **{rank["userRank"]} / {rank["teamCount"]} 队**。本次提交前为 {pre_rank["userRank"]}/{pre_rank["teamCount"]}、公开最佳 0.96131。当前最佳为 [outputs-round13/submission.csv](outputs-round13/submission.csv)，实验与复现见 [十折优化记录](TENFOLD_OPTIMIZATION.md)。此前最佳 `56853320` 及全部历史文件保留。')
    readme = replace(readme,
        '- 比赛每天最多提交 10 次，最终最多选择 2 个提交；共已提交 10 次，均评分成功。本轮只新增一次正式提交，最新回执显示今天剩 6 次。未手动选择最终提交时，Kaggle 会自动选择最佳评分候选。',
        '- 账号累计十二次提交，均已评分 Complete。第十二轮失败实验没有提交，第十三轮只新增一次正式提交；同文件再调用会跳过上传。最终提交选择和当日剩余额度以 Kaggle 当前页面为准。')
    readme += f'\n## 十折训练覆盖率与最新正式评分\n\n第十二轮一致性摘要、舱位专家、先验校正、原始老师接入树和不确定性叠加均没有通过筛选，详见 [失败实验记录](COHERENCE_OPTIMIZATION.md)。随后保持辅助 RealMLP 的特征及超参数，将外层五折改为十折：每次训练使用 90% 样本，测试预测由 80 个成员平均。\n\n替换旧辅助网络后，组合 OOF 从 **0.96168931 提高到 0.96173256**，五个旧评价区块及留块选择均提高。正式提交 **{ref}** 的公开 AUC 为 **{score}**，高于此前 0.96131；{stamp} 排名 **{rank["userRank"]}/{rank["teamCount"]}**。只新增一次提交，输入/代码/检查点和全部测试 ID 已核对，旧最佳保留。组合仍包含历史三/五折成员，不能视为全流程统一十折或嵌套验证。详见 [实验与复现](TENFOLD_OPTIMIZATION.md)、[正式预测](outputs-round13/submission.csv)、[验证](outputs-round13/validation.json) 和 [官方原始结果](evidence/round13-after-submit-official-results.json)。\n'
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')

    results = (ROOT / 'RESULTS.md').read_text(encoding='utf-8')
    results = replace(results,
        '已完成官方 SDK OAuth 接入、比赛数据下载、多轮建模、预测文件检查、十一次提交和官方评分查询。本轮新增辅助不确定性特征、联合神经网络和出行专家对照，最终少量融合出行专家，新组合已评分成功，成为当前最佳。组队已按用户要求暂时搁置，实际成员状态仍未确认。',
        '已完成官方 SDK OAuth 接入、比赛数据下载、多轮建模、预测文件检查、十二次提交和官方评分查询。本轮多个特征方向筛选失败后，将辅助神经网络改为十折训练；替换旧网络的组合已正式评分，成为当前最佳。组队继续按用户要求暂时搁置，实际成员状态仍未确认。')
    results = replace(results,
        '| 90% 原最佳 + 10% 商务/个人出行专家 | **0.961689** | **0.96131** | 56853320 | Complete |',
        f'| 90% 原最佳 + 10% 商务/个人出行专家 | **0.961689** | **0.96131** | 56853320 | Complete |\n| 十折辅助 RealMLP 替换旧网络 | **0.961733** | **{score}** | {ref} | Complete |')
    results = replace(results,
        '最新官方查询：北京时间 **2026-10-05 21:31**，公开最佳 **0.96131**，排名 **149 / 792 队**。本次提交前 21:28 为 153/792。此前 20:24 为 153/787、公开最佳 0.96129；19:34 为 154/785，当时公开最佳为 0.96128。',
        f'最新官方查询：北京时间 **{stamp}**，公开最佳 **{score}**，排名 **{rank["userRank"]} / {rank["teamCount"]} 队**。本次提交前为 {pre_rank["userRank"]}/{pre_rank["teamCount"]}、公开最佳 0.96131。账号累计十二次提交，本轮只新增一次。此前 2026-10-05 21:31 为 149/792、0.96131；20:24 为 153/787、0.96129。名次是查询时的榜单快照。')
    results = replace(results, '以及 Kaggle 十一次评分成功状态。', '以及 Kaggle 十二次评分成功状态。')
    results = replace(results,
        '- [当前最佳预测 CSV](outputs-round11/submission.csv)：299,844 行，列为 `id,satisfaction`。',
        '- [当前最佳预测 CSV](outputs-round13/submission.csv)：299,844 行，列为 `id,satisfaction`。')
    results = replace(results,
        '- [最新实验与正式评分](DIVERGENT_OPTIMIZATION.md)、[本次预测检查](outputs-round11/validation.json) 和 [最新官方原始记录](evidence/round11-after-submit-official-results.json)。',
        '- [最新实验与正式评分](TENFOLD_OPTIMIZATION.md)、[本次预测检查](outputs-round13/validation.json) 和 [最新官方原始记录](evidence/round13-after-submit-official-results.json)。')
    results += f'\n## 第十二、十三轮：特征失败与十折改善\n\n第十二轮固定候选未通过完整筛选，没有正式提交；原始/校正舱位专家和新增原始老师树在首折停止，一致性摘要完成五折但增量不稳定。见 [完整记录](COHERENCE_OPTIMIZATION.md)。\n\n第十三轮只改变辅助神经网络外层折数，保留 116 列、8 成员和 3 epoch。训练样本占比由 80% 增至 90%，测试集成从 40 个成员增至 80 个。完整固定四候选比较中，替换旧网络的 OOF 为 **0.9617325627**，增加 **0.0000432485**，五块均提高；留块选择五次均选中同一候选。完整组合仍混合历史三/五折成员，筛选具有适应性偏差。\n\n北京时间 **{submitted:%Y-%m-%d %H:%M:%S}**，提交 **{ref}**；状态 Complete，公开 **0.96131 → {score}**。{stamp} 排名 **{rank["userRank"]}/{rank["teamCount"]}**，此前为 {pre_rank["userRank"]}/{pre_rank["teamCount"]}。候选成为新官方最佳，旧文件和指纹完整保留；没有第二次上传。详细公式、逐块增量和验证边界见 [十折优化](TENFOLD_OPTIMIZATION.md)。\n'
    (ROOT / 'RESULTS.md').write_text(results, encoding='utf-8')

    sources = (ROOT / 'SOURCES.md').read_text(encoding='utf-8')
    sources += '\n第十二、十三轮继续由本项目独立实现：先测试无标签问卷一致性摘要、舱位专家及既有原始 XGBoost 在树中的接入，再将辅助 RealMLP 外层改为十折。没有新增外部数据、第三方预测、预训练权重或软件依赖；继续使用已核验的公开原始数据、现有 LightGBM/XGBoost 和 Apache-2.0 的 PyTabKit。十折改变训练覆盖率和测试集成成员数，采用与已有开源示例相通的交叉验证思路，未复制没有明确许可的 GitHub 代码。实验结果分别见 `COHERENCE_OPTIMIZATION.md` 和 `TENFOLD_OPTIMIZATION.md`。\n'
    (ROOT / 'SOURCES.md').write_text(sources, encoding='utf-8')

    record = (ROOT / 'TENFOLD_OPTIMIZATION.md').read_text(encoding='utf-8')
    record += f'\n## 正式提交与评分\n\n北京时间 **{submitted:%Y-%m-%d %H:%M:%S}** 完成提交，编号 **{ref}**，状态 Complete，公开 AUC **{score}**，较旧最佳 **0.96131** 提高 **0.00001**。北京时间 **{stamp}** 查询排名 **{rank["userRank"]}/{rank["teamCount"]}**，提交前为 **{pre_rank["userRank"]}/{pre_rank["teamCount"]}**。该榜单可能随其他队伍提交变化。\n\n提交前后官方记录从 11 条增至 12 条，仅新增这一次。`best-submission.json` 已指向第十三轮，内嵌此前最佳及指纹；第十一轮预测完整保留。上传完成后重复调用会根据相同 SHA256 跳过上传。TLS 查分错误期间保留旧快照，并未重新上传；成功查询后再核对正式评分。CLI 本次回执没有返回当日剩余额度，不据本地跨日推测 Kaggle 额度重置。\n\n公开改善小于本地 OOF 改善，只能确认本次显示精度下的增量；未宣称训练覆盖率或成员数量中哪个变化单独导致收益。原始结果在 `evidence/round13-before-upload-official-results.json` 和 `evidence/round13-after-submit-official-results.json`，收据在 `evidence/round13-submission-receipt.json`。\n'
    (ROOT / 'TENFOLD_OPTIMIZATION.md').write_text(record, encoding='utf-8')

    team = load('team-request.json')
    names = ('prepared_prediction', 'prepared_prediction_sha256', 'prepared_prediction_note')
    history = team.setdefault('prepared_prediction_history', [])
    history.append({k: team[k] for k in names})
    team.update(prepared_prediction='outputs-round13/submission.csv',
                prepared_prediction_sha256=metrics['submission_sha256'],
                prepared_prediction_note=f'Submitted during user-deferred teaming: {ref}, Complete, public AUC {score}; new official best. Team membership remains unverified.')
    (ROOT / 'team-request.json').write_text(json.dumps(team, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(updated=['README.md','RESULTS.md','SOURCES.md','TENFOLD_OPTIMIZATION.md','team-request.json'],
        official_submission=ref, public_score=score, checked_at_beijing=stamp), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
