"""Finalize user-facing research records after evaluation and any official scoring."""
from datetime import datetime,timedelta,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,text):(ROOT/name).write_text(text,encoding='utf-8')
def first_paragraph(text,prefix,replacement):
    paragraphs=text.split('\n\n');matched=[i for i,p in enumerate(paragraphs) if p.startswith(prefix)]
    if len(matched)!=1:raise ValueError('Expected one paragraph: '+prefix)
    paragraphs[matched[0]]=replacement
    return '\n\n'.join(paragraphs)


def main():
    m=load('analyses/round15-comparison.json');native=load('experiments/round15-lgb-10fold/metrics.json')
    best=load('best-submission.json');selected=m['selected_blend'];uploaded=False
    if selected:
        v=load('outputs-round15/validation.json')
        if not v['passed'] or not v['uploaded'] or v['official_status']!='COMPLETE':
            raise ValueError('Eligible candidate has not finished official scoring')
        uploaded=True;official=load('evidence/round15-after-submit-official-results.json')
        before=load('evidence/round15-before-upload-official-results.json')
        row=next(x for x in official['submissions'] if x['ref']==v['submission_ref'])
        outcome=f'正式提交 **{row["ref"]}**，已 Complete，公开 AUC **{row["publicScore"]}**。'
        if v['new_official_best']:outcome+='超过原最佳 0.96132，已更新最佳指针并保留旧文件。'
        else:outcome+='没有超过原最佳 0.96132，最佳文件继续保留第十三轮。'
        outcome+=f'只新增一次正式提交；上传后同文件重复调用已跳过。'
    else:
        official=load('evidence/round15-final-official-results.json');before=load('evidence/round14-before-training-official-results.json')
        outcome='第十五轮没有候选通过完整筛选，未上传；当前最佳继续保留第十三轮的 0.96132。'
    count=len(official['submissions']);rank=official['competition'][0];pr=before['competition'][0]
    local=datetime.fromisoformat(official['checked_at_utc']).astimezone(timezone(timedelta(hours=8)))
    stamp=local.strftime('%Y-%m-%d %H:%M')
    if hashlib.sha256((ROOT/best['path']).read_bytes()).hexdigest()!=best['sha256']:
        raise ValueError('Best prediction fingerprint mismatch')
    table='\n'.join(f'| {r["candidate"]} | {r["auc"]:.10f} | {r["delta"]:+.10f} | {"是" if r["all_blocks_positive"] else "否"} |' for r in m['coarse_scan'])
    record=(ROOT/'RESEARCH_OPTIMIZATION.md').read_text(encoding='utf-8')
    record+=f'\n## 第十五轮完整十折与最终状态\n\n新树原生十折 OOF **{native["selected_blend"]["auc"]:.10f}**，旧五折树为 **0.9611837935**；各折仍 292 列、原分箱上限 255，参数不变。复用首折后，其余九折及完整输出耗时 {native["seconds"]:.2f} 秒，首折单独约 114 秒；不把缓存日志的累计时间当作每折独立耗时相加。\n\n| 固定候选 | 组合 OOF AUC | 相对最佳增量 | 五块均提高 |\n|---|---:|---:|---|\n{table}\n'
    if selected:
        deltas=' / '.join(f'{x:+.10f}' for x in selected['fold_deltas'])
        record+=f'\n选择 **{selected["candidate"]}**，五个评价块增量为 **{deltas}**。留块 AUC **{m["selection_stability"]["auc"]:.10f}**，五个留出块均提高；选择与训练仍复用历史 OOF，不是完整嵌套验证。\n'
        record+=f'\n提交文件 SHA256 `{v["submission_sha256"]}`，299,844 行。已复算 OOF、验证完整测试 ID/概率/输入/代码与十个原生检查点，并核对原始来源预测及测试平均。固定预测的近似配对区间及选择偏差限制保存在指标中。\n'
    else:
        record+='\n没有达到全局 +0.00001、五块和留块检查同时通过的候选，保留基线；没有对失败候选微调权重或拼接。\n'
    record+=f'\n{outcome}\n\n北京时间 **{stamp}** 官方查询，最佳公开 **{best["public_score"]}**，排名 **{rank["userRank"]}/{rank["teamCount"]}**。对照查询为 **{pr["userRank"]}/{pr["teamCount"]}**；排名为当时快照，可能随其他提交变化。账号共 {count} 条提交，均需以原始官方状态为准。本轮规则、源码审计和失败保留在本报告，组合仍混合三/五/十折成员。\n'
    write('RESEARCH_OPTIMIZATION.md',record)

    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    readme=first_paragraph(readme,'已完成多轮建模、',f'已完成多轮建模、{count} 次官方 API 提交和查分。目前最佳公开 AUC 为 **{best["public_score"]}**，提交 `{best["submission_ref"]}` 已 Complete。北京时间 **{stamp}** 查询排名 **{rank["userRank"]} / {rank["teamCount"]} 队**。当前最佳为 [预测 CSV]({best["path"]})；本轮 GitHub/论文来源、失败实验、十折树和复现见 [研究优化记录](RESEARCH_OPTIMIZATION.md)。全部历史最佳和文件指纹保留。')
    marker='- 账号累计十二次提交，均已评分 Complete。第十二轮失败实验没有提交，第十三轮只新增一次正式提交；同文件再调用会跳过上传。最终提交选择和当日剩余额度以 Kaggle 当前页面为准。'
    if readme.count(marker)!=1:raise ValueError('README submission-count paragraph changed')
    readme=readme.replace(marker,f'- 账号累计 {count} 次提交，均已评分 Complete。本轮新增 {1 if uploaded else 0} 次正式提交；同文件再调用会跳过上传。最终提交选择和当日剩余额度以 Kaggle 当前页面为准。',1)
    readme+=f'\n## GitHub 与相关研究后的最新优化\n\n阅读 TabM、数值嵌入、航空满意度特征筛选研究和多个同题仓库，独立测试批次多样性、航程分箱、排序尺度及十折树覆盖率。前三条路线未通过，停止或保留失败缓存；十折树完成全量评估。{outcome}\n\n截至 {stamp}，最佳 **{best["public_score"]}**，排名 **{rank["userRank"]}/{rank["teamCount"]}**。详见 [实验、来源和复现](RESEARCH_OPTIMIZATION.md)、[完整十折对照](analyses/round15-comparison.json)。没有下载别人的预测或预训练权重，没有新增外部数据。\n'
    write('README.md',readme)

    results=(ROOT/'RESULTS.md').read_text(encoding='utf-8')
    results=first_paragraph(results,'已完成官方 SDK OAuth 接入、',f'已完成官方 SDK OAuth 接入、比赛数据下载、多轮建模、预测文件检查、{count} 次提交和官方评分查询。本轮吸收 GitHub 与相关论文经验，完成两阶段固定对照：第十四轮无可靠候选，第十五轮十折树完成评估。{outcome}组队继续按用户要求暂时搁置。')
    results=first_paragraph(results,'最新官方查询：',f'最新官方查询：北京时间 **{stamp}**，公开最佳 **{best["public_score"]}**，排名 **{rank["userRank"]} / {rank["teamCount"]} 队**。账号累计 {count} 次提交，本轮新增 {1 if uploaded else 0} 次。此前 2026-10-06 00:44 为 147/807、0.96132；2026-10-05 21:31 为 149/792、0.96131。名次为查询时快照。')
    if uploaded:
        marker='| 十折辅助 RealMLP 替换旧网络 | **0.961733** | **0.96132** | 56857129 | Complete |'
        if results.count(marker)!=1:raise ValueError('RESULTS round13 table changed')
        results=results.replace(marker,marker+f'\n| 十折辅助 LightGBM：{selected["candidate"]} | **{selected["auc"]:.6f}** | **{row["publicScore"]}** | {row["ref"]} | Complete |',1)
        results=results.replace('以及 Kaggle 十二次评分成功状态。',f'以及 Kaggle {count} 次评分成功状态。',1)
        results=results.replace('- [当前最佳预测 CSV](outputs-round13/submission.csv)：299,844 行，列为 `id,satisfaction`。',f'- [当前最佳预测 CSV]({best["path"]})：299,844 行，列为 `id,satisfaction`。',1)
        results=results.replace('- [最新实验与正式评分](TENFOLD_OPTIMIZATION.md)、[本次预测检查](outputs-round13/validation.json) 和 [最新官方原始记录](evidence/round13-after-submit-official-results.json)。','- [最新实验与正式评分](RESEARCH_OPTIMIZATION.md)、[本次预测检查](outputs-round15/validation.json) 和 [最新官方原始记录](evidence/round15-after-submit-official-results.json)。',1)
    results+=f'\n## 第十四、十五轮研究优化\n\n第十四轮原计划固定十二项，TabM 首折失败停止，所以完整对照八项，没有正式候选。高分辨率航程树整体替换增量 0.00000156，但两块下降；排序融合的最好增量不足 0.000001，也未通过。\n\n另行固定第十五轮后，原参数树完成十折：每次使用 90% 训练数据，测试平均十个模型。单树 OOF **{native["selected_blend"]["auc"]:.10f}**。{outcome}\n\n最终查询 {stamp}，最佳公开 **{best["public_score"]}**，排名 **{rank["userRank"]}/{rank["teamCount"]}**。完整逐块结果、固定候选、源码审计、复现和非嵌套验证限制见 [研究优化记录](RESEARCH_OPTIMIZATION.md)。\n'
    write('RESULTS.md',results)
    if uploaded:
        team=load('team-request.json');names=('prepared_prediction','prepared_prediction_sha256','prepared_prediction_note')
        team.setdefault('prepared_prediction_history',[]).append({k:team[k] for k in names})
        team.update(prepared_prediction='outputs-round15/submission.csv',prepared_prediction_sha256=v['submission_sha256'],
            prepared_prediction_note=f'Submitted during user-deferred teaming: {row["ref"]}, Complete, public AUC {row["publicScore"]}; official-best pointer updated only on a strict public improvement. Team membership remains unverified.')
        write('team-request.json',json.dumps(team,indent=2,ensure_ascii=False))
    print(json.dumps(dict(updated=['README.md','RESULTS.md','RESEARCH_OPTIMIZATION.md'],uploaded=uploaded,best_score=best['public_score'],checked_at_beijing=stamp,rank=rank['userRank'],teams=rank['teamCount']),indent=2))


if __name__=='__main__':main()
