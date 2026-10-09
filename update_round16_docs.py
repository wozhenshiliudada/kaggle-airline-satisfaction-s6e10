"""Prepare user records only after rounds16/17 have actual terminal results.

This script is outside every frozen modeling manifest. It never trains, scores
heldout predictions, changes the best pointer, uploads, or changes teaming.

Example (run only after final official refresh and any scoring):
  python update_round16_docs.py --rounds 16 17 --official evidence/FINAL.json \
      --production-record evidence/round16-17-production-outcome.json

Optional production-record schema: {"records": [{"round": 16, "candidate":
"raw13", "production_oof_auc": ..., "production_baseline_oof_auc": ...,
"compatibility_passed": false, "production_block_deltas": [...],
"uploaded": false, "submission_path": "...", "submission_sha256": "...",
"submission_ref": null}]}. Omit production fields if no carrier reached that
stage. Official scores always come from the final raw Kaggle snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
COMPETITION = 'playground-series-s6e10'
PRIMITIVES = ('compact', 'route_auc', 'route_tuned', 'cat', 'plain_nn',
              'lgb', 'context', 'aux', 'tabm', 'travel')
INITIAL_BEST_SHA = 'b3451e0310cf0892ebd89cab80f400a5c0fe7fe3725ff93965e0499bf9225f93'
SOURCE = 'https://www.kaggle.com/code/goodpjw2008/s6e10-more-aux-carriers-lb-0-96152'


def path(name):
    value = Path(name)
    value = (ROOT / value if not value.is_absolute() else value).resolve()
    value.relative_to(ROOT.resolve())
    return value


def load(name):
    return json.loads(path(name).read_text(encoding='utf-8'))


def sha(name):
    h = hashlib.sha256()
    with path(name).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rel(name):
    return str(path(name).relative_to(ROOT)).replace('\\', '/')


def paragraph(text, prefix, replacement):
    blocks = text.split('\n\n')
    found = [i for i, block in enumerate(blocks) if block.startswith(prefix)]
    if len(found) != 1:
        raise ValueError('Expected one current-state paragraph: ' + prefix)
    blocks[found[0]] = replacement
    return '\n\n'.join(blocks)


def auc(value):
    return f'{float(value):.10f}'


def delta(value):
    return f'{float(value):+.10f}'


def inspect_round(number):
    folder = f'experiments/round{number}-locked'
    protocol = load(folder + '/protocol.json')
    if (protocol['competition'] != COMPETITION or protocol['round'] != number
            or protocol['epochs'] != (4 if number == 16 else 8)
            or protocol['active_holdout'] != ('hold1' if number == 16 else 'hold2')):
        raise ValueError('Unexpected round/epoch/holdout protocol')
    for code, expected in protocol['source_code_sha256'].items():
        if sha(code) != expected:
            raise ValueError('Frozen modeling code changed before documentation: ' + code)
    if sha(protocol['split_path']) != protocol['split_sha256']:
        raise ValueError('Frozen development/holdout split changed')
    selection_path = f'analyses/round{number}-locked/development-selection.json'
    selection = load(selection_path)
    if selection['holdout_labels_evaluated'] is not False:
        raise ValueError('Development selection claims it evaluated holdout labels')
    final_path = f'analyses/round{number}-locked/final-comparison.json'
    final = load(final_path) if path(final_path).exists() else None
    if selection['selected_candidate'] is not None and final is None:
        raise ValueError('Development candidate is still awaiting its heldout check')
    if final is not None:
        if final['development_selection_sha256'] != sha(selection_path):
            raise ValueError('Heldout result does not reference the exact locked development choice')
        if final['eligible_for_production_transfer'] != final['heldout']['passed']:
            raise ValueError('Heldout result gate mismatch')
    # Reuse the independent pre-selection integrity/exclusion audit, without
    # parsing target labels or duplicating native prediction reconstruction.
    audit_path = f'evidence/round{number}-checkpoint-audit.json'
    audit = load(audit_path)
    if (audit['passed'] is not True or audit['round'] != number
            or audit['baseline_estimator_fits'] != 33 or audit['carrier_fits'] != 9
            or audit['holdout_labels_loaded'] is not False
            or audit['holdout_metrics_computed'] is not False
            or audit['script_sha256'] != sha('audit_round16_checkpoints.py')):
        raise ValueError('Missing/changed completed checkpoint audit')
    mutable_after_upload = {'best-submission.json', 'last-submission.json'}
    for filename, expected in audit['audited_file_sha256'].items():
        if filename not in mutable_after_upload and sha(filename) != expected:
            raise ValueError('Previously audited training artifact changed: ' + filename)
    fits = audit['baseline_estimator_fits']
    return dict(number=number, protocol=protocol, selection=selection, final=final,
                selection_path=selection_path, final_path=final_path if final else None,
                native_baseline_fits=fits, checkpoint_audit_path=audit_path)


def render_round(state):
    n, p, s, f = (state[key] for key in ('number', 'protocol', 'selection', 'final'))
    rows = '\n'.join('| ' + ' | '.join((row['candidate'], auc(row['auc']), delta(row['delta']),
        ' / '.join(delta(v) for v in row['block_deltas']), '通过' if row['eligible'] else '未通过')) + ' |'
        for row in s['candidates'])
    selected = s['selected_candidate']
    out = f'\n## 第{n}轮：{p["epochs"]} epoch，预定 {p["active_holdout"].upper()}\n\n'
    out += f'三种输入均实际完成三个开发折。开发集固定基准 AUC **{auc(s["baseline"]["auc"])}**。下表的 AUC 是开发 OOF 上组合器交叉拟合结果。\n\n'
    out += '| 输入 | 开发 AUC | 相对开发基准 | 开发三块增量 | 开发门槛 |\n|---|---:|---:|---|---|\n' + rows + '\n'
    if selected is None:
        out += '\n没有候选通过开发门槛，因此没有读取本轮预定复验块的标签。\n'
    else:
        h = f['heldout']
        out += f'\n开发集锁定 **{selected["candidate"]}** 和其系数后，仅检查 **{p["active_holdout"].upper()}**。复验基准 AUC **{auc(h["baseline_auc"])}**，候选 **{auc(h["selected_auc"])}**，差值 **{delta(h["delta"])}**；+0.00001 门槛 **{"通过" if h["passed"] else "未通过"}**。复验仅决定接受或拒绝，没有用它改系数、改输入或选择其他候选。\n'
        if 'paired_fixed_prediction_approximation' in h:
            ci = h['paired_fixed_prediction_approximation']['approx_95ci']
            out += f'\n固定预测的近似配对增量区间为 [{delta(ci[0])}, {delta(ci[1])}]，没有纳入训练波动、之前适应性探索和模型选择的不确定性。\n'
    out += f'\n[开发选择]({state["selection_path"]})'
    if f is not None:
        out += f' · [复验结果]({state["final_path"]})'
    out += f' · [原生检查点隔离审计]({state["checkpoint_audit_path"]})'
    return out + '\n'


def main(args):
    if args.rounds not in ([16], [16, 17]):
        raise ValueError('Report round16 alone or round16 followed by the predeclared round17')
    states = [inspect_round(n) for n in args.rounds]
    official, before = load(args.official), load(args.before)
    complete = lambda row: str(row['status']).upper().endswith('COMPLETE')
    if not official['competition'] or not all(complete(row) for row in official['submissions']):
        raise ValueError('Refresh official results after every pending submission has been scored')
    if not str(official['competition'][0]['ref']).rstrip('/').endswith('/' + COMPETITION):
        raise ValueError('Wrong competition in final official snapshot')
    before_refs = {int(row['ref']) for row in before['submissions']}
    refs = {int(row['ref']) for row in official['submissions']}
    if not before_refs.issubset(refs):
        raise ValueError('Final snapshot lost pre-existing submissions')
    additions = refs - before_refs
    best = load('best-submission.json')
    if sha(best['path']) != best['sha256'] or sha('outputs-round13/submission.csv') != INITIAL_BEST_SHA:
        raise ValueError('Best or protected original prediction changed')
    maximum = max(float(row['publicScore']) for row in official['submissions'])
    if float(best['public_score']) != maximum:
        raise ValueError('Best pointer is not the official maximum scored submission')
    row_best = next(row for row in official['submissions'] if int(row['ref']) == int(best['submission_ref']))
    if float(row_best['publicScore']) != float(best['public_score']):
        raise ValueError('Best pointer ref/score mismatch')
    if maximum <= .96132 and (best['path'] != 'outputs-round13/submission.csv' or best['sha256'] != INITIAL_BEST_SHA):
        raise ValueError('Original best was replaced without strict public improvement')
    records = load(args.production_record)['records'] if args.production_record else []
    by_round = {row['round']: row for row in records}
    if len(by_round) != len(records) or set(by_round) - set(args.rounds):
        raise ValueError('Unexpected duplicate/unrequested production outcome')
    uploaded_refs = set()
    production = ''
    for state in states:
        n = state['number']
        record = by_round.get(n)
        eligible = state['final'] is not None and state['final']['eligible_for_production_transfer']
        if not eligible:
            if record and record.get('uploaded'):
                raise ValueError('Uploaded candidate did not pass its locked check')
            production += f'\n第{n}轮没有进入正式全量训练迁移。\n'
            continue
        if record is None:
            raise ValueError('Locked candidate needs an actual production-transfer outcome')
        if record['candidate'] != state['selection']['selected_candidate']['candidate']:
            raise ValueError('Production input differs from locked development selection')
        production += (f'\n第{n}轮生产兼容性 AUC **{auc(record["production_oof_auc"])}**，'
                       f'历史组合基准 **{auc(record["production_baseline_oof_auc"])}**。'
                       f'五个历史评价块增量：**{" / ".join(delta(v) for v in record["production_block_deltas"])}**；'
                       f'兼容性门槛 **{"通过" if record["compatibility_passed"] else "未通过"}**。'
                       '这个结果来自全训练集的历史 OOF，不是开发 AUC 或 H1/H2 复验 AUC。\n')
        if record.get('submission_path'):
            if sha(record['submission_path']) != record['submission_sha256']:
                raise ValueError('Production CSV fingerprint mismatch')
        if record.get('uploaded'):
            if not record['compatibility_passed']:
                raise ValueError('Submission did not pass production compatibility gate')
            ref = int(record['submission_ref'])
            if ref not in additions:
                raise ValueError('Claimed upload is absent from actual new official refs')
            row = next(row for row in official['submissions'] if int(row['ref']) == ref)
            uploaded_refs.add(ref)
            production += f'\n正式提交 **{ref}**，已 Complete，Kaggle 公开 AUC **{row["publicScore"]}**。\n'
        else:
            production += '\n本轮没有正式上传。\n'
    if uploaded_refs != additions:
        raise ValueError('New official submissions do not exactly match reported outcomes')
    rank = official['competition'][0]
    stamp = datetime.fromisoformat(official['checked_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    counts = states[0]['selection']['counts']
    h2 = states[-1]['final'] if states[-1]['number'] == 17 else None
    h2_status = ('H2 已用于预定第二轮的单次接受/拒绝检查。' if h2 is not None else
                 'H2 标签没有用于评价，仍保留为预定后续复验块。')
    outcome = f'本次新增 **{len(additions)}** 条正式提交。最佳公开 AUC **{best["public_score"]}**，最佳提交 **{best["submission_ref"]}**；只在公开分数严格提高时更新最佳指针。北京时间 **{stamp}** 查询排名 **{rank["userRank"]}/{rank["teamCount"]}**，名次为当时快照。'
    record = '# 新划分重训与原始输入辅助网络：第十六、十七轮\n\n' + outcome + '\n'
    record += (f'\n训练数据固定划分为 80% 开发集与各 10% 的 H1/H2，开发集 {counts["development"]:,} 行，'
               f'两个排除复验块合计 {counts["excluded_holdouts"]:,} 行。开发集内三折，每个基准折实际完成 11 次 estimator fit：'
               '6 个 LightGBM、1 个 CatBoost、3 个八成员 RealMLP、1 个十六成员 TabM；共 33 次 fit。'
               '每次 fit 仅使用开发训练子集，开发验证子集用于检查点选择，H1/H2 目标由调用者置为 NaN。'
               '历史监督 OOF 不用于这些新划分的基准训练或复验。\n')
    record += (f'\n[公开方法来源]({SOURCE})启发了三个独立实现的输入对照：raw 原始输入及数值类别副本、'
               'raw13 再增加十三个辅助服务预计值、raw30 再增加全部三十个辅助特征。固定第十六轮最多 4 epoch，'
               '若未达到预定目标则第十七轮使用相同三个输入最多 8 epoch，复用同一已重训基准。'
               '未读取第三方预测或预训练权重。\n')
    record += '\n' + h2_status + '\n'
    for state in states:
        record += render_round(state)
    record += '\n## 全量生产迁移与正式评分\n' + production + '\n' + outcome + '\n'
    record += ('\n## 验证边界\n\n所有比赛标签都参与过先前研究，因此新的排除复验块不能称为从未使用过的数据。'
               '开发组合器使用基础 OOF 做交叉拟合，底层训练折重叠，不能称为完整嵌套验证。'
               '辅助任务和类别词表使用无标签训练/测试特征，属于 transductive 特征。'
               'H1/H2 检查评估的是重新训练的固定基准加新输入网络管道；生产迁移复用历史全量基准和新五折网络，'
               '训练覆盖率和原生折构成发生改变，不能把 H1/H2 成绩描述成最终生产 CSV 的无偏验证。'
               '历史生产 OOF 兼容性门控仍具有适应性，只是额外检查。开发、复验、生产兼容性和 Kaggle 公开 AUC 分开报告。\n')
    record += f'\n[最终官方原始快照]({rel(args.official)}) · [当前最佳预测]({best["path"]})\n'
    summary = f'新增排除复验划分，完整重训开发基准，并比较三种原始输入辅助神经网络。{outcome}开发、复验和全量生产兼容性结果见 [新划分优化记录](LOCKED_OPTIMIZATION.md)。'
    readme = path('README.md').read_text(encoding='utf-8')
    readme = paragraph(readme, '已完成多轮建模、', f'已完成多轮建模、{len(refs)} 次官方 API 提交和查分。{summary}当前最佳为 [预测 CSV]({best["path"]})；全部历史文件和最佳指纹保留。')
    readme, changed = re.subn(r'^- 账号累计 .*?次提交，.*$',
        f'- 账号累计 {len(refs)} 次提交，均已评分 Complete。本次新增 {len(additions)} 次正式提交；同文件重复调用会跳过上传。最终提交选择和当日剩余额度以 Kaggle 当前页面为准。', readme, count=1, flags=re.M)
    if changed != 1:
        raise ValueError('README submission count line not found')
    results = path('RESULTS.md').read_text(encoding='utf-8')
    results = paragraph(results, '已完成官方 SDK OAuth 接入、', f'已完成官方 SDK OAuth 接入、比赛数据下载、多轮建模、预测文件检查、{len(refs)} 次提交和官方评分查询。{summary}组队继续按用户要求暂时搁置。')
    results = paragraph(results, '最新官方查询：', f'最新官方查询：北京时间 **{stamp}**，公开最佳 **{best["public_score"]}**，排名 **{rank["userRank"]} / {rank["teamCount"]} 队**。账号累计 {len(refs)} 次提交，本次新增 {len(additions)} 次。名次为查询时快照。')
    results = re.sub(r'^- \[当前最佳预测 CSV\]\([^\n]*$', f'- [当前最佳预测 CSV]({best["path"]})：299,844 行，列为 `id,satisfaction`。', results, count=1, flags=re.M)
    for state in states:
        report_row = by_round.get(state['number'])
        if report_row and report_row.get('uploaded'):
            ref = int(report_row['submission_ref'])
            if re.search(r'\|\s*' + str(ref) + r'\s*\|', results):
                continue
            score = next(row['publicScore'] for row in official['submissions'] if int(row['ref']) == ref)
            insertion = (f'| 第{state["number"]}轮原始辅助网络生产迁移：{report_row["candidate"]} | '
                         f'**{float(report_row["production_oof_auc"]):.6f}** | **{score}** | {ref} | Complete |\n')
            marker = '\n最新官方查询：'
            if marker not in results:
                raise ValueError('RESULTS top score-table boundary missing')
            results = results.replace(marker, insertion + marker, 1)
    for filename, content in [('README.md', readme), ('RESULTS.md', results)]:
        begin, end = '<!-- ROUND16_LOCKED_LATEST_BEGIN -->', '<!-- ROUND16_LOCKED_LATEST_END -->'
        section = begin + '\n\n## 新划分重训后的最新优化\n\n' + summary + '\n\n' + end
        if begin in content:
            content = re.sub(re.escape(begin) + r'.*?' + re.escape(end), lambda _: section, content, count=1, flags=re.S)
        else:
            content += '\n\n' + section + '\n'
        path(filename).write_text(content, encoding='utf-8')
    path('LOCKED_OPTIMIZATION.md').write_text(record, encoding='utf-8')
    print(json.dumps(dict(updated=['README.md', 'RESULTS.md', 'LOCKED_OPTIMIZATION.md'],
                         rounds=args.rounds, added_official_submissions=len(additions),
                         best_score=best['public_score'], checked_at_beijing=stamp,
                         h2_status=h2_status), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', nargs='+', type=int, required=True)
    parser.add_argument('--official', required=True)
    parser.add_argument('--before', default='evidence/round16-before-training-official-results.json')
    parser.add_argument('--production-record')
    main(parser.parse_args())
