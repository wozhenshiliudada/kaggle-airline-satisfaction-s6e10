"""Read-only diagnostics of saved out-of-fold predictions; no model fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    'best_round13': 'outputs-round13/oof.csv',
    'old5': 'outputs-round5/oof.csv',
    'lgb5': 'experiments/round9-lgb-auxiliary-5fold/oof.csv',
    'context': 'experiments/realmlp-context-5fold-8ens/oof.csv',
    'aux10': 'experiments/round13-realmlp-route-auxiliary-10fold-8ens/oof.csv',
    'tabm': 'experiments/round9-tabm-auxiliary-5fold/oof.csv',
    'travel': 'experiments/round11-experts-5fold/oof.csv',
    'round15': 'outputs-round15/oof.csv',
    'lgb10': 'experiments/round15-lgb-10fold/oof.csv',
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])

def stats(y, p):
    e = (p >= .5) != y
    return {'n': len(y), 'positive_rate': float(y.mean()), 'auc': auc(y, p),
            'threshold_0_5_error_count': int(e.sum()),
            'threshold_0_5_error_rate': float(e.mean())}

def rank_error_endpoints(y, p):
    """Every misordered positive-negative pair contributes 1 to both endpoints.

    Ties contribute .5. Group endpoint shares sum to 1 after division by twice
    the total pair loss; they are descriptive allocations, not subgroup AUCs.
    """
    pos = np.sort(p[y]); neg = np.sort(p[~y]); out = np.empty(len(p))
    q = p[y]
    left = np.searchsorted(neg, q, side='left')
    right = np.searchsorted(neg, q, side='right')
    out[y] = len(neg) - right + .5 * (right - left)
    q = p[~y]
    left = np.searchsorted(pos, q, side='left')
    right = np.searchsorted(pos, q, side='right')
    out[~y] = left + .5 * (right - left)
    return out

def main():
    tr = pd.read_csv(ROOT / 'data/train.csv')
    y = tr.satisfaction.to_numpy(dtype=bool)
    pred = {}; source_sha = {}
    for name, path in PATHS.items():
        file = ROOT / path; frame = pd.read_csv(file)
        assert frame.id.equals(tr.id)
        assert np.array_equal(frame.satisfaction.to_numpy(dtype=bool), y)
        p = frame.prediction.to_numpy()
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
        pred[name] = p; source_sha[path] = sha(file)
    pred['contextual_pair'] = expit(.5 * logit(np.clip(pred['lgb5'], 1e-7, 1-1e-7))
        + .5 * logit(np.clip(pred['context'], 1e-7, 1-1e-7)))
    base = pred['best_round13']; eb = (base >= .5) != y
    endpoints = rank_error_endpoints(y, base)
    pair_loss = len(y[y]) * len(y[~y]) * (1 - auc(y, base))
    assert np.isclose(endpoints.sum(), 2 * pair_loss, rtol=1e-10)
    result = {'scope': 'Existing OOF predictions only; no fitting or parameter selection',
        'limitations': ['OOF model sources have historical mixed 3/5/10-fold training schemes.',
            'No fresh independent holdout; repeated model selection may bias OOF gains.',
            'Residual correlation uses y minus p and shares the target; class-conditional values are supplied.',
            'Threshold .5 errors are auxiliary and are not the competition ROC-AUC metric.',
            'Group AUC excludes cross-group pairs. Endpoint pair-loss shares include them but are descriptive only.',
            'Misclassification rescue counts cannot establish ROC-AUC improvement or hidden-label ceiling.'],
        'input_sha256': sha(ROOT / 'data/train.csv'), 'prediction_sha256': source_sha,
        'overall': {k: stats(y, v) for k, v in pred.items()},
        'components_vs_best': {}, 'pairwise_component_correlation': [], 'group_comparison': {},
        'probability_regions_best': [], 'rank_error_pair_loss': float(pair_loss)}
    for name, p in pred.items():
        if name == 'best_round13': continue
        e = (p >= .5) != y; both = int((e & eb).sum())
        result['components_vs_best'][name] = {
            'prediction_pearson': corr(base, p),
            'prediction_spearman': float(spearmanr(base, p).statistic),
            'residual_pearson': corr(y.astype(float)-base, y.astype(float)-p),
            'residual_pearson_y0': corr(base[~y], p[~y]),
            'residual_pearson_y1': corr(base[y], p[y]),
            'mean_absolute_prediction_difference': float(np.mean(np.abs(base-p))),
            'threshold_error_intersection_count': both,
            'threshold_error_jaccard': both / int((e | eb).sum()),
            'best_wrong_component_right_count': int((eb & ~e).sum()),
            'best_right_component_wrong_count': int((~eb & e).sum()),
            'threshold_disagreement_count': int((e != eb).sum())}
    names = ['old5','lgb5','context','aux10','tabm','travel','lgb10']
    for i, a in enumerate(names):
        for b in names[i+1:]:
            result['pairwise_component_correlation'].append({'a': a, 'b': b,
                'prediction_spearman': float(spearmanr(pred[a], pred[b]).statistic),
                'residual_pearson': corr(y.astype(float)-pred[a], y.astype(float)-pred[b]),
                'residual_pearson_y0': corr(pred[a][~y], pred[b][~y]),
                'residual_pearson_y1': corr(pred[a][y], pred[b][y])})
    all_error = int(eb.sum()); total_endpoint = endpoints.sum()
    for column in ['Class', 'Type of Travel', 'Customer Type']:
        rows = []
        for key, idx in tr.groupby(column).indices.items():
            m = np.asarray(idx); s = stats(y[m], base[m]); r = stats(y[m], pred['round15'][m])
            s.update({'group': str(key), 'sample_share': len(m)/len(y),
                'threshold_error_share': s['threshold_0_5_error_count']/all_error,
                'rank_pair_loss_endpoint_share': float(endpoints[m].sum()/total_endpoint),
                'round15_auc': r['auc'], 'round15_auc_delta': r['auc']-s['auc'],
                'round15_threshold_error_count': r['threshold_0_5_error_count']})
            rows.append(s)
        result['group_comparison'][column] = rows
    bands = [('below_0_1', base < .1), ('between_0_1_0_9', (base >= .1) & (base <= .9)),
        ('above_0_9', base > .9), ('between_0_4_0_6', (base >= .4) & (base <= .6))]
    for name, mask in bands:
        row = stats(y[mask], base[mask]); row.update({'region': name,
            'sample_share': float(mask.mean()),
            'threshold_error_share': int(eb[mask].sum()) / all_error,
            'rank_pair_loss_endpoint_share': float(endpoints[mask].sum()/total_endpoint)})
        result['probability_regions_best'].append(row)
    result['all_selected_components_wrong_at_0_5'] = {
        'components': ['old5','contextual_pair','aux10','tabm','travel'],
        'n': int(np.logical_and.reduce([((pred[k] >= .5) != y) for k in ['old5','contextual_pair','aux10','tabm','travel']]).sum())}
    partition = [('low', base < .1), ('middle', (base >= .1) & (base <= .9)), ('high', base > .9)]
    matrix = np.zeros((3,3))
    for i, (_, pos_region) in enumerate(partition):
        q = base[y & pos_region]
        for j, (_, neg_region) in enumerate(partition):
            negative = np.sort(base[~y & neg_region])
            left = np.searchsorted(negative, q, side='left')
            right = np.searchsorted(negative, q, side='right')
            matrix[i,j] = np.sum(len(negative)-right + .5*(right-left))
    assert np.isclose(matrix.sum(), pair_loss, rtol=1e-10)
    result['rank_pair_loss_probability_region_matrix'] = {
        'row_positive_probability_regions': ['below_0_1','0_1_to_0_9','above_0_9'],
        'column_negative_probability_regions': ['below_0_1','0_1_to_0_9','above_0_9'],
        'pair_loss_counts': matrix.tolist(), 'pair_loss_fractions': (matrix/pair_loss).tolist(),
        'pair_loss_fraction_with_positive_below_0_1_or_negative_above_0_9':
            float((matrix[0,:].sum()+matrix[:,2].sum()-matrix[0,2])/pair_loss),
        'confident_wrong_label_rows': int(((y & (base < .1)) | (~y & (base > .9))).sum())}
    feature_columns = [c for c in tr.columns if c not in ['id','satisfaction']]
    duplicate_only = tr[tr.duplicated(feature_columns, keep=False)]
    if len(duplicate_only):
        g = duplicate_only.groupby(feature_columns, dropna=False).satisfaction.agg(['size','sum'])
        conflict = (g['sum'] > 0) & (g['sum'] < g['size'])
        duplicate_info = {'rows_in_repeated_feature_patterns': len(duplicate_only),
            'repeated_patterns': len(g), 'contradictory_label_patterns': int(conflict.sum()),
            'rows_in_contradictory_label_patterns': int(g.loc[conflict,'size'].sum()),
            'minimum_0_5_errors_for_any_deterministic_exact_feature_mapping':
                int(np.minimum(g['sum'],g['size']-g['sum']).sum())}
    else:
        duplicate_info = {'rows_in_repeated_feature_patterns': 0, 'repeated_patterns': 0,
            'contradictory_label_patterns': 0, 'rows_in_contradictory_label_patterns': 0,
            'minimum_0_5_errors_for_any_deterministic_exact_feature_mapping': 0}
    result['exact_feature_duplicate_label_diagnostic'] = duplicate_info
    result['script_sha256'] = sha(Path(__file__))
    (ROOT / 'analyses/gap-error-diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['overall','components_vs_best','group_comparison',
        'probability_regions_best','all_selected_components_wrong_at_0_5',
        'rank_pair_loss_probability_region_matrix','exact_feature_duplicate_label_diagnostic']}, indent=2))

if __name__ == '__main__':
    main()
