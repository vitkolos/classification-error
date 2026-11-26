import torch
import polars as pl

COMPLEX_TYPES = (dict, str, list, torch.Tensor)


def unwrap_dict(source: dict, target: dict, prefix: str = ''):
    for k, v in source.items():
        if not isinstance(v, COMPLEX_TYPES) and v is not None:
            target[f'{prefix}{k}'] = v


def individual_stats(item: dict) -> dict:
    stats: dict = {}
    unwrap_dict(item, stats)

    for scope in ['point', 'sub', 'ext']:
        m = {}
        r = item[f'result_{scope}']
        target = item['target']

        max_pred_aprx = r['pred_aprx'].amax()
        m['correct'] = torch.isclose(r['pred_aprx'][target], max_pred_aprx, rtol=1e-7, atol=1e-9)
        m['correct_dist'] = max_pred_aprx - r['pred_aprx'][target]

        m['retention'] = r['pred_aprx'][target] / r['pred_orig'][target]

        for k, v in m.items():
            stats[f'{scope}_{k}'] = v.item()

        unwrap_dict(r, stats, prefix=f'{scope}_')

    return stats


def describe(results: pl.DataFrame) -> pl.DataFrame:
    pairs = {
        'sub_correct': 'sub_width',
        'sub_retention': 'sub_width',
        'ext_correct': 'ext_width',
        'ext_retention': 'ext_width',
    }
    exprs = {'statistic': pl.lit('weighted')}

    for col, w in pairs.items():
        if col in results.columns and w in results.columns:
            exprs[col] = (pl.col(col) * pl.col(w)).sum() / pl.col(w).sum()

    return pl.concat(
        [results.describe(percentiles=(0.5,)), results.select(**exprs)],
        how='diagonal_relaxed',
    )


def described_to_stats(described: pl.DataFrame) -> dict:
    stats = {row['statistic']: row for row in described.to_dicts()}
    return {
        'point_accuracy_mean': stats['mean']['point_correct'],
        'point_retention_mean': stats['mean']['point_retention'],
        'sub_accuracy_mean': stats['mean']['sub_correct'],
        'sub_retention_mean': stats['mean']['sub_retention'],
        'sub_accuracy_weighted': stats['weighted']['sub_correct'],
        'sub_retention_weighted': stats['weighted']['sub_retention'],
        'ext_accuracy_weighted': stats['weighted']['ext_correct'],
        'ext_retention_weighted': stats['weighted']['ext_retention'],
    }


def generalized_accuracy(results_list: list[dict]) -> dict[str, float]:
    stats: dict[str, float] = {}
    min_radius = float('inf')
    sub_widths_correct = 0.0
    sub_widths_all = 0.0
    ext_widths_correct = 0.0
    ext_widths_all = 0.0

    # def _process_subpolytope(radii, widths, target):
    #     num_classes = len(radii)
    #     width_sum =

    #     for c in range(num_classes):
    #         if radii[c] > 0.0:

    # for item in results_list:
    #     radii = item['result_sub']['radii_cls']
    #     widths = item['result_sub']['widths_cls']

    stats['sub_generalized'] = -1.0
    stats['ext_generalized'] = -1.0
    return stats
