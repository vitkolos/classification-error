import torch
import polars as pl

COMPLEX_TYPES = (dict, str, list, torch.Tensor)


def unwrap_dict(source: dict, target: dict, prefix: str = ''):
    for k, v in source.items():
        if not isinstance(v, COMPLEX_TYPES) and v is not None:
            target[f'{prefix}{k}'] = v


def results_list_to_df(results: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(individual_stats(r) for r in results)


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

        if r['radii_cls'] is not None:
            m['widths_sum_cor'], m['widths_sum_all'], m['radius_min'] = \
                generalized_accuracy_sums(target, r['radii_cls'], r['widths_cls'])

        for k, v in m.items():
            stats[f'{scope}_{k}'] = v.item() if isinstance(v, torch.Tensor) else v

        unwrap_dict(r, stats, prefix=f'{scope}_')

    return stats


def generalized_accuracy_sums(target: int, radii: list, widths: list) -> tuple:
    EPS = 0.0
    radius_min = float('inf')
    sum_cor = 0.0
    sum_all = 0.0

    if not isinstance(radii[0], list):
        radii = [radii]
        widths = [widths]

    for point_radii, point_widths in zip(radii, widths):
        for k, (radius, width) in enumerate(zip(point_radii, point_widths)):
            if radius > EPS:
                radius_min = min(radius_min, radius)
                sum_all += width

                if k == target:
                    sum_cor += width

    return sum_cor, sum_all, radius_min


def weighted_average(results: pl.DataFrame) -> pl.DataFrame:
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

    return results.select(**exprs)


def describe(results: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [results.describe(percentiles=(0.5,)), weighted_average(results)],
        how='diagonal_relaxed',
    )


def get_stats(results: pl.DataFrame) -> dict:
    mean = results.mean()
    summed = results.sum()
    weighted = weighted_average(results)
    stats = {
        'point_accuracy_mean': mean['point_correct'],
        'point_retention_mean': mean['point_retention'],
        'sub_accuracy_mean': mean['sub_correct'],
        'sub_retention_mean': mean['sub_retention'],
        'sub_accuracy_weighted': weighted['sub_correct'],
        'sub_retention_weighted': weighted['sub_retention'],
        'ext_accuracy_weighted': weighted['ext_correct'],
        'ext_retention_weighted': weighted['ext_retention'],
        'sub_generalized_accuracy': summed['sub_widths_sum_cor'] / summed['sub_widths_sum_all'],
        'ext_generalized_accuracy': summed['ext_widths_sum_cor'] / summed['ext_widths_sum_all'],
    }

    for k, v in stats.items():
        stats[k] = v.item()

    return stats
