import torch
import polars as pl

COMPLEX_TYPES = (dict, str, list, torch.Tensor)


def unwrap_dict(source: dict, target: dict, prefix: str = ''):
    for k, v in source.items():
        if not isinstance(v, COMPLEX_TYPES) and v is not None:
            target[f'{prefix}{k}'] = v


def individual_metrics(item: dict) -> dict:
    metrics = {}
    unwrap_dict(item, metrics)

    for scope in ['point', 'sub', 'ext']:
        m = {}
        r = item[f'result_{scope}']
        target = item['target']

        m['correct'] = torch.isclose(r['pred_aprx'][target], r['pred_aprx'].amax(), atol=1e-6)
        m['retention'] = r['pred_aprx'][target] / r['pred_orig'][target]

        for k, v in m.items():
            metrics[f'{scope}_{k}'] = v.item()

        unwrap_dict(r, metrics, prefix=f'{scope}_')

    return metrics


def describe(df_results: pl.DataFrame) -> pl.DataFrame:
    pairs = {
        'error_sample': 'polytope_width',
        'error_nearby': 'polytope_width',
        'union_error': 'union_width',
    }

    exprs = {'statistic': pl.lit('weighted')}
    for col, w in pairs.items():
        if col in df_results.columns and w in df_results.columns:
            exprs[col] = (pl.col(col) * pl.col(w)).sum() / pl.col(w).sum()

    return pl.concat(
        [df_results.describe(percentiles=(0.5,)), df_results.select(exprs)],
        how='diagonal_relaxed',
    )
