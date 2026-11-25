import torch
import numpy as np
import polars as pl


def individual_metrics(item: dict) -> dict:
    metrics = {}

    for scope in ['point', 'sub', 'ext']:
        m = {}
        r = item[f'result_{scope}']
        target = item['target']

        m['correct'] = torch.isclose(r['pred_aprx'][target], r['pred_aprx'].amax(), atol=1e-6)
        m['retention'] = r['pred_aprx'][target] / r['pred_orig'][target]

        for k, v in m.items():
            metrics[f'{scope}_{k}'] = v.item()

    return metrics


def describe(df_results: pl.DataFrame) -> pl.DataFrame:
    described = df_results.describe(percentiles=[0.5])
    weighted = {}
    polytope_widths = df_results.get('polytope_width')
    union_widths = df_results.get('union_width')

    def compute_weighted_average(column, weights, sum_only=False):
        if weights is not None and weights.sum() > 0 and not df_results[column].isna().any():
            if sum_only:
                weighted[column] = df_results[column].sum() / weights.sum()
            else:
                weighted[column] = np.average(df_results[column], weights=weights)

    compute_weighted_average('error_sample', polytope_widths)
    compute_weighted_average('error_nearby', polytope_widths)
    compute_weighted_average('integral', polytope_widths, sum_only=True)  # integrals are already "weighted"
    compute_weighted_average('union_error', union_widths)

    if weighted:
        described.loc['weighted'] = pl.Series(weighted)

    return described
