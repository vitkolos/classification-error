from pathlib import Path
import re
import json

import click
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import classerr.experiment
import classerr.applications


SEED = 42
EXPERIMENTS_DIR = Path('experiments')

ZOOM = 1.5
FIGSIZE = (6.4/ZOOM, 4.8/ZOOM)

rng = np.random.default_rng(SEED)


@click.command()
@click.argument('visualization')
@click.argument('dataset')
@click.argument('run-ids', default=['run'], nargs=-1)
@click.option('--plot-only', is_flag=True)
@click.option('--fresh-metadata', is_flag=True)
@click.option('--with-legend', is_flag=True)
def main(visualization, dataset, run_ids, plot_only, fresh_metadata, with_legend):
    dataset_path = EXPERIMENTS_DIR / dataset
    error_scaling = load_scaling(dataset, dataset_path, fresh_metadata)

    match visualization:
        case 'comparison':
            compare_results(dataset_path, run_ids, error_scaling)

        case _:
            raise NotImplementedError(f'{visualization} not implemented')


TEX_ALIASES = {
    'sample_max': r'E_T',
    'sample_mean': r'\overline{E}_T',
    'nearby_max': r'E_{\Xi_T}',
    'nearby_mean': r'\overline{E}_{\Xi_T}',
    'nearby_weighted_sum': r'\overline{E}^{\widetilde d}_{\Xi_T}',
    'integral_divided_sum': r'\overline{E}^{\widetilde d}_{\Xi_T^E}',
    'error_sample': r'E(x)',
    'error_nearby': r'E_{\Xi_x}',
    'polytope_width': r'\tilde d_n(\Xi_x)',
    'weight': r'\frac{\tilde d_n(\Xi_x)}{S}',
    'nearby_weighted': r'\frac{\tilde d_n(\Xi_x)}{S} E_{\Xi_x}',
    'integral_width': r'\tilde d_{n+1}(\Xi_x^E)',
    'integral_divided': r'\tilde d_{n+1}(\Xi_x^E)\over S',
    'union_max': r'E_{\overline{\Xi}_T}',
    'union_mean': r'\overline{E}_{\overline{\Xi}_T}',
    'union_weighted_sum': r'\overline{E}^{\widetilde d}_{\overline{\Xi}_T}',
}

NETS = {
    'california': 1,
    'year': 2,
    'utkface': 3,
}

LABEL_ALIASES = {
    'union_width': 'extended polytope width',
}


def to_display_label(label: str) -> str:
    label = LABEL_ALIASES.get(label, label)
    return label.replace('_', ' ')


def load_scaling(dataset: str, experiment_path: Path, fresh_metadata: bool) -> float:
    metadata_file = experiment_path / 'metadata.pt'

    if not fresh_metadata and metadata_file.is_file():
        metadata_dict = torch.load(metadata_file)
        return metadata_dict['error_scaling']

    bundle = classerr.applications.DataBundle(dataset)
    metadata_dict = bundle.data_split.metadata.to_dict()
    torch.save(metadata_dict, metadata_file)

    with open(experiment_path / f'metadata.json', 'w') as file_json:
        json.dump(metadata_dict, file_json)

    return metadata_dict['error_scaling']


def load_df_results(experiment_path: Path, run_id: str) -> pl.DataFrame:
    """loads the plain results (experiment output) into a DataFrame"""
    results = classerr.experiment.load_batch_results(experiment_path, run_id)
    df = pl.DataFrame(classerr.experiment.dict2flat(r) for r in results)
    df = df.set_index('sample_index').sort_index()
    return df


def extract_metrics(df_results: pl.DataFrame):
    """extracts our metrics from a DataFrame containing plain results"""
    described = classerr.experiment.describe(df_results)
    return {
        'sample_max': described.loc['max', 'error_sample'],
        'sample_mean': described.loc['mean', 'error_sample'],
        'nearby_max': described.loc['max', 'error_nearby'],
        'nearby_mean': described.loc['mean', 'error_nearby'],
        'nearby_weighted_sum': described.loc['weighted', 'error_nearby'],
        'integral_divided_sum': described.loc['weighted', 'integral'],
        'union_max': described.loc['max', 'union_error'],
        'union_mean': described.loc['mean', 'union_error'],
        'union_weighted_sum': described.loc['weighted', 'union_error'],
    }


def wrap_html_tables(tables, into_one=True):
    """wraps HTML tables so that the output looks as a nice webpage (which supports KaTeX)"""
    html = ''.join(tables)

    if into_one:
        html = '<table>' + re.sub(r'</?table.*?>', '', html) + '</table>'

    for column, alias in TEX_ALIASES.items():
        html = html.replace(f'>{column}</th>', f'>\\( {alias} \\)<small>{to_display_label(column)}</small></th>')

    style = 'body{font-family:sans-serif} table{border-collapse: collapse;} td,th{padding:0.5rem 1rem;} th{text-align:right} th:not(:first-child){vertical-align:bottom; text-align:left} small{display:block; margin-top:0.5rem}'
    katex = """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css" integrity="sha384-nH0MfJ44wi1dd7w6jinlyBgljjS8EJAh2JBoRad8a3VDw2K69vfaaqm4WnR+gXtA" crossorigin="anonymous">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.js" integrity="sha384-CwjPRVHTvLiMBFjEoij+QZViMV5rhTOIp7CJzl24JEqpRDA1sJFHVXXLURktbYYp" crossorigin="anonymous"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/contrib/auto-render.min.js" integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
    """
    return f'<!doctype html><html><head>{katex}<style>{style}</style></head><body>{html}</body></html>'


def compare_results(experiment_path: Path, run_ids: list[str], error_scaling: float):
    """writes nice tables containing the metrics and comparing them between runs"""
    dfs = {run_id: load_df_results(experiment_path, run_id) for run_id in run_ids}

    def analyze(name: str, df_results: pl.DataFrame):
        df_results.loc[:, classerr.experiment.UNSCALED_COLS] *= error_scaling
        net_index = NETS.get(experiment_path.name, experiment_path.name[0])
        bits = ''.join(char for char in name if char.isdigit()) or name[0]  # hack to extract bits from run_id
        return {
            'run': r' \( \boldsymbol{\widetilde{\mathcal{N}}_' + str(net_index) + '^{' + bits + '}} \\)',
            **extract_metrics(df_results),
        }

    df = pl.DataFrame(analyze(*item) for item in dfs.items())
    df = df.set_index('run')
    df.index.name = None

    FMT = '{:.5f}'.format  # precision

    target_dir = experiment_path / 'common_outputs'
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(target_dir / 'comparison.html', 'w') as f:
        f.write(wrap_html_tables([df.to_html(float_format=FMT)]))



if __name__ == '__main__':
    main()
