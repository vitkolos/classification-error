from pathlib import Path
import re
import json

import click
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import classerr.analysis
import classerr.experiment
import classerr.applications
import classerr.utils


ZOOM = 1.5
FIGSIZE = (6.4/ZOOM, 4.8/ZOOM)

SEED = 42
rng = np.random.default_rng(SEED)


@click.command()
@click.argument('visualization')
@click.argument('dataset')
@click.argument('run-ids', default=['run'], nargs=-1)
@click.option('--fresh-metadata', is_flag=True)
def main(visualization, dataset, run_ids, fresh_metadata):
    dataset_path = classerr.utils.DIR_OUTPUTS / dataset
    error_scaling = load_scaling(dataset, dataset_path, fresh_metadata)

    match visualization:
        case 'results':
            for run_id in run_ids:
                show_results(dataset_path / run_id)

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

LABEL_ALIASES = {
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


def analyze_results(directory: Path) -> tuple[pl.DataFrame, dict]:
    """loads the plain results (experiment output) into a DataFrame"""
    results = classerr.experiment.load_batch_results(directory)
    df = pl.DataFrame(classerr.analysis.individual_stats(r) for r in results)
    described = classerr.analysis.describe(df)
    stats = classerr.analysis.described_to_stats(described)
    stats |= classerr.analysis.generalized_accuracy(results)
    breakpoint()
    return df, stats


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


def show_results(directory: Path):
    print(analyze_results(directory))


if __name__ == '__main__':
    main()
