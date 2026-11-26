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
                show_results(dataset, run_id)

        case _:
            raise NotImplementedError(f'{visualization} not implemented')


TEX_ALIASES = {
    'point_accuracy_mean': r'\alpha_T',
    'point_retention_mean': r'\overline{y}_T',
    'sub_accuracy_mean': r'\alpha_{\Xi_T}',
    'sub_retention_mean': r'\overline{y}_{\Xi_T}',
    'sub_accuracy_weighted': r'\alpha_{\Xi_T}^{\widetilde d}',
    'sub_retention_weighted': r'\overline{y}_{\Xi_T}^{\widetilde d}',
    'ext_accuracy_weighted': r'\alpha_{\overline{\Xi}_{T_P}}^{\widetilde d}',
    'ext_retention_weighted': r'\overline{y}_{\overline{\Xi}_{T_P}}^{\widetilde d}',
    'sub_generalized_accuracy': r'\gamma_T',
    'ext_generalized_accuracy': r'\gamma_{T_P}',
}

LABEL_ALIASES: dict = {
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


def load_results(directory: Path) -> pl.DataFrame:
    """loads the plain results (experiment output) into a DataFrame"""
    results = classerr.experiment.load_batch_results(directory)
    return classerr.analysis.results_list_to_df(results)


def wrap_html_tables(tables):
    """wraps HTML tables so that the output looks as a nice webpage (which supports KaTeX)"""
    html = ''.join(tables)

    for column, alias in TEX_ALIASES.items():
        html = html.replace(f'>{column}</th>', f'>\\( {alias} \\)<small>{to_display_label(column)}</small></th>')

    style = 'body{font-family:sans-serif} table{border-collapse: collapse;} td,th{padding:0.5rem 1rem;} th{text-align:right} th:not(:first-child){vertical-align:bottom; text-align:left} small{display:block; margin-top:0.5rem}'
    katex = """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.css" integrity="sha384-nH0MfJ44wi1dd7w6jinlyBgljjS8EJAh2JBoRad8a3VDw2K69vfaaqm4WnR+gXtA" crossorigin="anonymous">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/katex.min.js" integrity="sha384-CwjPRVHTvLiMBFjEoij+QZViMV5rhTOIp7CJzl24JEqpRDA1sJFHVXXLURktbYYp" crossorigin="anonymous"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.47/dist/contrib/auto-render.min.js" integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
    """
    return f'<!doctype html><html><head>{katex}<style>{style}</style></head><body>{html}</body></html>'


def show_results(dataset_name: str, run_id: str):
    results = load_results(classerr.utils.DIR_OUTPUTS / dataset_name / run_id)
    described = classerr.analysis.describe(results)
    stats = classerr.analysis.get_stats(results)

    reports_dir = classerr.utils.DIR_REPORTS / dataset_name / run_id
    reports_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(reports_dir / 'results.csv')
    described.write_csv(reports_dir / 'described.csv')

    with open(reports_dir / f'stats.json', 'w') as file_json:
        json.dump(stats, file_json, indent=4)

    df = pl.DataFrame([{'run': run_id, **stats}])
    html_table = df.style.fmt_number(columns=pl.selectors.float(), decimals=5).as_raw_html()

    with open(reports_dir / "comparison.html", "w") as f:
        f.write(wrap_html_tables([html_table]))


if __name__ == '__main__':
    main()
