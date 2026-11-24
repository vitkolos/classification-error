from pathlib import Path
import time
import sys
import datetime
import json
import dataclasses
from typing import Any

import torch
import joblib
import joblib.externals.loky

import classerr.optimization as opt
import classerr.solving
import classerr.utils


def get_init_kwargs():
    return {'initializer': classerr.solving.init_worker, 'initargs': (classerr.solving._active_solver.get(),)}


def run_batch(
    directory: Path,
    metadata: classerr.utils.Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    samples: list[tuple[int, torch.Tensor, int]],
    num_jobs: int
):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # mkdir could cause problems if run in parallel (we can safely ignore this error)
        pass

    if len(samples) == 1:
        sample = samples[0]
        batch_step(directory, metadata, model_orig, model_aprx, *sample[:3])
    elif num_jobs == 1:
        for sample in classerr.utils.progress(samples, main=True):
            batch_step(directory, metadata, model_orig, model_aprx, *sample[:3])
    else:
        try:
            wrapped_step = joblib.delayed(batch_step)
            with joblib.Parallel(return_as='generator_unordered', **get_init_kwargs()) as para:
                results_gen = para(wrapped_step(directory, metadata, model_orig, model_aprx, *sample[:3])
                                   for sample in samples)
                for _ in classerr.utils.progress(results_gen, total=len(samples), smoothing=0, main=True):
                    # batch_step does not return anything
                    pass
        finally:
            # shutdown dangling loky workers
            joblib.externals.loky.get_reusable_executor().shutdown(wait=True, kill_workers=True)


def batch_step(
    directory: Path,
    metadata: classerr.utils.Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    sample_index: int,
    input_point: torch.Tensor,
    target: int,
):
    file_stem = f'point_{sample_index:04d}'

    if (directory / f'{file_stem}.pt').is_file():
        print('skipping', file_stem, file=sys.stderr)
        return

    data: dict[str, Any] = {}
    data['sample_index'] = sample_index
    data['date'] = str(datetime.datetime.now())

    start_time = time.time()
    results = single(metadata, model_orig, model_aprx, input_point, target)
    data['time'] = time.time() - start_time

    for key, result in results.items():
        data[key] = dataclasses.asdict(result)

    torch.save(data, directory / f'{file_stem}.pt')

    with open(directory / f'{file_stem}.json', 'w') as file_json:
        torch.set_printoptions(threshold=50)
        json.dump(data, file_json, default=str, indent=4)


def load_batch_results(directory: Path) -> list[dict]:
    results = [torch.load(f) for f in directory.glob('point_*.pt')]

    if not results:
        raise ValueError(f'batch results not found in {directory}')

    results.sort(key=lambda x: x['sample_index'])
    return results


def single(
    metadata: classerr.utils.Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    point: torch.Tensor,
    target: int,
    generalized_acc_widths: bool = False,
) -> dict[str, opt.PolytopeResult]:
    """'point' needs to be a single input (not a batch)"""
    lp_orig = opt.lp_from_net(model_orig, metadata.bounds, point)
    lp_aprx = opt.lp_from_net(model_aprx, metadata.bounds, point)
    spt = opt.Subpolytopes()

    result_point = opt.analyze_point(model_orig, model_aprx, point)
    result_sub = opt.analyze_polytope_sub(metadata, model_orig, model_aprx, lp_orig,
                                          lp_aprx, point, target, spt, generalized_acc_widths)
    result_ext = opt.analyze_polytope_ext(metadata, model_orig, model_aprx, lp_orig,
                                          point, target, spt, generalized_acc_widths)

    return {'result_point': result_point, 'result_sub': result_sub, 'result_ext': result_ext}
