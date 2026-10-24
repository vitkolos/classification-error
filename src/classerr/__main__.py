import click
import joblib

import classerr.applications
import classerr.optimization
import classerr.solving
import classerr.tree
import classerr.utils


@click.command()
@click.argument('experiment')
@click.argument('dataset')
@click.option('-n', '--run-name', default='', help='Choose a fitting name for the run (~ model variant).')
@click.option('-r', '--reduction', default='asymmetric', help='Kind of reduction/approximation/quantization.')
@click.option('-b', '--bits', default=8, help='Number of bits used by the quantized network (default: 8).')
@click.option('-s', '--solver', default=classerr.solving.SOLVER_DEFAULT, help='Best options: gurobi, gurobi-barrier, highs (default).')
@click.option('-i', '--items', default='', help='Usage: 3 (sample 3), 5:8 (samples 5, 6, 7); all the samples if left empty.')
@click.option('-j', '--jobs', default=1, help='Number of CPUs used (default: 1).')
def main(experiment, dataset, run_name, reduction, bits, solver, items, jobs):
    bundle = classerr.applications.DataBundle(dataset)
    data_split = bundle.data_split

    model_orig = bundle.load_model()
    model_aprx = bundle.load_model()
    model_aprx.round(bits=bits, qt=reduction)

    with joblib.parallel_config(backend='loky', n_jobs=jobs), classerr.solving.solver_config(solver):
        match experiment:
            case 'quality' | 'quality-test':
                metric = run_name or 'rmse'  # allowed values: rmse, mae, accuracy
                print(f'reporting {metric} on the {dataset}', 'TEST set' if experiment == 'quality-test' else 'dev set')
                dataset_subset = data_split.test if experiment == 'quality-test' else data_split.dev
                print(metric, 'original', model_orig.quality(metric, dataset_subset, data_split.metadata))
                print(metric, 'approx  ', model_aprx.quality(metric, dataset_subset, data_split.metadata))

            case 'single' | 'batch':
                dataset_subset = model_orig.subset(data_split.test, data_split.metadata.is_classification)
                samples_test = classerr.utils.get_samples(dataset_subset, items)

                if experiment == 'single':
                    i, point, target = samples_test[0]
                    # results = classerr.optimization.analyze_linear_region(
                    #     data_split.metadata, model_orig.layers, model_aprx.layers, point, target)
                    # print(results)
                    breakpoint()

            case 'tree':
                classerr.tree.build_tree(model_orig, data_split.dev)


if __name__ == '__main__':
    main()
