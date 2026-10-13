import click
import classerr.applications


@click.command()
@click.argument('experiment')
@click.argument('dataset')
@click.option('-n', '--run-name', default='', help='Choose a fitting name for the run (~ model variant).')
@click.option('-r', '--reduction', default='asymmetric', help='Kind of reduction/approximation/quantization.')
@click.option('-b', '--bits', default=8, help='Number of bits used by the quantized network (default: 8).')
@click.option('-i', '--items', default='', help='Usage: 3 (sample 3), 5:8 (samples 5, 6, 7); all the samples if left empty.')
@click.option('-j', '--jobs', default=1, help='Number of CPUs used (default: 1).')
def main(experiment, dataset, run_name, reduction, bits, items, jobs):
    bundle = classerr.applications.DataBundle(dataset)
    data_split = bundle.data_split

    model = bundle.load_model()
    model_approx = bundle.load_model()
    model_approx.round(bits=bits, qt=reduction)

    match experiment:
        case 'quality' | 'quality-test':
            metric = run_name or 'rmse'  # allowed values: rmse, mae, accuracy/<num_classes>
            print(f'reporting {metric} on the {dataset}', 'TEST set' if experiment == 'quality-test' else 'dev set')
            dataset_subset = data_split.test if experiment == 'quality-test' else data_split.dev
            print(metric, 'original', model.quality(metric, dataset_subset, data_split.metadata.error_scaling))
            print(metric, 'approx  ', model_approx.quality(metric, dataset_subset, data_split.metadata.error_scaling))


if __name__ == '__main__':
    main()
