"""CLI."""

import argparse

from ctgan.data import read_csv, read_tsv, write_tsv
from ctgan.synthesizers.ctgan import CTGAN


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer.')
    return value


def _layer_dimensions(value):
    try:
        return [_positive_int(item) for item in value.split(',')]
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError('Use comma-separated positive integers.') from None


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument(
        '-e', '--epochs', default=300, type=_positive_int, help='Number of training epochs'
    )
    parser.add_argument(
        '-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV'
    )
    parser.add_argument(
        '--no-header',
        dest='header',
        action='store_false',
        help='The CSV file has no header. Discrete columns will be indices.',
    )

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument(
        '-d', '--discrete', help='Comma separated list of discrete columns without whitespaces.'
    )
    parser.add_argument(
        '--infer-discrete',
        action='store_true',
        help='Infer text and boolean CSV categories; specify numeric categories with -d.',
    )
    parser.add_argument(
        '-n',
        '--num-samples',
        type=_positive_int,
        help='Number of rows to sample. Defaults to the training data size',
    )

    parser.add_argument(
        '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.'
    )
    parser.add_argument(
        '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.'
    )

    parser.add_argument(
        '--generator_decay', type=float, default=1e-6, help='Weight decay for the generator.'
    )
    parser.add_argument(
        '--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.'
    )

    parser.add_argument(
        '--embedding_dim',
        type=_positive_int,
        default=128,
        help='Dimension of input z to the generator.',
    )
    parser.add_argument(
        '--generator_dim',
        type=_layer_dimensions,
        default='256,256',
        help='Dimension of each generator layer. Comma separated integers with no whitespaces.',
    )
    parser.add_argument(
        '--discriminator_dim',
        type=_layer_dimensions,
        default='256,256',
        help='Dimension of each discriminator layer. Comma separated integers with no whitespaces.',
    )

    parser.add_argument(
        '--batch_size', type=_positive_int, default=500, help='Batch size. Must be divisible by 10.'
    )
    parser.add_argument(
        '--save', default=None, type=str, help='A filename to save the trained synthesizer.'
    )
    parser.add_argument(
        '--load', default=None, type=str, help='A filename to load a trained synthesizer.'
    )

    parser.add_argument(
        '--sample_condition_column', default=None, type=str, help='Select a discrete column name.'
    )
    parser.add_argument(
        '--sample_condition_column_value',
        default=None,
        type=str,
        help='Specify the value of the selected discrete column.',
    )

    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')

    args = parser.parse_args()
    if args.batch_size % 10:
        parser.error('--batch_size must be divisible by 10.')
    if args.tsv and not args.metadata:
        parser.error('--tsv requires --metadata.')
    if args.infer_discrete and (args.tsv or args.metadata or args.discrete is not None):
        parser.error('--infer-discrete cannot be combined with --tsv, --metadata, or --discrete.')
    if (args.sample_condition_column is None) != (args.sample_condition_column_value is None):
        parser.error('Provide both --sample_condition_column and --sample_condition_column_value.')
    return args


def main():
    """CLI."""
    args = _parse_args()
    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
    else:
        data, discrete_columns = read_csv(
            args.data, args.metadata, args.header, args.discrete, args.infer_discrete
        )

    if args.load:
        model = CTGAN.load(args.load)
    else:
        model = CTGAN(
            embedding_dim=args.embedding_dim,
            generator_dim=args.generator_dim,
            discriminator_dim=args.discriminator_dim,
            generator_lr=args.generator_lr,
            generator_decay=args.generator_decay,
            discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
        model.fit(data, discrete_columns)

    if args.save is not None:
        model.save(args.save)

    num_samples = len(data) if args.num_samples is None else args.num_samples

    sampled = model.sample(
        num_samples, args.sample_condition_column, args.sample_condition_column_value
    )

    if args.tsv:
        write_tsv(sampled, args.metadata, args.output)
    else:
        sampled.to_csv(args.output, index=False, header=args.header)


if __name__ == '__main__':
    main()
