"""Regression tests for the command-line workflow."""

from unittest.mock import Mock

import pandas as pd
import pytest

from ctgan import __main__ as cli


@pytest.mark.parametrize(
    'options',
    [
        ['--num-samples', '0'],
        ['--num-samples', '-1'],
        ['--epochs', '0'],
        ['--embedding_dim', '0'],
        ['--batch_size', '12'],
        ['--batch_size', '0'],
        ['--generator_dim', '256,0'],
        ['--discriminator_dim', 'abc'],
        ['--generator_dim', ''],
        ['--tsv'],
        ['--sample_condition_column', 'city'],
        ['--sample_condition_column_value', 'Boston'],
        ['--infer-discrete', '--discrete', 'city'],
        ['--infer-discrete', '--metadata', 'schema.json'],
        ['--infer-discrete', '--tsv', '--metadata', 'schema.txt'],
    ],
)
def test_invalid_arguments_fail_before_reading_data(monkeypatch, options):
    monkeypatch.setattr('sys.argv', ['ctgan', 'missing.csv', 'out.csv', *options])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_default_dimensions_are_parsed(monkeypatch):
    monkeypatch.setattr('sys.argv', ['ctgan', 'input.csv', 'out.csv'])
    args = cli._parse_args()
    assert args.generator_dim == [256, 256]
    assert args.discriminator_dim == [256, 256]


def test_load_samples_without_retraining(monkeypatch, tmp_path):
    source = tmp_path / 'input.csv'
    source.write_text('city\nBoston\nAustin\n')
    output = tmp_path / 'output.csv'
    model = Mock()
    model.sample.return_value = pd.DataFrame({'city': ['Boston']})
    synthesizer = Mock()
    synthesizer.load.return_value = model
    monkeypatch.setattr(cli, 'CTGAN', synthesizer)
    monkeypatch.setattr(
        'sys.argv',
        [
            'ctgan',
            str(source),
            str(output),
            '--load',
            'model.pkl',
            '-n',
            '1',
            '--sample_condition_column',
            'city',
            '--sample_condition_column_value',
            'Boston',
        ],
    )
    cli.main()
    synthesizer.load.assert_called_once_with('model.pkl')
    synthesizer.assert_not_called()
    model.fit.assert_not_called()
    model.sample.assert_called_once_with(1, 'city', 'Boston')
    assert output.read_text() == 'city\nBoston\n'


def test_training_infers_columns_and_saves(monkeypatch, tmp_path):
    source = tmp_path / 'input.csv'
    source.write_text('value,city\n1,Boston\n2,Austin\n')
    output = tmp_path / 'output.csv'
    synthesizer = Mock()
    model = synthesizer.return_value
    model.sample.return_value = pd.read_csv(source)
    monkeypatch.setattr(cli, 'CTGAN', synthesizer)
    monkeypatch.setattr(
        'sys.argv',
        [
            'ctgan',
            str(source),
            str(output),
            '--infer-discrete',
            '--save',
            'model.pkl',
        ],
    )
    cli.main()
    model.fit.assert_called_once()
    assert model.fit.call_args.args[1] == ['city']
    model.save.assert_called_once_with('model.pkl')
    model.sample.assert_called_once_with(2, None, None)
    pd.testing.assert_frame_equal(pd.read_csv(output), pd.read_csv(source))


def test_headerless_csv_output(monkeypatch, tmp_path):
    source = tmp_path / 'input.csv'
    source.write_text('1,Boston\n2,Austin\n')
    output = tmp_path / 'output.csv'
    synthesizer = Mock()
    synthesizer.return_value.sample.return_value = pd.read_csv(source, header=None)
    monkeypatch.setattr(cli, 'CTGAN', synthesizer)
    monkeypatch.setattr(
        'sys.argv',
        [
            'ctgan',
            str(source),
            str(output),
            '--no-header',
            '--discrete',
            '1',
        ],
    )
    cli.main()
    assert output.read_text() == source.read_text()
    assert synthesizer.return_value.fit.call_args.args[1] == [1]


def test_tsv_cli_preserves_categories(monkeypatch, tmp_path):
    source = tmp_path / 'input.tsv'
    source.write_text('1\tBoston\n2\tAustin\n')
    schema = tmp_path / 'schema.txt'
    schema.write_text('C 0 10\nD {Boston Austin}\n')
    output = tmp_path / 'output.tsv'
    synthesizer = Mock()
    synthesizer.return_value.sample.return_value = cli.read_tsv(source, schema)[0]
    monkeypatch.setattr(cli, 'CTGAN', synthesizer)
    monkeypatch.setattr(
        'sys.argv',
        [
            'ctgan',
            str(source),
            str(output),
            '--tsv',
            '--metadata',
            str(schema),
        ],
    )
    cli.main()
    assert output.read_text() == '1.0\tBoston\n2.0\tAustin\n'
