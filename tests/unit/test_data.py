"""Tests for tabular input schemas and output round trips."""

import numpy as np
import pytest

from ctgan.data import read_csv, read_tsv, read_tsv_metadata, write_tsv


def test_infer_discrete_is_opt_in(tmp_path):
    source = tmp_path / 'mixed.csv'
    source.write_text('amount,city,enabled,code\n1.5,Boston,True,1\n2.5,Austin,False,2\n')
    assert read_csv(source)[1] == []
    assert read_csv(source, infer_discrete=True)[1] == ['city', 'enabled']
    assert read_csv(source, discrete='code', infer_discrete=True)[1] == ['code']


def test_inference_without_header(tmp_path):
    source = tmp_path / 'mixed.csv'
    source.write_text('1.5,Boston,True\n2.5,Austin,False\n')
    assert read_csv(source, header=False, infer_discrete=True)[1] == [1, 2]


def test_metadata_takes_precedence(tmp_path):
    source = tmp_path / 'data.csv'
    source.write_text('amount,code\n1.5,1\n2.5,2\n')
    schema = tmp_path / 'schema.json'
    schema.write_text('{"columns": [{"name": "code", "type": "categorical"}]}')
    assert read_csv(source, meta_filename=schema, infer_discrete=True)[1] == ['code']


@pytest.mark.parametrize('metadata_type', ['string', 'path', 'dictionary'])
def test_tsv_round_trip(tmp_path, metadata_type):
    source = tmp_path / 'input.tsv'
    source.write_text('1.5\tred\n2.5\tblue\n')
    schema = tmp_path / 'schema.txt'
    schema.write_text('C 0 10\nD {red blue}\n')
    output = tmp_path / 'output.tsv'
    data, discrete = read_tsv(source, schema)
    metadata = {'string': str(schema), 'path': schema, 'dictionary': read_tsv_metadata(schema)}[
        metadata_type
    ]
    write_tsv(data, metadata, output)
    assert output.read_text() == source.read_text()
    restored, restored_discrete = read_tsv(output, schema)
    np.testing.assert_array_equal(restored, data)
    assert restored_discrete == discrete == [1]
