"""Persisted-output validation and workflow failure behavior."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

import cli
from config import AnalysisConfig, all_configs, prior_locations
from data import load_wing_lengths
from plotting import plot_result
from reporting import write_results
from validation import validate_all, validate_result


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def valid_outputs(tmp_path_factory):
    """Synthetic records for validator tests, never scientific analysis outputs.

    Build once in pytest's temporary directory; do not depend on results/.
    Posterior correctness and real-result regression are tested separately.
    """
    output = tmp_path_factory.mktemp('validation_fixture')
    y = load_wing_lengths(ROOT / 'data/raw/wing_lengths.csv')
    for config in all_configs():
        rows = []
        for mu0 in prior_locations(y):
            for target in config.targets:
                row = dict(method=config.method, target=target, mu0=mu0,
                           epss=0, fixed_risk=1., candidate_risk=1., boundary=0)
                if config.method == 'reimherr' or config.sampling == 'bootstrap':
                    row['bootstrap_rejections'] = 0
                row['risk_reps'] = config.risk_reps
                rows.append(row)
        write_results(rows, y, config, output)
        plot_result(output, y, config)
    return output, y


@pytest.fixture
def saved(tmp_path, valid_outputs):
    template, y = valid_outputs
    output = tmp_path / 'results'
    shutil.copytree(template, output)
    return output, y


def test_all_output_contracts(saved):
    output, y = saved
    report = validate_all(output, y)
    assert report['status'] == 'passed'
    assert report['combinations'] == 8 and report['rows'] == 1312
    assert len({c.output_stem for c in all_configs()}) == 8


@pytest.mark.parametrize('damage, message', [
    ('missing_row', 'expected 164'), ('duplicate', 'duplicate'),
    ('wrong_prior', 'prior-target'), ('wrong_target', 'prior-target'),
    ('nan_risk', 'nonfinite'), ('negative_risk', 'negative risk'),
    ('wrong_method', 'mismatched method'), ('wrong_reps', 'risk_reps'),
    ('wrong_epss', 'EPSS'), ('wrong_boundary', 'boundary'),
])
def test_reject_semantically_invalid_csv(saved, damage, message):
    output, y = saved
    table = output / 'tables/reimherr.csv'
    with table.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    if damage == 'missing_row':
        rows.pop()
    elif damage == 'duplicate':
        rows[-1] = rows[0].copy()
    else:
        field, value = {
            'wrong_prior': ('mu0', '999'), 'wrong_target': ('target', 'unknown'),
            'nan_risk': ('fixed_risk', 'nan'), 'negative_risk': ('candidate_risk', '-1'),
            'wrong_method': ('method', 'wiesenfarth'), 'wrong_reps': ('risk_reps', '1'),
            'wrong_epss': ('epss', '999'),
            'wrong_boundary': ('boundary', str(1-int(rows[0]['boundary']))),
        }[damage]
        rows[0][field] = value
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    # Even a refreshed file hash must not hide invalid scientific table contents.
    metadata_file = output / 'metadata/reimherr.json'
    metadata = json.loads(metadata_file.read_text())
    metadata['table_sha256'] = hashlib.sha256(table.read_bytes()).hexdigest()
    metadata_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        validate_result(output, y, AnalysisConfig('reimherr'))


@pytest.mark.parametrize('damage', ['hash', 'configuration', 'missing_combination', 'figure'])
def test_failed_validation_removes_old_success_report(saved, damage):
    output, y = saved
    validate_all(output, y)
    if damage == 'hash':
        table = output / 'tables/reimherr.csv'
        table.write_bytes(table.read_bytes() + b'\n')
    elif damage == 'configuration':
        path = output / 'metadata/reimherr.json'
        data = json.loads(path.read_text())
        data['config']['sampling'] = 'likelihood'
        path.write_text(json.dumps(data))
    elif damage == 'missing_combination':
        (output / 'tables/wiesenfarth__bootstrap__posterior_mse.csv').unlink()
    else:
        (output / 'figures/reimherr.png').write_bytes(b'broken PNG')
    with pytest.raises((ValueError, OSError)):
        validate_all(output, y)
    assert not (output / 'validation.json').exists()


def test_plot_stage_reads_saved_csv_without_computing(saved, monkeypatch):
    output, _ = saved
    table = output / 'tables/reimherr.csv'
    before = table.read_bytes()
    figure = output / 'figures/reimherr.png'
    figure.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('Plot stage must not recompute analysis')
    monkeypatch.setattr(cli, 'run_analysis', forbidden)
    cli.main(ROOT, ['--method', 'reimherr', '--stage', 'plot', '--output-dir', str(output)])
    assert table.read_bytes() == before
    assert figure.is_file()


@pytest.mark.parametrize('args', [
    ['--all', '--method', 'reimherr'], ['--all', '--sampling', 'bootstrap'],
    ['--all', '--loss', 'posterior_mse'], ['--all', '--target', 'mu'], [],
])
def test_reject_ambiguous_workflow_options(args):
    with pytest.raises(SystemExit) as error:
        cli.main(ROOT, args)
    assert error.value.code == 2
