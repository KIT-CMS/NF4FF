"""Regression checks for the legacy NF export and DNN plotting interface.

Load function definitions without the scripts' CLI parsing, GPU dependencies or
model-loading side effects. Run with unittest; only numpy/pandas are required.
"""
import ast
import logging
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class Inputs:
    def __init__(self, frame):
        self.X = frame[['pdf_ar', 'pdf_sr']].to_numpy()

    def to_torch(self):
        return self

    def to(self, device):
        return self


class NFExportTests(unittest.TestCase):
    def setUp(self):
        self.ns = {'np': np, 'pd': pd, 'logger': logging.getLogger(__name__),
                   'get_my_data_qcd': lambda frame, variables: Inputs(frame),
                   'evaluate_pdf': lambda model, values: values[:, model]}
        load_functions('Normalizing_Flow_tt/src/classes/Collection.py',
                       ['compute_eventwise_fake_factors'], self.ns)
        load_functions('Normalizing_Flow_tt/src/NF_ff_calculation.py',
                       ['normalizing_flow_ff', 'nf_output_columns', 'nf_process_groups',
                        'select_njets_category', 'main'], self.ns)
        load_functions('DNN_tt/src/classes/Plotting.py',
                       ['_nf_distribution_frame', '_nf_distribution_bins', 'validate_nf_weights',
                        'estimate_jet_fakes'], self.ns)

    def export(self, first, second=None, grouped=True):
        return self.ns['normalizing_flow_ff'](
            first, first if second is None else second, [], 0, 1, 0, 1,
            0.2, 0.2, 'cpu', include_njets=False, grouped=grouped)

    def test_small_densities_keep_the_density_ratio(self):
        frame = pd.DataFrame({'pdf_ar': [1e-8, 1e-10, 1e-12],
                              'pdf_sr': [1e-8, 1e-10, 1e-12]})
        result, _ = self.export(frame)
        np.testing.assert_allclose(result.ff_nf_tau1_njets, 0.1)

    def test_zero_weights_equal_reference_event_removal(self):
        frame = pd.DataFrame({'pdf_ar': [1e-10] * 3 + [0.0],
                              'pdf_sr': [1e-10, 2e-10, 2e-9, 1e-10],
                              'pt_1': [45., 65., 85., 105.],
                              'weight': [1., -0.5, 2., 1.]}, index=[7, 12, 15, 20])
        result, _ = self.export(frame)
        full, _, _, keep, _ = self.ns['compute_eventwise_fake_factors'](
            frame.pdf_ar.to_numpy(), frame.pdf_sr.to_numpy(), 0.2)
        np.testing.assert_array_equal(result.index, frame.index)
        self.assertEqual(result.ff_nf_tau1_njets_kept.tolist(), [True, True, False, False])
        self.assertEqual(result.ff_nf_tau1_njets.tolist(), [0.2, 0.4, 0., 0.])
        bins = [40, 60, 80, 100, 120]
        exported = np.histogram(frame.pt_1, bins, weights=frame.weight * result.ff_nf_tau1_njets)[0]
        reference = np.histogram(frame.pt_1[keep], bins,
                                 weights=frame.weight[keep] * full[keep] * 0.5)[0]
        np.testing.assert_allclose(exported, reference)
        display = self.ns['_nf_distribution_frame'](result, 'ff_nf_tau1_njets')
        self.assertEqual(display.index.tolist(), [7, 12])
        np.testing.assert_allclose(display.ff_nf_tau1_njets, full[keep])

    def test_empty_tau_does_not_skip_other_tau(self):
        frame = pd.DataFrame({'pdf_ar': [1e-10], 'pdf_sr': [1e-10]})
        empty, result = self.export(frame.iloc[:0], frame, grouped=False)
        self.assertIn('ff_nf_tau1', empty)
        np.testing.assert_allclose(result.ff_nf_tau2, [0.1])

    def test_all_rejected_events_have_finite_zero_prediction(self):
        result, _ = self.export(pd.DataFrame({'pdf_ar': [1e-10], 'pdf_sr': [1e-6]}))
        self.assertEqual(result.ff_nf_tau1_njets.tolist(), [0.])
        self.assertFalse(result.ff_nf_tau1_njets_kept.any())

    def test_nonfinite_densities_fail(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite NF densities'):
            self.export(pd.DataFrame({'pdf_ar': [np.nan], 'pdf_sr': [1.]}))

    def test_process_components_share_reference_acceptance_correction(self):
        mapping = {'data': 0, 'diboson_J': 1, 'diboson_L': 2, 'DYjets_J': 3,
                   'DYjets_L': 4, 'embedding': 5, 'ST_J': 6, 'ST_L': 7,
                   'ttbar_J': 8, 'ttbar_L': 9, 'Wjets': 10}
        groups = self.ns['nf_process_groups'](mapping)
        self.assertEqual(groups['diboson'], [1, 2])
        self.assertEqual(groups['embedding'], [5])
        self.assertEqual(sorted(sum(groups.values(), [])), list(range(11)))

    def test_closure_matches_reference_subtraction(self):
        processes = ('data', 'diboson', 'DYjets', 'ST', 'embedding', 'ttbar', 'wjets')
        frames = {}
        expected = np.zeros(2)
        bins = [40, 60, 80]
        for index, process in enumerate(processes):
            frame = pd.DataFrame({'pdf_ar': [1e-10, 1e-10],
                                  'pdf_sr': [1e-10, 2e-9 if index % 2 else 2e-10],
                                  'weight': [1., 1.] if index == 0 else [0.01, -0.02],
                                  'pt_1': [45., 65.]})
            first, second = self.export(frame)
            frames[process] = SimpleNamespace(AR_tau1=first, AR_tau2=second)
            full, _, _, keep, _ = self.ns['compute_eventwise_fake_factors'](
                frame.pdf_ar.to_numpy(), frame.pdf_sr.to_numpy(), 0.2)
            # Two identical tau estimates, each carrying 0.5.
            h = np.histogram(frame.pt_1[keep], bins, weights=frame.weight[keep] * full[keep])[0]
            expected += h if process == 'data' else -h
        self.ns['validate_nf_weights'](frames)
        actual, variance = self.ns['estimate_jet_fakes'](
            frames, bins, 'pt_1', 'ff_nf_tau1_njets', 'ff_nf_tau2_njets')
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.isfinite(variance).all())
        frames['embedding'].AR_tau1.loc[0, 'ff_nf_tau1_njets'] = np.nan
        with self.assertRaisesRegex(ValueError, 'embedding, AR_tau1: 1 invalid'):
            self.ns['validate_nf_weights'](frames)

    def test_distribution_range_includes_acceptance_corrected_tail(self):
        bins = self.ns['_nf_distribution_bins']([0.1, 3.5], [0.2])
        self.assertGreaterEqual(bins[-1], 3.5)

    def test_export_all_modes_refresh_both_column_sets_for_all_processes(self):
        mapping = dict(zip(('data', 'diboson_J', 'diboson_L', 'DYjets_J', 'DYjets_L',
                            'embedding', 'ST_J', 'ST_L', 'ttbar_J', 'ttbar_L', 'Wjets'), range(11)))
        rows = []
        for jets in (0, 1, 2):
            for process in range(11):
                for region in ('AR_tau1', 'AR_tau2'):
                    rows.append((jets, process, region, True, False))
            for region, count in (('AR_like_tau1', 1), ('AR_like_tau2', 1), ('SR_like', jets + 1)):
                rows.extend([(jets, 0, region, False, True)] * count)
        source = pd.DataFrame(rows, columns=['njets', 'process', 'region', 'OS', 'SS'])
        source['pdf_ar'] = source['pdf_sr'] = 1e-10
        source['ff_dnn_tau1'] = 17.0
        for grouped in (False, True):
            for column in self.ns['nf_output_columns'](grouped):
                source[column] = True if column.endswith('_kept') else 99.0
        for region in ('AR_tau1', 'AR_tau2', 'AR_like_tau1', 'AR_like_tau2', 'SR_like'):
            self.ns[region] = lambda frame, region=region: frame[frame.region == region].copy()
        checkpoints = []

        def model_loader(**kwargs):
            checkpoint = kwargs.get('checkpoint_path', kwargs.get('checkpoint_dir'))
            checkpoints.append(checkpoint)
            return 0 if '/AR-like/' in checkpoint else 1

        self.ns.update(
            Path=Path,
            yaml=SimpleNamespace(safe_load=lambda handle: {'variables_61': []}),
            torch=SimpleNamespace(device=lambda name: name, cuda=SimpleNamespace(is_available=lambda: False)),
            cfg_path={'variables': '/unused/variables', 'config_settings': '/unused/settings',
                      'NF_results': '/unused/models', 'datasets': '/unused/data',
                      'process_map': {'embedding': '/unused/process-map'}},
            config_path='/unused/config', MASKS_CONFIG_PATH='/unused/masks',
            load_config=lambda path: mapping if path.endswith('process-map') else {},
            load_masks_config=lambda path: {}, build_training_variables_tag=lambda variables: 'exact',
            resolve_training_tag=lambda variables, mode: 'resolved',
            load_saved_model_config=lambda *args: None,
            load_flow=model_loader, load_conditional_flow=model_loader,
            load_grouped_qcd_njets_router=model_loader,
            mask_preselection_for_estimator=lambda frame: frame,
        )
        for mode in ('single_nf', 'conditional_nf', 'grouped_njets_split'):
            with self.subTest(mode=mode):
                exported = []
                self.ns['args'] = SimpleNamespace(var='variables_61', model_mode=mode,
                                                  training_tag='', embedding='embedding')
                with patch('builtins.open', lambda *args: StringIO('')):
                    with patch.object(pd, 'read_feather', return_value=source.copy()):
                        with patch.object(pd.DataFrame, 'to_feather',
                                          lambda frame, path: exported.append(frame.copy())):
                            self.ns['main']()
                self.assertEqual(len(exported), 1)
                frame = exported[0]
                for tau in (1, 2):
                    ar = frame[frame.region == f'AR_tau{tau}']
                    np.testing.assert_allclose(ar[f'ff_nf_tau{tau}'], 1.0)
                    np.testing.assert_allclose(ar[f'ff_nf_tau{tau}_njets'], 0.5 * (ar.njets + 1))
                    self.assertTrue(ar[f'ff_nf_tau{tau}_njets_kept'].all())
                self.assertTrue(frame.loc[frame.SS, 'ff_nf_tau1_njets'].isna().all())
                np.testing.assert_allclose(frame.ff_dnn_tau1, 17.0)
        self.assertTrue(all('/training_resolved/' in path for path in checkpoints))


if __name__ == '__main__':
    unittest.main()
