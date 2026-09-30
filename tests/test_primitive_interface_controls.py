from __future__ import annotations

import importlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.registry import get_tool_registry


@pytest.fixture
def files(tmp_path):
    with workspace_scope(tmp_path):
        root = tmp_path / 'files'
        root.mkdir(exist_ok=True)
        yield root


def test_native_vasp_files_and_final_schemas(files):
    from catmaster.tools.geometry_inputs.vasp_prepare import vasp_prepare, VaspPrepareInput
    structure = Structure(Lattice.cubic(3), ['Fe'], [[0, 0, 0]])
    Poscar(structure).write_file(files / 'POSCAR')
    (files / 'pot').write_text('caller supplied POTCAR\n')
    kpoints = 'native shifted mesh\n0\nMonkhorst-Pack\n4 6 8\n0.5 0 0\n'
    (files / 'mesh').write_text(kpoints)
    _, artifact = vasp_prepare({'input_path': 'POSCAR', 'output_root': 'job', 'preset': 'static', 'regime': 'bulk', 'potcar_path': 'pot', 'kpoints_path': 'mesh'})
    assert (files / 'job/POTCAR').read_text() == 'caller supplied POTCAR\n'
    assert (files / 'job/KPOINTS').read_text() == kpoints
    assert not (files / 'job/POTCAR.spec').exists()
    for tool in get_tool_registry().as_openai_tools(allowlist=['vasp_prepare', 'vasp_neb_prepare', 'vasp_dimer_prepare', 'vasp_band_prepare']):
        assert tool['parameters']['properties']['potcar_functional']['default'] == 'PBE_54'
        assert 'potcar_functional' not in tool['parameters'].get('required', [])
        for key in ('potcar_path', 'potcar_settings', 'kpoints_path'):
            assert 'anyOf' not in tool['parameters']['properties'][key]
    assert VaspPrepareInput(input_path='POSCAR', output_root='other', preset='static', regime='bulk', potcar_settings=None).potcar_settings == {}
    assert VaspPrepareInput(input_path='POSCAR', output_root='other', preset='static', regime='bulk', potcar_settings={'Fe': 'Fe_pv'}).potcar_functional == 'PBE_54'


def test_mp_page_native_filters_and_cell_choice(files, monkeypatch):
    from catmaster.tools.retrieval import matdb
    captured = []
    class Client:
        materials = SimpleNamespace(summary=SimpleNamespace(count=lambda _: 5, search=lambda **kw: captured.append(kw) or [{'material_id': 'mp-3'}, {'material_id': 'mp-4'}]))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get_structure_by_material_id(self, _):
            return Structure(Lattice([[0, 2, 2], [2, 0, 2], [2, 2, 0]]), ['Cu'], [[0, 0, 0]])
    monkeypatch.setattr(matdb, '_mpr', lambda **kw: Client())
    _, result = matdb.mp_search_materials({'criteria': {'magnetic_ordering': 'FM'}, 'page': 2, 'limit': 2, 'fields': ['material_id'], 'output_csv': 'page.csv'})
    assert captured[0]['_page'] == 2 and captured[0]['magnetic_ordering'] == 'FM'
    assert result['data']['next_page'] == 3
    for cell, count in [('as_returned', 1), ('conventional', 4)]:
        _, result = matdb.mp_download_structure({'mp_ids': ['mp-1'], 'output_dir': cell, 'cell': cell})
        row = result['data']['results'][0]
        assert row['metadata']['natoms'] == count


def test_explicit_neb_pairs_preserve_good_case(files):
    from catmaster.tools.geometry_inputs.neb_tools import make_neb_geometry
    initial = Structure(Lattice.cubic(5), ['H', 'He'], [[0, 0, 0], [.4, .4, .4]])
    final = initial.copy()
    final.translate_sites([0], [.1, 0, 0])
    Poscar(initial).write_file(files / 'a.vasp')
    Poscar(final).write_file(files / 'b.vasp')
    content, artifact = make_neb_geometry({'endpoint_pairs': [{'initial_path': 'a.vasp', 'final_path': 'b.vasp'}, {'initial_path': 'missing.vasp', 'final_path': 'b.vasp'}], 'output_root': 'neb', 'n_images': 2, 'interp_method': 'linear'})
    assert artifact['data']['failed_count'] == 1
    assert (files / 'neb/case_0000/01.vasp').exists()
    assert 'failed' in content


def test_graph_handles_descriptions_and_paging(files):
    from catmaster.tools.misc import research_graph as graph
    first, a = graph.create_research_graph({'question': 'Which route?', 'initial_hypotheses': [{'claim': 'Route A is faster.'}]})
    initial_id = a['data']['changed']['initial_nodes'][0]['node_id']
    assert initial_id in first
    content, page = graph.list_research_graphs({'limit': 1})
    assert page['data']['matched_count'] == 1
    assert page['data']['graphs'][0]['graph_id'] in content
    for tool in get_tool_registry().as_openai_tools():
        if 'research' in tool['name'] or tool['name'].startswith('record_bound_'):
            assert all(value.get('description') for value in tool['parameters'].get('properties', {}).values()), tool['name']


def test_remote_batch_selection_and_all_preparation_errors(files, monkeypatch):
    remote = importlib.import_module('catmaster.tools.execution.remote_submission')
    (files / 'batch/nested/a').mkdir(parents=True)
    (files / 'batch/b').mkdir()
    assert remote._batch_stage_paths(files / 'batch', ['nested/a']) == [files / 'batch/nested/a']
    monkeypatch.setattr(remote, '_prepare_common', lambda _: (files / 'batch', 'x', None, None, '', None, 1, False, None))
    def bad(**kwargs): raise ValueError('bad ' + kwargs['stage_name'])
    monkeypatch.setattr(remote, '_build_task_spec', bad)
    monkeypatch.setattr(remote, '_submit', lambda **kw: pytest.fail('must not submit a failed preflight'))
    with pytest.raises(CatMasterToolExecutionError) as exc:
        remote.remote_submission_batch({'work_dir': 'batch', 'task_name': 'x', 'stage_paths': ['nested/a', 'b']})
    assert 'nested/a' in str(exc.value) and 'bad b' in str(exc.value)


def test_al_custom_descriptors_and_complete_scores(files):
    from catmaster.tools.machine_learning.mace_ml import calculate_al_candidates, _greedy_select
    frames = [Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]]) for _ in range(3)]
    write(files / 'pool.extxyz', frames)
    (files / 'features.json').write_text('[[0], [100], [1]]')
    _, result = calculate_al_candidates({'dataset_path': 'pool.extxyz', 'features_path': 'features.json', 'output_dir': 'al', 'selection_size': 1})
    output = json.loads((files / result['data']['summary_json_rel']).read_text())
    assert output['selected'][0]['candidate_index'] == 1
    assert len(output['all_candidates']) == 3
    # Compare O(ND) selection against a simple brute-force reference, including ties.
    features = np.random.default_rng(9).normal(size=(40, 6))
    expected = [int(np.argmax(np.linalg.norm(features - features.mean(axis=0), axis=1)))]
    while len(expected) < 8:
        scores = np.array([min(np.linalg.norm(row - features[j]) for j in expected) for row in features])
        scores[expected] = -np.inf
        expected.append(int(np.argmax(scores)))
    assert [row['candidate_index'] for row in _greedy_select(features, selection_size=8)] == expected


def test_dataset_export_retains_energy_forces_and_all_failure_details(files, monkeypatch):
    from catmaster.tools.machine_learning import dataset_tools as dataset
    atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-2., forces=np.ones((2, 3)), stress=np.zeros(6))
    (files / 'runs').mkdir()
    (files / 'runs/vasprun.xml').write_text('<modeling/>')
    monkeypatch.setattr(dataset, 'read_vasp_xml', lambda *a, **k: iter([atoms]))
    _, artifact = dataset.build_dataset_from_runs({'result_root': 'runs', 'output_dir': 'export', 'write_splits': False})
    exported = read(files / artifact['data']['dataset_rel'])
    assert exported.get_potential_energy() == -2.
    assert np.all(exported.get_forces() == 1)
    assert np.all(exported.arrays['REF_forces'] == 1)
    assert not (files / 'export/train.extxyz').exists()
    def broken(*a, **k): raise ValueError('broken XML')
    monkeypatch.setattr(dataset, 'read_vasp_xml', broken)
    with pytest.raises(CatMasterToolExecutionError) as exc:
        dataset.build_dataset_from_runs({'result_root': 'runs', 'output_dir': 'bad'})
    assert 'dataset_summary.json' in str(exc.value)
    assert 'broken XML' in (files / 'bad/dataset_summary.json').read_text()


def test_trajectory_thinning_unwraps_intervening_frames(files):
    from catmaster.tools.analysis.results_analysis import analyze_trajectory
    frames = [Atoms('Li', positions=[[x, 0, 0]], cell=[10, 10, 10], pbc=True) for x in [9, 3, 7]]
    write(files / 'motion.extxyz', frames)
    _, artifact = analyze_trajectory({'path': 'motion.extxyz', 'output_dir': 'md', 'coordinate_semantics': 'wrapped', 'frame_interval_fs': 1000, 'frame_stride': 2, 'fit_start_frame': 0, 'compute_rdf': False, 'make_plots': False})
    summary = json.loads((files / artifact['data']['summary_json_rel']).read_text())
    assert summary['final_msd_a2'] == pytest.approx(64.)
    assert summary['rdf_state'] == 'not_requested'
    assert not (files / 'md/trajectory_rdf.csv').exists()
    assert not (files / 'md/trajectory_msd.png').exists()
    assert '2.0' in (files / 'md/trajectory_msd.csv').read_text()


def test_rdf_only_does_not_require_diffusion_window(files):
    from catmaster.tools.analysis.results_analysis import analyze_trajectory
    write(files / 'single.extxyz', Atoms('He2', positions=[[0, 0, 0], [1, 0, 0]], cell=[8, 8, 8], pbc=True))
    _, artifact = analyze_trajectory({'path': 'single.extxyz', 'output_dir': 'rdf', 'coordinate_semantics': 'wrapped', 'frame_interval_fs': 1, 'compute_msd': False, 'make_plots': False})
    assert (files / 'rdf/trajectory_rdf.csv').exists()
    assert artifact['data']['diffusion_coefficient_a2_per_ps'] is None


def test_thermo_backend_and_frequency_policy(files, monkeypatch):
    from catmaster.tools.analysis import vaspkit_thermo as thermo
    (files / 'freq').mkdir()
    (files / 'freq/OUTCAR').write_text('1 f = 0.3 THz 1.9 2PiTHz 10 cm-1 1.24 meV\n2 f/i= 0.2 THz\n')
    monkeypatch.setattr(thermo, '_resolve_vaspkit_executable', lambda: None)
    with pytest.raises(ValueError, match='unavailable'):
        thermo.vaspkit_adsorbate_thermo_correction({'calculation_dir': 'freq', 'backend': 'vaspkit'})
    with pytest.raises(ValueError, match='imaginary'):
        thermo.vaspkit_adsorbate_thermo_correction({'calculation_dir': 'freq', 'backend': 'ase', 'imaginary_modes': 'error'})
    _, low = thermo.vaspkit_adsorbate_thermo_correction({'calculation_dir': 'freq', 'backend': 'ase', 'frequency_floor_cm1': 0})
    _, floored = thermo.vaspkit_adsorbate_thermo_correction({'calculation_dir': 'freq', 'backend': 'ase'})
    assert low['data']['e_zpe_ev'] < floored['data']['e_zpe_ev']


def test_corpus_modes_and_document_filter(files):
    from catmaster.runtime.literature.corpus import ingest_literature_files, query_literature_corpus
    (files / 'a.txt').write_text('alpha beta evidence')
    (files / 'b.txt').write_text('alpha gamma evidence')
    ingest_literature_files({'paths': ['a.txt', 'b.txt']})
    assert query_literature_corpus({'query': 'alpha beta', 'query_mode': 'any'})[1]['data']['total_count'] == 2
    assert query_literature_corpus({'query': 'alpha beta', 'query_mode': 'all'})[1]['data']['total_count'] == 1
    assert query_literature_corpus({'query': 'alpha AND gamma', 'query_mode': 'fts', 'source_paths': ['a.txt']})[1]['data']['total_count'] == 0


def test_acquisition_cache_title_and_batch_errors(files, monkeypatch):
    from catmaster.runtime.literature import acquisition as a
    a._reset_acquisition_cache_for_tests()
    calls = []
    for title in ['first title', 'different title', 'first title']:
        a._run_cached_acquisition(kind='doi', identifier='10.1234/test', expected_title=title, operation=lambda: calls.append(1) or ('ok', {'data': {'status': 'downloaded_pdf'}}))
    assert len(calls) == 2
    monkeypatch.setattr(a, '_require_scansci_version', lambda: None)
    seen = []
    monkeypatch.setattr(a, 'acquire_literature_source', lambda payload: seen.append(payload) or ('ok', {'data': {'status': 'cached_pdf', 'path': 'paper.pdf'}}))
    _, result = a.batch_acquire_literature_sources({'identifiers': ['nonsense', '10.1234/test'], 'expected_titles': {'10.1234/test': 'title'}})
    assert result['data']['status'] == 'partial'
    assert result['data']['status_counts']['invalid_request'] == 1
    assert seen[0]['expected_title'] == 'title'


def test_unicode_find_and_full_search_source(files, monkeypatch):
    from catmaster.runtime.literature import tools as web
    monkeypatch.setattr(web, '_literature_components', lambda: (None, None, None, None))
    monkeypatch.setattr(web, '_load_or_fetch_public_page', lambda **kw: (SimpleNamespace(model_dump=lambda **_: {'text': 'Straße target STRASSE', 'source_completeness': 'complete'}), 'page.json'))
    _, result = web.find_in_page({'source_path': 'page.json', 'pattern': 'target'})
    assert result['data']['result']['matches'][0]['start_char'] == 7
    _, result = web.find_in_page({'source_path': 'page.json', 'pattern': 'strasse'})
    assert [(m['start_char'], m['end_char']) for m in result['data']['result']['matches']] == [(0, 6), (14, 21)]
    original = 'long content ' * 1000
    content, result = web._web_search_result({'hits': [{'title': 'full', 'url': 'https://example.org', 'content': original}]}, max_results=1)
    assert result['data']['source_path'] in content
    assert original in (files / result['data']['source_path']).read_text()


def test_peer_reviews_survive_later_model_failure(files, monkeypatch):
    tool = importlib.import_module('catmaster.tools.analysis.peer_review_request')
    (files / 'paper.pdf').write_bytes(b'%PDF-mock')
    configs = [('one', SimpleNamespace(model='one')), ('two', SimpleNamespace(model='two'))]
    monkeypatch.setattr(tool, '_resolve_model_configs', lambda labels: configs if not labels else [row for row in configs if row[0] in labels])
    monkeypatch.setattr(tool.LLMProfile, 'from_env_or_file', lambda: SimpleNamespace())
    def build(cfg):
        if cfg.model == 'two': raise RuntimeError('provider unavailable')
        return SimpleNamespace(invoke=lambda _: SimpleNamespace(content='Scientific review retained.'))
    monkeypatch.setattr(tool, 'build_chat_model', build)
    content, result = tool.peer_review_request({'pdf_path': 'paper.pdf'})
    assert 'Scientific review retained.' in content and 'partial' in content
    saved = json.loads((files / result['data']['results_path']).read_text())
    assert saved['reviews'][0]['review_text'] == 'Scientific review retained.'
    assert saved['failures'][0]['model_label'] == 'two'


def test_failed_pdf_render_preserves_old_output(files, monkeypatch):
    tool = importlib.import_module('catmaster.tools.analysis.markdown_pdf')
    (files / 'doc.md').write_text('# Report')
    (files / 'doc.pdf').write_bytes(b'%PDF-original')
    monkeypatch.setattr(tool, '_resolve_executable', lambda env, names: Path(names[0]))
    monkeypatch.setattr(tool, '_resolve_font', lambda name: (name, name, '', False))
    def run(cmd, **kw):
        if cmd[0] == 'pandoc':
            Path(cmd[cmd.index('--output') + 1]).write_text('<html/>')
            return subprocess.CompletedProcess(cmd, 0, '', '')
        return subprocess.CompletedProcess(cmd, 1, '', 'render failed')
    monkeypatch.setattr(tool, '_run_command', run)
    with pytest.raises(CatMasterToolExecutionError): tool.render_markdown_pdf({'source_path': 'doc.md'})
    assert (files / 'doc.pdf').read_bytes() == b'%PDF-original'


def test_compile_engine_output_and_static_warning(files, monkeypatch):
    tool = importlib.import_module('catmaster.tools.analysis.agentic_compile_tex')
    (files / 'doc.tex').write_text('valid')
    monkeypatch.setattr(tool, '_static_diagnostics', lambda _: ([files / 'doc.tex'], ['heuristic warning']))
    captured = []
    def run(root, **kw):
        captured.append(kw)
        (kw['output_dir'] / 'doc.pdf').write_bytes(b'%PDF-test')
        return {'available': True, 'ok': True, 'name': kw['engine'], 'stdout': 'full log'}
    monkeypatch.setattr(tool, '_run_compiler', run)
    content, result = tool.compile_text({'source_path': 'doc.tex', 'output_dir': 'build', 'engine': 'xelatex', 'bibliography_tool': 'biber'})
    assert captured[0]['engine'] == 'xelatex' and captured[0]['bibliography_tool'] == 'biber'
    assert result['data']['compiled_ok'] is True
    assert result['data']['diagnostics_path'] in content
    assert 'full log' in (files / result['data']['diagnostics_path']).read_text()


def test_citation_destination_and_collision(files, monkeypatch):
    from catmaster.runtime.literature import citations
    monkeypatch.setattr(citations, '_resolve', lambda _: (None, 'missing'))
    citations.finalize_citations({'items': ['10.1234/abc'], 'output_path': 'article/refs.bib'})
    with pytest.raises(FileExistsError):
        citations.finalize_citations({'items': ['10.1234/abc'], 'output_path': 'article/refs.bib'})
    assert (files / 'article/refs.bib').is_file()


def test_aider_exact_file_allowance_and_sibling_exclusion():
    from catmaster.tools.misc.memory_patch_apply import _normalize_allowed_prefixes, _normalize_rel_path
    prefixes = _normalize_allowed_prefixes(['notes/a.md', 'data'])
    assert _normalize_rel_path('notes/a.md', prefixes) == 'notes/a.md'
    assert _normalize_rel_path('data/b.txt', prefixes) == 'data/b.txt'
    with pytest.raises(ValueError): _normalize_rel_path('notes/a.md.backup', prefixes)


def test_parallel_adsorption_index_updates_do_not_lose_members(files):
    from catmaster.tools.geometry_inputs.adsorbate_tool import _write_ads_indices_index
    index = files / 'ads_indices.json'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: _write_ads_indices_index(index, [{'output_poscar_rel': f'{i}.vasp', 'ads_indices': [i]}]), range(40)))
    rows = json.loads(index.read_text())['entries']
    assert len(rows) == 40
    assert {row['ads_indices'][0] for row in rows} == set(range(40))


def test_explicit_adsorption_coordinate_does_not_enumerate(files, monkeypatch):
    from catmaster.tools.geometry_inputs import adsorbate_tool as ads
    Poscar(Structure(Lattice.cubic(10), ['Cu'], [[0, 0, .3]])).write_file(files / 'slab.vasp')
    (files / 'H.xyz').write_text('1\nH\nH 0 0 0\n')
    monkeypatch.setattr(ads, 'enumerate_adsorption_sites_domain', lambda *a, **kw: pytest.fail('coordinate placement must not enumerate sites'))
    ads.place_adsorbate({'slab_file': 'slab.vasp', 'adsorbate_file': 'H.xyz', 'site_cart_coords': [0, 0, 5], 'output_poscar': 'ads.vasp'})
    assert len(Structure.from_file(files / 'ads.vasp')) == 2


def test_phonopy_matrix_and_native_metadata(files):
    pytest.importorskip('phonopy')
    from catmaster.tools.geometry_inputs.crystal_tool import generate_phonon_displacements
    source = Structure(Lattice.cubic(4), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
    Poscar(source).write_file(files / 'cell.vasp')
    matrix = [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    _, artifact = generate_phonon_displacements({'structure_file': 'cell.vasp', 'output_dir': 'phonons', 'supercell_matrix': matrix, 'backend': 'phonopy'})
    metadata = json.loads((files / artifact['data']['metadata_rel']).read_text())
    reference = Structure.from_file(files / metadata['reference_structure_rel'])
    assert np.allclose(reference.lattice.matrix, np.array(matrix) @ source.lattice.matrix)
    assert (files / metadata['phonopy_yaml_rel']).is_file()
    assert artifact['data']['structures_generated'] > 0


def test_combined_interstitials_and_fresh_output(files):
    from catmaster.tools.geometry_inputs.crystal_tool import insert_interstitial_at_coords
    Poscar(Structure(Lattice.cubic(5), ['Na'], [[0, 0, 0]])).write_file(files / 'base.vasp')
    insert_interstitial_at_coords({'structure_file': 'base.vasp', 'species': 'H', 'coords': [[.2, .2, .2], [.7, .7, .7]], 'mode': 'combined', 'output_path': 'combined.vasp'})
    assert len(Structure.from_file(files / 'combined.vasp')) == 3
    batch = {'structure_file': 'base.vasp', 'species': 'H', 'coords': [[.2, .2, .2], [.7, .7, .7]], 'output_dir': 'independent'}
    insert_interstitial_at_coords(batch)
    generated = {p.name: p.read_bytes() for p in (files / 'independent').iterdir()}
    with pytest.raises(CatMasterToolExecutionError):
        insert_interstitial_at_coords(batch)
    assert {p.name: p.read_bytes() for p in (files / 'independent').iterdir()} == generated


def test_new_controls_reach_structured_tool_schema(files):
    keys = {'generate_phonon_displacements': ['supercell_matrix'], 'vasp_prepare': ['potcar_settings', 'potcar_path', 'kpoints_path'],
            'remote_submission_batch': ['stage_paths'], 'make_neb_geometry': ['endpoint_pairs'],
            'query_literature_corpus': ['source_paths'], 'peer_review_request': ['model_labels']}
    registry = get_tool_registry()
    for tool in registry.as_langchain_tools(allowlist=list(keys), workspace=str(files.parent)):
        schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
        properties = schema['properties']
        for key in keys[tool.name]:
            assert 'anyOf' not in properties[key], (tool.name, key)
            assert properties[key].get('default', 'not-null') is not None
