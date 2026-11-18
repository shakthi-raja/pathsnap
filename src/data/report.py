"""Verify saved integration contracts and produce auditable aggregate reports."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
import pandas as pd
from .features import FEATURE_COLUMNS


def create_report(train='data/processed/train', inference='data/processed/student_test_inference', output='results'):
    train=Path(train); inference=Path(inference); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(train/'processing_summary.csv',dtype={'trip_id':str})
    run_metadata=json.loads((train/'metadata.json').read_text(encoding='utf8'))
    expected_trips={p.name for p in Path(run_metadata['dataset_folder']).iterdir() if p.is_dir()}
    assert set(summary.trip_id)==expected_trips, 'Training run is incomplete or has unexpected trip IDs'
    assert summary.trip_id.is_unique
    assert run_metadata['feature_columns']==FEATURE_COLUMNS
    audit=pd.read_csv(output/'exploration/trip_summary.csv',dtype={'trip_id':str})
    reasons=Counter(); ranks=Counter(); verified=0; warnings=[]
    confident_total=0
    expected_columns=None
    for row in summary.itertuples():
        if row.status!='success': continue
        frame=pd.read_parquet(train/f'{row.trip_id}.parquet')
        alignment=pd.read_parquet(train/f'{row.trip_id}.alignment.parquet')
        assert len(frame)==row.candidate_rows and len(alignment)==row.gps_count
        assert frame.gps_index.nunique()==len(alignment)
        assert not frame.duplicated(['trip_id','gps_index','arc_id']).any()
        assert not np.isinf(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).any()
        assert frame.label.isna().equals(~frame.training_eligible)
        assert ((frame.loc[frame.label.notna(),'label']==1)==(frame.loc[frame.label.notna(),'arc_id']==frame.loc[frame.label.notna(),'aligned_arc_id'])).all()
        positive=frame.loc[frame.label==1].groupby('gps_index').size()
        assert (positive==1).all()
        assert len(positive)==row.positive_labels
        if expected_columns is None: expected_columns=list(frame.columns)
        assert list(frame.columns)==expected_columns
        confident_total+=int(alignment.alignment_confident.sum())
        matched=frame.loc[frame.alignment_confident&(frame.arc_id==frame.aligned_arc_id)]
        for k in [1,5,8,10]: ranks[k]+=int(matched.loc[matched.candidate_rank<=k,'gps_index'].nunique())
        for status,count in alignment.alignment_status.value_counts().items(): reasons[status]+=int(count)
        verified+=1
    held=pd.read_csv(inference/'processing_summary.csv',dtype={'trip_id':str})
    held_metadata=json.loads((inference/'metadata.json').read_text(encoding='utf8'))
    assert set(held.trip_id)=={p.name for p in Path(held_metadata['dataset_folder']).iterdir() if p.is_dir()}
    assert held_metadata['feature_columns']==FEATURE_COLUMNS
    for row in held.itertuples():
        assert row.status=='success'
        frame=pd.read_parquet(inference/f'{row.trip_id}.parquet')
        assert not {'label','aligned_arc_id','route_occurrence','alignment_confidence'} & set(frame.columns)
        assert set(held_metadata['trips'][row.trip_id]['raw_sha256'])=={'track','nodes','arcs'}
        assert all(c in frame for c in FEATURE_COLUMNS)
        assert len(frame)==row.candidate_rows
    pd.DataFrame([dict(max_candidates=k,covered_confident_observations=v,
                       confident_observations=confident_total,recall=v/confident_total if confident_total else None)
                  for k,v in sorted(ranks.items())]).to_csv(output/'candidate_coverage.csv',index=False)
    pd.DataFrame([dict(status=k,observations=v) for k,v in reasons.most_common()]).to_csv(output/'alignment_status_counts.csv',index=False)
    sample=pd.read_parquet(train/f'{summary.loc[summary.status=="success","trip_id"].iloc[0]}.parquet')
    pd.DataFrame([dict(column=c,dtype=str(sample[c].dtype),model_feature=c in FEATURE_COLUMNS,
                       missing_rows_in_sample=int(sample[c].isna().sum())) for c in sample]).to_csv(output/'output_schema.csv',index=False)
    totals={c:int(summary[c].sum()) for c in ['gps_count','candidate_rows','confident_observations','uncertain_observations',
            'covered_confident_observations','positive_labels','negative_labels','excluded_candidate_rows','fallback_observations']}
    totals.update(successful_trips=int((summary.status=='success').sum()),failed_trips=int((summary.status!='success').sum()),
                  contract_verified_training_trips=verified,contract_verified_inference_trips=len(held),
                  inference_gps_count=int(held.gps_count.sum()),inference_candidate_rows=int(held.candidate_rows.sum()),
                  candidate_recall=totals['covered_confident_observations']/confident_total,
                  processing_seconds=float(summary.elapsed_s.sum()),peak_process_memory_mb=float(summary.process_peak_memory_mb.max()),
                  trips_without_confident_labels=int((summary.confident_observations==0).sum()))
    for flag in ['ambiguous_occurrence','disconnected_reference','alignment_reset','state_cap','time_gap']:
        totals[flag+'_observations']=sum(count for status,count in reasons.items() if flag in status.split('|'))
    import xml.etree.ElementTree as ET
    suites=ET.parse(output/'test-results.xml').getroot().findall('testsuite')
    totals['tests_passed']=sum(int(s.attrib['tests'])-int(s.attrib['failures'])-int(s.attrib['errors'])-int(s.attrib.get('skipped',0)) for s in suites)
    totals['tests_failed']=sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
    (output/'completion_metrics.json').write_text(json.dumps(totals,indent=2),encoding='utf8')
    sources=sorted(Path('src/data').glob('*.py'))+sorted(Path('tests').glob('*.py'))+[Path('requirements-data.txt'),Path('pytest.ini')]
    (output/'source_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf8')
    quality_cols=['gps_missing_values','gps_invalid_coordinates','nodes_missing_values','nodes_invalid_coordinates',
                  'invalid_node_references','invalid_route_references','duplicate_gps_rows','repeated_gps_coordinates',
                  'negative_time_gaps','repeated_timestamps','time_gaps_over_30s','speeds_over_60mps',
                  'duplicate_directed_connections','self_loop_arcs','route_connectivity_breaks']
    quality={c:int(audit[c].sum()) for c in quality_cols}
    (output/'quality_totals.json').write_text(json.dumps(quality,indent=2),encoding='utf8')
    anomaly_columns=['gps_invalid_coordinates','nodes_invalid_coordinates','invalid_node_references','invalid_route_references',
                     'duplicate_gps_rows','negative_time_gaps','time_gaps_over_30s','speeds_over_60mps','route_connectivity_breaks']
    anomalies=audit.loc[(audit[anomaly_columns]>0).any(axis=1)].copy()
    anomalies.to_csv(output/'unusual_trips.csv',index=False)
    coverage=pd.read_csv(output/'candidate_coverage.csv')
    coverage_lines='\n'.join(f'- Top {int(r.max_candidates)}: {r.recall:.4%} ({int(r.covered_confident_observations):,}/{confident_total:,}).' for r in coverage.itertuples())
    no_labels=', '.join(summary.loc[summary.confident_observations==0,'trip_id'])
    weak=summary.loc[summary.confident_observations>0].nsmallest(5,'candidate_recall')
    weak_lines='\n'.join(f'- {r.trip_id}: {r.candidate_recall:.2%}.' for r in weak.itertuples())
    report=f'''# Data engineering completion report

Verified workspace: `D:\\ML_HACKATHON\\pathsnap`. Branch: `feature/data-engineering`. No commit, push or merge was performed. Raw inputs were read only; student-test references were not read. Installed and used pytest in the existing Python 3.11 virtual environment.

## Executed results

- Automated suite: **{totals['tests_passed']} passed**, {totals['tests_failed']} failed; `results/test-results.xml`.
- Training export: **{totals['successful_trips']}/80 trips**, {totals['failed_trips']} failures; **{totals['gps_count']:,} GPS observations**, **{totals['candidate_rows']:,} candidate rows**.
- Label alignment: {totals['confident_observations']:,} confident, **{totals['uncertain_observations']:,} uncertain** observations.
- Ambiguous-occurrence flag: {totals['ambiguous_occurrence_observations']:,} observations; disconnected-reference flag: {totals['disconnected_reference_observations']:,}; alignment-reset flag: {totals['alignment_reset_observations']:,}. Flags can overlap and are not additive.
- Supervised labels: **{totals['positive_labels']:,} positive**, **{totals['negative_labels']:,} negative**, **{totals['excluded_candidate_rows']:,} null/excluded candidate rows**.
- Confident observations missing their aligned arc: {confident_total-totals['covered_confident_observations']:,}; all their candidate labels are null.
- Expanded/global search: {totals['fallback_observations']:,} observations.
- Held-out inference export: **{len(held)}/10 trips**, **{totals['inference_gps_count']:,} observations**, **{totals['inference_candidate_rows']:,} candidates**, no reference inputs or labels.
- Persisted-file integration contracts verified across {verified} training and {len(held)} inference trips.
- Sum of training trip processing times: {totals['processing_seconds']:.3f} seconds; process high-water memory: **{totals['peak_process_memory_mb']:.2f} MiB**, including native array/geometry allocations. This is one measured run, not a cross-platform benchmark.

## Candidate coverage

Weighted coverage over confident aligned observations only; uncertain observations are excluded from the denominator. This measures retrieval, not ground-truth alignment accuracy.

{coverage_lines}

Ten candidates improve retrieval but cannot guarantee coverage in dense/parallel/duplicate roads. Lowest nonempty per-trip coverage:

{weak_lines}

## Quality findings and implications

The 80-trip audit found {quality['gps_invalid_coordinates']} invalid GPS coordinates, {quality['nodes_invalid_coordinates']} invalid node coordinates, {quality['invalid_node_references']} invalid node references, {quality['invalid_route_references']} invalid route references and {quality['duplicate_gps_rows']} exact duplicate GPS rows. Missing GPS/node values total {quality['gps_missing_values']+quality['nodes_missing_values']}.

There are {quality['repeated_gps_coordinates']} repeated GPS coordinate pairs, {quality['duplicate_directed_connections']} duplicate directed road connections, {quality['self_loop_arcs']} self-loop arcs, {quality['time_gaps_over_30s']} time gaps over 30 seconds, and {quality['speeds_over_60mps']} approximate GPS speeds over 60 m/s. There are {quality['negative_time_gaps']} negative gaps and {quality['repeated_timestamps']} repeated timestamps. Maximum timestamp gap: {audit.max_time_gap_s.max():,.0f} seconds. Weak component counts range from {int(audit.weak_components.min())} to {int(audit.weak_components.max())}; use directed graph reachability in HMM transitions instead of assuming all nearby roads connect.

Reference connectivity breaks total {quality['route_connectivity_breaks']} across {int((audit.route_connectivity_breaks>0).sum())} trips. {totals['trips_without_confident_labels']} trips have no confident supervision: {no_labels}. Features remain available; do not train on their null labels. Trip 00000054 is a major fallback/outlier case (see per-trip summary). Long gaps should trigger cautious HMM handling; repeated locations can have undefined headings. Duplicates/reverse arcs remain distinct candidates.

## Files and outputs

Modified: `.gitignore`, `src/data/reader.py`, `exploration.py`, `candidates.py`, `features.py`, `labels.py`. Created: `src/data/network.py`, `pipeline.py`, `report.py`, `tests/test_data_pipeline.py`, `pytest.ini`, `requirements-data.txt`, `docs/DATA_ENGINEERING.md`, `docs/INTEGRATION_HANDOFF.md`, this report. Existing data-package init was retained. All work remains uncommitted.

Primary outputs:

- `data/processed/train/<trip_id>.parquet`: candidate metadata, 18 model features, nullable labels and alignment diagnostics.
- `data/processed/train/<trip_id>.alignment.parquet`: one alignment row per GPS observation.
- `data/processed/student_test_inference/<trip_id>.parquet`: same candidate features without route/label fields.
- Both output directories: `metadata.json` (versions, configs, input SHA256, per-trip CRS), `processing_summary.csv` (all statuses/counts/timings/memory).
- `results/exploration/trip_summary.csv` and `.parquet`, four trajectory plots, and `distributions.png`.
- `results/candidate_coverage.csv`, `alignment_status_counts.csv`, `output_schema.csv`, `unusual_trips.csv`, `quality_totals.json`, `completion_metrics.json`, `test-results.xml`, `source_sha256.json`, `environment-lock.txt`.

Raw data, virtual environments, results and processed Parquet remain Git-ignored. Final datasets are in `train/`; `smoke/` contains development runs and is not the final handoff.

## Model feature schema

Ordered model columns: {', '.join('`'+c+'`' for c in FEATURE_COLUMNS)}.

Feature definitions and units are in `docs/DATA_ENGINEERING.md`; exact saved column names/dtypes are in `results/output_schema.csv`. Arc IDs, observation IDs, trip IDs, absolute coordinates, route progress and labels are excluded from model predictors. Headings are clockwise from north; distance is metres, time seconds and speed metres/second. First-observation/invalid-time/stationary/zero-length cases retain documented NaNs, to be imputed using training-only fitted preprocessing.

## Teammate integration

Anuvidya: read only 8-digit training candidate files, filter `training_eligible`, select the exact ordered `FEATURE_COLUMNS`, split by trip before fitting preprocessing/model, and keep the three-key candidate identity. Do not treat null labels as negative or use student-test data in fitting.

Kruthika: call `build_inference_features` or the shared network/candidate/feature functions; apply the training model and preprocessing to the same columns. Group emissions by GPS index and use directed graph distances plus candidate offsets for HMM transitions. No route files are required. Road node row positions map through `network.node_ids`.

Shakthi Raja: use `projected_longitude`/`projected_latitude` for map points and join selected candidates by `(trip_id, gps_index, arc_id)`. Segment endpoints are also available in each trip's metric CRS. Do not mix projected coordinates between trips without a common transform. See `docs/INTEGRATION_HANDOFF.md`.

## Actual principal commands executed

```powershell
Get-Location
git branch --show-current
git status --short
.\\.venv\\Scripts\\python.exe -m pip install pytest
.\\.venv\\Scripts\\python.exe -m src.data.exploration
.\\.venv\\Scripts\\python.exe -m src.data.pipeline --trip 00000001 --output data/processed/smoke
.\\.venv\\Scripts\\python.exe -m src.data.pipeline --trip 00000001 --trip 00000002 --trip 00000003 --output data/processed/smoke
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=results/test-results.xml
.\\.venv\\Scripts\\python.exe -u -m src.data.pipeline --output data/processed/train
.\\.venv\\Scripts\\python.exe -u -m src.data.pipeline --folder data/raw/student_test --output data/processed/student_test_inference --inference
.\\.venv\\Scripts\\python.exe -m pip freeze | Set-Content -Encoding utf8 results/environment-lock.txt
.\\.venv\\Scripts\\python.exe -m compileall -q src tests
.\\.venv\\Scripts\\python.exe -m src.data.report
```

Additional commands inspected raw samples, versions, statistics and Git ignore rules. A development smoke run exposed an arc-index/column pandas merge ambiguity, which was fixed. The first test run was 27 passed/1 failed: a loop fixture repeated stationary coordinates at a turn and incorrectly required a unique occurrence; the corrected fixture traverses both turn endpoints and checks known interior occurrences. Subsequent suites passed 34, 35 and finally {totals['tests_passed']} tests. Pip initially failed under restricted DNS; the approved retry installed pytest. The matplotlib cache now stays under project results.

## Readiness and remaining limits

The data-engineering interfaces and exports are ready for teammate integration, with verified schemas and explicit exclusion policies. They do not establish final model/HMM accuracy. Labels are conservative heuristic sequence alignment, not manually verified point truth. Shared endpoints, overlapping route occurrences, state/search truncation and long gaps can remain ambiguous. Disconnected routes conservatively lose all supervision. Candidate retrieval misses some confident reference arcs; inspect per-trip recall. Road shape is approximated by supplied node-to-node segments; the local projection rejects networks extending over 500 km from its centre. Confidence is not calibrated. A trained classifier and HMM decoder are outside this assignment.

Official hackathon instructions were not present locally. Permitted reference use, timestamp semantics, official metrics, split rules, licensing and submission format remain unverified and require organiser confirmation. Student-test references were neither evaluated nor used for fitting. No claim of official compliance is made.
'''
    (output/'COMPLETION_REPORT.md').write_text(report,encoding='utf8')
    Path('docs/COMPLETION_REPORT.md').write_text(report,encoding='utf8')
    print(json.dumps(totals,indent=2))
    return totals

if __name__=='__main__': create_report()
