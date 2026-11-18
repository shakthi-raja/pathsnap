"""Reproducible, one-network-at-a-time training/inference exports."""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import platform
import time
import traceback
import numpy as np
import pandas as pd
from .reader import load_trip, normalize_trip_id
from .network import build_road_network
from .candidates import generate_candidates, CandidateConfig
from .features import build_candidate_features, FEATURE_COLUMNS


def _is_student_test(folder):
    return 'student_test' in {part.casefold() for part in Path(folder).resolve().parts}


def build_inference_features(dataset_folder, trip_id, candidate_config=None):
    """Return candidate metadata + FEATURE_COLUMNS. Never opens a .route file."""
    track,nodes,arcs,_=load_trip(dataset_folder,trip_id,include_route=False)
    network=build_road_network(nodes,arcs)
    candidates=generate_candidates(track,network,str(trip_id).zfill(8),candidate_config)
    result=build_candidate_features(track,network,candidates)
    result.attrs['crs']=network.crs
    return result


def build_training_dataset(dataset_folder, trip_id, candidate_config=None, alignment_config=None, allow_reference=False):
    """Return (candidate rows with nullable labels, one-row-per-GPS alignment)."""
    if _is_student_test(dataset_folder) and not allow_reference:
        raise ValueError('student_test is inference-only; reference loading is prohibited')
    from .labels import align_reference_route, attach_labels
    track,nodes,arcs,route=load_trip(dataset_folder,trip_id)
    network=build_road_network(nodes,arcs)
    candidates=generate_candidates(track,network,str(trip_id).zfill(8),candidate_config)
    features=build_candidate_features(track,network,candidates)
    alignment=align_reference_route(track,network,route,alignment_config)
    result,alignment=attach_labels(features,alignment)
    result.attrs['crs']=network.crs
    return result,alignment


def peak_memory_mb():
    """OS process high-water working set, including native geometry/array memory."""
    if platform.system()=='Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
                ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        api=ctypes.WinDLL('psapi',use_last_error=True)
        api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize/1024**2
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024**2 if platform.system()=='Darwin' else 1024)


def process_trips(folder, output, trip_ids=None, *, inference=False, candidate_config=None, allow_reference=False):
    from .labels import AlignmentConfig
    import importlib.metadata
    folder=Path(folder).resolve(); output=Path(output).resolve()
    # Protect held-out data even when accidentally passed to the training CLI.
    if not inference and _is_student_test(folder) and not allow_reference:
        raise ValueError('student_test is inference-only; training exports are prohibited')
    protected_input=folder.parent if folder.parent.name.casefold()=='raw' else folder
    project_raw=Path(__file__).resolve().parents[2]/'data'/'raw'
    if (output.is_relative_to(protected_input) or protected_input.is_relative_to(output)
            or output.is_relative_to(project_raw)):
        raise ValueError('Output directory must not overlap raw input data')
    config=candidate_config or CandidateConfig()
    trip_ids=sorted(p.name for p in folder.iterdir() if p.is_dir()) if trip_ids is None else trip_ids
    trip_ids=[normalize_trip_id(tid) for tid in trip_ids]
    if not trip_ids or len(set(trip_ids))!=len(trip_ids):
        raise ValueError('Choose at least one trip without duplicate trip IDs')
    mode='inference' if inference else ('evaluation' if allow_reference else 'training')
    metadata={'schema_version':'1.0','dataset_folder':str(folder),'mode':mode,
              'feature_columns':FEATURE_COLUMNS,'candidate_config':asdict(config),'alignment_config':asdict(AlignmentConfig()),
              'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in
              ['numpy','pandas','scipy','shapely','pyproj','pyarrow']},'trips':{}}
    prior_path=output/'metadata.json'
    if prior_path.exists():
        prior=json.loads(prior_path.read_text(encoding='utf8'))
        if (prior.get('mode')!=metadata['mode']
                or Path(prior.get('dataset_folder','')).resolve()!=folder
                or prior.get('candidate_config')!=metadata['candidate_config']
                or prior.get('feature_columns')!=metadata['feature_columns']):
            raise ValueError('Output directory belongs to another dataset, mode or configuration; use a fresh directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output directory has no run metadata; use a fresh directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    summaries=[]
    for tid in trip_ids:
        tid=str(tid).zfill(8); start=time.perf_counter()
        try:
            if inference:
                result=build_inference_features(folder,tid,config); alignment=None
            else:
                result,alignment=build_training_dataset(folder,tid,config,allow_reference=allow_reference)
            result.to_parquet(output/f'{tid}.parquet',index=False)
            row=dict(trip_id=tid,status='success',gps_count=int(result.gps_index.nunique()),candidate_rows=len(result),
                     fallback_observations=int(result.loc[result.fallback_used,'gps_index'].nunique()))
            if alignment is not None:
                alignment.to_parquet(output/f'{tid}.alignment.parquet',index=False)
                confident=alignment.alignment_confident
                row.update(gps_count=len(alignment),confident_observations=int(confident.sum()),
                           uncertain_observations=int((~confident).sum()),
                           covered_confident_observations=int((confident&alignment.reference_in_candidates).sum()),
                           positive_labels=int((result.label==1).sum()),negative_labels=int((result.label==0).sum()),
                           excluded_candidate_rows=int(result.label.isna().sum()))
                row['candidate_recall']=row['covered_confident_observations']/row['confident_observations'] if row['confident_observations'] else None
            suffixes=['track','nodes','arcs']+([] if inference else ['route'])
            metadata['trips'][tid]={'crs':result.attrs['crs'],'raw_sha256':{suffix:hashlib.sha256((folder/tid/f'{tid}.{suffix}').read_bytes()).hexdigest() for suffix in suffixes}}
        except Exception as exc:
            row=dict(trip_id=tid,status='failed',error=f'{type(exc).__name__}: {exc}')
            (output/f'{tid}.error.txt').write_text(traceback.format_exc(),encoding='utf8')
        row.update(elapsed_s=round(time.perf_counter()-start,3),process_peak_memory_mb=round(peak_memory_mb(),2))
        summaries.append(row)
        pd.DataFrame(summaries).to_csv(output/'processing_summary.csv',index=False)
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)
        # No network/STRtree is retained between trips.
    return pd.DataFrame(summaries)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',default='data/raw/train'); parser.add_argument('--output',default='data/processed/train')
    parser.add_argument('--trip',action='append'); parser.add_argument('--inference',action='store_true')
    parser.add_argument('--radius',type=float,default=50.); parser.add_argument('--max-radius',type=float,default=200.)
    parser.add_argument('--candidates',type=int,default=10)
    parser.add_argument('--evaluation-export',action='store_true',
                        help='Allow labelled export of student_test for development evaluation only')
    args=parser.parse_args()
    summary=process_trips(args.folder,args.output,args.trip,inference=args.inference,
                          candidate_config=CandidateConfig(args.radius,args.candidates,args.max_radius),
                          allow_reference=args.evaluation_export)
    raise SystemExit(int((summary.status=='failed').any()))
