from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from src.data.reader import load_trip
from src.data.network import build_road_network
from src.data.candidates import generate_candidates, CandidateConfig
from src.data.features import build_candidate_features, FEATURE_COLUMNS, angular_difference
from src.data.labels import align_reference_route, attach_labels, AlignmentConfig
from src.data.pipeline import build_inference_features, build_training_dataset, process_trips


def network_fixture():
    # Equatorial ~111 m edges, coincident forward duplicate and reverse arc.
    nodes=pd.DataFrame({'longitude':[0,.001,.002,.001],'latitude':[0,0,0,.001]})
    arcs=pd.DataFrame({'from_node_id':[0,1,0,1,1,3],'to_node_id':[1,0,1,2,3,1]},index=[10,11,12,13,14,15])
    arcs.index.name='arc_id'
    return nodes,arcs,build_road_network(nodes,arcs)

def track_at(lons,lats=None,times=None):
    t=pd.DataFrame({'longitude':lons,'latitude':lats if lats is not None else np.zeros(len(lons)),
                    'timestamp':times if times is not None else np.arange(len(lons))+1.})
    t.index.name='gps_index'
    return t

def save_trip(root, track=None, include_route=True):
    tid='00000001'; folder=root/tid; folder.mkdir(parents=True)
    (track if track is not None else track_at([.0002,.0007,.0013,.0018])).to_csv(folder/f'{tid}.track',sep=' ',header=False,index=False)
    nodes=pd.DataFrame([[0,0],[.001,0],[.002,0]])
    nodes.to_csv(folder/f'{tid}.nodes',sep=' ',header=False,index=False)
    pd.DataFrame([[0,1],[1,2]]).to_csv(folder/f'{tid}.arcs',sep=' ',header=False,index=False)
    if include_route:
        (folder/f'{tid}.route').write_text('0\n1\n')
    return folder

def test_reader_and_id_schema(tmp_path):
    save_trip(tmp_path)
    track,nodes,arcs,route=load_trip(tmp_path,1)
    assert track.index.name=='gps_index' and arcs.index.name=='arc_id'
    assert route.arc_id.tolist()==[0,1] and len(nodes)==3

@pytest.mark.parametrize('suffix,content',[
    ('track','181 0 1\n'),('track','0 nan 1\n'),('nodes','0 95\n'),
    ('arcs','0 9\n'),('arcs','0 1.5\n'),('route','9\n'),('route','0.5\n')])
def test_invalid_input_rejected(tmp_path,suffix,content):
    folder=save_trip(tmp_path); (folder/f'00000001.{suffix}').write_text(content)
    with pytest.raises(ValueError): load_trip(tmp_path,1)

def test_projection_distance_and_roundtrip():
    nodes,arcs,network=network_fixture()
    length=Geod(ellps='WGS84').inv(0,0,.001,0)[2]
    assert network.arcs.loc[10,'length_m']==pytest.approx(length,rel=1.e-5)
    xy=network.project(track_at([.0004])); lon,lat=network.inverse.transform(*xy[0])
    assert lon==pytest.approx(.0004,abs=1.e-9) and lat==pytest.approx(0,abs=1.e-9)
    assert network.adjacency[0,1]==pytest.approx(length,rel=1.e-5)
    assert network.adjacency[2,1]==0  # no implicit reverse connection

@pytest.mark.parametrize('longitude,latitude',[(8.6,49.8),(179.9998,0),(10,85)])
def test_local_projection_in_different_locations(longitude,latitude):
    lon2=(longitude+.001+180)%360-180
    nodes=pd.DataFrame({'longitude':[longitude,lon2],'latitude':[latitude,latitude]})
    network=build_road_network(nodes,pd.DataFrame({'from_node_id':[0],'to_node_id':[1]}))
    expected=Geod(ellps='WGS84').inv(longitude,latitude,lon2,latitude)[2]
    assert network.arcs.length_m.iloc[0]==pytest.approx(expected,rel=1.e-5)

def test_processing_failure_is_recorded(tmp_path):
    folder=save_trip(tmp_path/'raw')
    (folder/'00000001.arcs').write_text('0 99\n')
    output=tmp_path/'processed'
    summary=process_trips(tmp_path/'raw',output)
    assert summary.status.tolist()==['failed']
    assert 'Invalid node ID' in summary.error.iloc[0]
    assert (output/'00000001.error.txt').exists()
    assert pd.read_csv(output/'processing_summary.csv').status.tolist()==['failed']

def test_directed_ids_duplicates_and_deterministic_candidates():
    _,_,network=network_fixture(); track=track_at([.0004],[.0001])
    a=generate_candidates(track,network,'x'); b=generate_candidates(track,network,'x')
    pd.testing.assert_frame_equal(a,b)
    assert a.arc_id.tolist()[:3]==[10,11,12]
    assert a.distance_m.iloc[0]==pytest.approx(11.0574,abs=.02)
    assert a.projected_longitude.iloc[0]==pytest.approx(.0004,abs=1.e-8)
    assert network.arcs.loc[10,'orientation_rad']==pytest.approx(np.pi/2,abs=1.e-7)
    assert network.arcs.loc[11,'orientation_rad']==pytest.approx(3*np.pi/2,abs=1.e-7)
    assert a.candidate_rank.tolist()==list(range(1,len(a)+1))

def test_zero_length_segment():
    nodes=pd.DataFrame({'longitude':[0,0],'latitude':[0,0]})
    arcs=pd.DataFrame({'from_node_id':[0],'to_node_id':[1]})
    network=build_road_network(nodes,arcs); track=track_at([0])
    c=generate_candidates(track,network,'x'); f=build_candidate_features(track,network,c)
    assert f.zero_length.iloc[0]==1 and f.position_fraction.iloc[0]==0
    assert np.isnan(f.road_sin.iloc[0]) and f.distance_m.iloc[0]==0

def test_fallback_far_outside_radius():
    _,_,network=network_fixture()
    c=generate_candidates(track_at([.05]),network,'x',CandidateConfig(1,5,2))
    assert len(c)>0 and c.fallback_used.all() and c.distance_m.min()>2

def test_empty_candidate_observations():
    _,_,network=network_fixture()
    c=generate_candidates(track_at([np.nan]),network,'x')
    assert c.empty

@pytest.mark.parametrize('a,b,expected',[(359,1,2),(0,180,180),(90,90,0)])
def test_angle_wrap(a,b,expected):
    assert angular_difference(np.deg2rad(a),np.deg2rad(b))==pytest.approx(np.deg2rad(expected))

def test_heading_and_invalid_time_gaps():
    _,_,network=network_fixture(); track=track_at([.0001,.0002,.0003,.0003],times=[2,2,1,4])
    c=generate_candidates(track,network,'x'); f=build_candidate_features(track,network,c)
    for i in [0,1,2]: assert f.loc[f.gps_index==i,'speed_mps'].isna().all()
    assert f.loc[f.gps_index==1,'heading_sin'].iloc[0]==pytest.approx(1,abs=1.e-7)
    assert f.loc[f.gps_index==3,'heading_sin'].isna().all()
    assert not np.isinf(f[FEATURE_COLUMNS].to_numpy()).any()

def test_known_route_alignment_and_multiple_observations():
    _,_,network=network_fixture(); track=track_at([.0001,.0004,.0013,.0018])
    route=pd.DataFrame({'arc_id':[10,13]})
    a=align_reference_route(track,network,route)
    assert a.aligned_arc_id.tolist()==[10,10,13,13]
    assert a.alignment_confident.all()
    c=generate_candidates(track,network,'x'); f=build_candidate_features(track,network,c)
    labelled,_=attach_labels(f,a)
    assert (labelled.label==1).sum()==4
    # Coincident arc 12 is NOT positive just because it has matching geometry.
    assert (labelled.loc[labelled.arc_id==12,'label']==0).all()

def test_repeated_arcs_and_route_loop():
    _,_,network=network_fixture()
    track=track_at([.0002,.0008,.001,.0008,.0002,0,.0002,.0008,.0015],times=np.arange(9)*3)
    a=align_reference_route(track,network,pd.DataFrame({'arc_id':[10,11,10,13]}))
    # At turn endpoints either adjacent occurrence has the same projection.
    assert a.route_occurrence.iloc[[0,1,3,4,6,7,8]].tolist()==[0,0,1,1,2,2,3]
    assert (a.route_occurrence.diff().dropna()>=0).all()

@pytest.mark.parametrize('content',['0 0 1 2\n','0 0\n','a 0 1\n'])
def test_bad_track_schema(tmp_path,content):
    folder=save_trip(tmp_path); (folder/'00000001.track').write_text(content)
    with pytest.raises(ValueError): load_trip(tmp_path,1)

def test_route_can_skip_intermediate_arcs():
    nodes=pd.DataFrame({'longitude':[0,.001,.002,.003],'latitude':[0,0,0,0]})
    arcs=pd.DataFrame({'from_node_id':[0,1,2],'to_node_id':[1,2,3]})
    network=build_road_network(nodes,arcs)
    a=align_reference_route(track_at([.0002,.0028],times=[0,10]),network,pd.DataFrame({'arc_id':[0,1,2]}))
    assert a.aligned_arc_id.tolist()==[0,2] and a.alignment_confident.all()

def test_state_cap_is_explicitly_uncertain():
    _,_,network=network_fixture()
    a=align_reference_route(track_at([.0005]),network,pd.DataFrame({'arc_id':[10,11,10,11,10]}),AlignmentConfig(max_states=2))
    assert a.alignment_status.str.contains('state_cap').all()

def test_invalid_candidates_config():
    _,_,network=network_fixture()
    with pytest.raises(ValueError): generate_candidates(track_at([0]),network,'x',CandidateConfig(0))

def test_ambiguous_repeated_occurrence_excluded():
    _,_,network=network_fixture(); track=track_at([.0004])
    a=align_reference_route(track,network,pd.DataFrame({'arc_id':[10,11,10]}))
    assert not a.alignment_confident.any()
    assert a.alignment_status.str.contains('ambiguous').all()

def test_missing_reference_candidate_not_negative():
    _,_,network=network_fixture(); track=track_at([.0004])
    a=align_reference_route(track,network,pd.DataFrame({'arc_id':[12]}))
    c=generate_candidates(track,network,'x',CandidateConfig(max_candidates=1))
    f=build_candidate_features(track,network,c); labelled,a=attach_labels(f,a)
    assert not a.reference_in_candidates.any() and labelled.label.isna().all()

def test_disconnected_route_excluded():
    _,_,network=network_fixture()
    a=align_reference_route(track_at([.0004,.0014]),network,pd.DataFrame({'arc_id':[10,10,13]}))
    assert not a.alignment_confident.any()
    assert a.alignment_status.str.contains('disconnected').all()

def test_long_gap_flagged():
    _,_,network=network_fixture()
    a=align_reference_route(track_at([.0004,.0014],times=[0,100]),network,pd.DataFrame({'arc_id':[10,13]}))
    assert 'time_gap' in a.alignment_status.iloc[1]

def test_no_route_leakage_and_matching_feature_schema(tmp_path,monkeypatch):
    from src.data import reader
    folder=save_trip(tmp_path)
    training,_=build_training_dataset(tmp_path,1)
    original=reader._read_table
    opened=[]
    def deny_reference(path,columns):
        assert path.suffix!='.route', 'Inference attempted to open a reference route'
        opened.append(path.suffix)
        return original(path,columns)
    monkeypatch.setattr(reader,'_read_table',deny_reference)
    inference=build_inference_features(tmp_path,1)
    assert opened==['.track','.nodes','.arcs']
    assert len(FEATURE_COLUMNS)==18 and len(set(FEATURE_COLUMNS))==18
    pd.testing.assert_frame_equal(training[FEATURE_COLUMNS],inference[FEATURE_COLUMNS])
    assert training[FEATURE_COLUMNS].dtypes.equals(inference[FEATURE_COLUMNS].dtypes)
    assert not {'label','route_occurrence','aligned_arc_id'} & set(inference.columns)

def test_inference_without_any_reference_file(tmp_path):
    save_trip(tmp_path,include_route=False)
    inference=build_inference_features(tmp_path,1)
    assert not inference.empty and 'label' not in inference

@pytest.mark.parametrize('trip_id',['../00000001','..','-1','123456789','1.5',''])
def test_trip_id_cannot_escape_dataset(tmp_path,trip_id):
    with pytest.raises(ValueError,match='one to eight digits'):
        load_trip(tmp_path,trip_id)

@pytest.mark.parametrize('output_name',['raw','raw/exports','.'])
def test_output_cannot_overlap_inputs(tmp_path,output_name):
    root=tmp_path/'raw'; save_trip(root)
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='overlap raw input'):
        process_trips(root,tmp_path/output_name)
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}

@pytest.mark.parametrize('change',['mode','dataset','configuration'])
def test_output_cannot_mix_runs(tmp_path,change):
    root=tmp_path/'raw'; save_trip(root); output=tmp_path/'processed'
    process_trips(root,output)
    before=(output/'metadata.json').read_bytes()
    kwargs={}; folder=root
    if change=='mode': kwargs['inference']=True
    if change=='dataset': folder=tmp_path/'other'; save_trip(folder)
    if change=='configuration': kwargs['candidate_config']=CandidateConfig(max_candidates=5)
    with pytest.raises(ValueError,match='another dataset, mode or configuration'):
        process_trips(folder,output,**kwargs)
    assert (output/'metadata.json').read_bytes()==before

def test_empty_dataset_rejected_before_writes(tmp_path):
    root=tmp_path/'empty'; root.mkdir(); output=tmp_path/'processed'
    with pytest.raises(ValueError,match='at least one trip'): process_trips(root,output)
    assert not output.exists()

def test_student_test_guard_is_case_insensitive(tmp_path):
    root=tmp_path/'STUDENT_TEST'; save_trip(root)
    with pytest.raises(ValueError,match='inference-only'): build_training_dataset(root,1)
    with pytest.raises(ValueError,match='inference-only'): process_trips(root,tmp_path/'output')

@pytest.mark.parametrize('config',[
    {'radius_m':np.nan},{'max_radius_m':np.inf},{'max_candidates':1.5},{'max_candidates':True}])
def test_invalid_candidate_configuration_rejected_early(config):
    with pytest.raises(ValueError,match='Invalid candidate configuration'): CandidateConfig(**config)

@pytest.mark.parametrize('config',[
    {'radius_m':np.nan},{'max_states':2.5},{'ambiguity_margin':-1},{'sigma_m':np.inf}])
def test_invalid_alignment_configuration_rejected_early(config):
    with pytest.raises(ValueError,match='Invalid alignment configuration'): AlignmentConfig(**config)

def test_direct_alignment_rejects_fractional_arc_ids():
    _,_,network=network_fixture()
    with pytest.raises(ValueError,match='finite integers'):
        align_reference_route(track_at([.0004]),network,pd.DataFrame({'arc_id':[10.5]}))

def test_candidate_large_arc_id_precision_preserved():
    nodes,arcs,_=network_fixture()
    arcs.index=arcs.index+2**53
    network=build_road_network(nodes,arcs)
    candidates=generate_candidates(track_at([.0004]),network,'x')
    assert set(candidates.arc_id)==set(arcs.index[:3])

def test_held_out_training_guard(tmp_path):
    root=tmp_path/'student_test'; save_trip(root)
    with pytest.raises(ValueError,match='inference-only'): process_trips(root,tmp_path/'output')
    with pytest.raises(ValueError,match='inference-only'): build_training_dataset(root,1)

@pytest.fixture
def real_training_root():
    from src.config import ConfigurationError, get_settings

    try:
        return get_settings().data_root / "train"
    except ConfigurationError as exc:
        pytest.skip(f"Raw training data not available: {exc}")


@pytest.mark.real
def test_real_training_inference_schema_and_values(real_training_root):
    if not (real_training_root / "00000001").exists():
        pytest.skip("Raw training trip 00000001 not available")
    training,_=build_training_dataset(real_training_root,1)
    inference=build_inference_features(real_training_root,1)
    keys=['trip_id','gps_index','arc_id']
    pd.testing.assert_frame_equal(training[keys+FEATURE_COLUMNS],inference[keys+FEATURE_COLUMNS])

@pytest.mark.real
@pytest.mark.parametrize('tid',['00000001','00000002','00000003'])
def test_real_trip_end_to_end(tid, real_training_root):
    if not (real_training_root / tid).exists(): pytest.skip(f'Raw training trip {tid} not available')
    track,nodes,arcs,route=load_trip(real_training_root,tid)
    if tid=='00000001': assert tuple(map(len,[track,nodes,arcs,route]))==(1315,88809,165790,525)
    frame,alignment=build_training_dataset(real_training_root,tid)
    assert frame.gps_index.nunique()==len(track)==len(alignment)
    assert not frame.duplicated(['trip_id','gps_index','arc_id']).any()
    assert frame.arc_id.isin(arcs.index).all()
    assert frame.label.isna().equals(~frame.training_eligible)
    assert frame[FEATURE_COLUMNS].columns.tolist()==FEATURE_COLUMNS
