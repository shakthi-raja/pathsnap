"""Stable route-free numeric model schema. Angles are clockwise from north."""
import numpy as np

FEATURE_COLUMNS=[
    'distance_m','candidate_rank','position_fraction','position_m','length_m',
    'road_sin','road_cos','movement_m','heading_sin','heading_cos',
    'heading_difference_rad','dt_s','speed_mps','valid_time_gap',
    'stationary','zero_length','gaussian_emission_20m','log_emission_20m',
]

def angular_difference(a,b):
    return np.abs((np.asarray(a)-np.asarray(b)+np.pi)%(2*np.pi)-np.pi)

def build_candidate_features(track, network, candidates):
    """Previous-observation features only; no future data or reference inputs.

    Unknown heading/time/speed remain NaN. Consumers must fit imputation on the
    training split only. Identifier and absolute coordinates are metadata.
    """
    xy=network.project(track)
    delta=np.full_like(xy,np.nan); delta[1:]=np.diff(xy,axis=0)
    movement=np.hypot(delta[:,0],delta[:,1]); heading=np.mod(np.arctan2(delta[:,0],delta[:,1]),2*np.pi)
    heading[movement<1.]=np.nan
    dt=track.timestamp.diff().to_numpy(dtype=float)
    valid=np.isfinite(dt)&(dt>0)
    speed=np.divide(movement,dt,out=np.full(len(dt),np.nan),where=valid)
    observation=track.copy()
    for key,value in dict(movement_m=movement,gps_heading_rad=heading,dt_s=dt,speed_mps=speed,
                          valid_time_gap=valid.astype(float),stationary=(movement<1.).astype(float)).items():
        observation[key]=value
    columns=['movement_m','gps_heading_rad','dt_s','speed_mps','valid_time_gap','stationary']
    result=candidates.merge(observation[columns],left_on='gps_index',right_index=True,validate='many_to_one')
    result=result.merge(network.arcs[['arc_id','from_node_id','to_node_id','length_m','orientation_rad','start_x','start_y','end_x','end_y']].reset_index(drop=True),on='arc_id',validate='many_to_one')
    result['road_sin']=np.sin(result.orientation_rad); result['road_cos']=np.cos(result.orientation_rad)
    result['heading_sin']=np.sin(result.gps_heading_rad); result['heading_cos']=np.cos(result.gps_heading_rad)
    result['heading_difference_rad']=angular_difference(result.gps_heading_rad,result.orientation_rad)
    result['zero_length']=(result.length_m==0).astype(float)
    result['log_emission_20m']=-.5*(result.distance_m/20.)**2
    result['gaussian_emission_20m']=np.exp(result.log_emission_20m)
    result[FEATURE_COLUMNS]=result[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan)
    return result
