"""Trip-by-trip data quality audit. Run python -m src.data.exploration."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pyproj import Geod
from .reader import load_trip


def inspect_trip(folder, trip_id):
    track, nodes, arcs, route = load_trip(folder, trip_id, validate=False)
    result = dict(trip_id=trip_id, gps_count=len(track), node_count=len(nodes),
                  arc_count=len(arcs), route_count=len(route))
    for name, frame in [('gps', track), ('nodes', nodes)]:
        result[name + '_missing_values'] = int(frame.isna().sum().sum())
        result[name + '_invalid_coordinates'] = int((~frame.longitude.between(-180,180) |
                                                   ~frame.latitude.between(-90,90)).sum())
    valid = arcs.ge(0).all(axis=1) & arcs.lt(len(nodes)).all(axis=1)
    valid &= (arcs == np.floor(arcs)).all(axis=1)
    result['invalid_node_references'] = int((~valid).sum())
    result['invalid_route_references'] = int((~route.arc_id.isin(arcs.index)).sum())
    result['duplicate_gps_rows'] = int(track.duplicated().sum())
    result['repeated_gps_coordinates'] = int(track.duplicated(['longitude','latitude']).sum())
    dt = track.timestamp.diff().to_numpy()
    result['negative_time_gaps'] = int((dt < 0).sum())
    result['repeated_timestamps'] = int(track.timestamp.duplicated().sum())
    result['time_gaps_over_30s'] = int((dt > 30).sum())
    result['max_time_gap_s'] = float(np.nanmax(dt)) if len(dt)>1 else 0.
    lon, lat = track.longitude.to_numpy(), track.latitude.to_numpy()
    _, _, distance = Geod(ellps='WGS84').inv(lon[:-1],lat[:-1],lon[1:],lat[1:])
    speed = np.divide(distance, dt[1:], out=np.full_like(distance,np.nan), where=dt[1:]>0)
    result['median_spacing_m'] = float(np.nanmedian(distance)) if len(distance) else 0.
    result['max_spacing_m'] = float(np.nanmax(distance)) if len(distance) else 0.
    result['max_speed_mps'] = float(np.nanmax(speed)) if np.isfinite(speed).any() else None
    result['speeds_over_60mps'] = int((speed>60).sum())
    a = arcs.loc[valid].to_numpy(dtype=int)
    graph = coo_matrix((np.ones(len(a),dtype=np.int8),(a[:,0],a[:,1])),shape=(len(nodes),len(nodes))).tocsr()
    nc, components = connected_components(graph,directed=True,connection='weak')
    result['weak_components'] = int(nc)
    result['strong_components'] = int(connected_components(graph,directed=True,connection='strong',return_labels=False))
    result['largest_weak_component_nodes'] = int(np.bincount(components).max())
    result['isolated_nodes'] = int(((graph.getnnz(axis=0)+graph.getnnz(axis=1))==0).sum())
    result['self_loop_arcs'] = int((a[:,0]==a[:,1]).sum())
    result['duplicate_directed_connections'] = int(arcs.duplicated().sum())
    if result['invalid_route_references']==0:
        ra = arcs.loc[route.arc_id].to_numpy()
        result['route_connectivity_breaks'] = int((ra[:-1,1]!=ra[1:,0]).sum())
    return result, track


def explore(folder='data/raw/train', output='results/exploration'):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(Path(output).resolve()/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i, path in enumerate(sorted(Path(folder).iterdir())):
        if not path.is_dir(): continue
        try:
            row, track=inspect_trip(folder,path.name)
            row['status']='success'
            if i<4:
                fig, ax=plt.subplots(figsize=(7,5))
                ax.plot(track.longitude,track.latitude,linewidth=.6)
                ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title=f'Trip {path.name}')
                ax.set_aspect(1/np.cos(np.deg2rad(track.latitude.mean())))
                fig.tight_layout(); fig.savefig(output/f'trajectory_{path.name}.png',dpi=140); plt.close(fig)
        except Exception as exc:
            row=dict(trip_id=path.name,status='failed',error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'trip_summary.csv',index=False); frame.to_parquet(output/'trip_summary.parquet',index=False)
    fig, axes=plt.subplots(2,2,figsize=(10,7))
    for ax, column in zip(axes.flat,['gps_count','median_spacing_m','max_time_gap_s','weak_components']):
        values=frame[column].dropna()
        if column=='max_time_gap_s':
            ax.hist(values,bins=np.geomspace(max(1.,values.min()),max(2.,values.max()),20)); ax.set_xscale('log')
        else:
            ax.hist(values,bins=20)
        label={'gps_count':'GPS observations per trip','median_spacing_m':'Median GPS spacing (m)',
               'max_time_gap_s':'Maximum timestamp gap (s, log scale)','weak_components':'Weakly connected components'}[column]
        ax.set(xlabel=label,ylabel='Trips')
    fig.tight_layout(); fig.savefig(output/'distributions.png',dpi=140); plt.close(fig)
    print(frame[['trip_id','status']].to_string(index=False))
    return frame

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--folder',default='data/raw/train'); parser.add_argument('--output',default='results/exploration')
    args=parser.parse_args(); summary=explore(args.folder,args.output)
    raise SystemExit(int((summary.status=='failed').any()))
