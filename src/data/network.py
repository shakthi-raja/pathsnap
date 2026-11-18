"""Directed, ID-preserving local metric road geometry (Shapely >=2)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pyproj import CRS, Transformer
from scipy.sparse import coo_matrix

@dataclass
class RoadNetwork:
    arcs: pd.DataFrame
    geometries: np.ndarray
    tree: STRtree
    transformer: Transformer
    inverse: Transformer
    crs: str
    adjacency: object
    node_ids: np.ndarray

    def project(self, track):
        x,y=self.transformer.transform(track.longitude.to_numpy(),track.latitude.to_numpy())
        return np.column_stack([x,y])


def build_road_network(nodes, arcs):
    """IDs come from input indexes; sparse adjacency uses node row positions.

    Use a local azimuthal equidistant CRS, centred on the node extent. Suitable
    for these local road networks, including UTM-boundary/polar locations.
    Zero-length, coincident and opposite directed arcs are retained.
    """
    if not nodes.index.is_unique or not arcs.index.is_unique or nodes.empty or arcs.empty:
        raise ValueError('Node and arc IDs must be unique and network nonempty')
    if not (nodes.longitude.between(-180,180).all() and nodes.latitude.between(-90,90).all()):
        raise ValueError('Invalid node coordinates')
    if not np.isfinite(nodes.to_numpy(dtype=float)).all(): raise ValueError('Nonfinite nodes')
    refs=arcs[['from_node_id','to_node_id']].to_numpy()
    if not np.isfinite(refs).all() or not np.equal(refs,np.floor(refs)).all(): raise ValueError('Invalid node references')
    indices=nodes.index.get_indexer(refs.ravel()).reshape(-1,2)
    if (indices<0).any(): raise ValueError('Unknown node reference')
    # Circular mean handles the antimeridian.
    longitude=np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(nodes.longitude)))))
    latitude=float(nodes.latitude.mean())
    crs=CRS.from_proj4(f'+proj=aeqd +lat_0={latitude} +lon_0={longitude} +datum=WGS84 +units=m')
    forward=Transformer.from_crs(4326,crs,always_xy=True)
    inverse=Transformer.from_crs(crs,4326,always_xy=True)
    x,y=forward.transform(nodes.longitude.to_numpy(),nodes.latitude.to_numpy())
    xy=np.column_stack([x,y]); ends=xy[indices]
    if not np.isfinite(ends).all() or np.max(np.hypot(x,y))>500_000:
        raise ValueError('Network exceeds supported local projection extent (500 km)')
    geometries=shapely.linestrings(ends)
    lengths=shapely.length(geometries)
    # Shapely considers a collapsed LineString invalid; retain it as a Point.
    zero=lengths==0
    geometries[zero]=shapely.points(ends[zero,0,:])
    if not shapely.is_valid(geometries).all(): raise ValueError('Invalid road geometry')
    table=arcs.copy(); table['arc_id']=arcs.index
    for col, values in zip(['start_x','start_y','end_x','end_y'],[ends[:,0,0],ends[:,0,1],ends[:,1,0],ends[:,1,1]]): table[col]=values
    table['length_m']=lengths
    delta=ends[:,1]-ends[:,0]
    table['orientation_rad']=np.where(zero,np.nan,np.mod(np.arctan2(delta[:,0],delta[:,1]),2*np.pi))
    adjacency=coo_matrix((lengths,(indices[:,0],indices[:,1])),shape=(len(nodes),len(nodes))).tocsr()
    # Parallel arcs must not have summed traversal weights.
    if pd.DataFrame(indices).duplicated().any():
        edge=pd.DataFrame({'u':indices[:,0],'v':indices[:,1],'w':lengths}).groupby(['u','v'],as_index=False).w.min()
        adjacency=coo_matrix((edge.w,(edge.u,edge.v)),shape=(len(nodes),len(nodes))).tocsr()
    return RoadNetwork(table,geometries,STRtree(geometries),forward,inverse,crs.to_string(),adjacency,nodes.index.to_numpy())
