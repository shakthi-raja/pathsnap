"""Indexed candidate search with bounded expansion and explicit global fallback."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
import pandas as pd
from shapely.geometry import Point

@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.
    max_candidates: int = 10
    max_radius_m: float = 200.

    def __post_init__(self):
        if (not np.isfinite([self.radius_m, self.max_radius_m]).all()
                or self.radius_m <= 0 or self.max_radius_m < self.radius_m
                or isinstance(self.max_candidates, bool)
                or not isinstance(self.max_candidates, Integral) or self.max_candidates < 1):
            raise ValueError('Invalid candidate configuration')

CANDIDATE_COLUMNS=['trip_id','gps_index','arc_id','candidate_rank','distance_m',
                   'projected_x','projected_y','projected_longitude','projected_latitude',
                   'position_m','position_fraction','search_radius_m','fallback_used']

def generate_candidates(track, network, trip_id, config=None):
    config=config or CandidateConfig()
    if config.radius_m<=0 or config.max_radius_m<config.radius_m or config.max_candidates<1:
        raise ValueError('Invalid candidate configuration')
    xy=network.project(track); rows=[]
    table=network.arcs
    for gps_index, coords in zip(track.index,xy):
        if not np.isfinite(coords).all(): continue
        point=Point(coords); radius=config.radius_m
        indices=network.tree.query(point,predicate='dwithin',distance=radius)
        while len(indices)==0 and radius<config.max_radius_m:
            radius=min(radius*2,config.max_radius_m)
            indices=network.tree.query(point,predicate='dwithin',distance=radius)
        fallback=radius>config.radius_m
        if len(indices)==0:
            # query_nearest returns ALL tied geometries, preserving reverse arcs.
            indices=network.tree.query_nearest(point,all_matches=True); fallback=True
        scored=sorted(((point.distance(network.geometries[j]),int(table['arc_id'].iloc[j]),int(j)) for j in indices))
        for rank,(distance,arc_id,j) in enumerate(scored[:config.max_candidates],1):
            arc=table.iloc[j]; length=arc.length_m
            position=float(network.geometries[j].project(point)) if length>0 else 0.
            projection=network.geometries[j].interpolate(position) if length>0 else network.geometries[j]
            lon,lat=network.inverse.transform(projection.x,projection.y)
            rows.append((str(trip_id),gps_index,arc_id,rank,distance,projection.x,projection.y,lon,lat,
                         position,position/length if length else 0.,radius,fallback))
    return pd.DataFrame(rows,columns=CANDIDATE_COLUMNS)
