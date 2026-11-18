"""Reference-occurrence sequence alignment; never imported by feature creation."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely import STRtree

@dataclass(frozen=True)
class AlignmentConfig:
    radius_m: float = 100.
    max_states: int = 48
    sigma_m: float = 20.
    confidence_distance_m: float = 35.
    ambiguity_margin: float = 2.
    backwards_tolerance_m: float = 5.
    gap_threshold_s: float = 30.

    def __post_init__(self):
        values=[self.radius_m,self.sigma_m,self.confidence_distance_m,self.ambiguity_margin,
                self.backwards_tolerance_m,self.gap_threshold_s]
        if (not np.isfinite(values).all() or min(values) < 0
                or self.radius_m == 0 or self.sigma_m == 0 or self.gap_threshold_s == 0
                or isinstance(self.max_states, bool) or not isinstance(self.max_states, Integral)
                or self.max_states < 2):
            raise ValueError('Invalid alignment configuration')

def align_reference_route(track, network, route, config=None):
    """Min-sum forward/backward alignment over ordered route occurrences.

    State = occurrence + projection onto that directed arc. Emission is squared
    distance/(2 sigma^2). Transitions penalise road-vs-GPS displacement, backwards
    jitter and implausible speed. Occurrence index cannot decrease; route progress
    can retreat at most 5 m for GPS jitter. Arbitrarily many arcs may be skipped.
    Path min-marginal differences detect competing occurrences (including loops).
    These are conservative heuristic confidence scores, not calibrated probability.
    A disconnected reference invalidates supervision rather than inventing a path.
    """
    cfg=config or AlignmentConfig()
    if cfg.max_states<2 or cfg.sigma_m<=0 or cfg.radius_m<=0: raise ValueError('Invalid alignment configuration')
    raw_ids=route.arc_id.to_numpy()
    if not np.isfinite(raw_ids).all() or not np.equal(raw_ids,np.floor(raw_ids)).all():
        raise ValueError('Reference arc IDs must be finite integers')
    ids=raw_ids.astype(np.int64)
    if len(ids)==0 or not np.isin(ids,network.arcs.arc_id).all(): raise ValueError('Empty/unknown reference arcs')
    arc=network.arcs.set_index('arc_id').loc[ids]
    breaks=int((arc.to_node_id.to_numpy()[:-1]!=arc.from_node_id.to_numpy()[1:]).sum())
    lengths=arc.length_m.to_numpy(); starts=np.r_[0,np.cumsum(lengths)[:-1]]
    positions=network.arcs.index.get_indexer(ids)
    # network.arcs retains original arc index, including custom nonconsecutive IDs.
    geoms=network.geometries[positions]; tree=STRtree(geoms)
    xy=network.project(track); states=[]; emissions=[]; capped=[]
    for coords in xy:
        if not np.isfinite(coords).all():
            raise ValueError('Cannot align invalid coordinates')
        point=Point(coords); occurrences=tree.query(point,predicate='dwithin',distance=cfg.radius_m)
        if not len(occurrences): occurrences=tree.query_nearest(point,all_matches=True)
        scored=sorted((point.distance(geoms[j]),int(j)) for j in occurrences)
        capped.append(len(scored)>cfg.max_states)
        values=[]
        for distance,j in scored[:cfg.max_states]:
            offset=geoms[j].project(point) if lengths[j]>0 else 0.
            values.append((j,starts[j]+offset,distance))
        states.append(np.array(values,dtype=float))
        emissions.append(.5*(states[-1][:,2]/cfg.sigma_m)**2)
    if not states: return pd.DataFrame(columns=['gps_index','aligned_arc_id','alignment_confident'])
    dt=track.timestamp.diff().to_numpy(dtype=float)
    moves=np.r_[0,np.hypot(*np.diff(xy,axis=0).T)]
    def transition(i):
        previous,current=states[i-1],states[i]
        ds=current[None,:,1]-previous[:,None,1]
        allowed=(current[None,:,0]>=previous[:,None,0])&(ds>=-cfg.backwards_tolerance_m)
        scale=max(20.,moves[i]*.5)
        cost=np.abs(ds-moves[i])/scale+np.maximum(-ds,0)/5.
        if np.isfinite(dt[i]) and dt[i]>0:
            cost+=np.maximum(ds-60*dt[i],0)/max(20.,20*dt[i])
        return np.where(allowed,cost,np.inf)
    # Start a new independently aligned block when no monotonic path is possible.
    forward=[emissions[0].copy()]; links=[]; resets=np.zeros(len(track),dtype=bool)
    for i in range(1,len(states)):
        score=forward[-1][:,None]+transition(i)
        link=np.argmin(score,axis=0); f=emissions[i]+np.min(score,axis=0)
        if not np.isfinite(f).any():
            resets[i]=True; f=emissions[i].copy(); link[:]=-1
        forward.append(f); links.append(link)
    backward=[np.zeros(len(s)) for s in states]
    for i in range(len(states)-2,-1,-1):
        if not resets[i+1]:
            backward[i]=np.min(transition(i+1)+emissions[i+1][None,:]+backward[i+1][None,:],axis=1)
    # Recover a single coherent best path per block, rather than independent MAP ties.
    chosen=np.zeros(len(states),dtype=int)
    for i in range(len(states)-1,-1,-1):
        if i==len(states)-1 or resets[i+1]: chosen[i]=np.argmin(forward[i])
        if i>0 and not resets[i]: chosen[i-1]=links[i-1][chosen[i]]
    rows=[]
    for i,(gps_index,s) in enumerate(zip(track.index,states)):
        k=chosen[i]; occurrence=int(s[k,0]); costs=forward[i]+backward[i]
        alternatives=np.delete(costs,k)
        margin=float(max(0.,np.min(alternatives)-costs[k])) if len(alternatives) else 1.e6
        distance=float(s[k,2]); reasons=[]
        if breaks: reasons.append('disconnected_reference')
        if capped[i]: reasons.append('state_cap')
        if distance>cfg.confidence_distance_m: reasons.append('distant')
        if margin<cfg.ambiguity_margin: reasons.append('ambiguous_occurrence')
        if resets[i] or (i+1<len(states) and resets[i+1]): reasons.append('alignment_reset')
        if i>0 and (not np.isfinite(dt[i]) or dt[i]<=0 or dt[i]>cfg.gap_threshold_s): reasons.append('time_gap')
        confidence=float(np.exp(-.5*(distance/cfg.sigma_m)**2)*(1-np.exp(-min(margin,700))))
        rows.append(dict(gps_index=gps_index,aligned_arc_id=int(ids[occurrence]),route_occurrence=occurrence,
                         route_progress_m=float(s[k,1]),alignment_distance_m=distance,alignment_margin=margin,
                         alignment_confidence=confidence,alignment_confident=not reasons,
                         alignment_status='confident' if not reasons else '|'.join(reasons)))
    return pd.DataFrame(rows)

def attach_labels(features, alignment):
    matched=features[['gps_index','arc_id']].merge(alignment[['gps_index','aligned_arc_id']],on='gps_index')
    covered=matched.loc[matched.arc_id==matched.aligned_arc_id,'gps_index'].unique()
    alignment=alignment.copy(); alignment['reference_in_candidates']=alignment.gps_index.isin(covered)
    result=features.merge(alignment,on='gps_index',validate='many_to_one')
    result['training_eligible']=result.alignment_confident & result.reference_in_candidates
    result['label']=(result.arc_id==result.aligned_arc_id).astype('Int8').where(result.training_eligible)
    return result,alignment
