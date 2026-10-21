"""Rule-based research prototype; not a validated synthesis predictor."""
from pathlib import Path
import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title='RAFT Multiblock Designer', page_icon='🧪', layout='wide')
st.title('RAFT Multiblock Designer')
st.caption('Rule-based research prototype | Clothier et al., Angew. Chem. Int. Ed. (2024), DOI: 10.1002/anie.202320154')

# Approximate literature-style reference Tg values (degC). USER MUST VERIFY against
# preferred sources, tacticity, MW, humidity, and copolymer composition before use.
MONOMERS={
 'nBMA':('methacrylate',20.), 'iBMA':('methacrylate',53.),
 'tBMA':('methacrylate',107.), 'BzMA':('methacrylate',54.),
 'HMA':('methacrylate',-5.), 'EMA':('methacrylate',65.),
 'MMA':('methacrylate',105.), 'St':('styrene',100.),
 'MS':('styrene',105.), 'BuA':('acrylate',-54.),
 'tBuAm':('acrylamide',120.),
}
# Values are adjustable DEMONSTRATION assumptions, not data from the manuscript.
HALF_LIFE_H_80C={'KPS':8.0,'ACPA':2.5,'AIBN':1.3,'BPO':5.0}
DATA=Path(__file__).with_name('si_block_intervals.csv')
df=pd.read_csv(DATA)

if 'blocks' not in st.session_state:
    st.session_state.blocks=[{'monomer':'nBMA','DP':200}]

st.subheader('1 · Existing polymer architecture')
for i,b in enumerate(st.session_state.blocks):
    a,c,d=st.columns([3,2,1])
    b['monomer']=a.selectbox(f'Block {i+1}: monomer',list(MONOMERS),index=list(MONOMERS).index(b['monomer']),key=f'm{i}')
    b['DP']=c.selectbox(f'Block {i+1}: DP',[50,100,150,200,275],index=[50,100,150,200,275].index(b['DP']) if b['DP'] in [50,100,150,200,275] else 1,key=f'd{i}')
    if d.button('Remove',key=f'r{i}',disabled=len(st.session_state.blocks)==1):
        st.session_state.blocks.pop(i);st.rerun()
if st.button('＋ Add existing block'):
    st.session_state.blocks.append({'monomer':'nBMA','DP':100});st.rerun()

st.subheader('2 · Proposed extension')
a,b=st.columns(2)
new_m=a.selectbox('Next monomer',list(MONOMERS),index=list(MONOMERS).index('St'))
new_dp=b.selectbox('Target DP',[50,100,150,200],index=1)
with st.expander('Assumptions & model settings',expanded=False):
    st.info('Tg values and half-lives are editable illustrative defaults, not measurements or validated literature values.')
    tg_override={}
    for mon,(family,tg) in MONOMERS.items():
        tg_override[mon]=st.number_input(f'Tg of P{mon} (°C)',value=float(tg),step=1.,key=f'tg{mon}')
    st.write('Initiator half-lives at the chosen reaction temperature, supplied by user (h)')
    half={}
    for name,guess in HALF_LIFE_H_80C.items():
        half[name]=st.number_input(f'{name}: t½ (h)',min_value=0.01,value=guess,step=0.1,key=f'hl{name}')
    mix_fraction=st.number_input('Previous monomer fraction in new feed, when needed',0.0,0.9,0.10,0.01)
    old_limit=st.number_input('Maximum nominal total DP',min_value=100,value=1500,step=100)

if st.button('Predict next block',type='primary'):
    prior=st.session_state.blocks
    sequence=prior+[{'monomer':new_m,'DP':new_dp}]
    total_dp=sum(b['DP'] for b in sequence)
    weighted_tg=sum(tg_override[b['monomer']]*b['DP'] for b in sequence)/total_dp
    penalty=1+0.1*(total_dp//100-2) if total_dp>=300 else 1.0
    adjusted_tg=weighted_tg*penalty
    initiator='KPS' if adjusted_tg<70 else 'ACPA' if adjusted_tg<90 else 'AIBN' if adjusted_tg<=110 else 'BPO'

    # Qualitative RAFT-family heuristic. Not a universal reactivity hierarchy.
    prev_family=MONOMERS[prior[-1]['monomer']][0]
    next_family=MONOMERS[new_m][0]
    backwards=(prev_family in ('styrene','acrylate','acrylamide') and next_family=='methacrylate') or (prev_family=='acrylate' and next_family=='styrene')
    caution=[]
    if backwards:
        caution.append(f'Potentially unfavourable RAFT sequence ({prev_family} → {next_family}). Consider ~{mix_fraction:.0%} of prior monomer in the next feed as an experimental bridging strategy, not a guarantee.')
    if total_dp>old_limit:caution.append(f'Total DP {total_dp} exceeds your assumed instability threshold ({old_limit}). High risk; do not label feasible.')
    if new_m=='BuA':caution.append('Butyl acrylate: consider backbiting/branching; paper reduced temperature to 70 °C for the last chain extension.')
    if new_m=='tBuAm':caution.append('tBuAm was dissolved in ethanol in the Supporting Information; aqueous dispersion/solubility requires consideration.')
    if prior[-1]['monomer']==new_m:caution.append('Same-monomer extension does not create a chemically distinct new block.')
    if initiator=='BPO':caution.append('BPO threshold is your proposed extrapolation; BPO was not used as an initiator in the uploaded experimental tables.')

    # First use exact recorded scheduled intervals for this monomer and DP.
    ext=df.loc[(df.monomer==new_m)&(df.DP==new_dp)&(~df.includes_seed)]
    basis='observed scheduled interval (median for exact monomer/DP)'
    if len(ext):
        hours=float(ext.time_min.median())/60
    else:
        # Class/monomer typical scheduled interval normalized by DP; NOT kinetics.
        baseline=df.loc[(df.monomer==new_m)&(~df.includes_seed)]
        basis='DP-scaled scheduled interval (extrapolated; low confidence)'
        if baseline.empty:
            baseline=df.loc[df.monomer.map(lambda m:MONOMERS[m][0])==next_family]
            baseline=baseline.loc[~baseline.includes_seed]
            basis='family-level DP-scaled interval (extrapolated; very low confidence)'
        if baseline.empty:baseline=df.loc[~df.includes_seed];basis='pooled DP-scaled interval (very low confidence)'
        # Weighted median minutes per DP; limits imposed to avoid extreme unphysical extrapolation.
        hours=max(0.25,float((baseline.time_min/baseline.DP).median()*new_dp)/60)
    remaining=100*2**(-hours/half[initiator])

    st.subheader('3 · Recommendation')
    st.markdown(' **→** '.join(f"P{b['monomer']}({b['DP']})" for b in sequence))
    if total_dp>old_limit:st.error('High instability risk under the user-defined DP cutoff; feasibility not established.')
    elif backwards:st.warning('Conditionally plausible, but sequence requires experimental validation.')
    else:st.success('No hard rule violated. Feasibility remains unvalidated.')
    x,y,z=st.columns(3)
    x.metric('Total DP',total_dp)
    y.metric('DP-weighted Tg / adjusted Tg',f'{weighted_tg:.1f} / {adjusted_tg:.1f} °C')
    z.metric('Selected initiator',initiator)
    st.metric('Suggested reaction interval',f'{hours*60:.0f} min')
    st.caption('Basis: '+basis+'. Published intervals are scheduled sampling/addition periods, not fitted conversion kinetics.')
    st.metric('Initiator remaining at end (single shot)',f'{remaining:.1f}%')
    if remaining<10:st.warning('Less than 10% initial initiator predicted to remain: consider a supplementary shot. This is a heuristic; the amount and timing require kinetic modelling.')
    else:st.info('More than 10% of initial initiator predicted to remain; no extra shot triggered by this criterion.')
    for warning in caution:st.warning(warning)
    st.caption('Tg method: user-specified DP-weighted Celsius average; penalty applies to the FINAL total DP including proposed extension. This is NOT a measured effective latex Tg. Initiator depletion assumes constant isothermal first-order decay, with user-supplied half-life.')
    st.dataframe(df.loc[df.monomer==new_m,['table','block_index','monomer','DP','time_min']].reset_index(drop=True),hide_index=True)

st.divider()
st.caption('Reference: SI Tables 2–15 (automatically transcribed and checked for monomer, DP, time). Rows document scheduled step intervals. Check source PDF before planning experiments. This model does not account for conversion, end-group fidelity, phase partitioning, oxygen, particle sizes, mixing, or initiator efficiency.')
