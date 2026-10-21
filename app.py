"""Research prototype: transparent heuristics; never a validated synthesis guarantee."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title='RAFT Multiblock Designer v0.2',page_icon='🧪',layout='wide')
st.title('RAFT Multiblock Designer · v0.2')
st.caption('Exploratory rules + SI schedule regression · Clothier et al., Angew. Chem. Int. Ed. 2024 · DOI: 10.1002/anie.202320154')

# Approximate editable reference Tg placeholders (°C), NOT manuscript measurements.
MONOMERS={'nBMA':('methacrylate',20.),'iBMA':('methacrylate',53.),'tBMA':('methacrylate',107.),'BzMA':('methacrylate',54.),'HMA':('methacrylate',-5.),'EMA':('methacrylate',65.),'MMA':('methacrylate',105.),'St':('styrene',100.),'MS':('styrene',105.),'BuA':('acrylate',-54.),'tBuAm':('acrylamide',120.)}
DP_CHOICES=[50,100,150,200,300,400]
# Deliberately placeholders. Replace t_half at T_ref and E_a with verified supplier data.
KINETICS={'KPS':(8.,110.),'ACPA':(2.5,125.),'AIBN':(1.3,130.),'BPO':(5.,120.)}  # hours at 80 °C, kJ mol^-1
R=8.314462618
DATA=Path(__file__).with_name('si_block_intervals.csv')
raw=pd.read_csv(DATA)
raw['includes_seed']=raw['includes_seed'].astype(str).str.lower().eq('true')


def half_life_at_temp(tref_hours, ea_kj, temp_c, ref_c=80.):
    """Arrhenius extrapolation from verified/reference half-life at ref_c."""
    return tref_hours*math.exp((ea_kj*1000/R)*(1/(temp_c+273.15)-1/(ref_c+273.15)))


def fraction_remaining(half_h, elapsed_h):
    return 2**(-elapsed_h/half_h)


def predict_schedule(df, monomer, dp, block_idx):
    """Ridge model of log SCHEDULED INTERVAL, not experimentally observed conversion time."""
    train=df.loc[~df.includes_seed].copy()
    train=train.loc[train.monomer.isin(MONOMERS) & train.DP.gt(0) & train.time_min.gt(0)]
    train['family']=train.monomer.map(lambda m: MONOMERS[m][0]);train['log_dp']=np.log(train.DP)
    x=train[['monomer','family','log_dp','block_index']]
    model=Pipeline([('prep',ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['monomer','family']),('num',StandardScaler(),['log_dp','block_index'])])),('ridge',Ridge(alpha=8.0))])
    model.fit(x,np.log(train.time_min))
    query=pd.DataFrame([{'monomer':monomer,'family':MONOMERS[monomer][0],'log_dp':math.log(dp),'block_index':block_idx}])
    predicted=float(np.exp(model.predict(query)[0]))
    exact=train.loc[(train.monomer==monomer)&(train.DP==dp)]
    # Always show exact SI scheduling observations alongside fitted result, never substitute for kinetic measurements.
    return max(10.,predicted), len(exact), (float(exact.time_min.median()) if len(exact) else None), len(train)


def weighted_tg(blocks,tgs):
    n=sum(b['DP'] for b in blocks)
    return sum(tgs[b['monomer']]*b['DP'] for b in blocks)/n


def dp_penalty(total_dp):
    return 1.0 if total_dp<300 else 1+0.1*(total_dp//100-2)


def decay_shots(shots, final_h, half_lives):
    """Concentrations expressed in equivalent mol per mol RAFT; ignores withdrawal/dilution."""
    rows=[]
    for s in shots:
        remaining=s['equivalents']*fraction_remaining(half_lives[s['initiator']],max(0.,final_h-s['time_h']))
        rows.append({**s,'remaining_equiv_per_RAFT':remaining,'remaining_pct_of_shot':100*remaining/s['equivalents']})
    return pd.DataFrame(rows)

if 'blocks' not in st.session_state:st.session_state.blocks=[{'monomer':'nBMA','DP':200}]

st.subheader('1 · Existing architecture (seed included)')
for i,b in enumerate(st.session_state.blocks):
    c1,c2,c3=st.columns([3,2,1])
    b['monomer']=c1.selectbox(f'Block {i+1}',list(MONOMERS),index=list(MONOMERS).index(b['monomer']),key=f'm{i}')
    b['DP']=c2.selectbox(f'DP {i+1}',DP_CHOICES,index=DP_CHOICES.index(b['DP']) if b['DP'] in DP_CHOICES else 1,key=f'd{i}')
    if c3.button('Remove',key=f'remove{i}',disabled=len(st.session_state.blocks)<=1):
        st.session_state.blocks.pop(i);st.rerun()
if st.button('＋ Add existing block'):
    st.session_state.blocks.append({'monomer':'nBMA','DP':100});st.rerun()

st.subheader('2 · Next block and reaction conditions')
a,b,c=st.columns(3)
new_m=a.selectbox('Proposed monomer',list(MONOMERS),index=list(MONOMERS).index('St'))
new_dp=b.selectbox('Target DP',DP_CHOICES,index=1)
temp_c=c.slider('Reaction temperature (°C)',min_value=70,max_value=100,value=80,step=1)

st.subheader('3 · Initial initiator loading / legacy shots')
st.caption('Default new shot: [RAFT]/[I] = 20, i.e. 0.05 mol initiator per mol RAFT, as in the SI. This is **not** 5 wt%, nor the amount already present from earlier blocks. Existing initiator inventory defaults to zero until entered.')
c1,c2,c3=st.columns(3)
raft_to_initiator=c1.number_input('New shot [RAFT]/[initiator] molar ratio',min_value=0.1,max_value=1000.,value=20.,step=1.)
legacy_name=c2.selectbox('Initiator previously used',list(KINETICS))
legacy_equiv=c3.number_input('Existing undecomposed initiator (mol per mol RAFT)',min_value=0.,max_value=10.,value=0.,step=0.005,format='%.4f')
initial_equiv=1./raft_to_initiator

with st.expander('Adjust references, screening choices and kinetics (important)',expanded=False):
    st.warning('Tg and kinetic constants below are demonstration placeholders, not verified manufacturer or publication constants. Review against data sheets before laboratory use. Arrhenius extrapolation is only valid to the extent the chosen parameters represent the decomposition mechanism.')
    tgs={m:st.number_input(f'P{m} Tg (°C)',value=float(v[1]),step=1.,key=f'tg_{m}') for m,v in MONOMERS.items()}
    st.markdown('**Temperature-dependent decomposition:** enter half-life at 80 °C and decomposition activation energy for each initiator.')
    params={}
    for name,(th,ea) in KINETICS.items():
        k1,k2=st.columns(2)
        half_ref=k1.number_input(f'{name}: t½ at 80 °C (hours)',min_value=0.01,value=th,step=0.1,key=f'hl{name}')
        act=k2.number_input(f'{name}: Ea (kJ/mol)',min_value=1.,value=ea,step=5.,key=f'ea{name}')
        params[name]=(half_ref,act)
    tg_basis=st.selectbox('Initiator Tg screening metric',['Recent two blocks Tg (local proxy)','Overall DP-weighted Tg'])
    recent_n=st.number_input('Recent-block window',min_value=1,max_value=10,value=2,step=1)
    max_dp=st.number_input('Unstable above total DP',min_value=100,value=1500,step=100)
    previous_fraction=st.number_input('Bridge: previous monomer fraction',0.,0.9,0.10,0.01)
    min_half_count=st.number_input('Maximum reaction duration in initiator half-lives before reject (no redosing)',min_value=0.1,max_value=20.,value=3.32193,step=0.1)
    target_remain_pct=st.number_input('Trigger another shot if new shot remaining falls below (%)',min_value=0.1,max_value=90.,value=10.,step=1.)

if st.button('Predict next block',type='primary'):
    prior=list(st.session_state.blocks)
    chain=prior+[{'monomer':new_m,'DP':new_dp}]
    total_dp=sum(x['DP'] for x in chain)
    overall_tg=weighted_tg(chain,tgs)
    recent_tg=weighted_tg(chain[-int(recent_n):],tgs)
    penalty=dp_penalty(total_dp)
    basis_tg=recent_tg if tg_basis.startswith('Recent') else overall_tg
    penalized=basis_tg*penalty
    target='KPS' if penalized<70 else 'ACPA' if penalized<90 else 'AIBN' if penalized<=110 else 'BPO'
    time_min,n_exact,exact_median,n_train=predict_schedule(raw,new_m,new_dp,len(chain))
    duration_h=time_min/60
    hl={k:half_life_at_temp(*params[k],temp_c) for k in KINETICS}
    half_count={k:duration_h/hl[k] for k in KINETICS}
    valid={k:half_count[k]<=min_half_count for k in KINETICS}
    # No forced switch to unsuitable Tg initiator. Flag mismatch explicitly.
    chosen=target if valid[target] else None
    old_family=MONOMERS[prior[-1]['monomer']][0];new_family=MONOMERS[new_m][0]
    backwards=(old_family in ('styrene','acrylate','acrylamide') and new_family=='methacrylate') or (old_family=='acrylate' and new_family=='styrene')
    st.subheader('4 · Assessment')
    st.write(' → '.join(f"P{x['monomer']}({x['DP']})" for x in chain))
    if total_dp>max_dp:st.error(f'User-defined total DP instability cutoff exceeded: {total_dp} > {max_dp}.')
    elif backwards:st.warning('Monomer family transition is potentially unfavourable; consider testing a bridging comonomer strategy.')
    else:st.info('No family-order rule flagged; experimental feasibility is NOT guaranteed.')
    a,b,c,d=st.columns(4)
    a.metric('Final DP including seed',total_dp)
    b.metric('Overall Tg proxy',f'{overall_tg:.1f} °C')
    c.metric('Recent blocks Tg proxy',f'{recent_tg:.1f} °C')
    d.metric('Penalized Tg (screening)',f'{penalized:.1f} °C')
    st.caption(f'DP penalty ×{penalty:.2f} is applied after including proposed block AND seed. Tg screening basis: {tg_basis}. DP-weighted Tg is a proxy, not a thermodynamic glass-transition calculation for a multiphase particle.')
    st.metric('Ridge regression: programmed interval',f'{time_min:.0f} min')
    st.caption(f'Fit on {n_train} SI non-seed interval records. Exact matching SI records: {n_exact}.'+(f' Their median scheduled interval was {exact_median:.0f} min.' if n_exact else '')+' This is a regression of *programmed scheduling*, not rate-to-conversion or validated temperature-dependent kinetics. Temperature is not a fitted time predictor because SI data provide insufficient temperature variation.')
    st.markdown('**Initiator-screening outcome**')
    table=[]
    for name in KINETICS:
        table.append({'Initiator':name,'Half-life at chosen T (h)':round(hl[name],3),'Half-lives elapsed':round(half_count[name],2),'New shot remaining (%)':round(100*fraction_remaining(hl[name],duration_h),2),'Duration screen':('PASS' if valid[name] else 'REJECT')})
    st.dataframe(pd.DataFrame(table),hide_index=True)
    if chosen is None:
        st.error(f'{target} is favoured by the Tg heuristic, but rejected by the {min_half_count:.2f}-half-life duration screen at {temp_c} °C. No automatic substitution: revise time, temperature, initiator strategy, or plan staggered shots.')
    else:st.success(f'Conditional initiator recommendation: {chosen} (passes user-defined Tg and duration screens).')
    if backwards:st.warning(f'Potentially difficult {old_family} → {new_family}. Test ~{previous_fraction:.0%} of prior monomer in new feed; resultant segment is a copolymer, not a pure homopolymer block.')
    if new_m=='BuA':st.warning('BuA backbiting/branching caution: SI Run 3 lowered temperature to 70 °C in the final chain extension; selected T may increase branching risk.')
    if new_m=='tBuAm':st.warning('Solid tBuAm required a cosolvent/monomer dissolution strategy in the SI.')
    if target=='BPO':st.warning('BPO was not used in the uploaded multiblock dataset; Tg-screening choice extrapolates beyond the paper.')
    if prior[-1]['monomer']==new_m:st.caption('Same monomer consecutive addition is not a chemically distinct block.')
    st.markdown('**Initiator inventory: what was originally present?**')
    simname=target
    shots=[]
    if legacy_equiv>0:shots.append({'initiator':legacy_name,'time_h':0.,'equivalents':legacy_equiv,'origin':'user-entered residual at start'})
    shots.append({'initiator':simname,'time_h':0.,'equivalents':initial_equiv,'origin':'proposed new shot'})
    remnants=decay_shots(shots,duration_h,hl)
    new_fraction=fraction_remaining(hl[simname],duration_h)
    a,b,c=st.columns(3)
    a.metric('New shot initial loading',f'{initial_equiv:.4f} mol/mol RAFT')
    b.metric('New shot remaining',f'{100*new_fraction:.2f}%')
    c.metric('New shot remaining amount',f'{initial_equiv*new_fraction:.5f} mol/mol RAFT')
    st.dataframe(remnants,hide_index=True)
    total_remaining=float(remnants.remaining_equiv_per_RAFT.sum())
    st.write(f'All tracked initiators remaining, in mol initiator per mol initial RAFT: **{total_remaining:.5f}**.')
    if 100*new_fraction<target_remain_pct:
        time_at_threshold=hl[simname]*math.log2(100/target_remain_pct)
        st.warning(f'The proposed new shot crosses {target_remain_pct:.1f}% at ~{time_at_threshold*60:.0f} min. A second shot may be worth assessing, but the dose should be chosen using radical flux, chain-end fidelity and total initiator balance—not based only on residual percentage.')
    else:st.info('Proposed new shot remains above the selected percentage threshold over the predicted interval.')
    st.caption('The inventory does NOT invent historic initiator doses. Defaults: zero pre-existing initiator, and a new dose of [I]/[RAFT] = 1/20. If you know earlier shots, add their estimated *undecomposed* equivalent as the residual field; prior elapsed times cannot be reconstructed without a complete shot history. No correction yet for latex sampling/removal or changes in RAFT chain count.')
    st.dataframe(raw.loc[(raw.monomer==new_m)&(~raw.includes_seed),['table','block_index','monomer','DP','time_min']].reset_index(drop=True),hide_index=True)

st.divider()
st.caption('Prototype limitations: unverified Tg / kinetic placeholders; emulsion particle morphology, radical entry, oxygen, conversions and kinetics not modeled. SI records reflect scheduled intervals and mixed outcomes, including failed later extensions. Use for experimental planning hypotheses only.')
