"""Research prototype: transparent heuristics; never a validated synthesis guarantee."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title='RAFT Multiblock Designer v0.3',page_icon='🧪',layout='wide')
st.title('RAFT Multiblock Designer · v1.19')
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


# Approximate repeat-unit molar masses in g/mol (check substitutions and structures).
REPEAT_MASS={'nBMA':142.20,'iBMA':142.20,'tBMA':142.20,'BzMA':176.22,
             'HMA':198.31,'EMA':114.14,'MMA':100.12,'St':104.15,
             'MS':118.18,'BuA':128.17,'tBuAm':127.19}


def fox_mixture_tg_c(blocks, tgs, k_fox_flory):
    """Hypothetical ONE miscible polymer phase; not generally a multiblock Tg."""
    masses=[b['DP']*REPEAT_MASS[b['monomer']] for b in blocks]
    mn=sum(masses)
    inv=sum(m/(tgs[b['monomer']]+273.15) for b,m in zip(blocks,masses))/mn
    tg_inf_k=1/inv
    return tg_inf_k-273.15-k_fox_flory/mn, mn


def local_phase_tg_c(prior, tgs, k_fox_flory):
    """Last existing block is a local phase proxy, NOT an observed phase Tg."""
    last=prior[-1]
    mn=last['DP']*REPEAT_MASS[last['monomer']]
    return tgs[last['monomer']]-k_fox_flory/mn, mn


def wet_tg_c(dry_c, monomer_mass_fraction, monomer_tg_c, gt_k):
    """Gordon-Taylor approximation for one miscible polymer/monomer phase."""
    w=monomer_mass_fraction
    poly=1-w
    if w==0:return dry_c
    return (poly*(dry_c+273.15)+gt_k*w*(monomer_tg_c+273.15))/(poly+gt_k*w)-273.15


def mobility_score(reaction_c, wet_c, delta50_c, width_c):
    z=(reaction_c-wet_c-delta50_c)/width_c
    return 1/(1+math.exp(-max(-700,min(700,z))))


def fit_outcome_model(labels, reaction_c, effective_tg_c):
    """Optional empirical calibration only from independently labelled tests."""
    subset=labels.dropna(subset=['delta_t_c','success']).copy()
    subset=subset[subset['success'].isin([0,1])]
    if len(subset)<10 or subset['success'].nunique()<2:
        return None,'At least 10 labelled rows and both 0 and 1 outcomes are required.'
    counts=subset.success.value_counts()
    if counts.min()<3:return None,'At least three successes and three failures are required.'
    x=subset[['delta_t_c']].astype(float)
    y=subset['success'].astype(int)
    model=LogisticRegression(C=1.0,solver='lbfgs').fit(x,y)
    return model,None


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
st.caption('Default new shot: [RAFT]/[I] = 20, i.e. 0.05 mol initiator per mol RAFT.')
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

st.subheader('4 · Physical Tg and mobility model (new)')
st.caption('Dry polymer Tg values are approximate editable references. The local last-existing-block calculation is a proxy for the chain-end environment, not a measured particle Tg. Fox mixture assumes miscibility; swelling assumes a homogeneous polymer/monomer phase.')
c1,c2,c3=st.columns(3)
phase_mode=c1.selectbox('Phase approximation',['Local last existing block (recommended proxy)','Hypothetical fully mixed existing polymer'])
flory_k=c2.number_input('Fox–Flory K (K·g/mol; illustrative)',min_value=0.,max_value=500000.,value=0.,step=1000.,help='Zero disables molecular weight correction until a literature/experimental K has been entered. Applies to a phase proxy, not a universal block Tg.')
swell_w=c3.slider('Estimated monomer mass fraction inside particles',0.,0.60,0.,0.01,help='Set to zero unless you have a defensible particle swelling estimate.')
c1,c2,c3=st.columns(3)
monomer_tg=c1.number_input('Hypothetical Tg of absorbed monomer (°C)',min_value=-250.,max_value=100.,value=-100.,step=5.,help='Illustrative ONLY; verify for the specific monomer before enabling swelling.')
gt_k=c2.number_input('Gordon–Taylor k (dimensionless)',min_value=0.01,max_value=20.,value=1.,step=0.1)
chosen=c3.selectbox('Proposed initiator (independent of Tg)',list(KINETICS),help='Initiator identity is not physically determined by polymer Tg alone; assess aqueous/organic partitioning, RAFT end groups, pH, radical entry, and initiator kinetics.')
c1,c2=st.columns(2)
delta50=c1.number_input('Mobility transition ΔT50 (°C; illustrative)',value=0.,step=2.)
width=c2.number_input('Mobility sigmoid width (°C; illustrative)',min_value=0.1,value=10.,step=1.)
with st.expander('Optional: fit mobility-to-success calibration from independently labelled experiments'):
    st.write('Upload CSV with **delta_t_c** = reaction temperature minus effective wet Tg (°C), and **success** = 0 or 1. Use identical Tg methodology when preparing the CSV. These labels are NOT obtainable from scheduled SI intervals alone.')
    outcomes_file=st.file_uploader('Outcome calibration CSV',type=['csv'])
    outcome_df=None
    if outcomes_file is not None:
        try:
            outcome_df=pd.read_csv(outcomes_file)
            if not {'delta_t_c','success'}.issubset(outcome_df.columns):
                st.error('Required columns: delta_t_c, success');outcome_df=None
        except Exception as exc:
            st.error(f'Cannot read calibration file: {exc}')



if st.button('Predict next block',type='primary'):
    prior=list(st.session_state.blocks)
    chain=prior+[{'monomer':new_m,'DP':new_dp}]
    total_dp=sum(x['DP'] for x in chain)
    overall_tg=weighted_tg(chain,tgs)  # legacy descriptor only, no penalty
    recent_tg=weighted_tg(chain[-int(recent_n):],tgs)  # legacy descriptor only
    if phase_mode.startswith('Local'):
        dry_tg,phase_mn=local_phase_tg_c(prior,tgs,flory_k)
    else:
        dry_tg,phase_mn=fox_mixture_tg_c(prior,tgs,flory_k)
    wet_tg=wet_tg_c(dry_tg,swell_w,monomer_tg,gt_k)
    delta_t=temp_c-wet_tg
    mob=mobility_score(temp_c,wet_tg,delta50,width)
    calib=None
    if outcome_df is not None:
        try:
            calib,calib_err=fit_outcome_model(outcome_df,temp_c,wet_tg)
            if calib_err:st.warning(f'Calibration not used: {calib_err}')
        except Exception as exc:
            st.warning(f'Calibration not used: {exc}')
    time_min,n_exact,exact_median,n_train=predict_schedule(raw,new_m,new_dp,len(chain))
    duration_h=time_min/60
    hl={k:half_life_at_temp(*params[k],temp_c) for k in KINETICS}
    half_count={k:duration_h/hl[k] for k in KINETICS}
    valid={k:half_count[k]<=min_half_count for k in KINETICS}
    # No forced switch to unsuitable Tg initiator. Flag mismatch explicitly.
    candidate=chosen if valid[chosen] else None
    old_family=MONOMERS[prior[-1]['monomer']][0];new_family=MONOMERS[new_m][0]
    backwards=(old_family in ('styrene','acrylate','acrylamide') and new_family=='methacrylate') or (old_family=='acrylate' and new_family=='styrene')
    st.subheader('4 · Assessment')
    st.write(' → '.join(f"P{x['monomer']}({x['DP']})" for x in chain))
    if total_dp>max_dp:st.error(f'User-defined total DP instability cutoff exceeded: {total_dp} > {max_dp}.')
    elif backwards:st.warning('Monomer family transition is potentially unfavourable; consider testing a bridging comonomer strategy.')
    else:st.info('No family-order rule flagged.')
    a,b,c,d=st.columns(4)
    a.metric('Final DP including seed',total_dp)
    b.metric('Dry local-phase Tg proxy',f'{dry_tg:.1f} °C')
    c.metric('Swollen phase Tg proxy',f'{wet_tg:.1f} °C')
    d.metric('T reaction − Tg wet',f'{delta_t:.1f} °C')
    st.metric('Illustrative chain mobility score (NOT success probability)',f'{mob:.1%}')
    st.caption(f'Phase proxy Mn = {phase_mn:,.0f} g/mol; Fox–Flory K = {flory_k:,.0f} K·g/mol. Monomer loading = {swell_w:.0%}. Sigmoid ΔT50 = {delta50:.1f} °C, width = {width:.1f} °C. No total-DP multiplier or Tg-based initiator ranking is applied.')
    if swell_w>0:st.warning('Swollen Tg estimate uses unverified monomer Tg and Gordon–Taylor k unless you replaced those entries with measurements. It is a scenario analysis, not a measured particle Tg.')
    if phase_mode.startswith('Hypothetical'):st.warning('The Fox-mixed Tg assumes a single miscible polymer phase. A microphase-separated multiblock can have distinct phase Tgs.')
    if calib is not None:
        prob=float(calib.predict_proba(pd.DataFrame({'delta_t_c':[delta_t]}))[0,1])
        st.metric('Empirically fitted success probability (dataset-specific)',f'{prob:.1%}')
        st.caption(f'Calibrated on {len(outcome_df)} submitted rows using regularized logistic regression and ΔT alone; not independently validated. Use held-out tests and add family/order, DP and initiator effects before using quantitatively.')
    else:st.info('No measured success/failure dataset provided. SI intervals can fit schedules, not an outcome probability; mobility percentage is only a configurable screening score.')
    st.metric('Ridge regression: programmed interval',f'{time_min:.0f} min')
    st.caption(f'Fit on {n_train} SI non-seed interval records. Exact matching SI records: {n_exact}.'+(f' Their median scheduled interval was {exact_median:.0f} min.' if n_exact else ''))
    st.markdown('**Initiator-screening outcome**')
    table=[]
    for name in KINETICS:
        table.append({'Initiator':name,'Half-life at chosen T (h)':round(hl[name],3),'Half-lives elapsed':round(half_count[name],2),'New shot remaining (%)':round(100*fraction_remaining(hl[name],duration_h),2),'Duration screen':('PASS' if valid[name] else 'REJECT')})
    st.dataframe(pd.DataFrame(table),hide_index=True)
    if candidate is None:
        st.error(f'Chosen initiator {chosen} fails the {min_half_count:.2f}-half-life duration screen at {temp_c} °C. Reassess kinetics, dosing and aqueous compatibility.')
    else:st.success(f'Selected initiator {chosen} passes the user-defined half-life duration screen; this is NOT a chemical suitability guarantee.')
    if backwards:st.warning(f'Potentially difficult {old_family} → {new_family}. Test ~{previous_fraction:.0%} of prior monomer in new feed; resultant segment is a copolymer, not a pure homopolymer block.')
    if new_m=='BuA':st.warning('BuA backbiting/branching caution: SI Run 3 lowered temperature to 70 °C in the final chain extension; selected T may increase branching risk.')
    if new_m=='tBuAm':st.warning('Solid tBuAm required a cosolvent/monomer dissolution strategy in the SI.')
    if chosen=='BPO':st.warning('BPO was not used in the uploaded multiblock dataset; Tg-screening choice extrapolates beyond the paper.')
    if prior[-1]['monomer']==new_m:st.caption('Same monomer consecutive addition is not a chemically distinct block.')
    st.markdown('**Initiator inventory: what was originally present?**')
    simname=chosen
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
    st.dataframe(raw.loc[(raw.monomer==new_m)&(~raw.includes_seed),['table','block_index','monomer','DP','time_min']].reset_index(drop=True),hide_index=True)

st.divider()
st.caption('Prototype limitations: Fox–Flory correction and Gordon–Taylor swelling require measured/verified parameters; mobility sigmoid is illustrative unless calibrated with outcome labels; unverified Tg / kinetic placeholders; emulsion particle morphology, radical entry, oxygen, conversions and kinetics not modeled. SI records reflect scheduled intervals and mixed outcomes, including failed later extensions.')
