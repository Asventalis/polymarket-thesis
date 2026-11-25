import pandas as pd, numpy as np
a = pd.read_parquet('hand_check/day=2026-03-13.parquet'); b = pd.read_parquet('/data/v.pampinella/agy_project/data/derived/hand_features/day=2026-03-13.parquet')
print('dup keys', a.duplicated(['market_id','ts']).mean(), b.duplicated(['market_id','ts']).mean(), flush=True)
hist = ['t_last_move','since_trade_s','last_move_sign','moves_60m','rv_60m','ret_15m','ret_1h','ret_3h','ret_6h','ret_12h','log_dt_prev','range_pos_60m','mid_minus_mean60','n_60m','n_15m','n_5m','n_1m','ofi_5','ofi_20','ofi_15m','ofi_20_z','ofi_15m_z','log_vol_15m','flow_5m','flow_15m','flow_60m','usd_15m','usd_60m','ntr_15m','flow_ratio_15m','last_trade_minus_mid_ticks']
rowc = [c for c in a.columns if c not in hist + ['market_id','ts']]
sa = a.sort_values(['market_id','ts'] + rowc, kind='stable').reset_index(drop=True); sb = b.sort_values(['market_id','ts'] + rowc, kind='stable').reset_index(drop=True)
for c in rowc:
    x, y = sa[c].astype(float).values, sb[c].astype(float).values
    ok = np.isfinite(x) & np.isfinite(y); bad = (np.abs(x[ok]-y[ok]) > 1e-6*(1+np.abs(y[ok]))).mean()
    if bad > 0: print('row column differs after full sort:', c, bad, flush=True)
print('row columns compared', flush=True)
t = pd.to_datetime(sa.ts); late = (t.dt.hour >= 13).values
for c in ['moves_60m','rv_60m','n_60m','ret_12h','ofi_20','last_move_sign','since_trade_s']:
    x, y = sa[c].astype(float).values[late], sb[c].astype(float).values[late]
    ok = np.isfinite(x) & np.isfinite(y); bad = (np.abs(x[ok]-y[ok]) > 1e-6*(1+np.abs(y[ok]))).mean(); nm = (np.isfinite(x)!=np.isfinite(y)).mean()
    print(f'{c:16s} afternoon differ {bad:.4%} nan-mismatch {nm:.4%}', flush=True)
print('DONE')
