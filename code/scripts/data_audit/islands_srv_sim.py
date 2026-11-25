import numpy as np, json
D="/data/v.pampinella/data_audit/graphviz/"
lay=np.load(D+"layout_anchor.npz"); xy=lay["xy"]; nW=int(lay["nW"]); XM=xy[nW:]
lab=np.load(D+"market_cluster_labels.npy"); e=np.load(D+"edges.npz"); fills=e["fillsM"]
lo,hi=np.quantile(xy,0.003,axis=0),np.quantile(xy,0.997,axis=0)
out={}
for l in np.unique(lab):
    if l<0: continue
    m=lab==l
    out[int(l)]=dict(x=float(np.median(XM[m,0])),y=float(np.median(XM[m,1])),n=int(m.sum()),fills=float(fills[m].sum()))
print(json.dumps(dict(lo=lo.tolist(),hi=hi.tolist(),islands=out)))
