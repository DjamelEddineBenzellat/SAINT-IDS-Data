import json,sys,glob
import numpy as np
X=sys.argv[1]; A=sys.argv[2]
R={m:[json.load(open(f"{X}/results/stage_r1_{m}_run{r}.json")) for r in (1,2,3)] for m in ("deployed","default")}
print("=== recompute per run (normal mix / burst)")
for m,rs in R.items():
    for d in rs:
        n,b=d["normal_mix"],d["busy_mix_attacks_only"]
        print(f" {m:8} run{d['run']}: explain {n['explain']['mean_ms']:.3f} total {n['total']['mean_ms']:.3f} p50 {n['total']['p50_ms']:.2f} p95 {n['total']['p95_ms']:.2f} p99 {n['total']['p99_ms']:.2f} | burst total {b['total']['mean_ms']:.3f} p99 {b['total']['p99_ms']:.2f} | cores {n['cores_used']:.2f} MHz {n['core_mhz']['mean']:.0f} [{n['core_mhz']['min']:.0f}-{n['core_mhz']['max']:.0f}] | kernel {d['machine_before']['platform'][6:22]} load {d['machine_before']['loadavg'][0]:.2f}")
print("\n=== explanation stage distribution and relay")
for m,rs in R.items():
    for d in rs[:1]:
        for mix in ("normal_mix","busy_mix_attacks_only"):
            x=d[mix]; e=x["explain"]; r=x["relay"]
            print(f" {m:8} {mix:22} explain p50 {e['p50_ms']:.2f} p95 {e['p95_ms']:.2f} p99 {e['p99_ms']:.2f} max {e['max_ms']:.2f} | ciphertexts {r['ciphertexts']} alerts/ct {r['alerts_per_ciphertext_mean']:.1f} encrypt mean {r['encrypt_ms']['mean']:.2f} p99 {r['encrypt_ms']['p99']:.2f} | delivery mean {r['delivery_ms']['mean']:.0f} p99 {r['delivery_ms']['p99']:.0f} | wall {x['wall_s']:.1f}s")
print("\n=== is the tail caused by encryption? flows slower than median+2.5 ms, order in time")
for m in ("deployed","default"):
    for r in (1,2,3):
        z=np.load(f"{X}/results/stage_r1_{m}_run{r}_per_flow_ms.npz")
        for lab in ("normal","busy"):
            t=z[f"{lab}_total"]; e=z[f"{lab}_explain"]; med=np.median(t)
            slow=np.where(t>med+2.5)[0]; gaps=np.diff(slow)
            d=R[m][r-1]["normal_mix" if lab=="normal" else "busy_mix_attacks_only"]
            cts=d["relay"]["ciphertexts"]
            print(f" {m:8} run{r} {lab:6} flows {len(t)} median {med:.2f} | slow flows {len(slow)} ({100*len(slow)/len(t):.1f}%) vs ciphertexts {cts} ({100*cts/len(t):.1f}%) | mean extra on slow {t[slow].mean()-med:.2f} ms | median gap {np.median(gaps):.0f} flows = {np.median(gaps)*med:.0f} ms | mean of the fast flows {t[t<=med+2.5].mean():.2f} p99 of fast {np.percentile(t[t<=med+2.5],99):.2f}")
z=np.load(f"{X}/results/stage_r1_deployed_run2_per_flow_ms.npz")["normal_total"]
h,edges=np.histogram(z,bins=[7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,16,20])
print("\n histogram deployed run2 normal (ms: count):",{f"{edges[i]:g}-{edges[i+1]:g}":int(h[i]) for i in range(len(h))})
print("\n=== old run, no relay at all (E4, default ONNX) vs this default run with relay")
e=json.load(open(f"{A}/reports/comment4/e4/latency_e4_run1.json"))["normal_mix"]["total"]
print(f" E4 no relay  : mean {e['mean_ms']:.2f} p50 {e['p50_ms']:.2f} p95 {e['p95_ms']:.2f} p99 {e['p99_ms']:.2f} max {e['max_ms']:.2f}")
d=R['default'][0]['normal_mix']['total']; print(f" now, relay on: mean {d['mean_ms']:.2f} p50 {d['p50_ms']:.2f} p95 {d['p95_ms']:.2f} p99 {d['p99_ms']:.2f} max {d['max_ms']:.2f}")
print("\n=== over HTTP, same code, SHAP without relay (L2) vs with relay (L3)")
h=json.load(open(f"{A}/reports/comment13/x2_x3.json"))
for rate in (60,80,100):
    for k in ("X3_L2","X3_L3"):
        r=[x for x in h[k]["ramp"] if x["offered_rate"]==rate][0]; s=r["service_ms"]
        print(f" {k} {rate:>3}/s p50 {s['p50']:.2f} p95 {s['p95']:.2f} p99 {s['p99']:.2f} max {s['max']:.1f} cpu {r['cpu_percent_mean']:.1f}")
print("\n=== derived replacement numbers (median of 3)")
md=lambda f:float(np.median([f(d) for d in R['deployed']]))
n=lambda s,st='mean_ms':md(lambda d:d['normal_mix'][s][st]); b=lambda s,st='mean_ms':md(lambda d:d['busy_mix_attacks_only'][s][st])
dec=n('ingest')+n('score')+n('route'); print(f" normal: ingest {n('ingest'):.3f} score {n('score'):.3f} explain {n('explain'):.3f} route {n('route'):.3f} total {n('total'):.3f} p99 {n('total','p99_ms'):.2f}")
print(f" burst : ingest {b('ingest'):.3f} score {b('score'):.3f} explain {b('explain'):.3f} route {b('route'):.3f} total {b('total'):.3f} p99 {b('total','p99_ms'):.2f}")
print(f" decision {dec:.3f} ms; explanation share {100*n('explain')/n('total'):.1f}%; capacity {1000/n('total'):.0f} flows/s; alerts-only mean {dec+0.18*n('explain'):.2f} ms; alert cost {n('total'):.1f} ms; 353/{n('total'):.2f} = {353/n('total'):.0f}x")
print(f" frequency share: {4274/4000:.3f} vs SHAP ratio {11.003/8.224:.3f}")
print(" explain p99 deployed, normal:",round(md(lambda d:d['normal_mix']['explain']['p99_ms']),2)," E3 isolated p99 8.74")
