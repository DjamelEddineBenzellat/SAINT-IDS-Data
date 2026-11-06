"""
C10 — X1: real TenSEAL CKKS microbenchmarks for the alert payload (Comment 13; replaces the
simulator-based V1 Table 5). One core, n_threads=1. Read-only towards the repository.

  python c10_ckks_bench.py --repo <repo> --sample <latency_sample_v2.npz> --out <dir> [--core 2]

Alert vector (D4): 28 PHI-sanitized features (CLOUD_SAFE_INDICES) + s_attack + 6 class probabilities
= 35 float64 values, built from 1,000 real alerting v2 test flows (s_attack >= τ).
For N = 8192 [60,40,40,60] and N = 16384 [60,40,40,40,60], scale 2^40:
  context generation (with / without Galois keys) and public-context size;
  public-key encryption + serialization (+ HMAC-SHA256) for K alerts packed per ciphertext,
  K in {1, 8, 32, max}; ciphertext size raw/base64 per ciphertext and per alert;
  deserialization + decryption with the secret context; round-trip error per field;
  homomorphic sum of 100 ciphertexts (D7) and its error.
Also: the CURRENT code path (CKKSEncryptor + KeyManager: secret context with Galois keys, 28 fields)
for continuity with the C1 laptop figure (6.57 ms, 441,400 B).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

_T0 = time.time()
PARAMS = {"N8192": (8192, [60, 40, 40, 60]), "N16384": (16384, [60, 40, 40, 40, 60])}
SCALE_BITS = 40
SECRET = b"x1-benchmark-hmac-key"


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def st(v):
    a = np.asarray(v, dtype=float)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def alert_vectors(root, sample, n=1000):
    import onnxruntime as ort
    from contracts.flow_feature_vector import CLOUD_SAFE_INDICES, FEATURE_NAMES
    from edge.inference.decision import get_default_policy
    z = np.load(sample); X = z["X"].astype(np.float32)
    sess = ort.InferenceSession(str(root / "edge/models_v2/lgbm_classifier.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    P = np.zeros((len(X), 6))
    for r, d in enumerate(sess.run(None, {name: X})[1]):
        for k, v in d.items():
            P[r, int(k)] = v
    s = 1 - P[:, 0]
    idx = np.where(s >= get_default_policy().alert_threshold)[0][:n]
    V = np.concatenate([X[idx][:, list(CLOUD_SAFE_INDICES)].astype(np.float64), s[idx, None], P[idx]], axis=1)
    names = [FEATURE_NAMES[i] for i in CLOUD_SAFE_INDICES] + ["s_attack"] + [f"p{k}" for k in range(6)]
    return V, names


def bench_params(ts, tag, N, coeff, V, names):
    out = {"N": N, "coeff_mod_bit_sizes": coeff, "scale_bits": SCALE_BITS, "slots": N // 2}
    # context generation
    for galois in (False, True):
        t = time.perf_counter()
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N, coeff_mod_bit_sizes=coeff, n_threads=1)
        ctx.global_scale = 2 ** SCALE_BITS
        if galois:
            ctx.generate_galois_keys()
        gen_ms = (time.perf_counter() - t) * 1e3
        pub = ctx.copy(); pub.make_context_public()
        out[f"context_{'with' if galois else 'without'}_galois"] = {
            "generation_ms": gen_ms, "secret_context_bytes": len(ctx.serialize(save_secret_key=True)),
            "public_context_bytes": len(pub.serialize())}
    # benchmark contexts: no Galois keys (not needed for encryption, packing or addition)
    sec = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N, coeff_mod_bit_sizes=coeff, n_threads=1)
    sec.global_scale = 2 ** SCALE_BITS
    pub = ts.context_from(sec.serialize(save_secret_key=False))          # public-key-only context
    assert not pub.has_secret_key() and sec.has_secret_key()
    d = V.shape[1]; kmax = (N // 2) // d
    out["packing"] = {}
    for K in sorted({1, 8, 32, kmax}):
        reps = max(math.ceil(len(V) / K), 100 if K > 1 else len(V))
        enc, hm, dec, size_raw, size_b64 = [], [], [], [], []
        abs_err = np.zeros(d); rel_err = np.zeros(d); n_chk = 0
        for r in range(reps):
            rows = [(r * K + j) % len(V) for j in range(K)]
            vals = V[rows].reshape(-1)
            t0 = time.perf_counter()
            ct = ts.ckks_vector(pub, vals.tolist()); raw = ct.serialize()
            t1 = time.perf_counter()
            sig = hmac.new(SECRET, raw, hashlib.sha256).hexdigest()
            t2 = time.perf_counter()
            b64 = base64.b64encode(raw)
            t3 = time.perf_counter()
            back = ts.ckks_vector_from(sec, raw).decrypt()
            t4 = time.perf_counter()
            enc.append((t1 - t0) * 1e3); hm.append((t2 - t1) * 1e3); dec.append((t4 - t3) * 1e3)
            size_raw.append(len(raw)); size_b64.append(len(b64))
            got = np.asarray(back[:len(vals)]).reshape(K, d); ref = V[rows]
            e = np.abs(got - ref)
            abs_err = np.maximum(abs_err, e.max(0))
            rel_err = np.maximum(rel_err, (e / np.maximum(np.abs(ref), 1e-12)).max(0))
            n_chk += K
            assert sig
        out["packing"][str(K)] = {
            "alerts_per_ciphertext": K, "ciphertexts": reps,
            "encrypt_serialize_ms": st(enc), "hmac_ms": st(hm), "deserialize_decrypt_ms": st(dec),
            "ciphertext_bytes_raw": int(np.median(size_raw)), "ciphertext_bytes_base64": int(np.median(size_b64)),
            "per_alert": {"encrypt_ms": float(np.mean(enc)) / K, "bytes_base64": float(np.median(size_b64)) / K},
            "max_abs_error_per_field": dict(zip(names, map(float, abs_err))),
            "max_abs_error": float(abs_err.max()), "max_rel_error_nonzero_fields": float(rel_err[np.abs(V).max(0) > 0].max()),
            "values_checked": int(n_chk * d)}
        say(f"  {tag} K={K:4d}: encrypt {np.mean(enc):7.2f} ms/ct ({np.mean(enc) / K:6.3f} ms/alert), "
            f"{np.median(size_b64) / 1024:8.1f} KiB/ct ({np.median(size_b64) / K / 1024:7.2f} KiB/alert), "
            f"decrypt {np.mean(dec):6.2f} ms, max abs err {abs_err.max():.3g}")
    # homomorphic sum (D7): 100 single-alert ciphertexts
    cts = [ts.ckks_vector(pub, V[i].tolist()) for i in range(100)]
    t = time.perf_counter(); acc = cts[0].copy()
    for c in cts[1:]:
        acc += c
    add_ms = (time.perf_counter() - t) * 1e3
    total = ts.ckks_vector_from(sec, acc.serialize()).decrypt()      # only the key holder can decrypt
    err = np.abs(np.asarray(total[:d]) - V[:100].sum(0))
    try:
        acc.decrypt(); public_can_decrypt = True
    except ValueError:
        public_can_decrypt = False
    out["public_context_can_decrypt"] = public_can_decrypt
    out["homomorphic_sum_100"] = {"total_ms": add_ms, "per_add_ms": add_ms / 99, "max_abs_error": float(err.max()),
                                  "max_rel_error": float((err / np.maximum(np.abs(V[:100].sum(0)), 1e-12)).max())}
    say(f"  {tag} homomorphic sum of 100: {add_ms / 99:.3f} ms/add, max abs err {err.max():.3g}")
    return out


def current_code_path(root, V):
    """The repository's CKKSEncryptor + KeyManager as they are today (28 fields, secret ctx + Galois)."""
    from edge.crypto.ckks_encryptor import CKKSEncryptor
    from edge.crypto.key_manager import KeyManager
    t = time.perf_counter(); km = KeyManager(); gen = (time.perf_counter() - t) * 1e3
    enc = CKKSEncryptor(km)
    lat, size = [], []
    for v in V[:1000, :28]:
        p = enc.encrypt(v); lat.append(p.encrypt_time_ms); size.append(len(p.ciphertext_b64))
    back = enc.decrypt(p)
    return {"keymanager_init_ms": gen, "encrypt_ms": st(lat), "ciphertext_bytes_base64": int(np.median(size)),
            "roundtrip_max_abs_error_last": float(np.abs(back - V[min(len(V), 1000) - 1, :28]).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--core", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.repo).resolve(); sys.path.insert(0, str(root))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"x1_ckks_{socket.gethostname()}.json"
    if dst.exists():
        say(f"{dst.name} exists - skipping"); return
    import tenseal as ts
    assert "tenseal_sim" not in ts.__file__, "simulator in use"
    os.sched_setaffinity(0, {a.core})
    say(f"TenSEAL {ts.__version__} from {ts.__file__}")
    V, names = alert_vectors(root, a.sample)
    say(f"{len(V)} alerting flows, {V.shape[1]} values per alert; max |value| {np.abs(V).max():.3g}")
    res = {"tenseal": ts.__version__, "host": socket.gethostname(), "core": a.core, "alert_vector_fields": names,
           "n_alerts": int(len(V)), "max_abs_value": float(np.abs(V).max()),
           "repo_commit": os.popen(f"git -C {root} rev-parse HEAD").read().strip()}
    for tag, (N, coeff) in PARAMS.items():
        say(f"{tag}: N={N}, coeff={coeff}")
        res[tag] = bench_params(ts, tag, N, coeff, V, names)
    say("current code path (CKKSEncryptor + KeyManager)")
    res["current_code_path"] = current_code_path(root, V)
    c = res["current_code_path"]
    say(f"  current: encrypt {c['encrypt_ms']['mean']:.2f} ms (p99 {c['encrypt_ms']['p99']:.2f}), "
        f"{c['ciphertext_bytes_base64']:,} B base64")
    res["elapsed_s"] = round(time.time() - _T0, 1)
    dst.write_text(json.dumps(res, indent=1)); say(f"written {dst}")


if __name__ == "__main__":
    import logging
    import warnings
    logging.disable(logging.WARNING); warnings.filterwarnings("ignore")
    main()
