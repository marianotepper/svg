"""
Experiments-section verification for the covering claim:
 (a) SVG-L0 attains navigability at an out-degree = small multiple of C_beta, on
     metric (ada-002/RBF) AND non-metric (Netflix/MIP, t!=k);
 (b) SVG-L0 improves over truncated MRNG when covering is non-trivial, ties when trivial.
Reports: C_beta (greedy covering number; mean/max; n-stability), recall@1 vs M for
SVG-L0 and truncated MRNG (greedy + short backtracking), and M*/C_beta with M* the
smallest tested M reaching recall@1 >= 0.99.
"""
import heapq
import numpy as np
from index.optimization import qp_fista

ADA = "./datasets/wikipedia_squad/100k/ada_002_100000_base_vectors.fvec"
NET = "./datasets/netflix/netflix_base.fvecs"


def load(path, dim, n):
    with open(path, "rb") as fh:
        buf = np.fromfile(fh, dtype=np.float32, count=n * (dim + 1)).reshape(n, dim + 1)
    return buf[:, 1:].astype(np.float64)


def sp_edges(K, v, M, budget, T=6):
    n = len(K)
    s = np.zeros(n)
    N = []
    for _ in range(T):
        corr = K[v] - K @ s
        corr[v] = -np.inf
        C = sorted(set(np.argsort(corr)[-M:].tolist()) | set(N))
        x = qp_fista(K[np.ix_(C, C)], K[v, C], budget=budget, iters=600)
        order = np.argsort(x)[::-1][:M]
        Nn = sorted(C[i] for i in order if x[i] > 1e-9)
        s = np.zeros(n)
        for i in order:
            if x[i] > 1e-9: s[C[i]] = x[i]
        if Nn == N: break
        N = Nn
    return N


def trunc_edges(K, v, M):
    return [j for j in np.argsort(K[v])[::-1] if j != v][:M]


def cover_C(K, v, beta, cap=64):
    n = len(K)
    O = K > K[v][None, :]
    tsel = (np.arange(n) != v) & (K[v] >= beta)
    Ot = O[:, tsel]; cov = np.zeros(Ot.shape[1], bool); c = 0
    while not cov.all() and c < cap:
        g = (Ot & ~cov[None, :]).sum(1).astype(float); g[v] = -1
        a = int(g.argmax())
        if g[a] <= 0: break
        cov |= Ot[a]; c += 1
    return c


def reach(adj, K, v, k, tgt, budget):
    visited = {v}
    frontier = [(-K[v, k], v)]
    exp = 0
    while frontier and exp < budget:
        _, c = heapq.heappop(frontier)
        if c == tgt: return True
        exp += 1
        for j in adj[c]:
            if j not in visited:
                visited.add(j)
                heapq.heappush(frontier, (-K[j, k], j))
    return tgt in visited


def recall(adj, K, tgt, srcs, qs, beta, budget=32):
    ok = tot = 0
    for k in qs:
        for v in srcs:
            if v == k or K[v, k] < beta: continue
            tot += 1
            ok += reach(adj, K, v, k, tgt[k], budget)
    return ok / max(tot, 1)


def run(name, X, kernel):
    n = len(X)
    rng = np.random.default_rng(0)
    if kernel == "rbf":
        d2 = np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T
        sig = np.sqrt(np.median(d2[d2 > 1e-9]))
        K = np.exp(-np.maximum(d2, 0) / (2 * sig**2))
        beta = np.quantile(K[np.triu_indices(n, 1)], 0.5)
    else:                                            # mip / linear
        K = X @ X.T
        beta = np.quantile(K[np.triu_indices(n, 1)], 0.7)
    tgt = np.argmax(K, axis=1)
    frac = np.mean(tgt != np.arange(n))
    budget = 1.0 / n
    Cs = [cover_C(K, v, beta) for v in rng.choice(n, 50, replace=False)]
    srcs = rng.choice(n, 120, replace=False)
    qs = rng.choice(n, 120, replace=False)
    Ms = [4, 8, 16, 32]
    rS, rT = {}, {}
    for M in Ms:
        aS = {v: sp_edges(K, v, M, budget) for v in range(n)}
        aT = {v: trunc_edges(K, v, M) for v in range(n)}
        rS[M] = recall(aS, K, tgt, srcs, qs, beta)
        rT[M] = recall(aT, K, tgt, srcs, qs, beta)
    Cb = float(np.mean(Cs))
    Mstar = next((M for M in Ms if rS[M] >= 0.99), None)
    print(f"\n=== {name} ({kernel}, n={n}) ===  t!=k frac={frac:.2f}  beta={beta:.3f}")
    print(f"  C_beta = {Cb:.1f} (max {int(np.max(Cs))})")
    print(f"  {'M':>3} | SVG-L0 recall@1 | truncated recall@1 | SVG-trunc")
    for M in Ms:
        print(f"  {M:>3} |    {rS[M]:.4f}      |     {rT[M]:.4f}       | {rS[M]-rT[M]:+.4f}")
    if Mstar:
        print(f"  M*(recall>=0.99) = {Mstar}  =>  M*/C_beta = {Mstar/Cb:.1f}x")
    else:
        print(f"  recall<0.99 for all tested M (max M={Ms[-1]})")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    N = 2000
    if which in ("ada", "both"):
        run("ada-002", load(ADA, 1536, N), "rbf")
    if which in ("net", "both"):
        run("Netflix", load(NET, 300, N), "mip")
