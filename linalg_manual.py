"""Manual linear algebra used by compression methods.

Numpy is used only as an array container and for elementwise arithmetic
(+, -, *, /, sqrt, sums). No numpy.linalg (or any other solver) is called;
every decomposition step is written out here.
"""

import numpy as np

EPS = 1e-12


def round_robin_pairs(n):
    """Split all column pairs (p, q) of an n-column matrix into rounds.

    Uses the "circle method" from tournament scheduling: in each round every
    column appears in at most one pair, so all rotations in a round touch
    disjoint columns and can be applied at the same time. n - 1 rounds (n even)
    cover every one of the n(n-1)/2 pairs exactly once.
    """
    players = list(range(n))
    if n % 2:
        players.append(None)  # "bye" so the count is even
    m = len(players)
    rounds = []
    for _ in range(m - 1):
        ps, qs = [], []
        for i in range(m // 2):
            a, b = players[i], players[m - 1 - i]
            if a is not None and b is not None:
                ps.append(min(a, b))
                qs.append(max(a, b))
        rounds.append((np.array(ps, dtype=int), np.array(qs, dtype=int)))
        # Keep the first player fixed, rotate everyone else one seat.
        players = [players[0], players[-1]] + players[1:-1]
    return rounds


def jacobi_svd(matrices, tol=1e-8, max_sweeps=40, progress=None):
    """One-sided (Hestenes) Jacobi SVD of a stack of equally-sized matrices.

    matrices: array of shape (b, m, n). Returns (U, S, V) with shapes
    (b, m, r), (b, r), (b, n, r), r = min(m, n), such that for every i
        matrices[i] = U[i] @ diag(S[i]) @ V[i].T
    and S[i] sorted largest-first.

    Idea: apply plane (Givens) rotations J to the columns of A until every
    pair of columns is orthogonal. Then A V = W with orthogonal columns, so
    sigma_j = ||w_j||, u_j = w_j / sigma_j and A = U diag(sigma) V^T

    For a pair of columns a_p, a_q with
        alpha = a_p . a_p,  beta = a_q . a_q,  gamma = a_p . a_q
    the rotation that makes them orthogonal is
        zeta = (beta - alpha) / (2 gamma)
        t    = sign(zeta) / (|zeta| + sqrt(1 + zeta^2))   (tan of the angle)
        c    = 1 / sqrt(1 + t^2),  s = c t
        a_p <- c a_p - s a_q,  a_q <- s a_p + c a_q
    """
    a = np.asarray(matrices, dtype=float)
    if a.ndim == 2:
        a = a[None, :, :]
    b, m, n = a.shape

    # Work on the orientation with fewer columns (fewer pairs to rotate).
    transposed = n > m
    if transposed:
        a = a.transpose(0, 2, 1)
        m, n = n, m

    # Store W^T and V^T so each column is a contiguous row in memory; the
    # pair gathers below are then fast row copies instead of strided reads.
    wt = a.transpose(0, 2, 1).copy()  # (b, n, m): row j = column j of W
    vt = np.zeros((b, n, n), dtype=float)  # row j = column j of V
    for i in range(n):
        vt[:, i, i] = 1.0

    rounds = round_robin_pairs(n)
    total_rounds = max(1, len(rounds))

    for sweep in range(max_sweeps):
        worst = 0.0
        for r_idx, (p, q) in enumerate(rounds):
            wp = wt[:, p, :]
            wq = wt[:, q, :]
            alpha = (wp * wp).sum(axis=2)
            beta = (wq * wq).sum(axis=2)
            gamma = (wp * wq).sum(axis=2)

            scale = np.sqrt(alpha * beta)
            active = (scale > EPS) & (np.abs(gamma) > tol * scale)
            if not active.any():
                continue
            off = np.abs(gamma[active]) / scale[active]
            worst = max(worst, float(off.max()))

            safe_gamma = np.where(active, gamma, 1.0)
            zeta = (beta - alpha) / (2.0 * safe_gamma)
            sign = np.where(zeta >= 0.0, 1.0, -1.0)
            t = sign / (np.abs(zeta) + np.sqrt(1.0 + zeta * zeta))
            t = np.where(active, t, 0.0)  # inactive pairs: identity rotation
            c = 1.0 / np.sqrt(1.0 + t * t)
            s = c * t

            c3 = c[:, :, None]
            s3 = s[:, :, None]
            wt[:, p, :] = c3 * wp - s3 * wq
            wt[:, q, :] = s3 * wp + c3 * wq

            vp = vt[:, p, :]
            vq = vt[:, q, :]
            vt[:, p, :] = c3 * vp - s3 * vq
            vt[:, q, :] = s3 * vp + c3 * vq

            if progress is not None and r_idx % 16 == 0:
                progress(sweep, r_idx / total_rounds)
        if worst <= tol:
            break

    # Singular values are the column norms of W = A V.
    sigma = np.sqrt((wt * wt).sum(axis=2))  # (b, n)
    u = np.zeros((b, m, n), dtype=float)
    s_sorted = np.zeros_like(sigma)
    v_sorted = np.zeros((b, n, n), dtype=float)
    for i in range(b):
        order = sorted(range(n), key=lambda j: sigma[i, j], reverse=True)
        for new_j, old_j in enumerate(order):
            sig = sigma[i, old_j]
            s_sorted[i, new_j] = sig
            v_sorted[i, :, new_j] = vt[i, old_j, :]
            if sig > EPS:
                u[i, :, new_j] = wt[i, old_j, :] / sig

    if transposed:
        # We decomposed A^T = U S V^T, so A = V S U^T.
        u, v_sorted = v_sorted, u
    return u, s_sorted, v_sorted


def reconstruct(u, s, v, k):
    """Rank-k approximation A_k = sum_{i<k} s_i u_i v_i^T = U_k diag(s_k) V_k^T."""
    return (u[:, :k] * s[:k]) @ v[:, :k].T


def frobenius_norm_sq(a):
    """||A||_F^2 = sum of squared entries."""
    a = np.asarray(a, dtype=float)
    return float((a * a).sum())

