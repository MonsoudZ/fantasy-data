"""Neural ant: a sequence model that reads a player's trajectory.

The colony (ensemble.py) stalled because its ants were all snapshot models --
they see the same trailing averages and err together. This ant is different by
construction: a GRU consumes a player's last K games as an ordered *sequence*
of raw per-game outcomes, so it can learn momentum, ramps, and breakout curves
that fixed r3/r5 windows blur. A learned position embedding lets QB/RB/WR/TE
trajectories be read differently. Current-week pre-game context (opponent
defense, Vegas lines, injury status) is concatenated onto the sequence summary.

The point isn't just accuracy -- it's *decorrelation*. A model with a different
inductive bias should make different mistakes than the trees, which is exactly
what stacking needs. backtest_neural() reports both: the neural ant's standalone
accuracy AND its error correlation with a GBM trained on the same cutoff.

Protocol: unlike the tree ants (retrained weekly), a neural net trains once on
all seasons before the test season and predicts it -- leak-free (test is the
future) and practical (weekly NN retraining is not).

FINDING (2024, single run): on equal footing (both trained on < 2024) the
sequence ant is the most accurate single model here -- it beats the GBM on MAE
(4.49 vs 4.57), RMSE (6.16 vs 6.22), and weekly rank (0.694 vs 0.682). Yet its
errors correlate 0.97 with the GBM's. So a fundamentally different architecture,
reading the data as a trajectory, still makes nearly the same mistakes: the
residual is dominated by irreducible game-to-game fantasy variance, not by
signal one model sees and another misses. Consequence: this ant is valuable as
a candidate *primary* model, NOT as an ensemble member (stacking near-identical
errors buys nothing -- see ensemble.py). Caveat: one seed, one season; verify
across seeds/seasons (backtest_neural accepts both) before promoting it.

OpenMP note: LightGBM and PyTorch each bundle their own OpenMP runtime, and on
macOS the second to initialize aborts the process. torch is therefore imported
lazily (inside train_predict) and all LightGBM work runs first -- do not hoist
`import torch` to module scope.

    from ffdata.neural import backtest_neural
    print(backtest_neural(test_season=2024))
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from .features import build_features
from .projections import GBMProjector

POSITIONS = ("QB", "RB", "WR", "TE")
# Per-game outcomes fed as the sequence (what happened in each prior game).
SEQ_STATS = ["fp", "targets", "receptions", "receiving_yards", "carries",
             "rushing_yards", "snap_pct", "passing_yards", "passing_tds",
             "target_share", "air_yards_share", "attempts"]
# Current-week signals known before kickoff (no leakage).
CONTEXT = ["def_fp_allowed_r5", "team_implied_total", "opp_implied_total",
           "team_spread", "game_total", "is_home",
           "inj_on_report", "inj_status", "practice_dnp", "practice_limited"]
K = 8  # games of history in each sequence


def _standardize(mat: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Z-score columns using train-row stats; NaN -> 0 (the mean after scaling)."""
    mu = np.nanmean(mat[train_mask], axis=0)
    sd = np.nanstd(mat[train_mask], axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.nan_to_num((mat - mu) / sd, nan=0.0)


def build_sequences(feats: pd.DataFrame, test_season: int):
    """Assemble right-aligned K-game sequences + context; standardized on train."""
    f = feats.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    train_mask = (f["season"] < test_season).to_numpy()

    seq = _standardize(f[SEQ_STATS].to_numpy(float), train_mask)
    ctx = _standardize(f[CONTEXT].to_numpy(float), train_mask)
    pos = f["position"].map({p: i for i, p in enumerate(POSITIONS)}).fillna(0).astype(int).to_numpy()
    y = f["fp"].to_numpy(float)

    n, d = len(f), len(SEQ_STATS)
    xseq = np.zeros((n, K, d), dtype=np.float32)
    for idx in f.groupby("player_id").indices.values():
        for j, row in enumerate(idx):
            prior = idx[max(0, j - K):j]              # strictly earlier games only
            if len(prior):
                xseq[row, K - len(prior):] = seq[prior]
    hist = f["fp_r3"].notna().to_numpy()              # rows with enough history to score
    meta = f[["season", "week", "player_id", "position"]].copy()
    return xseq, ctx.astype(np.float32), pos, y.astype(np.float32), train_mask, hist, meta


def train_predict(feats: pd.DataFrame, test_season: int, epochs: int = 25, seed: int = 0) -> pd.DataFrame:
    """Train once on seasons < test_season, predict the test season.

    torch is imported here (not at module scope) so that any LightGBM work in
    the caller runs before torch grabs the OpenMP runtime. See module docstring.
    """
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    class SeqModel(nn.Module):
        def __init__(self, n_seq, n_ctx, n_pos=4, hidden=64, emb=4):
            super().__init__()
            self.gru = nn.GRU(n_seq, hidden, batch_first=True)
            self.pos_emb = nn.Embedding(n_pos, emb)
            self.head = nn.Sequential(
                nn.Linear(hidden + n_ctx + emb, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

        def forward(self, xseq, xctx, pos):
            _, h = self.gru(xseq)
            z = torch.cat([h.squeeze(0), xctx, self.pos_emb(pos)], dim=1)
            return self.head(z).squeeze(1)

    xseq, ctx, pos, y, train_mask, hist, meta = build_sequences(feats, test_season)
    test_mask = (meta["season"] == test_season).to_numpy() & hist

    Xs, Xc, P, Y = (torch.tensor(a) for a in (xseq, ctx, pos, y))
    model = SeqModel(len(SEQ_STATS), len(CONTEXT))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    tr = np.where(train_mask)[0]
    model.train()
    for _ in range(epochs):
        perm = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(model(Xs[b], Xc[b], P[b]), Y[b]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        te = np.where(test_mask)[0]
        preds = model(Xs[te], Xc[te], P[te]).numpy()
    out = meta.iloc[te].copy()
    out["fp"], out["pred"] = y[te], preds
    return out


def backtest_neural(train_from: int = 2019, test_season: int = 2024, epochs: int = 25) -> dict:
    """Neural ant accuracy on the test season + error correlation with a GBM."""
    from scipy.stats import spearmanr

    feats = build_features(seasons=list(range(train_from, test_season + 1)))

    # GBM FIRST -- LightGBM must touch OpenMP before torch does (see docstring).
    gbm = GBMProjector()
    gbm.fit(feats[feats["season"] < test_season])
    test = feats[(feats["season"] == test_season) & feats["fp_r3"].notna()].copy()
    test["gbm"] = gbm.predict(test)

    neural = train_predict(feats, test_season, epochs=epochs)
    m = neural.merge(test[["player_id", "season", "week", "gbm"]],
                     on=["player_id", "season", "week"], how="inner")

    def metrics(pred):
        err = m[pred] - m["fp"]
        wk = m.groupby(["season", "week"]).apply(
            lambda g: spearmanr(g[pred], g["fp"]).correlation if g["fp"].nunique() > 1 else np.nan)
        return {"MAE": round(float(err.abs().mean()), 3),
                "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
                "weekly_spearman": round(float(wk.mean()), 4)}

    err_corr = float(np.corrcoef(m["pred"] - m["fp"], m["gbm"] - m["fp"])[0, 1])
    return {"n": len(m), "neural": metrics("pred"), "gbm": metrics("gbm"),
            "error_corr_neural_vs_gbm": round(err_corr, 3)}


if __name__ == "__main__":
    import json
    print(json.dumps(backtest_neural(), indent=2))
