"""Grounded natural-language advice over the draft/keeper/trade engine.

The decision tools already produce the *numbers* -- projected points, VOR,
auction $, overall/positional rank, keeper surplus, trade value per side. What
they don't produce is the *why*: a plain-English read of what those numbers mean
for your specific league. This module asks Claude to write that read, but on a
tight leash -- it reasons **only** from the numbers we hand it, cites them, and
is told in no uncertain terms not to invent a stat that isn't in the facts.

    from ffdata.advice import available, explain
    if available():
        text = explain("compare", facts)   # facts = the engine's own output

Design:
- **Grounded, not generative.** Every fact the model is allowed to use is in the
  `facts` dict (built from `draft_board`/`keeper_value`/`trade_value` output plus
  the league's scoring). The system prompt forbids outside numbers. So the model
  can phrase and weigh trade-offs, but it can't hallucinate "he's due for 1,400
  yards" -- that number was never provided.
- **Optional dependency.** `anthropic` is an extra (`pip install '.[advice]'`)
  and needs `ANTHROPIC_API_KEY`. `available()` gates the feature so the rest of
  the app runs fine without it; the web UI only shows "Explain" when it's on.
- **League-aware.** The scoring rules travel in the facts, so the explanation is
  about *your* league (a TE-premium or superflex read differs from vanilla PPR).

NOTE: the live API path can't be exercised here (no outbound network in this
environment). The prompt assembly and the `available()` gate are unit-tested
with a mocked client; validate end-to-end once you have an ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os

MODEL = "claude-opus-4-8"
_MAX_TOKENS = 1024

# The whole safety story is in this prompt: reason only from the JSON facts, cite
# the numbers, never introduce a stat that isn't there, and stay short.
SYSTEM = """You are a fantasy-football draft assistant embedded in an analytics \
tool. You will be given a JSON object of FACTS produced by the tool's own models \
-- projected fantasy points, value-over-replacement (VOR), auction dollar value, \
overall and positional rank, and the league's scoring rules. Sometimes also \
keeper surplus (value minus cost) or per-side trade totals.

Your job: explain, in plain English, what these numbers mean for THIS manager's \
decision, and give a clear recommendation.

Hard rules:
- Reason ONLY from the numbers in FACTS. Do not introduce any statistic, \
projection, injury, depth-chart note, or news that is not present in FACTS. If \
you don't have a number, don't imply one.
- Cite the actual figures you rely on (e.g. "VOR 42 vs 31", "$18 keeper cost \
against $34 of value"). The manager should be able to check every claim against \
the table.
- Respect the scoring rules given: VOR and value already reflect them, so a \
superflex or TE-premium slant is baked in -- read the numbers, don't re-derive \
the format.
- Be decisive but honest. If two players are close, say they're close and name \
the tiebreak (position scarcity via positional rank, or auction value). Don't \
manufacture separation the numbers don't support.
- Be concise: 2-4 short paragraphs or a few tight bullets. No preamble, no \
restating the question, no disclaimers about being an AI."""

# What each request kind is asking the model to decide. Kept here (not in the
# prompt) so the endpoint and the tests share one source of truth.
_PROMPTS = {
    "compare": "Compare these players for a draft pick and say who to take, and "
               "for whom that answer would differ.",
    "keeper": "Assess these keeper decisions: which are worth keeping at their "
              "cost, and which to let go back into the draft pool.",
    "trade": "Evaluate this trade: who wins, by how much, and whether the side "
             "getting less value has a reason (e.g. positional need) to still do it.",
}


def available() -> bool:
    """True when the advice layer can actually run (SDK installed + key set)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _client():
    import anthropic
    return anthropic.Anthropic()


def explain(kind: str, facts: dict, client=None) -> str:
    """Return a grounded explanation for a decision `kind` given `facts`.

    `kind` is one of compare/keeper/trade; `facts` is the engine's own output
    (plus scoring context). `client` is injectable for tests. Raises ValueError
    on an unknown kind so a typo can't silently produce an unguided prompt.
    """
    if kind not in _PROMPTS:
        raise ValueError(f"unknown advice kind: {kind!r}")
    user = (
        f"{_PROMPTS[kind]}\n\n"
        f"FACTS (the only numbers you may use):\n```json\n"
        f"{json.dumps(facts, indent=2, sort_keys=True)}\n```"
    )
    client = client or _client()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    # Adaptive thinking prepends thinking blocks; keep only the text output.
    return "".join(b.text for b in msg.content if b.type == "text").strip()
