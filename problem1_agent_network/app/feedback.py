"""Closed-loop learning: which notifications actually led to engagement?

Per (agent, concept) we keep `sent` and `engaged` counts. A smoothed engagement rate nudges future scores:

    rate = (engaged + k*p0) / (sent + k)          # Beta-style smoothing toward a neutral prior p0
    multiplier = clamp(1 + gain*(rate - p0), lo, hi)

Why this instead of "shift the agent's embedding away from ignored posts" (a common suggestion):
 * it is explainable and debuggable (you can read the counts),
 * it is bounded - the multiplier floor means a very relevant post can still get through, so a topic is never
   permanently starved and an agent can recover (a built-in form of exploration),
 * it cannot drift the stored interest vector into nonsense.
Explicit signals (a comment, 'relevant', 'not_relevant') and implicit ones (ignored past the TTL) feed the same counts.
"""
from __future__ import annotations

from collections import defaultdict

PRIOR_RATE = 0.30     # neutral expected engagement
PRIOR_STRENGTH = 4.0  # pseudo-observations: how many real observations before the data outweighs the prior
GAIN = 0.6
LO, HI = 0.70, 1.20


def multiplier_for(sent: int, engaged: int) -> float:
    rate = (engaged + PRIOR_STRENGTH * PRIOR_RATE) / (sent + PRIOR_STRENGTH)
    return max(LO, min(HI, 1 + GAIN * (rate - PRIOR_RATE)))


def multipliers(db, agent_ids: list, concepts: list) -> dict:
    """agent_id -> multiplier, averaged over the post's concepts that the agent has history on."""
    if not agent_ids or not concepts:
        return {}
    out: dict = {}
    by_agent: dict = defaultdict(list)
    cq = ",".join("?" * len(concepts))
    for chunk_start in range(0, len(agent_ids), 400):
        chunk = agent_ids[chunk_start: chunk_start + 400]
        aq = ",".join("?" * len(chunk))
        for r in db.all(f"SELECT agent_id, concept, sent, engaged FROM agent_topic_stats WHERE agent_id IN ({aq}) AND concept IN ({cq})", (*chunk, *concepts)):
            by_agent[r["agent_id"]].append(multiplier_for(r["sent"], r["engaged"]))
    for a, ms in by_agent.items():
        out[a] = sum(ms) / len(ms)
    return out


def record_sent(db, agent_id: str, concepts: list) -> None:
    db.many("INSERT INTO agent_topic_stats(agent_id, concept, sent, engaged) VALUES (?,?,1,0) "
            "ON CONFLICT(agent_id, concept) DO UPDATE SET sent = sent + 1", [(agent_id, c) for c in concepts])


def record_engaged(db, agent_id: str, concepts: list) -> None:
    db.many("INSERT INTO agent_topic_stats(agent_id, concept, sent, engaged) VALUES (?,?,0,1) "
            "ON CONFLICT(agent_id, concept) DO UPDATE SET engaged = engaged + 1", [(agent_id, c) for c in concepts])
