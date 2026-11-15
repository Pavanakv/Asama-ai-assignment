from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Topic = str


def _clean_topics(v: list) -> list:
    out = []
    for t in v:
        t = " ".join(str(t).lower().split())[:40]
        if t and t not in out:
            out.append(t)
    return out[:20]


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    industry: str = Field(min_length=2, max_length=80)
    description: str = Field(min_length=10, max_length=2000, description="Plain-language statement of what this agent cares about.")
    topics: list[str] = Field(default_factory=list, max_length=20)
    webhook_url: str = Field(min_length=8, max_length=500)

    @field_validator("topics")
    @classmethod
    def _t(cls, v):
        return _clean_topics(v)


class AgentPatch(BaseModel):
    description: Optional[str] = Field(default=None, min_length=10, max_length=2000)
    topics: Optional[list[str]] = Field(default=None, max_length=20)
    webhook_url: Optional[str] = Field(default=None, min_length=8, max_length=500)

    @field_validator("topics")
    @classmethod
    def _t(cls, v):
        return None if v is None else _clean_topics(v)


class PostIn(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    content: str = Field(min_length=10, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=10)
    kind: Literal["offer", "announcement", "market_update", "question"] = "announcement"
    expires_in_hours: Optional[int] = Field(default=None, ge=1, le=24 * 90)

    @field_validator("tags")
    @classmethod
    def _t(cls, v):
        return _clean_topics(v)[:10]


class CommentIn(BaseModel):
    content: str = Field(min_length=2, max_length=1500)
    parent_id: Optional[str] = None


class ReportIn(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class FeedbackIn(BaseModel):
    signal: Literal["relevant", "not_relevant"]


class SubscriptionIn(BaseModel):
    topic: str = Field(min_length=2, max_length=40)

    @field_validator("topic")
    @classmethod
    def _t(cls, v):
        return " ".join(v.lower().split())


class BlockIn(BaseModel):
    agent_id: str


class TrustIn(BaseModel):
    trust: float = Field(ge=0.0, le=1.0)
