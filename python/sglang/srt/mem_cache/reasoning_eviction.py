"""CPU reference for reasoning eviction at completed assistant boundaries.

This module defines the attention policy, not a GPU allocator or serving API.
All offsets are in the original, unshortened token transcript. A boundary at
``end`` takes effect *after* the forward pass for token ``end - 1``. Replaying
only retained token IDs does not reproduce the computation defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


def _nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class ReasoningEvictionPolicy:
    """``None`` means full attention; W includes the protected closing marker.

    W=0 and W=1 both retain no interior tokens. Both delimiters and every
    non-reasoning token are protected regardless of W. The current assistant
    retains all its own reasoning through its terminal message token.
    """

    window: Optional[int] = None
    boundary: str = "assistant_message"
    version: int = 1

    def __post_init__(self) -> None:
        if self.window is not None:
            _nonnegative_int("window", self.window)
        if self.boundary != "assistant_message":
            raise ValueError("Only the assistant_message boundary is supported")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported reasoning eviction policy version")

    def dropped_positions(self, span: ReasoningSpan) -> range:
        if self.window is None:
            return range(span.start, span.start)
        keep_interior = max(self.window - 1, 0)
        return range(span.start, max(span.start, span.stop - keep_interior))


@dataclass(frozen=True)
class ReasoningSpan:
    """Interior [start, stop); stop is the closing delimiter's position.

    ``expires_at`` is the exclusive assistant message end, including its
    terminal token. Empty reasoning is valid and has no eligible KV.
    """

    start: int
    stop: int
    expires_at: int

    def __post_init__(self) -> None:
        for name in ("start", "stop", "expires_at"):
            _nonnegative_int(name, getattr(self, name))
        if not 0 < self.start <= self.stop < self.expires_at:
            raise ValueError("Expected 0 < start <= stop < expires_at")


@dataclass(frozen=True)
class MessageBoundary:
    """Trusted template/session metadata; never infer roles from body text.

    [start, end) includes the IFM message delimiters. ``body_start`` is just
    after the role/header, optionally before one template newline. Completion
    must be confirmed by the caller: length limits, cancellation and disconnect
    do not complete a message. Only the final message may be incomplete.
    """

    role: str
    start: int
    body_start: int
    end: int
    completed: bool = True

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported message role: {self.role}")
        for name in ("start", "body_start", "end"):
            _nonnegative_int(name, getattr(self, name))
        if not self.start < self.body_start <= self.end:
            raise ValueError("Expected start < body_start <= end")
        if type(self.completed) is not bool:
            raise ValueError("completed must be a boolean")
        if self.completed and self.body_start == self.end:
            raise ValueError("A completed message must include its terminal token")


@dataclass(frozen=True)
class IFMReasoningMarkers:
    message_start: int
    message_end: int
    think_pairs: tuple[tuple[int, int], ...]
    newline: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "think_pairs", tuple(tuple(p) for p in self.think_pairs)
        )
        object.__setattr__(self, "newline", tuple(self.newline))
        if not self.think_pairs or any(len(p) != 2 for p in self.think_pairs):
            raise ValueError("think_pairs must contain opening/closing token pairs")
        markers = [self.message_start, self.message_end]
        markers.extend(token for pair in self.think_pairs for token in pair)
        for token in (*markers, *self.newline):
            _nonnegative_int("marker token ID", token)
        if len(set(markers)) != len(markers):
            raise ValueError("IFM markers must have distinct token IDs")
        if not self.newline or set(self.newline).intersection(markers):
            raise ValueError("newline must be nonempty and contain no IFM markers")

    @classmethod
    def from_tokenizer(cls, tokenizer) -> IFMReasoningMarkers:
        """Resolve Horizon high/medium/low markers with a HF-style tokenizer."""

        def single(text: str) -> int:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1 or ids[0] == getattr(tokenizer, "unk_token_id", None):
                raise ValueError(f"IFM marker must be a single known token: {text}")
            return ids[0]

        return cls(
            message_start=single("<|ifm|im_start|>"),
            message_end=single("<|ifm|im_end|>"),
            think_pairs=tuple(
                (single(f"<ifm|{kind}>"), single(f"</ifm|{kind}>"))
                for kind in ("think", "think_fast", "think_faster")
            ),
            newline=tuple(tokenizer.encode("\n", add_special_tokens=False)),
        )


def annotate_reasoning(
    token_ids: Sequence[int],
    messages: Sequence[MessageBoundary],
    markers: IFMReasoningMarkers,
) -> tuple[ReasoningSpan, ...]:
    """Find unambiguous leading reasoning in completed assistant messages.

    Prompt-supplied reasoning priming is included, regardless of which tokens
    were sampled. Think tags in user/tool bodies or later answer text are not
    reasoning. A malformed/nested think block is retained in full. Ambiguous
    message delimiters reject the trace instead of guessing a message boundary.
    The caller must supply boundaries from trusted message structure, not by
    scanning arbitrary user/tool text for apparent headers.
    """
    pairs = dict(markers.think_pairs)
    think_ids = set(pairs) | set(pairs.values())
    spans = []
    previous_end = 0
    for index, message in enumerate(messages):
        if message.start < previous_end or message.end > len(token_ids):
            raise ValueError(
                "Messages must be ordered, disjoint and inside the transcript"
            )
        previous_end = message.end
        if not message.completed and (
            index != len(messages) - 1 or message.end != len(token_ids)
        ):
            raise ValueError(
                "An incomplete message must be the transcript's final suffix"
            )
        if token_ids[message.start] != markers.message_start:
            raise ValueError("Message start metadata does not match IFM token IDs")
        if message.completed and token_ids[message.end - 1] != markers.message_end:
            raise ValueError("Completed message is missing its terminal IFM token")
        body_end = message.end - int(message.completed)
        if any(
            token in (markers.message_start, markers.message_end)
            for token in token_ids[message.start + 1 : body_end]
        ):
            raise ValueError("Ambiguous message delimiter inside a message")
        if message.role != "assistant" or not message.completed:
            continue
        body = message.body_start
        if tuple(token_ids[body : body + len(markers.newline)]) == markers.newline:
            body += len(markers.newline)
        if body >= body_end or token_ids[body] not in pairs:
            continue
        close_id = pairs[token_ids[body]]
        closes = [i for i in range(body + 1, body_end) if token_ids[i] == close_id]
        if len(closes) != 1:
            continue
        stop = closes[0]
        if any(token in think_ids for token in token_ids[body + 1 : stop]):
            continue
        spans.append(ReasoningSpan(body + 1, stop, message.end))
    return tuple(spans)


@dataclass(frozen=True)
class ReferenceLayout:
    """Logical residency only; positions are never renumbered by compaction."""

    token_ids: tuple[int, ...]
    original_positions: tuple[int, ...]
    next_absolute_position: int


@dataclass(frozen=True)
class CompactionPlan:
    """Gather indices into the pre-boundary resident sequence, not GPU slots.

    The allocator must separately handle page ownership, moves and freeing.
    These indices alone must never be passed to a page allocator's free().
    """

    keep_indices: tuple[int, ...]
    drop_indices: tuple[int, ...]
    layout: ReferenceLayout


@dataclass(frozen=True)
class ReplayStep:
    """Teacher-force [start, stop), then drop these original KV positions.

    This includes the terminal message token's forward pass before eviction.
    Chunks never cross an eviction boundary, even when the window drops nothing.
    """

    start: int
    stop: int
    drop_after: tuple[int, ...]


class ReasoningEvictionReference:
    """Immutable CPU oracle for a single chronological token transcript.

    The dense mask, logical layout and replay plan are validation tools. They
    do not integrate with the scheduler or claim production serving support.
    ``spans`` must come from trusted annotations (or ``annotate_reasoning``).
    """

    def __init__(
        self,
        token_ids: Iterable[int],
        spans: Iterable[ReasoningSpan],
        policy: ReasoningEvictionPolicy,
    ) -> None:
        self._token_ids = tuple(token_ids)
        self._spans = tuple(spans)
        self._policy = policy
        for token in self.token_ids:
            _nonnegative_int("token ID", token)
        previous_stop = -1
        previous_expiry = 0
        for span in self.spans:
            if span.start - 1 <= previous_stop or span.expires_at < previous_expiry:
                raise ValueError("Reasoning spans must be ordered and disjoint")
            if span.expires_at > len(self.token_ids):
                raise ValueError("Reasoning boundary lies beyond the transcript")
            if span.expires_at != previous_expiry and span.start <= previous_expiry:
                raise ValueError("Reasoning spans from different messages overlap")
            previous_stop, previous_expiry = span.stop, span.expires_at
        expiries = [None] * len(self.token_ids)
        for span in self.spans:
            for position in policy.dropped_positions(span):
                expiries[position] = span.expires_at
        self._expires_at = tuple(expiries)

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self._token_ids

    @property
    def spans(self) -> tuple[ReasoningSpan, ...]:
        return self._spans

    @property
    def policy(self) -> ReasoningEvictionPolicy:
        return self._policy

    @property
    def expires_at(self) -> tuple[Optional[int], ...]:
        """Per-key first disallowed query position; None means never evicted."""
        return self._expires_at

    def visible_positions(self, query_position: int) -> tuple[int, ...]:
        _nonnegative_int("query_position", query_position)
        if query_position >= len(self.token_ids):
            raise ValueError("Query position lies beyond the transcript")
        return tuple(
            key
            for key in range(query_position + 1)
            if self.expires_at[key] is None or query_position < self.expires_at[key]
        )

    def attention_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Small-trace dense boolean oracle; True permits attention (O(N^2))."""
        return tuple(
            tuple(
                key <= query and (expiry is None or query < expiry)
                for key, expiry in enumerate(self.expires_at)
            )
            for query in range(len(self.token_ids))
        )

    def layout_after(self, processed_tokens: int) -> ReferenceLayout:
        """Residency after forwards [0, processed_tokens) and due evictions."""
        _nonnegative_int("processed_tokens", processed_tokens)
        if processed_tokens > len(self.token_ids):
            raise ValueError("Processed length lies beyond the transcript")
        positions = tuple(
            key
            for key in range(processed_tokens)
            if self.expires_at[key] is None or processed_tokens < self.expires_at[key]
        )
        return ReferenceLayout(
            tuple(self.token_ids[key] for key in positions), positions, processed_tokens
        )

    def plan_compaction(
        self,
        boundary: int,
        resident_positions: Sequence[int],
        *,
        committed_through: int,
    ) -> CompactionPlan:
        """Validate a completed boundary before constructing a logical gather.

        committed_through is an exclusive ORIGINAL position, not a resident
        length. Sampling the terminal token is insufficient. No post-boundary
        tokens may be forwarded until this transition has been applied.
        """
        _nonnegative_int("boundary", boundary)
        _nonnegative_int("committed_through", committed_through)
        if boundary not in {span.expires_at for span in self.spans}:
            raise ValueError("Not an annotated completed assistant boundary")
        if committed_through != boundary:
            raise ValueError(
                "Commit through the terminal token before boundary compaction"
            )
        before = self.visible_positions(boundary - 1)
        if tuple(resident_positions) != before:
            raise ValueError("Resident positions do not match the pre-boundary layout")
        layout = self.layout_after(boundary)
        retained = set(layout.original_positions)
        return CompactionPlan(
            tuple(i for i, position in enumerate(before) if position in retained),
            tuple(i for i, position in enumerate(before) if position not in retained),
            layout,
        )

    def replay_steps(
        self, max_chunk_tokens: Optional[int] = None
    ) -> tuple[ReplayStep, ...]:
        """Reconstruct full computation history, including discarded reasoning."""
        if max_chunk_tokens is not None:
            _nonnegative_int("max_chunk_tokens", max_chunk_tokens)
            if max_chunk_tokens == 0:
                raise ValueError("max_chunk_tokens must be positive")
        boundaries = sorted(
            {span.expires_at for span in self.spans} | {len(self.token_ids)}
        )
        drops = {}
        for position, expiry in enumerate(self.expires_at):
            if expiry is not None:
                drops.setdefault(expiry, []).append(position)
        steps = []
        start = 0
        for boundary in boundaries:
            while start < boundary:
                stop = (
                    min(boundary, start + max_chunk_tokens)
                    if max_chunk_tokens
                    else boundary
                )
                steps.append(ReplayStep(start, stop, tuple(drops.get(stop, ()))))
                start = stop
        return tuple(steps)
