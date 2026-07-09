from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    event_name: str
    payload: dict
    created_at: str = datetime.now(UTC).isoformat()

