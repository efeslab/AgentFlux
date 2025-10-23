from typing import List, Optional, Tuple
from datetime import datetime

from pydantic import BaseModel

from rena_runtime.proto import browserd_pb2


class Trace(BaseModel):
    label: str
    latency: Optional[int] = None  # ms
    metadata: Optional[str] = None
    inner_traces: Optional[List["Trace"]] = None

    @classmethod
    def start(cls, label: str) -> Tuple["Trace", datetime]:
        return (cls(label=f"(runtime) {label}"), datetime.now())

    def end(
        self,
        start: datetime,
        metadata: Optional[str] = None,
        inner_traces: Optional[List["Trace"]] = None,
    ) -> None:
        self.latency = int((datetime.now() - start).total_seconds() * 1000)
        self.metadata = metadata
        self.inner_traces = inner_traces

    def to_proto(self) -> browserd_pb2.Trace:
        inner_traces = (
            browserd_pb2.TraceList(
                traces=[inner.to_proto() for inner in self.inner_traces]
            )
            if self.inner_traces
            else None
        )

        return browserd_pb2.Trace(
            label=self.label,
            latency=self.latency,
            metadata=self.metadata,
            inners=inner_traces,
        )

    @classmethod
    def from_proto(cls, trace: browserd_pb2.Trace) -> "Trace":
        return cls(
            label=trace.label,
            latency=trace.latency,
            metadata=trace.metadata,
            inners=(
                [cls.from_proto(inner) for inner in trace.inner_traces.traces]
                if trace.inner_traces
                else None
            ),
        )
