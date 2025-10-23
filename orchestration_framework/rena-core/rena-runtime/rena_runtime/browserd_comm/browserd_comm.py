from typing import AsyncGenerator, Dict, Optional, Tuple, List
import logging
import uuid
import grpc
import asyncio

from rena_runtime.proto import browserd_pb2, browserd_pb2_grpc
from rena_runtime.browserd_comm.base import BaseBrowserdCommClient, BaseBrowserdComm

logger = logging.getLogger("rena_runtime")


class BrowserdClient:
    def __init__(self, browserd_url: str):
        self._stub = browserd_pb2_grpc.BrowserdStub(
            grpc.aio.insecure_channel(browserd_url)
        )

    def connect(
        self, to_browserd_stream: AsyncGenerator[browserd_pb2.Event, None]
    ) -> AsyncGenerator[browserd_pb2.Event, None]:
        """Connect to the browserd server."""
        return self._stub.Connect(to_browserd_stream)


class BrowserdCommClient(BaseBrowserdCommClient):
    def __init__(
        self,
        browserd_comm: "BrowserdComm",
        to_browserd_comm: asyncio.Queue,
    ):
        """Initialize the BrowserdCommClient.

        Args:
            to_browserd_comm (asyncio.Queue): Queue for sending events to browserd_comm.
        """
        self._browserd_comm = browserd_comm
        self.to_browserd_comm = to_browserd_comm
        self.requests: Dict[str, asyncio.Queue] = {}

    # TODO(sean): deregister client once done
    async def subscribe(self) -> AsyncGenerator[browserd_pb2.Event, None]:
        from_browserd_comm = asyncio.Queue()
        self._browserd_comm.register_client(to_client=from_browserd_comm)

        while True:
            event = await from_browserd_comm.get()
            event_id = str(uuid.UUID(bytes=event.id))

            if event_id in self.requests:
                self.requests[event_id].put_nowait(event)
                self.requests.pop(event_id)
                continue

            yield event

    async def publish(
        self,
        event: browserd_pb2.Event,
        expect_res: Optional[bool] = False,
        timeout: Optional[int] = None,
    ) -> Optional[browserd_pb2.Event]:
        if not expect_res:
            await self.to_browserd_comm.put(event)
            return None

        if expect_res and timeout is None:
            raise ValueError("timeout must be specified when expect_res is True")

        await self.to_browserd_comm.put(event)

        event_id = str(uuid.UUID(bytes=event.id))
        rx = asyncio.Queue()
        self.requests[event_id] = rx

        try:
            return await asyncio.wait_for(rx.get(), timeout)
        except asyncio.TimeoutError as e:
            raise ValueError(
                f"timeout waiting for res event. req_event_id: {event_id}"
            ) from e


class BrowserdComm(BaseBrowserdComm):
    """Browserd communication protocol."""

    browserd_client: BrowserdClient
    from_clients: asyncio.Queue
    to_clients: List[asyncio.Queue]

    def __init__(
        self,
        browserd_client: BrowserdClient,
        from_clients: asyncio.Queue,
    ):
        self.browserd_client = browserd_client
        self.from_clients = from_clients
        self.to_clients = []

    @classmethod
    def new(
        cls, browserd_client: BrowserdClient
    ) -> Tuple["BrowserdComm", BrowserdCommClient]:
        """Factory method to create a (BrowserdComm, BrowserdCommClient) pair."""
        from_clients = asyncio.Queue()

        browserd_comm = BrowserdComm(browserd_client, from_clients)

        return (
            browserd_comm,
            BrowserdCommClient(browserd_comm, from_clients),
        )

    async def run(self):
        logger.info("browserd_comm started")
        async for event in self.browserd_client.connect(self._to_browserd_stream()):
            event_id = str(uuid.UUID(bytes=event.id))
            logger.info(
                "received event from browserd. event_id: %s, event_type: %s",
                event_id,
                event.payload.WhichOneof("msg"),
            )

            if len(self.to_clients) == 0:
                logger.warning(
                    "no clients registered. dropping event. event_id: %s", event_id
                )
                continue

            for to_client in self.to_clients:
                await to_client.put(event)

    def register_client(self, to_client: asyncio.Queue):
        self.to_clients.append(to_client)

    async def _to_browserd_stream(self):
        """Stream of events from browserd_comm client to be sent to browserd."""
        while True:
            event = await self.from_clients.get()
            logger.info(
                "sending event to browserd. event_id: %s, event_type: %s",
                uuid.UUID(bytes=event.id),
                event.payload.WhichOneof("msg"),
            )
            yield event
