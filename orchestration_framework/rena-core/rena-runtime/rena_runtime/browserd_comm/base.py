from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator
import asyncio

from rena_runtime.proto import browserd_pb2


class BaseBrowserdCommClient(ABC):
    """
    Base class for browserd communication client.
    """

    @abstractmethod
    async def subscribe(self) -> AsyncGenerator[browserd_pb2.Event, None]:
        """
        Subscribe to the browserd events.
        """

    async def publish(
        self,
        event: browserd_pb2.Event,
        expect_res: Optional[bool] = False,
        timeout: Optional[int] = None,
    ) -> Optional[browserd_pb2.Event]:
        """
        Publish an event to the browserd.

        Args:
            expect_res (Optional[bool]): Whether to expect a response from the browserd.
                If True, subscribes to incomming browserd events,
                if incomming_event.id matchs with event.id, returns the incomming_event.
            timeout (Optional[int]): Only when expect_res is True, the timeout for waiting
                for res event.

        Raises:
            asyncio.TimeoutError: If expect_res is True and timeout is reached.
            Exception: If expect_res is True and timeout is None.
        """


class BaseBrowserdComm(ABC):
    """
    Base class for browserd communication protocol.
    It is responsible to setup (and run) the transport for the communication.
    To communicate with browserd, one needs to get a BrowserdCommClient instance
    which works with the BaseBrowserdComm.
    """

    @abstractmethod
    async def run(self):
        """Run the browserd communication protocol"""

    @abstractmethod
    def register_client(self, to_client: asyncio.Queue) -> "BaseBrowserdCommClient":
        """
        Register a new cloned client in order to notify the client when subscribed

        Args:
            to_client (asyncio.Queue): The client's tx queue to send events to.
        """
