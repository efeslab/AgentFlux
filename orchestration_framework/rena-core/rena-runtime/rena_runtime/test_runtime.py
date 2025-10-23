import logging
import asyncio
from uuid import UUID

from rena_runtime.runtime import Runtime
from rena_runtime.browserd_comm.browserd_comm import BrowserdComm, BrowserdClient
from rena_runtime.utils import setup_logging


# steps to test the runtim-browserd connection
# 1. test app registry by first push an app budle to it and then pull it (passed)
# 2. then run the browserd, it should spawn the gRPC server
# (but its a rust lib, so no way to test it without building and running through electorn app)
# then run the runtim, it should connect to browserd
# the container_id to use inside browserd let container_id = uuid::Uuid::from_str("c3b56fb6-7f34-4ce7-9b90-3c1f03e72012").unwrap();
async def main():
    setup_logging(name="rena_runtime", verbose=True)

    # runtime = Runtime(container_id="c3b56fb6-7f34-4ce7-9b90-3c1f03e72012")
    # await runtime.run(browserd_url="localhost:50051")

    # first create browserd_comm
    browserd_comm, browserd_comm_client = BrowserdComm.new(
        browserd_client=BrowserdClient("localhost:50051")
    )
    runtime = Runtime(
        container_id=UUID("c3b56fb6-7f34-4ce7-9b90-3c1f03e72012"),
        browserd_comm_client=browserd_comm_client,
    )

    # spawn browserd_comm task
    browserd_comm_task = asyncio.create_task(browserd_comm.run())

    # spawn runtime task
    runtime_task = asyncio.create_task(runtime.run())

    # wait for tasks, if any one failes raise error
    try:
        await asyncio.gather(
            browserd_comm_task,
            runtime_task,
        )
    except Exception as e:
        logging.getLogger("rena_runtime").exception(
            "browserd_comm or runtime tasks failed. error %s", e
        )


if __name__ == "__main__":
    asyncio.run(main())
