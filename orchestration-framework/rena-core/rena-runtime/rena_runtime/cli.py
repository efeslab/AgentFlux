import argparse
import logging
import asyncio
from uuid import UUID

from rena_runtime.runtime import Runtime
from rena_runtime.browserd_comm.browserd_comm import BrowserdComm, BrowserdClient
from rena_runtime.utils import setup_logging

RENA_RUNTIME = "rena_runtime"


async def run(args: argparse.Namespace) -> None:
    # first create browserd_comm
    browserd_comm, browserd_comm_client = BrowserdComm.new(
        browserd_client=BrowserdClient(args.browserd_url)
    )
    runtime = Runtime(
        container_id=UUID(args.container_id),
        browserd_comm_client=browserd_comm_client,
    )

    # spawn browserd_comm task
    browserd_comm_task = asyncio.create_task(browserd_comm.run())

    # spawn runtime task
    runtime_task = asyncio.create_task(runtime.run())

    # wait for tasks, if any one failes log and cancel the other
    try:
        await asyncio.gather(
            browserd_comm_task,
            runtime_task,
        )
    except Exception as e:
        logging.getLogger(RENA_RUNTIME).exception(
            "browserd_comm or runtime tasks failed. error %s", e
        )
    finally:
        browserd_comm_task.cancel()
        runtime_task.cancel()

        await asyncio.gather(
            browserd_comm_task,
            runtime_task,
            return_exceptions=True,
        )


# TODO(sean): change the cli call logic within browserd rust crate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rena Runtime CLI")
    parser.add_argument(
        "container_id",
        metavar="CONTAINER_ID",
        type=str,
        help="Container ID where this runtime is running on.",
    )
    parser.add_argument(
        "browserd_url",
        metavar="BROWSERD_URL",
        type=str,
        help="Browserd URL to connect to.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    cli_args = parser.parse_args()
    setup_logging(name=RENA_RUNTIME, verbose=cli_args.verbose)

    asyncio.run(run(cli_args))
