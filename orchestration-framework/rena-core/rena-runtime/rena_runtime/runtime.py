from typing import Dict, List
import asyncio
import logging
import uuid
from pathlib import Path
import shutil
from uuid import UUID
import json

from rena_runtime.data_types import AppBundle
from rena_runtime.proto import browserd_pb2
from rena_runtime.browserd_comm.base import BaseBrowserdCommClient
from rena_runtime.agent_protocol.base import BaseAgentProtocol
from rena_runtime.agent_protocol.mcp import MCPAgent


logger = logging.getLogger("rena_runtime")


RENA_RUNTIME = "RENA_RUNTIME"
APPS_BASE_PATH = Path("./rena_runtime/apps")
DEFAULT_INFERENCE_ENGINE = "anthropic"


class Runtime:
    def __init__(
        self, container_id: UUID, browserd_comm_client: BaseBrowserdCommClient
    ):
        self.container_id = container_id
        self.browserd_comm_client = browserd_comm_client

        self.running_apps = []

    # TODO(sean): handle the case where app preparation succeeds but app.run() fails
    async def run(self):
        """
        Listens to browserd incomming events, and handle app_req
        (i.e., run_app, and query_app) and container_req (i.e., get_container_id) events.
        """
        async for event in self.browserd_comm_client.subscribe():
            match event.payload.WhichOneof("msg"):
                case "app_req":
                    container_id = uuid.UUID(bytes=event.payload.app_req.container_id)
                    if self.container_id != container_id:
                        logger.warning(
                            "container id mismatch. expected %s, received %s",
                            self.container_id,
                            container_id,
                        )
                        continue

                    match event.payload.app_req.WhichOneof("request"):
                        case "run_app_req":
                            _ = asyncio.create_task(
                                _run_app(
                                    app_bundle=AppBundle.from_proto(
                                        event.payload.app_req.run_app_req.app_bundle
                                    ),
                                    res_event_id=event.id,
                                    browserd_comm_client=self.browserd_comm_client,
                                    running_apps=self.running_apps,
                                )
                            )
                        case "list_tools_req":
                            _ = asyncio.create_task(
                                _list_tools(
                                    res_event_id=event.id,
                                    running_apps=self.running_apps,
                                    browserd_comm_client=self.browserd_comm_client,
                                )
                            )
                        case "call_tool_req":
                            _ = asyncio.create_task(
                                _call_tool(
                                    id_=event.payload.app_req.call_tool_req.id,
                                    tool_name=event.payload.app_req.call_tool_req.tool_name,
                                    tool_input=event.payload.app_req.call_tool_req.tool_input,
                                    res_event_id=event.id,
                                    browserd_comm_client=self.browserd_comm_client,
                                    running_apps=self.running_apps,
                                )
                            )
                        case _:
                            pass
                case "container_req":
                    match event.payload.container_req.WhichOneof("request"):
                        case "get_container_id_req":
                            await self.browserd_comm_client.publish(
                                event=browserd_pb2.Event(
                                    id=event.id,
                                    payload=browserd_pb2.Event.Payload(
                                        container_res=browserd_pb2.ContainerResponse(
                                            get_container_id_res=browserd_pb2.GetContainerIdResponse(
                                                container_id=self.container_id.bytes
                                            )
                                        )
                                    ),
                                )
                            )
                        case _:
                            pass
                case _:
                    pass


async def _run_app(
    app_bundle: AppBundle,
    res_event_id: bytes,
    browserd_comm_client: BaseBrowserdCommClient,
    running_apps: List[BaseAgentProtocol],
) -> None:
    error = None

    if running_apps:
        running_apps.pop()

    try:
        app = MCPAgent(server_path=_dump_app(app_bundle) if app_bundle.files else None)

        if app_bundle.config.scripts:
            await app.run_scripts(app_bundle.config.scripts)

        await app.run(
            app_bundle.config.command,
            app_bundle.config.args,
            app_bundle.config.env,
        )

        running_apps.append(app)
    except Exception as e:
        error = e

    try:
        await browserd_comm_client.publish(
            event=browserd_pb2.Event(
                id=res_event_id,
                payload=browserd_pb2.Event.Payload(
                    app_res=browserd_pb2.AppResponse(
                        run_app_res=browserd_pb2.RunAppResponse(
                            success=not bool(error),
                            error=None if not error else str(error),
                        )
                    )
                ),
            )
        )
    except Exception as e:
        logger.exception(e)
        raise e


async def _list_tools(
    res_event_id: bytes,
    browserd_comm_client: BaseBrowserdCommClient,
    running_apps: List[BaseAgentProtocol],
):
    error = None

    try:
        if not running_apps:
            raise ValueError("app not running")

        tools = await running_apps[0].list_tools()
    except Exception as e:
        error = str(e)

    try:
        await browserd_comm_client.publish(
            event=browserd_pb2.Event(
                id=res_event_id,
                payload=browserd_pb2.Event.Payload(
                    app_res=browserd_pb2.AppResponse(
                        list_tools_res=browserd_pb2.ListToolsResponse(
                            tools=(
                                [tool.to_proto() for tool in tools] if not error else []
                            ),
                            error=error,
                        )
                    )
                ),
            )
        )
    except Exception as e:
        logger.exception(e)
        raise e


async def _call_tool(
    id_: str,
    tool_name: str,
    tool_input: str,
    res_event_id: bytes,
    browserd_comm_client: BaseBrowserdCommClient,
    running_apps: List[BaseAgentProtocol],
):
    call_tool_result = None
    error = None

    try:
        if not running_apps:
            raise ValueError("app not running")

        call_tool_result = await running_apps[0].call_tool(
            tool_name, tool_input=json.loads(tool_input)
        )
    except Exception as e:
        logger.exception(e)
        error = str(e)

    try:
        await browserd_comm_client.publish(
            event=browserd_pb2.Event(
                id=res_event_id,
                payload=browserd_pb2.Event.Payload(
                    app_res=browserd_pb2.AppResponse(
                        call_tool_res=(
                            call_tool_result.to_proto(tool_use_id=id_)
                            if not error
                            else browserd_pb2.CallToolResponse(
                                id=id_,
                                tool_results=[],
                                error=error,
                            )
                        )
                    )
                ),
            )
        )
    except Exception as e:
        logger.exception(e)
        raise e


def _dump_app(app_bundle: AppBundle) -> str:
    app_path = APPS_BASE_PATH.joinpath(app_bundle.config.name)
    if app_path.exists():
        shutil.rmtree(app_path)

    for file in app_bundle.files:
        file_path = app_path.joinpath(Path(file.relative_path))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(file.content)

    return str(app_path)
