from enum import Enum
from typing import List, Union, Dict, Iterable, Any, Optional
import logging
import json
from typing_extensions import Literal

from pydantic import BaseModel

from rena_runtime.proto import common_pb2, browserd_pb2

logger = logging.getLogger("rena_runtime")


class Runtime(Enum):
    PYTHON = "python"
    NODEJS = "nodejs"


class AgentProtocol(Enum):
    MCP = "mcp"


AGENT_PROTOCOL_MAPPINGS = {
    browserd_pb2.AgentProtocol.MCP: AgentProtocol.MCP,
    AgentProtocol.MCP: browserd_pb2.AgentProtocol.MCP,
}


class AppConfig(BaseModel):
    name: str
    agent_protocol: AgentProtocol
    command: str
    args: List[str]
    env: Dict[str, str]
    scripts: Optional[List[str]]

    @classmethod
    def from_proto(cls, app_config: browserd_pb2.AppConfig) -> "AppConfig":
        return AppConfig(
            name=app_config.name,
            agent_protocol=AGENT_PROTOCOL_MAPPINGS[app_config.agent_protocol],
            command=app_config.command,
            args=list(app_config.args),
            env=app_config.env,
            scripts=list(app_config.scripts) if app_config.scripts else None,
        )

    def to_proto(self) -> browserd_pb2.AppConfig:
        return browserd_pb2.AppConfig(
            name=self.name,
            agent_protocol=AGENT_PROTOCOL_MAPPINGS[self.agent_protocol],
            command=self.command,
            args=self.args,
            env=self.env,
            scripts=self.scripts,
        )


class AppFile(BaseModel):
    relative_path: str
    content: bytes

    @classmethod
    def from_proto(cls, app_file: common_pb2.AppFile) -> "AppFile":
        return AppFile(
            relative_path=app_file.relative_path,
            content=app_file.content,
        )

    def to_proto(self) -> common_pb2.AppFile:
        return common_pb2.AppFile(
            relative_path=self.relative_path,
            content=self.content,
        )


class AppBundle(BaseModel):
    config: AppConfig
    files: List[AppFile]

    @classmethod
    def from_proto(cls, app_bundle: browserd_pb2.AppBundle) -> "AppBundle":
        return AppBundle(
            config=AppConfig.from_proto(app_bundle.config),
            files=[
                AppFile(
                    relative_path=app_file.relative_path,
                    content=app_file.content,
                )
                for app_file in app_bundle.files.files
            ],
        )

    def to_proto(self) -> browserd_pb2.AppBundle:
        return common_pb2.AppBundle(
            config=self.config.to_proto(),
            files=common_pb2.AppFileList(
                files=[app_file.to_proto() for app_file in self.files]
            ),
        )


class TextBlock(BaseModel):
    text: str

    type: Literal["text"]


class ToolUseBlock(BaseModel):
    id: str

    input: object

    name: str

    type: Literal["tool_use"]


class ToolResultBlockParam(BaseModel):
    type: Literal["tool_result"]

    content: Union[str, Iterable[TextBlock]]

    is_error: bool

    def to_proto(self, tool_use_id: str) -> browserd_pb2.CallToolResponse:
        results = (
            [self.content]
            if isinstance(self.content, str)
            else [block.text for block in self.content]
        )
        return browserd_pb2.CallToolResponse(
            id=tool_use_id,
            tool_results=([] if self.is_error else results),
            error=None if not self.is_error else " ".join(results),
        )


class ToolParam(BaseModel):
    input_schema: Dict[str, Any]

    name: str

    description: str

    def to_proto(self) -> browserd_pb2.Tool:
        return browserd_pb2.Tool(
            input_schema=json.dumps(self.input_schema),
            name=self.name,
            description=self.description,
        )
