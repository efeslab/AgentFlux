#!/bin/bash

category=filesys
port=9015
classifier=../config/filesys/classifier.json
tool_adapters=../config/filesys/tool_adaptor.json
tool_list=../config/filesys/tool_list.json

python -m agentflux.proxy \
  --category $category \
  --classifier $classifier \
  --tool_adapters $tool_adapters \
  --tool_list $tool_list \
  --port $port
