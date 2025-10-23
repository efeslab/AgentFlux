#!/bin/bash

category=${1:-filesys}

port=9015
classifier=config/$category/classifier.json
tool_adapters=config/$category/tool_adapters.json
tool_list=config/$category/tool_list.json

python -m agentflux.proxy \
  --category $category \
  --classifier $classifier \
  --tool_adapters $tool_adapters \
  --tool_list $tool_list \
  --port $port
