# Rena Runtime

A Python module that handles incomming requests (e.g., run_app, query_app) from `browserd`.

## Usage

This module is not intended to be used directly. `browserd` uses this module as a container base image that once spawned, by default connects to specified `browserd` gRPC server url to handle to incomming requests.

## Appendix

Note: The reason it's not implemented in Rust, is the fact that it depends on agent-protocl (e.g., `mcp`) SDK be provided for the runtime's language. The `mcp` SDK is not provided for Rust yet. Python is one of the languages that `mcp` SDK is provided for.
