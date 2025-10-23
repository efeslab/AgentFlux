# Evals

This directory contains two types of evals:

- real-world
- basic (test)

To run these evaluations please refer to [run-evals](../../rena-browserd/browserd-eval/README.md#use-cases-evaluation). To see the results of already runned evaluations, please refer to [eval-results](../../rena-browserd/browserd-eval/README.md#use-cases-evaluation-results).

## Real World Evals

[Real World Evals](./eval.toml) were mainly hand-selected to represent a real-world use-case.

## Basic Evals

To derive [Basic Evals](./test-eval.toml), we used the following approach:

- The query shouldn't mention the steps required to execute it (to test the ability of orchestrator to come up with right plan)
- At the same time there needs to be at least one plan capable of executing the query correctly
- All possible plans should reach to the same ideal
- Prefer evaluation queries that generated ExecutionPlan for them can be executed in parallel (or at least some part of it)
- Use cases should be ordered from simple to complex
- Only use a set of ready MCPs (currently tested that can run within browserd):
  - Filesystem
  - Playwright
  - Repomix
  - Obsidian (needs to run obsidian app locally and derive obsidian api-key following this [guide](https://github.com/MarkusPfundstein/mcp-obsidian?tab=readme-ov-file#install))
  - BraveSearch (needs to derive brave-search api-key)
