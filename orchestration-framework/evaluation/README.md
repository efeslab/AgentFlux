Filesys:

Results:

* small workspace
  * eval trajs: /home/zhan/rena/rena-eval-judge/filesys/eval-results/small_workspace_onlyread
  * judge results: /home/zhan/rena/rena-eval-judge/filesys/judge-results/small_workspace_onlyread
* large workspace
  * eval trajs: /home/zhan/rena/rena-eval-judge/filesys/eval-results/onlyread/
  * judge results: TODO (need to be generated)



combination eval: python/run_start_end_rerun_ife_1_times_query.py

* export WORKSPACE=/home/zhan/rena/filesys/8-21-evaluation/workspaces/workspace
* --query 
  * large workspace
    * /home/zhan/rena/rena-eval-judge/filesys/queries/filesys_detailed_wo_modify.txt
    * /home/zhan/rena/rena-eval-judge/filesys/queries/filesys_fuzzing_wo_modify.txt
  * small workspace
    * /home/zhan/rena/mcp-bench/filesys_detailed_small_workspace_onlyread.txt
    * /home/zhan/rena/mcp-bench/filesys_fuzzing_small_workspace_onlyread.txt
* --proxy_log_dir xxx



baseline eval: python run_baseline.py

* /home/zhan/rena/rena-eval-judge/filesys/scripts/baseline.sh



Judging script: filesys/scripts/gptsum_judge_batch_error.py

* export WORKSPACE=
  * /home/zhan/rena/filesys/8-21-evaluation/workspaces/workspace
  * /home/zhan/rena/small_workspace/workspace
* trajs: 
* output: 



Get score and ratio: python/llm_gt_judge_score_success.py

* --llm_judge_path