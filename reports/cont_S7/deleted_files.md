# Deleted weights and partial runs (2026-10-15, disk cleanup at the user's request)

Results are in the reports (summary.md / rounds.md / conclusions.md / routing_v4.md / learner_checks.txt). Re-running scripts/cont_check_learners.py now needs the round-1 cls3_oracle state regenerated (S7 chunks 0-6).

```
eval_outputs/cont_S2/cls2_free/chunk_00_I/continual_state.pt
eval_outputs/cont_S2/cls2_free/chunk_01_square/continual_state.pt
eval_outputs/cont_S2/cls2_free/chunk_02_T/continual_state.pt
eval_outputs/cont_S2/cls2_free/chunk_03_I/continual_state.pt
eval_outputs/cont_S2/cls2_free/chunk_04_square/continual_state.pt
eval_outputs/cont_S2/cls2_oracle/chunk_00_I/continual_state.pt
eval_outputs/cont_S2/cls2_oracle/chunk_01_square/continual_state.pt
eval_outputs/cont_S2/cls2_oracle/chunk_02_T/continual_state.pt
eval_outputs/cont_S2/cls2_oracle/chunk_03_I/continual_state.pt
eval_outputs/cont_S2/cls2_oracle/chunk_04_square/continual_state.pt
eval_outputs/cont_S2/w0.003/chunk_00_I/continual_state.pt
eval_outputs/cont_S2/w0.003/chunk_01_square/continual_state.pt
eval_outputs/cont_S2/w0.003/chunk_02_T/continual_state.pt
eval_outputs/cont_S2/w0.003/chunk_03_I/continual_state.pt
eval_outputs/cont_S2/w0.003/chunk_04_square/continual_state.pt
eval_outputs/cont_S2/w0/chunk_00_I/continual_state.pt
eval_outputs/cont_S2/w0/chunk_01_square/continual_state.pt
eval_outputs/cont_S2/w0/chunk_02_T/continual_state.pt
eval_outputs/cont_S2/w0/chunk_03_I/continual_state.pt
eval_outputs/cont_S2/w0/chunk_04_square/continual_state.pt
eval_outputs/cont_S7/cls2_free/chunk_06_T/continual_state.pt
eval_outputs/cont_S7/cls2_free/chunk_13_Z/continual_state.pt
eval_outputs/cont_S7/cls3_free/chunk_06_T/continual_state.pt
eval_outputs/cont_S7/cls3_free/chunk_13_Z/continual_state.pt
eval_outputs/cont_S7/cls3_oracle/chunk_06_T/continual_state.pt
eval_outputs/cont_S7/cls3_oracle/chunk_13_Z/continual_state.pt
eval_outputs/cont_S7/cls_oracle/chunk_06_T/continual_state.pt
eval_outputs/cont_S7/cls_oracle/chunk_13_Z/continual_state.pt
eval_outputs/cont_S7/w0/chunk_06_T/continual_state.pt
eval_outputs/cont_S7/w0/chunk_13_Z/continual_state.pt
checkpoints/slow_predall_I_S2b0
checkpoints/slow_predall_square_S2b1
eval_outputs/cont_S7b/w0 (stopped in chunk 0)
eval_outputs/cont_S7b/frozen (stopped after 5 chunks: I 20%, L 65%, square 20%, Z 85%, small_tee 85%)
eval_outputs/cont_S7b/cls3_free (stopped in chunk 0)
```
