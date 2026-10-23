# Frozen-checkpoint routing diagnostics

Selection protocol: original_test_as_val. Selection-set results are not independent-test results.
Interventions are inference-time diagnostics, not retrained architectural ablations.
Path weights are explicit mixture weights, not causal modality contributions.

| Evaluation | Video Top1 | Video Top5 | Video NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| conditional__normal__prefix_1 | 0.96320939 | 0.98943249 | 0.19640851 | 0.07053169 | 0.04765371 |
| conditional__reset_each_window__prefix_1 | 0.96320939 | 0.98943249 | 0.19564709 | 0.07046372 | 0.04759551 |
| conditional__zero_memory_readout__prefix_1 | 0.96320939 | 0.98943249 | 0.19842673 | 0.07067344 | 0.04786479 |
| cstr3__normal__prefix_1 | 0.92915851 | 0.98356164 | 0.31347329 | 0.12163750 | 0.07085974 |
| current__normal__prefix_1 | 0.96320939 | 0.98943249 | 0.19598949 | 0.07043972 | 0.04748753 |
| current__reset_each_window__prefix_1 | 0.96320939 | 0.98943249 | 0.19598949 | 0.07043972 | 0.04748753 |
| current__zero_memory_readout__prefix_1 | 0.96320939 | 0.98943249 | 0.19598949 | 0.07043972 | 0.04748753 |
| current_ema__normal__prefix_1 | 0.96281800 | 0.99099804 | 0.19671258 | 0.07152074 | 0.05281356 |
| frame__normal__prefix_1 | 0.97103718 | 0.99295499 | 0.28017991 | 0.09918141 | 0.15298506 |
| frame_point__normal__prefix_1 | 0.95342466 | 0.98864971 | 0.33491225 | 0.12371915 | 0.15590675 |
| fusion__normal__prefix_1 | 0.96046967 | 0.98864971 | 0.35925712 | 0.07326396 | 0.04096238 |
| fusion_frame__normal__prefix_1 | 0.97025440 | 0.99295499 | 0.21757199 | 0.07302189 | 0.10380440 |
| gru__normal__prefix_1 | 0.96320939 | 0.98943249 | 0.19685698 | 0.07052090 | 0.04764450 |
| gru__reset_each_window__prefix_1 | 0.96320939 | 0.98943249 | 0.19567217 | 0.07041245 | 0.04747703 |
| gru__zero_memory_readout__prefix_1 | 0.96320939 | 0.98943249 | 0.19833077 | 0.07067085 | 0.04782443 |
| point__normal__prefix_1 | 0.88923679 | 0.95890411 | 0.58719954 | 0.19592403 | 0.08888249 |
| static__normal__prefix_1 | 0.96242661 | 0.99099804 | 0.19723609 | 0.07165138 | 0.05338162 |
| static__reset_each_window__prefix_1 | 0.96242661 | 0.99099804 | 0.19723609 | 0.07165138 | 0.05338162 |
| static__zero_memory_readout__prefix_1 | 0.96242661 | 0.99099804 | 0.19723609 | 0.07165138 | 0.05338162 |
| three_way__normal__prefix_1 | 0.96790607 | 0.99021526 | 0.26841378 | 0.09227172 | 0.13265827 |
| ty_p__normal__prefix_1 | 0.92407045 | 0.98238748 | 0.36603939 | 0.13507894 | 0.08492054 |
| xt_p__normal__prefix_1 | 0.89980431 | 0.97847358 | 0.45017036 | 0.16863302 | 0.08757944 |

| Comparison | Prediction changes | Left only correct | Right only correct | Δ NLL (right−left) |
|---|---:|---:|---:|---:|
| conditional__reset_each_window__prefix_1 | 0 | 0 | 0 | -0.00076142 |
| conditional__zero_memory_readout__prefix_1 | 0 | 0 | 0 | 0.00201822 |
| current__reset_each_window__prefix_1 | 0 | 0 | 0 | 0.00000000 |
| current__zero_memory_readout__prefix_1 | 0 | 0 | 0 | 0.00000000 |
| gru__reset_each_window__prefix_1 | 0 | 0 | 0 | -0.00118481 |
| gru__zero_memory_readout__prefix_1 | 1 | 0 | 0 | 0.00147379 |
| static__reset_each_window__prefix_1 | 0 | 0 | 0 | 0.00000000 |
| static__zero_memory_readout__prefix_1 | 0 | 0 | 0 | 0.00000000 |
| static_vs_current__prefix_1 | 4 | 0 | 2 | -0.00124660 |
| current_vs_gru__prefix_1 | 0 | 0 | 0 | 0.00086749 |
| current_vs_conditional__prefix_1 | 1 | 0 | 0 | 0.00041901 |
| gru_vs_conditional__prefix_1 | 1 | 0 | 0 | -0.00044848 |
