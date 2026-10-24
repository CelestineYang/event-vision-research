# Frozen expert Train versus selection diagnostics

Train is in-sample for the experts. Selection uses original Test; not independent testing.

| Method | Train Top1 | Train NLL | Selection Top1 | Selection NLL |
|---|---:|---:|---:|---:|
| fusion | 1.00000000 | 0.00081342 | 0.96046967 | 0.35925712 |
| point | 0.99728050 | 0.04147509 | 0.88923679 | 0.58719954 |
| cstr3 | 0.99660062 | 0.03740706 | 0.92915851 | 0.31347329 |
| xt_p | 0.99922300 | 0.01702656 | 0.89980431 | 0.45017036 |
| ty_p | 0.99961150 | 0.01675316 | 0.92407045 | 0.36603939 |
| frame | 1.00000000 | 0.02140641 | 0.97103718 | 0.28017991 |
| frame_point | 0.99980575 | 0.02885853 | 0.95342466 | 0.33491225 |
| fusion_frame | 1.00000000 | 0.01083890 | 0.97025440 | 0.21757199 |
| three_way | 1.00000000 | 0.01906693 | 0.96790607 | 0.26841378 |
