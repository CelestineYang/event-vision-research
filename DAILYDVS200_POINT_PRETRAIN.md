# Independent EventMamba V2 for current DailyDVS-200 windows

Historical protocol. For the new paper-method run (Adam, batch32, 350 epochs,
full-event offline preprocessing), use [EVENTMAMBA_PAPER_REPRODUCTION.md](EVENTMAMBA_PAPER_REPRODUCTION.md).

Entry: train_dailydvs200_point_prepared.py. New run from random initialization; no public, historical, image, or FusionA weight is used. This is our 200-class adaptation, not reproduction of the author's 12-class DailyAction protocol. Existing train_eventmamba_v2_only.py and its 1.5-second/350-epoch protocol are unchanged.

Data: prepared_v1 HDF5,110841 Train windows/14773 videos,31120 Test windows/4093 videos. 1-second half-open windows,0.5-second stride,short recordings and end-aligned tails retained. Input float32 [3,2048], t,x,y. Fixed-window normalized time is preserved; no sampled-time renormalization. Train uses deterministic per-window/epoch sampled-point rotation +/-15 degrees and random spatial crop up to10% per edge, then time-ordered resampling to2048. This augments sampled points, not the full events used by the image branches. Evaluation reads exact unaugmented prepared points.

Model: V2 embedding32,three stage outputs64/128/256,groups1024/512/256,K24,feature KNN and temporal FPS index sorting,200-class head. All model weights and BN statistics train normally. Seed0,batch32,150epochs,AdamW LR1e-3,wd1e-4,cosine eta_min0,FP32. Loss follows upstream smoothing: true class0.8,other classes0.2/(C-1). No early stopping. Singleton final Train batches are merged without dropping any samples.

Original Test is evaluated each epoch; best videoTop1 (lowerNLL tie) and bestNLL saved separately. Video metrics use arithmetic mean window probabilities. Test-as-validation, independent_test=false. This differs from the old final-only evaluation run; no claim of an independent held-out final score.

Checkpoints: last.pt (net,optimizer,scheduler,RNG,config/data/source identities), best_video_top1.pt,best_video_nll.pt. Exact configuration/source identity required on --resume. Do not edit source during active training.

Smoke output: experiments/dailydvs200_point_smoke_20261031;65 Train windows, two optimizer updates (32+33),Train-only evaluation,then disk resume to epoch2. This deliberately tests singleton minibatch handling. The formal model starts fresh.
