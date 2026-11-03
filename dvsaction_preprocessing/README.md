# DVSAction preprocessing

`prepare.py` extracts the original SNN class from the archived author script without executing its dataset entry point. Source: https://github.com/rhwxmx/EventMamba/blob/main/dataprocess/generate_action.py (retrieved 2026-10-31).

Explicit Train/Test manifests are authoritative. Files absent from both are excluded. Author label order is preserved with folder-name aliases for spaces. No random resplitting.

The full DAVIS decoded stream is cached as int64 `[t_us,x,y,p]`. For point compatibility, the author's omission of the last physical AEDAT record is reproduced before selecting event indices `[N//2:N//2+int(2*N/5)]`. No timestamp repair is performed; decreasing timestamps inside the selected interval fail the run. Raw timestamp decreases outside that interval are preserved and recorded. One recording wraps uint32 before the selected interval; no timestamp repair is needed for its selected events. The original SNN, including its coordinate ordering and fixed 180 inhibition bounds, is preserved. Its reconstructed points have no polarity. These are cached separately as `[t_us,x,y]`.

Windows are 0.5 s, stride 0.25 s, end strictly before last reconstructed timestamp. The last event index <= window end is excluded by slicing, matching upstream. More than 5000 reconstructed points are required. Sample 2048 points in source order; normalize sampled time range and divide x,y by 150. Sampling uses deterministic per-window RNG instead of upstream unseeded randomness. Zero sampled time spans are invalid. No offline augmentation copies are created.

Outputs: train/test HDF5 (`data,label,mark`), per-split file and window JSONL manifests, full raw event and SNN caches, config with source hashes, audit with excluded files and zero-valid recordings, completion status. Marks are split-local. Raw window indices are temporal correspondences only; image rendering must also apply the documented spatial transform. Raw indices are restricted to the selected event range. The prepare step does not render images; run the separate render step below. Do not feed SNN points to polarity image renderers.

Only `status.json` with `status=complete,scope=full` represents complete data. No resume; use a new output directory after a failure. The smoke scope processes one recording per split. Four CPU workers are used for the full run; no GPU.

Validation performed before full run: vectorized decoder exactly matched the original decoder on a real 1,090,113-event recording; original utility window indices exactly matched both real smoke recordings; HDF5 shapes, finite values and normalization passed.

## Three-view cache

`render_frames.py` generates `cstr3`, `xt_p`, `ty_p` for every valid point window from full-polarity raw events at its recorded raw indices. It applies the matching spatial ROI: `x_cached-100`, `259-y_cached-50`, keeping coordinates 0..149. It does not apply SNN reconstruction or morphology to raw events. Thus this is a project fusion representation, not the author's SNN input.

Each view is float32 `[3,224,224]` in [0,1], with fixed window temporal normalization and existing THU projection math. Filenames are HDF5 `point_row`, under `train|test/view/row.npy`. Frame manifests include sample/window IDs, labels, source slice, event counts and paths. Every file is reread and compared exactly during generation. Empty crops produce explicit zero images and are audited without dropping windows. ImageNet normalization is deferred to the loader.

This is the deterministic unaugmented cache. Event-space augmentation during training requires online rerendering of the shared augmented events; a fixed cache is not equivalent. The image adapter is `image_dataset.py`; the fusion adapter is `../dvsaction_gm_data.py`. GM training integration is documented in `../DVSACTION_GM_RUNBOOK.md`.


## Reproduce the prepared data

The user-supplied recording lists are versioned as `splits/train_files.txt` and `splits/test_files.txt`. Keep them unchanged when reproducing the reported 226/57-video, 926/239-window split. From the repository root:

```bash
python dvsaction_preprocessing/prepare.py --root LOCAL_PATH/DVSAction --train-list dvsaction_preprocessing/splits/train_files.txt --test-list dvsaction_preprocessing/splits/test_files.txt --output LOCAL_PATH/dvsaction_prepared_v2 --workers 4
python dvsaction_preprocessing/render_frames.py --store LOCAL_PATH/dvsaction_prepared_v2 --output LOCAL_PATH/dvsaction_frames_v1
```

Dependencies include NumPy, h5py, PyTorch, torchvision, Pillow and OpenCV, in addition to the EventMamba CUDA extensions for model training. The upstream script is archived for provenance and its SNN extraction; run the project entry point above, not the author's original machine-specific entry point. Raw events, HDF5 and rendered NPY files are local artifacts, not repository contents.

Source license: [EventMamba MIT](../eventmamba_v2_upstream/LICENSE). Exact source/split hashes and split coverage are recorded in [provenance.json](provenance.json).
