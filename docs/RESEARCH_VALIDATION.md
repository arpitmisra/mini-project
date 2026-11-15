# Local research workflow validation — 15 November 2026

Handoff review: the README now follows five sequential steps. Bash syntax validation and 16 targeted launcher/CLI/inspection tests passed, including both `--backend mlx` and `--backend=mlx` and preservation of quoted paths.

The follow-up found a real Mac fine-tuning failure in the earlier CI run: the upstream positional resize dispatched to a custom Metal kernel without a VJP. The local adapter now uses differentiable gathers and weighted sums. Independent float64 cubic interpolation/adjoint tests verify values and gradients at three output sizes; another test forces Metal availability and rejects any custom-kernel dispatch. After this fix, the complete opted-in MLX suite passed: **100 passed, 2 skipped, 61.58 seconds** on Windows CPU. Real full and LoRA integration executed; the two separate baseline integration tests were intentionally not enabled in the MLX environment. Version 3 of the adapter/training contract separates these results from older MLX checkpoints and caches.

**Published code commit `96bec10504bb285a0415f100b16001e3c773c4c6` passed all three workflows:**

| Workflow | Actual results |
|---|---|
| [Real pretrained baseline](https://github.com/arpitmisra/mini-project/actions/runs/43724574199) | macOS 15 ARM64, Windows and Linux CPU passed. Mac installation uses `bash oct.sh setup`; self-test, extraction/export/cache tests and attention inspection executed. |
| [Real MLX CPU](https://github.com/arpitmisra/mini-project/actions/runs/43724574195) | macOS 15 ARM64: **100 passed, 2 skipped in 24.52 s**. Windows: **100 passed, 2 skipped in 31.92 s**. Both passed dependency checks, real pretrained self-test and attention/normalization inspection. Full and LoRA adaptation integration tests executed. |
| [Portable tests](https://github.com/arpitmisra/mini-project/actions/runs/43724574221) | Windows and Linux passed. |

The two MLX-suite skips are PyTorch baseline integration tests, executed in the separate baseline workflow. CI uses generated small image fixtures, not the full OCT dataset. This establishes the CPU path on a real Apple Silicon runner; it does not test the professor's exact machine or Apple GPU training. These records were added in a documentation-only follow-up to the tested code commit.

Published commit `9c1c84b` passed baseline checks on Windows/Linux, portable tests on both, and MLX CPU checks on Windows. Its Mac jobs did not start because the runner was retired. Both workflows now use the supported `macos-15` Apple Silicon runner. A fresh clone of `9c1c84b` also passed the real pretrained self-test and random-noise dashboard locally, using the existing baseline interpreter and cached model.

After the version-3 correction, all four real local OCT conditions were rerun successfully without downloads: `outputs/research-validation/v3-{lora,full}-{imagenet,unit}/`. Each did one training step and extracted 12 scans. Initial losses match across LoRA/full within each condition (9.057210 for imagenet; 8.932680 for unit). Export via `oct.py show` succeeded for the new LoRA run. Corrected MLX attention/features for the same OCT scan again matched the separately computed PyTorch reference at atol=5e-4, rtol=1e-3; maximum attention difference 1.365e-5 and maximum patch difference 2.270e-4. Evidence is saved locally in `outputs/research-validation/v3-parity.json` and `v3-inspection-oct/`.

Executed on the local Windows CPU, with the previously installed separate baseline and MLX environments. No dataset or pretrained-weight download was performed. These are software checks, not evidence of improved segmentation.

- Before the Mac interpolation correction: entire opted-in MLX suite **90 passed, 2 skipped**, 50.23 seconds. The skips are the separate PyTorch integration tests. Includes real pretrained full/LoRA adaptation, strict merged checkpoint reload, adapted extraction/cache reuse, native gradients/EMA/resume and new LoRA/normalization tests.
- Four real local OCT smoke conditions passed: LoRA/full crossed with pretrained mean/std (`imagenet`) or RGB/255-only (`unit`). Each uses one optimizer step on two selected training scans, then extracts four scans per split (12 total). Results: `outputs/research-validation/{lora,full}-{imagenet,unit}/`.
- Equal initial SSL loss within each preprocessing condition confirms the controlled initialization on these inputs: full and LoRA both 9.057213 for imagenet, and both 8.932679 for unit. These are pre-update one-step losses, not evidence that one normalization is better.
- Trainable backbone parameters: LoRA rank 8 = **147,456**; full = **22,056,576**. Including smoke configuration SSL heads: 204,992 versus 22,114,112. LoRA tests verify unchanged frozen weights and unchanged key slice, nonzero updates, and merged/unmerged agreement.
- Actual random-noise PyTorch inspection passed. The same real OCT scan passed in both MLX and PyTorch, and the merged LoRA checkpoint passed MLX inspection.
- Exact inspection arrays passed attention shape/row-sum/finite checks, repeated ordinary feature inference agreement and NPZ round-trip equality. Input ablation uses identical crops and weights.
- Independent MLX versus PyTorch comparison of the real OCT scan passed at atol=5e-4, rtol=1e-3. Largest absolute differences: imagenet attention 1.435e-5, patches 2.384e-4, CLS 3.338e-5; unit attention 6.661e-6, patches 2.282e-4, CLS 2.861e-5.
- Offline HTML opened in the local Opera browser. Patch clicks changed the query and map; Pearson selection changed relationship maps and disabled the head selector; normalization selection changed the displayed matrices.

The first LoRA export exposed expected numerical differences from merging matrix operations. The code now checks merge agreement with a documented float32 tolerance separately from strict serialization agreement of the merged model. Another review caught LoRA RNG consumption changing SSL-head initialization; the implementation now seeds heads independently and records the new training-contract version.

Apple GPU training, full-dataset throughput, convergence, segmentation accuracy and patient independence remain unverified by these smoke checks.
