# Local research workflow validation — 15 November 2026

Executed on the local Windows CPU, with the previously installed separate baseline and MLX environments. No dataset or pretrained-weight download was performed. These are software checks, not evidence of improved segmentation.

- Entire opted-in MLX suite: **90 passed, 2 skipped**, 50.23 seconds. The skips are the separate PyTorch integration tests. Includes real pretrained full/LoRA adaptation, strict merged checkpoint reload, adapted extraction/cache reuse, native gradients/EMA/resume and new LoRA/normalization tests.
- Four real local OCT smoke conditions passed: LoRA/full crossed with pretrained mean/std (`imagenet`) or RGB/255-only (`unit`). Each uses one optimizer step on two selected training scans, then extracts four scans per split (12 total). Results: `outputs/research-validation/{lora,full}-{imagenet,unit}/`.
- Equal initial SSL loss within each preprocessing condition confirms the controlled initialization on these inputs: full and LoRA both 9.057213 for imagenet, and both 8.932679 for unit. These are pre-update one-step losses, not evidence that one normalization is better.
- Trainable backbone parameters: LoRA rank 8 = **147,456**; full = **22,056,576**. Including smoke configuration SSL heads: 204,992 versus 22,114,112. LoRA tests verify unchanged frozen weights and unchanged key slice, nonzero updates, and merged/unmerged agreement.
- Actual random-noise PyTorch inspection passed. The same real OCT scan passed in both MLX and PyTorch, and the merged LoRA checkpoint passed MLX inspection.
- Exact inspection arrays passed attention shape/row-sum/finite checks, repeated ordinary feature inference agreement and NPZ round-trip equality. Input ablation uses identical crops and weights.
- Independent MLX versus PyTorch comparison of the real OCT scan passed at atol=5e-4, rtol=1e-3. Largest absolute differences: imagenet attention 1.435e-5, patches 2.384e-4, CLS 3.338e-5; unit attention 6.661e-6, patches 2.282e-4, CLS 2.861e-5.
- Offline HTML opened in the local Opera browser. Patch clicks changed the query and map; Pearson selection changed relationship maps and disabled the head selector; normalization selection changed the displayed matrices.

The first LoRA export exposed expected numerical differences from merging matrix operations. The code now checks merge agreement with a documented float32 tolerance separately from strict serialization agreement of the merged model. Another review caught LoRA RNG consumption changing SSL-head initialization; the implementation now seeds heads independently and records the new training-contract version.

CI includes the new attention/normalization inspection on Windows/Linux/macOS baseline and Windows/macOS MLX CPU. These added remote jobs have not been claimed as passed based on local results. Apple GPU training, full-dataset throughput, convergence, segmentation accuracy and patient independence remain unverified by these smoke checks.
