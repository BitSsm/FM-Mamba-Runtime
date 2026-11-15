# Short-run validation

Environment: Windows x64 / RTX 5090 D / Python 3.11.16 / PyTorch 2.10.0+cu128.

Each group was randomly initialized and used 99 training images and 16 validation images, with training batches of 64 and 35. A checkpoint was saved after the first step, followed by a process restart and resume at the second step. Evaluation, benchmarking, resource logging, and CUDA trace export then completed. Validation processes rejected JIT compilation and excluded compiler directories from PATH.

Numerical checks compared the original Python implementation with the binary implementation using identical seeds, inputs, and native kernels. Checks covered forward logits, BCE, all parameter gradients, and weights after one SGD update. Per-tensor tolerances were atol=2e-5 and rtol=2e-4.

|Group|Train/resume/evaluate/benchmark|Maximum absolute logit difference|Maximum absolute gradient difference|CUDA kernel events (5 forward passes)|
|---|---|---:|---:|---:|
|`vim`|Passed|0|4.66e-09|5740|
|`vmamba`|Passed|0|9.31e-09|5665|
|`2dmamba`|Passed|0|1.12e-08|5665|
|`m2d_2m2a`|Passed|0|0|8235|
|`m2d_4m0a`|Passed|0|0|7535|
|`fm_2m2a_c1`|Passed|0|0|8055|
|`fm_2m2a_c2`|Passed|0|0|8055|
|`fm_2m2a_c3`|Passed|0|0|8115|
|`fm_4m0a_c1`|Passed|0|0|6995|
|`fm_4m0a_c2`|Passed|0|0|6995|
|`fm_4m0a_c3`|Passed|0|0|7175|
|`fm_pcd_c1`|Passed|0|0|3195|
|`fm_pcd_c2`|Passed|0|0|3195|
|`fm_pcd_c3`|Passed|0|0|3375|

All three PCD groups also passed Graph-versus-eager forward-output checks. After Graph release, the final 35-image batch completed backward propagation and a parameter update, with all forward states retained and recomputation disabled.

Additional source-versus-binary Graph training checks invoked 18 Graph modules per PCD group. Outputs, losses, all gradients, and weights after one SGD update were elementwise identical.

Complete machine-readable results are available in [validation/results.json](validation/results.json). These records cover the short-run workflow checks.

Distribution audit: 14 CUDA kernel binaries and 22 compiled Python modules; zero FM core source files or manuscript files.

Installation verification on 2026-11-15: the repository was cloned again from GitHub, release assets were downloaded, and the wheel SHA256 was verified. A new virtual environment with independently installed dependencies completed PCD condition II training for two steps, checkpoint resume, 16-image validation, benchmarking, resource sampling, and CUDA trace export with four DataLoader workers.

The 14-group workflow checks used a separate runtime installation with the existing local PyTorch dependency directory. The subsequent GitHub download verification used a fully independent dependency installation.
