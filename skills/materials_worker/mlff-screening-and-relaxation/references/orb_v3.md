# ORB-v3 backend reference

Use `backend="orb_v3"` only when it appears in `available_backends`. Use one of the exact official OMat model names returned by `enabled_models`: `orb-v3-conservative-inf-omat`, `orb-v3-conservative-20-omat`, `orb-v3-direct-inf-omat`, or `orb-v3-direct-20-omat`. These map mechanically to provider loader functions by replacing hyphens with underscores; there is no CatMaster model alias.

Accepted precision values are `float32-high`, `float32-highest`, and `float64`; the official recommendation and CatMaster default is `float32-high`. ORB reports confidence-head shape and peak-bin metadata when the calculator returns it, but confidence is not a calibrated error bar by itself.

`compile_mode=auto` follows the provider CUDA/MPS policy, `edge_method=knn_alchemi` is the preferred neighbor implementation, and `half_supercell=auto` leaves the large-cell threshold to ORB. Use explicit `on` or `off` only for a controlled comparable benchmark.

The initial CatMaster adapter supports fixed-cell SP and relaxation. Keep the model and precision identical across ranked structures.
