# MatterSim backend reference

Use `backend="mattersim"` only when it appears in `available_backends`. The initial alias `mattersim-v1-1m` resolves to the official MatterSim-v1.0.0-1M checkpoint; record that exact checkpoint identity in reports.

The backend schema exposes `dtype`, `compute_stress`, `direct_graph`, and `compile`. Use `dtype=float32` for normal inference and disable stress for fixed-cell work that does not consume it. In the pinned MatterSim 1.2.5 deployment, `direct_graph=true` returned non-finite energy for both primitive and orthogonal periodic 1024-Si cells, with or without stress or compilation. Managed validation therefore requires `direct_graph=false` and `compile=false` until a provider upgrade passes the same regression.

MatterSim-v1 is intended for bulk materials. Surface, interface, molecular, and long-range-interaction applications require external validation and should be labeled qualitative screening.

The initial CatMaster adapter supports fixed-cell SP and relaxation. Cell relaxation remains disabled until separately validated.
