# DFT validation of MACE-screened H* adsorption (HER)

Validated set: 4 slab/site candidates (each with corresponding bare slab) + H2(g) reference.

## Definitions
Using the same convention as the MACE screening:

- ΔE_H* = E(slab+H) − E(slab) − 1/2 E(H2)
- ΔG_H* = ΔE_H* + 0.24 eV

DFT reference energy used: E(H2) = -6.7723967 eV (so 1/2 E(H2) = -3.3861983 eV).

## Best HER candidate by DFT (among validated cases)
Criterion: smallest |ΔG_H*| (closest to 0 eV).

- **Candidate:** `mp-12086_h100_t1__bridge_0`
  - material_id: mp-12086
  - facet: (100), termination: 1, site: bridge_0

### Energetics
| source | ΔE_H* (eV) | ΔG_H* (eV) |
|---|---:|---:|
| DFT | -0.278 | -0.038 |
| MACE | -0.245 | -0.005 |

### Surrogate error (MACE − DFT)
- error(ΔE_H*) = +0.033 eV (|error|=0.033 eV)
- error(ΔG_H*) = +0.033 eV (|error|=0.033 eV)

## All validated cases (sorted by |ΔG_H*| from DFT)

| candidate | facet | term | site | ΔE_H* DFT | ΔG_H* DFT | ΔE_H* MACE | ΔG_H* MACE | err ΔG (M−D) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `mp-12086_h100_t1__bridge_0` | (100) | 1 | bridge_0 | -0.278 | -0.038 | -0.245 | -0.005 | +0.033 |
| `mp-12086_h110_t1__bridge_0` | (110) | 1 | bridge_0 | -0.333 | -0.093 | -0.235 | +0.005 | +0.098 |
| `mp-1186117_h110_t0__bridge_4` | (110) | 0 | bridge_4 | -0.140 | +0.100 | -0.231 | +0.009 | -0.090 |
| `mp-12608_h110_t0__bridge_7` | (110) | 0 | bridge_7 | -0.092 | +0.148 | -0.241 | -0.001 | -0.149 |
