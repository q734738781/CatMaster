# mlff_neb

Prepare exactly one locally interpolated and checked path per stage:

```text
stage/
  input/
    path/
      00.vasp
      01.vasp
      ...
      NN.vasp
```

The numbered files must be contiguous from `00`, contain both endpoints and at least one intermediate image, and have identical atom count/order, cell, PBC, and constraints. Build them locally with `make_neb_geometry` after endpoint validation/remapping. Endpoint-only stages are invalid.

Do not place several path directories under one stage. Use one stage per path and `remote_submission_batch` for independent paths. Only fixed-image `plain` mode is accepted. Remote AutoNEB insertion is not part of this contract.
