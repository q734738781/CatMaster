# Native CP2K AIMD task fragments

These fragments are deliberately incomplete. They show the MD, optional thermostat, output, and restart sections without choosing an electronic-structure method, system, ensemble, timestep, temperature, coupling time, trajectory length, or numerical standard. Build the complete CP2K input from the sampling objective and the selected electronic model.

## MD operation fragment

```text
&GLOBAL
  RUN_TYPE MD
&END GLOBAL

&MOTION
  &MD
    ENSEMBLE ENSEMBLE_NAME
    STEPS NSTEPS
    TIMESTEP TIMESTEP_FS
  &END MD
&END MOTION
```

Select `ENSEMBLE_NAME`, timestep, and trajectory length for the system and observable. Do not add thermostat or barostat sections to an ensemble that does not use them.

## Optional CSVR thermostat fragment

Insert this inside `&MD` only if NVT with CSVR has already been selected:

```text
TEMPERATURE TARGET_TEMPERATURE_K
&THERMOSTAT
  REGION GLOBAL
  &CSVR
    TIMECON THERMOSTAT_TIMECON_FS
  &END CSVR
&END THERMOSTAT
```

CSVR is one possible thermostat, not a default implied by this reference. Choose the thermostat family and coupling time independently.

## Output-stride fragments

Insert these under `&MOTION / &PRINT` when the corresponding artifacts are required:

```text
&TRAJECTORY
  &EACH
    MD TRAJECTORY_EVERY
  &END EACH
&END TRAJECTORY

&RESTART
  &EACH
    MD RESTART_EVERY
  &END EACH
&END RESTART
```

Strides are analysis and recovery choices, not fixed values.

## Explicit restart fragment

Add this top-level section to a complete input and map the referenced file into the stage:

```text
&EXT_RESTART
  RESTART_FILE_NAME RESTART_FILE
&END EXT_RESTART
```

Keep the new `&MOTION / &MD` intent explicit. If continuity is required, decide which restart fields to retain and do not replace restored velocities with newly generated ones.
