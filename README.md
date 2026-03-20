# Morphology Clamp Explorer

Modern browser-based UI for running current-clamp experiments on a single multi-compartment pyramidal neuron imported from `pyramidal_neuron.swc` and simulated with NetPyNe/NEURON.

## Current scope

- Import the full `pyramidal_neuron.swc` morphology through NetPyNe.
- Edit one active multi-compartment neuron with uniform HH-style membrane plus chloride/shunt conductances.
- Launch a local Dash app that opens in your browser.
- Use a rotatable Plotly 3D morphology viewer to place:
  - pulse-train stimulation sites
  - the recording patch site
  - the F-I stimulation site
- Generate an in-app live Plotly current-clamp dashboard with membrane voltage, current traces, ionic currents, gating variables, and conductances.
- Run F-I sweeps and export the latest clamp or F-I results to CSV.
- Launch from a clickable macOS `.app` bundle that creates `.venv` and installs the simulation stack on first run.

## Launch

Double-click [`NetPyNe Modeler.app`](/Users/szymonlic/Desktop/NetPyNe modelling/NetPyNe Modeler.app), or run:

```bash
cd "/Users/szymonlic/Desktop/NetPyNe modelling"
python3 bootstrap.py
```

This starts a local web UI and opens it in your default browser.

## Notes

- The default cell parameters follow your selected Hodgkin-Huxley table: `Cm=1`, `gNa=120`, `gK=36`, `gL=0.3`, `ENa=63.54`, `EK=-74.16`, `E_leak=-54.3`, `V_init=-65`, `dt=0.01 ms`, `duration=50 ms`.
- The default current-clamp example is a soma pulse of `0.2 nA` from `5 ms` to `30 ms`, which gives a visible spike on the imported morphology.
- Current injection now defaults to `nA` for local patch-style stimulation, with a toggle back to `uA/cm2` if you explicitly want density-based input.
- Chloride reversal is derived from `Cl_i` and `Cl_o`; the passive inhibitory pathway is exposed as `gCl` and `g_shunt`.
- The current implementation applies the same HH-style active membrane densities to all imported sections. It is morphology-aware, but not yet a section-specific pyramidal conductance map.
- The SWC geometry is kept exact for visualization and site picking, while the electrical discretization is coarser than the raw SWC point count so live updates stay responsive.
- The live Plotly dashboard is downsampled for display speed; CSV export still preserves the full recorded traces.
- The default live view shows only the applied and command current panels in addition to voltage so updates stay fast; the heavier ionic/gating/conductance panels can be enabled when needed.
- Plotly outputs are written to `/Users/szymonlic/Desktop/NetPyNe modelling/plotly_outputs`.
# circuit_modeller
