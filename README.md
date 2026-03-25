# Morphology Clamp Explorer

Modern browser-based UI for running current-clamp experiments on a single multi-compartment pyramidal neuron imported from `pyramidal_neuron.swc` and simulated with NetPyNe/NEURON.

## Current scope

- Import the full `pyramidal_neuron.swc` morphology through NetPyNe.
- Edit one active multi-compartment neuron with a deterministic region-specific membrane scaffold plus chloride/shunt conductances.
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

- The default membrane scaffold uses `Ra=100 ohm*cm`, `cm=1 uF/cm2` in soma/axon, `cm=2 uF/cm2` in dendrites, `g_pas=0.00003 S/cm2` in soma/axon, `g_pas=0.00006 S/cm2` in dendrites, `E_pas=-90 mV`, `ENa=50 mV`, `EK=-85 mV`, `V_init=-72 mV`, `dt=0.01 ms`, and `duration=50 ms`.
- The default current-clamp example is a soma pulse of `0.2 nA` from `5 ms` to `30 ms`, which gives a visible spike on the imported morphology.
- Current injection now defaults to `nA` for local patch-style stimulation, with a toggle back to `uA/cm2` if you explicitly want density-based input.
- Chloride reversal is derived from `Cl_i` and `Cl_o`; the passive inhibitory pathway is exposed as `gCl` and `g_shunt`.
- The current implementation uses an explicit deterministic region map (`AIS`, `distal axon`, `soma`, `basal dendrite`, `apical dendrite`, `apical hotzone`, optional `tuft`) with a fixed passive scaffold and region-scoped optional mechanisms.
- Dendritic asymmetry is currently carried by `Ih` in apical dendrites / hotzone / tuft and `KA` in dendrites only; soma and axon do not receive those channels.
- The apical hotzone now carries a hotzone-only `Ca_LVAst` + `CaDynamics_E2` threshold mechanism with `gCa_LVAstbar=0.002 S/cm2`, `cai_rest=5e-5 mM`, `cao=2.0 mM`, `tau_cai=40 ms`, and reduced gating tuned to support local distal events without placing calcium conductances elsewhere.
- `Ca_HVA` is now present in apical dendrites (`0.0002 S/cm2`) and the apical hotzone (`0.0007 S/cm2`), while `SK_E2` is restricted to the hotzone (`0.0015 S/cm2`, `Kd=0.00035 mM`, Hill `4`, `tau=4 ms`) so the distal branch can generate a calcium-assisted event and then terminate it locally.
- Persistent sodium is now present in the AIS (`0.0045 S/cm2`), distal axon (`0.0020 S/cm2`), and soma (`0.0004 S/cm2`) with a slightly retuned reduced activation threshold (`Vhalf=-55 mV`, `k=4.5 mV`, `tau=1 ms`) to smooth entry into the burst regime.
- `Im` is restricted to the soma (`0.00008 S/cm2`) and apical dendrites / hotzone (`0.0007 S/cm2`) so slow potassium control is present without adding extra axonal or basal adaptation currents.
- The SWC geometry is kept exact for visualization and site picking, while the electrical discretization is coarser than the raw SWC point count so live updates stay responsive.
- The live Plotly dashboard is downsampled for display speed; CSV export still preserves the full recorded traces.
- The default live view shows only the applied and command current panels in addition to voltage so updates stay fast; the heavier ionic/gating/conductance panels can be enabled when needed.
- Plotly outputs are written to `/Users/szymonlic/Desktop/NetPyNe modelling/plotly_outputs`.
# circuit_modeller
