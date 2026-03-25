from __future__ import annotations

from dataclasses import dataclass
import math
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .plotly_views import OUTPUT_DIR, write_fi_curve, write_live_trace_dashboard
from .simulator import (
    CURRENT_INJECTION_UNITS,
    CURRENT_UNIT_DENSITY,
    CURRENT_UNIT_NA,
    CURRENT_CLAMP,
    FISweepConfig,
    FISweepResult,
    MorphologyPreview,
    MorphologySection,
    MorphologySite,
    NeuronConfig,
    SimulationResult,
    VoltagePulseTrain,
    chloride_reversal_mV,
    current_nA_to_value,
    current_value_to_nA,
    default_recording_site,
    default_setup,
    estimate_site_segment_area_cm2,
    interpolate_section_site,
    load_pyramidal_morphology_preview,
    simulate_current_clamp,
    simulate_fi_sweep,
)


PLOT_PANEL_LABELS = {
    "applied_trace": "Applied current trace",
    "command_trace": "Stimulus / command trace",
    "ionic_currents": "I_Na, I_K, I_leak, I_Cl, I_shunt",
    "gating": "m, h, n gating variables",
    "conductances": "g_Na, g_K, g_leak, g_Cl, g_shunt",
}

PLOT_PANEL_DEFAULTS = {
    "applied_trace": True,
    "command_trace": True,
    "ionic_currents": False,
    "gating": False,
    "conductances": False,
}

SECTION_COLORS = {
    "soma": "#111111",
    "axon": "#d17a00",
    "dend": "#2a6dbb",
    "apic": "#228b5a",
}


@dataclass(frozen=True, slots=True)
class SliderSpec:
    minimum: float
    maximum: float
    resolution: float
    digits: int = 2
    integer: bool = False


NEURON_SLIDER_SPECS = {
    "duration_ms": SliderSpec(5.0, 500.0, 1.0, digits=1),
    "dt_ms": SliderSpec(0.005, 0.2, 0.001, digits=3),
    "v_rest_mV": SliderSpec(-100.0, 20.0, 0.5, digits=1),
    "gl_mS_cm2": SliderSpec(0.0, 5.0, 0.01, digits=3),
    "ena_mV": SliderSpec(20.0, 100.0, 0.5, digits=2),
    "ek_mV": SliderSpec(-120.0, -20.0, 0.5, digits=2),
    "eleak_mV": SliderSpec(-100.0, 20.0, 0.5, digits=2),
    "gcl_mS_cm2": SliderSpec(0.0, 5.0, 0.01, digits=3),
    "gshunt_mS_cm2": SliderSpec(0.0, 5.0, 0.01, digits=3),
    "cli_mM": SliderSpec(1.0, 50.0, 0.1, digits=2),
    "clo_mM": SliderSpec(10.0, 200.0, 0.5, digits=2),
}

HOLDING_CURRENT_SPECS = {
    CURRENT_UNIT_NA: SliderSpec(-5.0, 5.0, 0.01, digits=3),
    CURRENT_UNIT_DENSITY: SliderSpec(-300.0, 300.0, 0.5, digits=2),
}

TRAIN_SLIDER_SPECS = {
    "start_ms": SliderSpec(0.0, 500.0, 0.5, digits=1),
    "pulse_width_ms": SliderSpec(0.1, 500.0, 0.5, digits=1),
    "interval_ms": SliderSpec(0.1, 500.0, 0.5, digits=1),
    "pulse_count": SliderSpec(1.0, 20.0, 1.0, digits=0, integer=True),
}

TRAIN_AMPLITUDE_SPECS = {
    CURRENT_UNIT_NA: SliderSpec(-5.0, 5.0, 0.01, digits=3),
    CURRENT_UNIT_DENSITY: SliderSpec(-300.0, 300.0, 0.5, digits=2),
}

FI_SLIDER_SPECS = {
    CURRENT_UNIT_NA: {
        "start_current": SliderSpec(-1.0, 5.0, 0.01, digits=3),
        "end_current": SliderSpec(-1.0, 5.0, 0.01, digits=3),
        "step_current": SliderSpec(0.01, 1.0, 0.01, digits=3),
    },
    CURRENT_UNIT_DENSITY: {
        "start_current": SliderSpec(-100.0, 300.0, 0.5, digits=2),
        "end_current": SliderSpec(-100.0, 300.0, 0.5, digits=2),
        "step_current": SliderSpec(0.5, 100.0, 0.5, digits=2),
    },
    "pulse_start_ms": SliderSpec(0.0, 500.0, 0.5, digits=1),
    "pulse_width_ms": SliderSpec(0.5, 500.0, 0.5, digits=1),
}


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        for widget in (self.canvas, self.content):
            widget.bind("<Enter>", self._bind_mousewheel, add="+")
            widget.bind("<Leave>", self._unbind_mousewheel, add="+")

    def _on_content_configure(self, event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.delta == 0:
            return
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(1, "units")


class SliderEntry(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        textvariable: tk.StringVar,
        spec: SliderSpec,
        on_commit,
    ) -> None:
        super().__init__(parent)
        self.textvariable = textvariable
        self.spec = spec
        self.on_commit = on_commit
        self._updating = False
        self.scale_var = tk.DoubleVar()

        self.scale = tk.Scale(
            self,
            from_=spec.minimum,
            to=spec.maximum,
            resolution=1 if spec.integer else spec.resolution,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self.scale_var,
            command=self._on_scale,
            length=250,
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.entry = ttk.Entry(self, textvariable=self.textvariable, width=10)
        self.entry.pack(side=tk.LEFT, padx=(8, 0))
        self.entry.bind("<Return>", self._on_entry_commit)
        self.entry.bind("<FocusOut>", self._on_entry_commit)

    def _format_value(self, value: float) -> str:
        if self.spec.integer:
            return str(int(round(value)))
        return f"{value:.{self.spec.digits}f}"

    def update_spec(self, spec: SliderSpec) -> None:
        self.spec = spec
        self.scale.configure(
            from_=spec.minimum,
            to=spec.maximum,
            resolution=1 if spec.integer else spec.resolution,
        )
        self.sync_from_variable()

    def _coerce(self, raw: str) -> float:
        value = float(raw)
        return float(int(round(value))) if self.spec.integer else value

    def _expand_range_for(self, value: float) -> None:
        minimum = float(self.scale.cget("from"))
        maximum = float(self.scale.cget("to"))
        if value < minimum:
            self.scale.configure(from_=value)
        if value > maximum:
            self.scale.configure(to=value)

    def _set_value(self, value: float, trigger: bool) -> None:
        value = float(int(round(value))) if self.spec.integer else value
        self._expand_range_for(value)
        self._updating = True
        try:
            self.scale_var.set(value)
            self.textvariable.set(self._format_value(value))
        finally:
            self._updating = False
        if trigger and self.on_commit is not None:
            self.on_commit()

    def _on_scale(self, raw_value: str) -> None:
        if self._updating:
            return
        self._set_value(float(raw_value), trigger=True)

    def _on_entry_commit(self, event: tk.Event | None = None) -> None:
        if self._updating:
            return
        try:
            value = self._coerce(self.textvariable.get())
        except ValueError:
            return
        self._set_value(value, trigger=True)

    def sync_from_variable(self) -> None:
        try:
            value = self._coerce(self.textvariable.get())
        except ValueError:
            return
        self._set_value(value, trigger=False)


class ClampExplorerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.withdraw()

        self.morphology_preview = load_pyramidal_morphology_preview()
        self.neuron_config, self.pulse_trains = default_setup(CURRENT_CLAMP)
        self.fi_config = FISweepConfig()
        self.recording_site = default_recording_site()
        self.fi_stimulation_site = MorphologySite(
            section_name=self.recording_site.section_name,
            section_x=self.recording_site.section_x,
            x_um=self.recording_site.x_um,
            y_um=self.recording_site.y_um,
            z_um=self.recording_site.z_um,
        )
        self.selected_train_index: int | None = None
        self.last_result: SimulationResult | None = None
        self.last_fi_result: FISweepResult | None = None
        self.last_trace_target: str | Path | None = None
        self.last_fi_path: Path | None = None
        self.live_update_enabled = False
        self.live_update_after_id: str | None = None
        self.live_update_running = False
        self.live_update_pending = False

        self.status_var = tk.StringVar(
            value="The pyramidal morphology is loaded via NetPyNe/NEURON. Use the morphology window to place train sites and the recording patch."
        )
        self.ecl_var = tk.StringVar()
        self.current_unit_var = tk.StringVar()
        self.holding_current_label_var = tk.StringVar()
        self.train_amplitude_label_var = tk.StringVar()
        self.fi_start_label_var = tk.StringVar()
        self.fi_end_label_var = tk.StringVar()
        self.fi_step_label_var = tk.StringVar()
        self.recording_site_var = tk.StringVar()
        self.fi_site_var = tk.StringVar()
        self.train_site_var = tk.StringVar()
        self.viewer_info_var = tk.StringVar(value="Click a branch to assign a site.")
        self.assign_mode_var = tk.StringVar(value="train")
        self.projection_var = tk.StringVar(value="xy")
        self.plot_panel_vars = {
            key: tk.BooleanVar(value=PLOT_PANEL_DEFAULTS.get(key, False)) for key in PLOT_PANEL_LABELS
        }

        self.neuron_vars = {
            "duration_ms": tk.StringVar(),
            "dt_ms": tk.StringVar(),
            "v_rest_mV": tk.StringVar(),
            "holding_current": tk.StringVar(),
            "gl_mS_cm2": tk.StringVar(),
            "ena_mV": tk.StringVar(),
            "ek_mV": tk.StringVar(),
            "eleak_mV": tk.StringVar(),
            "gcl_mS_cm2": tk.StringVar(),
            "gshunt_mS_cm2": tk.StringVar(),
            "cli_mM": tk.StringVar(),
            "clo_mM": tk.StringVar(),
        }
        self.neuron_controls: dict[str, SliderEntry] = {}

        self.train_vars = {
            "label": tk.StringVar(),
            "start_ms": tk.StringVar(),
            "pulse_width_ms": tk.StringVar(),
            "interval_ms": tk.StringVar(),
            "pulse_count": tk.StringVar(),
            "amplitude": tk.StringVar(),
        }
        self.train_controls: dict[str, SliderEntry] = {}

        self.fi_vars = {
            "start_current": tk.StringVar(),
            "end_current": tk.StringVar(),
            "step_current": tk.StringVar(),
            "pulse_start_ms": tk.StringVar(),
            "pulse_width_ms": tk.StringVar(),
        }
        self.fi_controls: dict[str, SliderEntry] = {}

        self.neuron_window: tk.Toplevel | None = None
        self.experiment_window: tk.Toplevel | None = None
        self.morphology_window: tk.Toplevel | None = None
        self.morphology_canvas: tk.Canvas | None = None

        self._build_windows()
        self._load_neuron_form()
        self._load_fi_form()
        self._refresh_site_labels()
        self._refresh_train_list()
        self._load_train_form(0)
        self._redraw_morphology()

    def _window_alive(self, window: tk.Toplevel | None) -> bool:
        return window is not None and window.winfo_exists()

    def _dialog_parent(self) -> tk.Misc:
        if self._window_alive(self.experiment_window):
            return self.experiment_window  # type: ignore[return-value]
        if self._window_alive(self.morphology_window):
            return self.morphology_window  # type: ignore[return-value]
        if self._window_alive(self.neuron_window):
            return self.neuron_window  # type: ignore[return-value]
        return self.root

    def _build_windows(self) -> None:
        self.neuron_window = tk.Toplevel(self.root)
        self.neuron_window.title("Neuron Properties")
        self.neuron_window.geometry("660x920+40+60")
        self.neuron_window.minsize(540, 620)
        self.neuron_window.protocol("WM_DELETE_WINDOW", lambda: self._handle_window_close("neuron"))
        neuron_outer = ttk.Frame(self.neuron_window, padding=10)
        neuron_outer.pack(fill=tk.BOTH, expand=True)
        neuron_scroll = ScrollableFrame(neuron_outer)
        neuron_scroll.pack(fill=tk.BOTH, expand=True)
        self._build_neuron_window(neuron_scroll.content)
        ttk.Label(
            neuron_outer,
            text=(
                "These controls define the single multi-compartment pyramidal neuron imported from "
                "`pyramidal_neuron.swc`. The morphology itself is fixed; only the biophysics and protocol sites change."
            ),
            wraplength=600,
            justify="left",
        ).pack(fill=tk.X, pady=(10, 0))

        self.experiment_window = tk.Toplevel(self.root)
        self.experiment_window.title("Current Clamp And F-I")
        self.experiment_window.geometry("760x920+730+60")
        self.experiment_window.minsize(620, 620)
        self.experiment_window.protocol("WM_DELETE_WINDOW", lambda: self._handle_window_close("experiment"))
        experiment_outer = ttk.Frame(self.experiment_window, padding=10)
        experiment_outer.pack(fill=tk.BOTH, expand=True)
        experiment_scroll = ScrollableFrame(experiment_outer)
        experiment_scroll.pack(fill=tk.BOTH, expand=True)
        self._build_experiment_window(experiment_scroll.content)
        ttk.Label(experiment_outer, textvariable=self.status_var, anchor="w", justify="left").pack(
            fill=tk.X, pady=(10, 0)
        )

        self.morphology_window = tk.Toplevel(self.root)
        self.morphology_window.title("Morphology Viewer")
        self.morphology_window.geometry("760x920+1490+60")
        self.morphology_window.minsize(620, 620)
        self.morphology_window.protocol("WM_DELETE_WINDOW", lambda: self._handle_window_close("morphology"))
        self._build_morphology_window()

    def _build_neuron_window(self, parent: ttk.Frame) -> None:
        neuron_frame = ttk.LabelFrame(parent, text="Neuron Parameters", padding=10)
        neuron_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(neuron_frame, text="Current injection unit").grid(row=0, column=0, sticky="w", pady=2)
        current_unit_combo = ttk.Combobox(
            neuron_frame,
            textvariable=self.current_unit_var,
            values=list(CURRENT_INJECTION_UNITS),
            state="readonly",
            width=12,
        )
        current_unit_combo.grid(row=0, column=1, sticky="w", pady=2)
        current_unit_combo.bind("<<ComboboxSelected>>", self._on_current_unit_change)

        fields = [
            ("Duration (ms)", "duration_ms"),
            ("dt (ms)", "dt_ms"),
            ("Initial V (mV)", "v_rest_mV"),
            ("gL (mS/cm2)", "gl_mS_cm2"),
            ("ENa (mV)", "ena_mV"),
            ("EK (mV)", "ek_mV"),
            ("E_leak (mV)", "eleak_mV"),
            ("gCl (mS/cm2)", "gcl_mS_cm2"),
            ("g_shunt (mS/cm2)", "gshunt_mS_cm2"),
            ("Cl_i (mM)", "cli_mM"),
            ("Cl_o (mM)", "clo_mM"),
        ]
        holding_row = 1
        ttk.Label(neuron_frame, textvariable=self.holding_current_label_var).grid(
            row=holding_row, column=0, sticky="w", pady=2
        )
        control = SliderEntry(
            neuron_frame,
            textvariable=self.neuron_vars["holding_current"],
            spec=HOLDING_CURRENT_SPECS[CURRENT_UNIT_NA],
            on_commit=self._on_neuron_control_change,
        )
        control.grid(row=holding_row, column=1, sticky="ew", pady=2)
        self.neuron_controls["holding_current"] = control

        for row_offset, (label, key) in enumerate(fields, start=holding_row + 1):
            ttk.Label(neuron_frame, text=label).grid(row=row_offset, column=0, sticky="w", pady=2)
            control = SliderEntry(
                neuron_frame,
                textvariable=self.neuron_vars[key],
                spec=NEURON_SLIDER_SPECS[key],
                on_commit=self._on_neuron_control_change,
            )
            control.grid(row=row_offset, column=1, sticky="ew", pady=2)
            self.neuron_controls[key] = control

        ttk.Label(neuron_frame, textvariable=self.ecl_var, foreground="#245e2b").grid(
            row=len(fields) + 2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )
        ttk.Label(
            neuron_frame,
            text=(
                "Chloride reversal is derived from Cl_i and Cl_o. "
                "The imported morphology now uses a fixed region-specific passive scaffold with deterministic AIS, axon, soma, basal, apical, and hotzone assignments."
            ),
            wraplength=580,
            justify="left",
        ).grid(row=len(fields) + 3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        neuron_frame.columnconfigure(1, weight=1)

    def _build_experiment_window(self, parent: ttk.Frame) -> None:
        site_frame = ttk.LabelFrame(parent, text="Selected Sites", padding=10)
        site_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(site_frame, text="Recording patch site").grid(row=0, column=0, sticky="w")
        ttk.Label(site_frame, textvariable=self.recording_site_var).grid(row=0, column=1, sticky="w")
        ttk.Label(site_frame, text="F-I stimulation site").grid(row=1, column=0, sticky="w")
        ttk.Label(site_frame, textvariable=self.fi_site_var).grid(row=1, column=1, sticky="w")
        ttk.Label(
            site_frame,
            text="Train sites are assigned by selecting a pulse train here, then clicking the morphology window in Train mode.",
            wraplength=640,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        site_frame.columnconfigure(1, weight=1)

        trains_frame = ttk.LabelFrame(parent, text="Current Clamp Pulse Trains", padding=10)
        trains_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.train_listbox = tk.Listbox(trains_frame, exportselection=False, height=7)
        self.train_listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.train_listbox.bind("<<ListboxSelect>>", self._on_train_select)

        train_fields = [
            ("Label", "label"),
            ("Start (ms)", "start_ms"),
            ("Pulse width (ms)", "pulse_width_ms"),
            ("Interval (ms)", "interval_ms"),
            ("Pulse count", "pulse_count"),
            (None, "amplitude"),
        ]
        for offset, (label, key) in enumerate(train_fields, start=1):
            if key == "amplitude":
                ttk.Label(trains_frame, textvariable=self.train_amplitude_label_var).grid(
                    row=offset, column=0, sticky="w", pady=2
                )
            else:
                ttk.Label(trains_frame, text=label).grid(row=offset, column=0, sticky="w", pady=2)
            if key == "label":
                entry = ttk.Entry(trains_frame, textvariable=self.train_vars[key])
                entry.bind("<Return>", self._on_train_form_commit)
                entry.bind("<FocusOut>", self._on_train_form_commit)
                entry.grid(row=offset, column=1, sticky="ew", pady=2)
            else:
                spec = TRAIN_AMPLITUDE_SPECS[CURRENT_UNIT_NA] if key == "amplitude" else TRAIN_SLIDER_SPECS[key]
                control = SliderEntry(
                    trains_frame,
                    textvariable=self.train_vars[key],
                    spec=spec,
                    on_commit=self._on_train_form_change,
                )
                control.grid(row=offset, column=1, sticky="ew", pady=2)
                self.train_controls[key] = control

        ttk.Label(trains_frame, text="Selected train site").grid(row=len(train_fields) + 1, column=0, sticky="w")
        ttk.Label(trains_frame, textvariable=self.train_site_var).grid(
            row=len(train_fields) + 1, column=1, sticky="w"
        )
        ttk.Label(
            trains_frame,
            text="Negative amplitudes inject hyperpolarizing pulses.",
            wraplength=640,
            justify="left",
        ).grid(row=len(train_fields) + 2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        button_row = len(train_fields) + 3
        ttk.Button(trains_frame, text="Add New Train", command=self.add_train).grid(
            row=button_row, column=0, columnspan=2, sticky="ew", pady=(10, 2)
        )
        ttk.Button(trains_frame, text="Update Selected Train", command=self.update_train).grid(
            row=button_row + 1, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(trains_frame, text="Delete Selected Train", command=self.delete_train).grid(
            row=button_row + 2, column=0, columnspan=2, sticky="ew", pady=2
        )
        trains_frame.columnconfigure(1, weight=1)
        trains_frame.rowconfigure(0, weight=1)

        fi_frame = ttk.LabelFrame(parent, text="F-I Sweep", padding=10)
        fi_frame.pack(fill=tk.X, pady=(0, 10))
        fi_fields = [
            (None, "start_current"),
            (None, "end_current"),
            (None, "step_current"),
            ("Pulse start (ms)", "pulse_start_ms"),
            ("Pulse width (ms)", "pulse_width_ms"),
        ]
        for row, (label, key) in enumerate(fi_fields):
            if key == "start_current":
                ttk.Label(fi_frame, textvariable=self.fi_start_label_var).grid(row=row, column=0, sticky="w", pady=2)
            elif key == "end_current":
                ttk.Label(fi_frame, textvariable=self.fi_end_label_var).grid(row=row, column=0, sticky="w", pady=2)
            elif key == "step_current":
                ttk.Label(fi_frame, textvariable=self.fi_step_label_var).grid(row=row, column=0, sticky="w", pady=2)
            else:
                ttk.Label(fi_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            control = SliderEntry(
                fi_frame,
                textvariable=self.fi_vars[key],
                spec=FI_SLIDER_SPECS[CURRENT_UNIT_NA][key] if key in {"start_current", "end_current", "step_current"} else FI_SLIDER_SPECS[key],
                on_commit=lambda: None,
            )
            control.grid(row=row, column=1, sticky="ew", pady=2)
            self.fi_controls[key] = control
        ttk.Label(
            fi_frame,
            text="The F-I sweep uses the selected F-I stimulation site and the selected recording patch site.",
            wraplength=640,
            justify="left",
        ).grid(row=len(fi_fields), column=0, columnspan=2, sticky="w", pady=(8, 0))
        fi_frame.columnconfigure(1, weight=1)

        plot_frame = ttk.LabelFrame(parent, text="Clamp Dashboard Plots", padding=10)
        plot_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            plot_frame,
            text=(
                "Membrane voltage is always shown. Extra panels add more recorded variables and larger Plotly payloads, "
                "so keeping only the traces you need will make live updates faster."
            ),
            wraplength=640,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        for key, label in PLOT_PANEL_LABELS.items():
            ttk.Checkbutton(
                plot_frame,
                text=label,
                variable=self.plot_panel_vars[key],
                command=self._schedule_live_update,
            ).pack(anchor="w", pady=1)

        actions_frame = ttk.LabelFrame(parent, text="Run Experiments", padding=10)
        actions_frame.pack(fill=tk.X)
        ttk.Button(
            actions_frame,
            text="Run Clamp Dashboard",
            command=self.run_clamp_dashboard,
        ).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(
            actions_frame,
            text="Run F-I Sweep",
            command=self.run_fi_sweep,
        ).pack(fill=tk.X, pady=4)
        ttk.Button(
            actions_frame,
            text="Reset Current-Clamp Defaults",
            command=self.reset_defaults,
        ).pack(fill=tk.X, pady=(10, 4))
        ttk.Button(
            actions_frame,
            text="Export Last Clamp CSV",
            command=self.export_trace_csv,
        ).pack(fill=tk.X, pady=4)
        ttk.Button(
            actions_frame,
            text="Export Last F-I CSV",
            command=self.export_fi_csv,
        ).pack(fill=tk.X, pady=4)
        ttk.Label(
            actions_frame,
            text=f"Plotly outputs are written into: {OUTPUT_DIR}",
            wraplength=640,
            justify="left",
        ).pack(fill=tk.X, pady=(8, 0))

    def _build_morphology_window(self) -> None:
        assert self.morphology_window is not None
        outer = ttk.Frame(self.morphology_window, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(outer, text="Selection Controls", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="Projection").grid(row=0, column=0, sticky="w")
        projection_combo = ttk.Combobox(
            controls,
            textvariable=self.projection_var,
            values=["xy", "xz", "yz"],
            state="readonly",
            width=8,
        )
        projection_combo.grid(row=0, column=1, sticky="w", padx=(6, 16))
        projection_combo.bind("<<ComboboxSelected>>", lambda event: self._redraw_morphology())

        ttk.Label(controls, text="Click mode").grid(row=0, column=2, sticky="w")
        ttk.Radiobutton(
            controls,
            text="Train site",
            variable=self.assign_mode_var,
            value="train",
        ).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Radiobutton(
            controls,
            text="Recording patch",
            variable=self.assign_mode_var,
            value="record",
        ).grid(row=0, column=4, sticky="w", padx=(6, 0))
        ttk.Radiobutton(
            controls,
            text="F-I site",
            variable=self.assign_mode_var,
            value="fi",
        ).grid(row=0, column=5, sticky="w", padx=(6, 0))

        ttk.Label(
            controls,
            text=(
                "Use XY, XZ, and YZ projections to disambiguate overlapping branches. "
                "The nearest branch in the active projection is converted back to a section and position."
            ),
            wraplength=700,
            justify="left",
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(8, 0))

        canvas_frame = ttk.Frame(outer)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.morphology_canvas = tk.Canvas(canvas_frame, background="#ffffff", highlightthickness=1, highlightbackground="#cfcfcf")
        self.morphology_canvas.pack(fill=tk.BOTH, expand=True)
        self.morphology_canvas.bind("<Configure>", lambda event: self._redraw_morphology())
        self.morphology_canvas.bind("<Button-1>", self._on_morphology_click)

        footer = ttk.LabelFrame(outer, text="Current Selection", padding=10)
        footer.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(footer, textvariable=self.viewer_info_var, justify="left", wraplength=700).pack(anchor="w")

    def _handle_window_close(self, window_name: str) -> None:
        if window_name == "neuron" and self._window_alive(self.neuron_window):
            self.neuron_window.destroy()
            self.neuron_window = None
        elif window_name == "experiment" and self._window_alive(self.experiment_window):
            self.experiment_window.destroy()
            self.experiment_window = None
        elif window_name == "morphology" and self._window_alive(self.morphology_window):
            self.morphology_window.destroy()
            self.morphology_window = None
        if (
            not self._window_alive(self.neuron_window)
            and not self._window_alive(self.experiment_window)
            and not self._window_alive(self.morphology_window)
        ):
            self.root.destroy()

    def _refresh_site_labels(self) -> None:
        self.recording_site_var.set(self._format_site(self.recording_site))
        self.fi_site_var.set(self._format_site(self.fi_stimulation_site))
        if self.selected_train_index is None or not (0 <= self.selected_train_index < len(self.pulse_trains)):
            self.train_site_var.set("No train selected")
        else:
            train = self.pulse_trains[self.selected_train_index]
            self.train_site_var.set(self._format_train_site(train))

    def _format_site(self, site: MorphologySite) -> str:
        return f"{site.section_name} @ {site.section_x:.2f}  ({site.x_um:.1f}, {site.y_um:.1f}, {site.z_um:.1f}) um"

    def _format_train_site(self, train: VoltagePulseTrain) -> str:
        return f"{train.section_name} @ {train.section_x:.2f}"

    def _load_neuron_form(self) -> None:
        self.current_unit_var.set(self.neuron_config.current_injection_unit)
        for key, variable in self.neuron_vars.items():
            variable.set(str(getattr(self.neuron_config, key)))
        self._apply_current_unit_labels()
        for control in self.neuron_controls.values():
            control.sync_from_variable()
        self._refresh_derived_values()

    def _apply_neuron_form(self) -> None:
        self.neuron_config = NeuronConfig(
            duration_ms=float(self.neuron_vars["duration_ms"].get()),
            dt_ms=float(self.neuron_vars["dt_ms"].get()),
            v_rest_mV=float(self.neuron_vars["v_rest_mV"].get()),
            holding_mV=self.neuron_config.holding_mV,
            current_injection_unit=self.current_unit_var.get(),
            holding_current=float(self.neuron_vars["holding_current"].get()),
            gl_mS_cm2=float(self.neuron_vars["gl_mS_cm2"].get()),
            ena_mV=float(self.neuron_vars["ena_mV"].get()),
            ek_mV=float(self.neuron_vars["ek_mV"].get()),
            eleak_mV=float(self.neuron_vars["eleak_mV"].get()),
            gcl_mS_cm2=float(self.neuron_vars["gcl_mS_cm2"].get()),
            gshunt_mS_cm2=float(self.neuron_vars["gshunt_mS_cm2"].get()),
            cli_mM=float(self.neuron_vars["cli_mM"].get()),
            clo_mM=float(self.neuron_vars["clo_mM"].get()),
        )

    def _load_fi_form(self) -> None:
        for key, variable in self.fi_vars.items():
            variable.set(str(getattr(self.fi_config, key)))
        for control in self.fi_controls.values():
            control.sync_from_variable()

    def _apply_fi_form(self) -> None:
        self.fi_config = FISweepConfig(
            start_current=float(self.fi_vars["start_current"].get()),
            end_current=float(self.fi_vars["end_current"].get()),
            step_current=float(self.fi_vars["step_current"].get()),
            pulse_start_ms=float(self.fi_vars["pulse_start_ms"].get()),
            pulse_width_ms=float(self.fi_vars["pulse_width_ms"].get()),
        )

    def _refresh_train_list(self) -> None:
        self.train_listbox.delete(0, tk.END)
        for index, train in enumerate(self.pulse_trains, start=1):
            self.train_listbox.insert(
                tk.END,
                f"{index}. {train.label} | start={train.start_ms} ms | "
                f"width={train.pulse_width_ms} ms | interval={train.interval_ms} ms | "
                f"n={train.pulse_count} | amp={train.amplitude:.6g} {self._current_unit()} | "
                f"site={train.section_name}@{train.section_x:.2f}",
            )

    def _selected_plot_panels(self) -> set[str]:
        return {key for key, variable in self.plot_panel_vars.items() if variable.get()}

    def _load_train_form(self, index: int | None) -> None:
        if index is None or not (0 <= index < len(self.pulse_trains)):
            self.selected_train_index = None
            blank = VoltagePulseTrain(label=f"Pulse Train {len(self.pulse_trains) + 1}")
            for key, variable in self.train_vars.items():
                variable.set(str(getattr(blank, key)))
            for control in self.train_controls.values():
                control.sync_from_variable()
            self.train_listbox.selection_clear(0, tk.END)
            self._refresh_site_labels()
            self._redraw_morphology()
            return

        self.selected_train_index = index
        train = self.pulse_trains[index]
        for key, variable in self.train_vars.items():
            variable.set(str(getattr(train, key)))
        for control in self.train_controls.values():
            control.sync_from_variable()
        self.train_listbox.selection_clear(0, tk.END)
        self.train_listbox.selection_set(index)
        self.train_listbox.see(index)
        self._refresh_site_labels()
        self._redraw_morphology()

    def _train_from_form(self) -> VoltagePulseTrain:
        if self.selected_train_index is not None and 0 <= self.selected_train_index < len(self.pulse_trains):
            existing = self.pulse_trains[self.selected_train_index]
            section_name = existing.section_name
            section_x = existing.section_x
        else:
            section_name = self.recording_site.section_name
            section_x = self.recording_site.section_x
        return VoltagePulseTrain(
            label=self.train_vars["label"].get().strip() or f"Pulse Train {len(self.pulse_trains) + 1}",
            start_ms=float(self.train_vars["start_ms"].get()),
            pulse_width_ms=float(self.train_vars["pulse_width_ms"].get()),
            interval_ms=float(self.train_vars["interval_ms"].get()),
            pulse_count=int(float(self.train_vars["pulse_count"].get())),
            amplitude=float(self.train_vars["amplitude"].get()),
            section_name=section_name,
            section_x=section_x,
        )

    def _on_train_select(self, event: tk.Event) -> None:
        selection = self.train_listbox.curselection()
        if selection:
            self._load_train_form(selection[0])

    def _open_generated_file(self, target: str | Path | None) -> None:
        if target is None:
            messagebox.showinfo(
                "No output",
                "Run a simulation first to generate the Plotly HTML file.",
                parent=self._dialog_parent(),
            )
            return
        path = Path(target)
        if not path.exists():
            messagebox.showinfo(
                "No output",
                "Run a simulation first to generate the Plotly HTML file.",
                parent=self._dialog_parent(),
            )
            return
        webbrowser.open(path.resolve().as_uri())

    def _refresh_derived_values(self) -> None:
        try:
            ecl_mV = chloride_reversal_mV(
                float(self.neuron_vars["cli_mM"].get()),
                float(self.neuron_vars["clo_mM"].get()),
            )
            self.ecl_var.set(f"Derived E_Cl: {ecl_mV:.3f} mV")
        except ValueError:
            self.ecl_var.set("Derived E_Cl: invalid Cl_i / Cl_o")

    def _current_unit(self) -> str:
        return self.current_unit_var.get() or self.neuron_config.current_injection_unit

    def _convert_current_value_between_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        area_cm2: float,
    ) -> float:
        if from_unit == to_unit:
            return value
        current_nA = current_value_to_nA(value, from_unit, area_cm2)
        return current_nA_to_value(current_nA, to_unit, area_cm2)

    def _apply_current_unit_labels(self) -> None:
        unit = self._current_unit()
        self.holding_current_label_var.set(f"Holding current ({unit})")
        self.train_amplitude_label_var.set(f"Pulse amplitude from baseline ({unit})")
        self.fi_start_label_var.set(f"Start current ({unit})")
        self.fi_end_label_var.set(f"End current ({unit})")
        self.fi_step_label_var.set(f"Current increment ({unit})")

        if "holding_current" in self.neuron_controls:
            self.neuron_controls["holding_current"].update_spec(HOLDING_CURRENT_SPECS[unit])
        if "amplitude" in self.train_controls:
            self.train_controls["amplitude"].update_spec(TRAIN_AMPLITUDE_SPECS[unit])
        for key in ("start_current", "end_current", "step_current"):
            if key in self.fi_controls:
                self.fi_controls[key].update_spec(FI_SLIDER_SPECS[unit][key])

    def _on_current_unit_change(self, event: tk.Event | None = None) -> None:
        previous_unit = self.neuron_config.current_injection_unit
        new_unit = self.current_unit_var.get()
        if not new_unit or new_unit == previous_unit:
            self._apply_current_unit_labels()
            return

        try:
            holding_value = float(self.neuron_vars["holding_current"].get())
            holding_area_cm2 = estimate_site_segment_area_cm2(self.recording_site)
            self.neuron_vars["holding_current"].set(
                f"{self._convert_current_value_between_units(holding_value, previous_unit, new_unit, holding_area_cm2):.6f}"
            )

            for index, train in enumerate(self.pulse_trains):
                area_cm2 = estimate_site_segment_area_cm2(self._site_from_train(train))
                train.amplitude = self._convert_current_value_between_units(train.amplitude, previous_unit, new_unit, area_cm2)
                if self.selected_train_index == index:
                    self.train_vars["amplitude"].set(f"{train.amplitude:.6f}")

            fi_area_cm2 = estimate_site_segment_area_cm2(self.fi_stimulation_site)
            for key in ("start_current", "end_current", "step_current"):
                value = float(self.fi_vars[key].get())
                converted = self._convert_current_value_between_units(value, previous_unit, new_unit, fi_area_cm2)
                self.fi_vars[key].set(f"{converted:.6f}")
        except ValueError:
            pass

        self.neuron_config.current_injection_unit = new_unit
        self._apply_current_unit_labels()
        self._refresh_train_list()
        self._schedule_live_update()

    def _on_neuron_control_change(self) -> None:
        self._refresh_derived_values()
        self._schedule_live_update()

    def _on_train_form_commit(self, event: tk.Event | None = None) -> None:
        self._on_train_form_change()

    def _on_train_form_change(self) -> None:
        if self.selected_train_index is None:
            return
        try:
            self.pulse_trains[self.selected_train_index] = self._train_from_form()
        except ValueError:
            return
        self._refresh_train_list()
        self.train_listbox.selection_clear(0, tk.END)
        self.train_listbox.selection_set(self.selected_train_index)
        self.train_listbox.see(self.selected_train_index)
        self._refresh_site_labels()
        self._redraw_morphology()
        self._schedule_live_update()

    def _schedule_live_update(self) -> None:
        if not self.live_update_enabled:
            return
        if self.live_update_after_id is not None:
            self.root.after_cancel(self.live_update_after_id)
        self.live_update_after_id = self.root.after(300, self._run_live_update)

    def _run_live_update(self) -> None:
        self.live_update_after_id = None
        if self.live_update_running:
            self.live_update_pending = True
            return
        self.live_update_running = True
        try:
            self._execute_clamp_dashboard(open_browser=False, allow_dialogs=False)
        finally:
            self.live_update_running = False
            if self.live_update_pending:
                self.live_update_pending = False
                self._schedule_live_update()

    def add_train(self) -> None:
        try:
            train = self._train_from_form()
        except ValueError as exc:
            messagebox.showerror("Invalid train", f"Could not parse train values:\n{exc}", parent=self._dialog_parent())
            return
        self.pulse_trains.append(train)
        self._refresh_train_list()
        self._load_train_form(len(self.pulse_trains) - 1)
        self.status_var.set(f"Added {train.label}.")
        self._schedule_live_update()

    def update_train(self) -> None:
        if self.selected_train_index is None:
            messagebox.showinfo("No selection", "Select a pulse train to update.", parent=self._dialog_parent())
            return
        try:
            self.pulse_trains[self.selected_train_index] = self._train_from_form()
        except ValueError as exc:
            messagebox.showerror("Invalid train", f"Could not parse train values:\n{exc}", parent=self._dialog_parent())
            return
        self._refresh_train_list()
        self._load_train_form(self.selected_train_index)
        self.status_var.set("Updated the selected pulse train.")
        self._schedule_live_update()

    def delete_train(self) -> None:
        if self.selected_train_index is None:
            messagebox.showinfo("No selection", "Select a pulse train to delete.", parent=self._dialog_parent())
            return
        label = self.pulse_trains[self.selected_train_index].label
        del self.pulse_trains[self.selected_train_index]
        self._refresh_train_list()
        next_index = min(self.selected_train_index, len(self.pulse_trains) - 1)
        self._load_train_form(next_index if self.pulse_trains else None)
        self.status_var.set(f"Deleted {label}.")
        self._schedule_live_update()

    def reset_defaults(self) -> None:
        self.neuron_config, self.pulse_trains = default_setup(CURRENT_CLAMP)
        self.fi_config = FISweepConfig()
        self.recording_site = default_recording_site()
        self.fi_stimulation_site = MorphologySite(
            section_name=self.recording_site.section_name,
            section_x=self.recording_site.section_x,
            x_um=self.recording_site.x_um,
            y_um=self.recording_site.y_um,
            z_um=self.recording_site.z_um,
        )
        self.last_result = None
        self.last_fi_result = None
        self.last_trace_target = None
        self.last_fi_path = None
        self._load_neuron_form()
        self._load_fi_form()
        self._refresh_train_list()
        self._load_train_form(0)
        self._refresh_site_labels()
        self._redraw_morphology()
        self.status_var.set("Reset to the default morphology-aware current-clamp example.")
        self._schedule_live_update()

    def run_clamp_dashboard(self) -> None:
        self.live_update_enabled = True
        self._execute_clamp_dashboard(open_browser=True, allow_dialogs=True)

    def _execute_clamp_dashboard(self, open_browser: bool, allow_dialogs: bool) -> None:
        try:
            self._apply_neuron_form()
        except ValueError as exc:
            if allow_dialogs:
                messagebox.showerror("Invalid neuron settings", str(exc), parent=self._dialog_parent())
            else:
                self.status_var.set(f"Live update paused: {exc}")
            return

        if allow_dialogs:
            self.status_var.set("Running current-clamp simulation on the imported morphology...")
            self.root.update_idletasks()
        try:
            result = simulate_current_clamp(
                self.neuron_config,
                list(self.pulse_trains),
                recording_site=self.recording_site,
                selected_panels=self._selected_plot_panels(),
            )
            dashboard_target = write_live_trace_dashboard(
                result,
                title="Current Clamp Dashboard",
                selected_panels=self._selected_plot_panels(),
            )
        except (RuntimeError, ValueError) as exc:
            if allow_dialogs:
                messagebox.showerror("Simulation error", str(exc), parent=self._dialog_parent())
                self.status_var.set("Clamp simulation failed.")
            else:
                self.status_var.set(f"Live update paused: {exc}")
            return

        self.last_result = result
        self.last_trace_target = dashboard_target
        self.status_var.set(
            "Clamp dashboard is live. Adjust neuron parameters or click new sites in the morphology window to update the traces."
            if open_browser
            else "Clamp dashboard updated."
        )
        if open_browser:
            self._open_generated_file(dashboard_target)

    def run_fi_sweep(self) -> None:
        try:
            self._apply_neuron_form()
            self._apply_fi_form()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self._dialog_parent())
            return

        self.status_var.set("Running F-I sweep...")
        self.root.update_idletasks()
        try:
            result = simulate_fi_sweep(
                self.neuron_config,
                self.fi_config,
                stimulation_site=self.fi_stimulation_site,
                recording_site=self.recording_site,
            )
            html_path = write_fi_curve(result)
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("F-I sweep error", str(exc), parent=self._dialog_parent())
            self.status_var.set("F-I sweep failed.")
            return

        self.last_fi_result = result
        self.last_fi_path = html_path
        self.status_var.set(
            f"F-I sweep complete from {self.fi_stimulation_site.section_name}@{self.fi_stimulation_site.section_x:.2f}."
        )
        self._open_generated_file(html_path)

    def export_trace_csv(self) -> None:
        if self.last_result is None:
            messagebox.showinfo("No traces", "Run a clamp dashboard first.", parent=self._dialog_parent())
            return
        path = filedialog.asksaveasfilename(
            parent=self._dialog_parent(),
            title="Export clamp CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="clamp_traces.csv",
        )
        if not path:
            return

        current_header = (
            self.last_result.current_trace_label.lower().replace(" ", "_")
            + "_"
            + self.last_result.current_trace_unit.replace("/", "_per_")
        )
        command_header = (
            self.last_result.command_trace_label.lower().replace(" ", "_")
            + "_"
            + self.last_result.command_trace_unit.replace("/", "_per_")
        )
        extra_columns: list[tuple[str, list[float]]] = []
        for label, values in self.last_result.ionic_currents_uA_cm2.items():
            extra_columns.append((f"{label.lower()}_uA_per_cm2", values))
        for label, values in self.last_result.gating_variables.items():
            extra_columns.append((f"{label}_gate", values))
        for label, values in self.last_result.conductances_mS_cm2.items():
            extra_columns.append((f"{label.lower()}_mS_per_cm2", values))

        headers = ["time_ms", "voltage_mV", current_header, command_header] + [header for header, _ in extra_columns]
        lines = [",".join(headers)]
        for index, time_ms in enumerate(self.last_result.times_ms):
            row_values = [
                time_ms,
                self.last_result.voltage_mV[index],
                self.last_result.current_trace[index],
                self.last_result.command_trace[index],
            ]
            row_values.extend(values[index] for _, values in extra_columns)
            lines.append(",".join(f"{value:.6f}" for value in row_values))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.status_var.set(f"Exported clamp CSV to {path}.")

    def export_fi_csv(self) -> None:
        if self.last_fi_result is None:
            messagebox.showinfo("No F-I sweep", "Run an F-I sweep first.", parent=self._dialog_parent())
            return
        path = filedialog.asksaveasfilename(
            parent=self._dialog_parent(),
            title="Export F-I CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="fi_curve.csv",
        )
        if not path:
            return
        current_header = f"current_{self.last_fi_result.current_unit.replace('/', '_per_')}"
        lines = [f"{current_header},spike_count,firing_rate_hz"]
        for values in zip(
            self.last_fi_result.current_values,
            self.last_fi_result.spike_count,
            self.last_fi_result.firing_rate_hz,
        ):
            current_value, spike_count, firing_rate = values
            lines.append(f"{current_value:.6f},{spike_count:d},{firing_rate:.6f}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.status_var.set(f"Exported F-I CSV to {path}.")

    def _projection_axes(self) -> tuple[int, int]:
        return {
            "xy": (0, 1),
            "xz": (0, 2),
            "yz": (1, 2),
        }[self.projection_var.get()]

    def _canvas_dimensions(self) -> tuple[int, int]:
        if self.morphology_canvas is None:
            return 700, 780
        width = max(200, self.morphology_canvas.winfo_width())
        height = max(200, self.morphology_canvas.winfo_height())
        return width, height

    def _project_point(self, x_um: float, y_um: float, z_um: float) -> tuple[float, float]:
        first_axis, second_axis = self._projection_axes()
        coords = (x_um, y_um, z_um)
        return coords[first_axis], coords[second_axis]

    def _canvas_transform(self) -> tuple[float, float, float]:
        width, height = self._canvas_dimensions()
        margin = 30.0
        projected_points = [
            self._project_point(point[0], point[1], point[2])
            for section in self.morphology_preview.sections
            for point in section.points_3d
        ]
        min_u = min(point[0] for point in projected_points)
        max_u = max(point[0] for point in projected_points)
        min_v = min(point[1] for point in projected_points)
        max_v = max(point[1] for point in projected_points)
        span_u = max(max_u - min_u, 1.0)
        span_v = max(max_v - min_v, 1.0)
        scale = min((width - 2 * margin) / span_u, (height - 2 * margin) / span_v)
        offset_u = margin - min_u * scale + max(0.0, (width - 2 * margin - span_u * scale) / 2.0)
        offset_v = margin - min_v * scale + max(0.0, (height - 2 * margin - span_v * scale) / 2.0)
        return scale, offset_u, offset_v

    def _canvas_coords(self, x_um: float, y_um: float, z_um: float) -> tuple[float, float]:
        scale, offset_u, offset_v = self._canvas_transform()
        u, v = self._project_point(x_um, y_um, z_um)
        _, height = self._canvas_dimensions()
        canvas_x = u * scale + offset_u
        canvas_y = height - (v * scale + offset_v)
        return canvas_x, canvas_y

    def _redraw_morphology(self) -> None:
        if self.morphology_canvas is None or not self._window_alive(self.morphology_window):
            return
        canvas = self.morphology_canvas
        canvas.delete("all")

        for section in self.morphology_preview.sections:
            color = SECTION_COLORS.get(section.section_type, "#6b6b6b")
            width = 4 if section.section_type == "soma" else 2
            coords: list[float] = []
            for point in section.points_3d:
                canvas_x, canvas_y = self._canvas_coords(point[0], point[1], point[2])
                coords.extend([canvas_x, canvas_y])
            if len(coords) >= 4:
                canvas.create_line(*coords, fill=color, width=width, capstyle=tk.ROUND, smooth=False)

        self._draw_site_marker(self.recording_site, color="#c62828", radius=6)
        self._draw_site_marker(self.fi_stimulation_site, color="#7b1fa2", radius=6)

        for index, train in enumerate(self.pulse_trains, start=1):
            site = self._site_from_train(train)
            highlight = self.selected_train_index == index - 1
            self._draw_site_marker(site, color="#ff8f00", radius=5 if highlight else 4, label=str(index))

    def _draw_site_marker(
        self,
        site: MorphologySite,
        color: str,
        radius: float,
        label: str | None = None,
    ) -> None:
        if self.morphology_canvas is None:
            return
        canvas_x, canvas_y = self._canvas_coords(site.x_um, site.y_um, site.z_um)
        self.morphology_canvas.create_oval(
            canvas_x - radius,
            canvas_y - radius,
            canvas_x + radius,
            canvas_y + radius,
            fill=color,
            outline="#ffffff",
            width=1,
        )
        if label:
            self.morphology_canvas.create_text(canvas_x + radius + 6, canvas_y, text=label, anchor="w", fill=color)

    def _site_from_train(self, train: VoltagePulseTrain) -> MorphologySite:
        section = next((item for item in self.morphology_preview.sections if item.name == train.section_name), None)
        if section is None:
            return MorphologySite(section_name=train.section_name, section_x=train.section_x)
        return interpolate_section_site(section, train.section_x)

    def _nearest_site(self, click_x: float, click_y: float) -> MorphologySite | None:
        best_site: MorphologySite | None = None
        best_dist2 = float("inf")
        for section in self.morphology_preview.sections:
            if len(section.points_3d) < 2:
                continue
            projected = [self._canvas_coords(point[0], point[1], point[2]) for point in section.points_3d]
            cumulative = section.cumulative_lengths_um
            for index in range(len(projected) - 1):
                ax, ay = projected[index]
                bx, by = projected[index + 1]
                dx = bx - ax
                dy = by - ay
                length2 = dx * dx + dy * dy
                if length2 <= 0:
                    continue
                t = ((click_x - ax) * dx + (click_y - ay) * dy) / length2
                t = min(1.0, max(0.0, t))
                nearest_x = ax + t * dx
                nearest_y = ay + t * dy
                dist2 = (click_x - nearest_x) ** 2 + (click_y - nearest_y) ** 2
                if dist2 < best_dist2:
                    segment_start = cumulative[index]
                    segment_end = cumulative[index + 1]
                    path_length = segment_start + t * (segment_end - segment_start)
                    section_x = path_length / max(section.total_length_um, 1e-9)
                    best_site = interpolate_section_site(section, section_x)
                    best_dist2 = dist2
        if best_site is None or best_dist2 > 16.0 * 16.0:
            return None
        return best_site

    def _on_morphology_click(self, event: tk.Event) -> None:
        site = self._nearest_site(float(event.x), float(event.y))
        if site is None:
            self.viewer_info_var.set("No branch was close enough to that click. Try another projection or click closer to a branch.")
            return

        if self.assign_mode_var.get() == "record":
            self.recording_site = site
            self.viewer_info_var.set(f"Recording patch moved to {self._format_site(site)}")
        elif self.assign_mode_var.get() == "fi":
            self.fi_stimulation_site = site
            self.viewer_info_var.set(f"F-I stimulation site moved to {self._format_site(site)}")
        else:
            if self.selected_train_index is None or not (0 <= self.selected_train_index < len(self.pulse_trains)):
                self.viewer_info_var.set("Select a pulse train first, then click again to assign its stimulation site.")
                return
            train = self.pulse_trains[self.selected_train_index]
            train.section_name = site.section_name
            train.section_x = site.section_x
            self.viewer_info_var.set(f"{train.label} moved to {self._format_site(site)}")
            self._refresh_train_list()
            self.train_listbox.selection_clear(0, tk.END)
            self.train_listbox.selection_set(self.selected_train_index)
            self.train_listbox.see(self.selected_train_index)

        self._refresh_site_labels()
        self._redraw_morphology()
        self._schedule_live_update()


VoltageClampApp = ClampExplorerApp


def run() -> int:
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.1)
    except tk.TclError:
        pass
    ClampExplorerApp(root)
    root.mainloop()
    return 0
