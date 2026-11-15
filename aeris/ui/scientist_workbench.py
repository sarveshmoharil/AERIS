"""
AERIS Scientist Workbench — district-grid edition.
Loads the existing 15-cell district grid and its zonal environmental profiles.
Legacy SRO-01 outputs are deliberately kept separate because they belong to the old
sample-derived 8-cell grid and are not valid for the district grid.
Run from E:\\TrialShit:
    python .\\aeris\\ui\\scientist_workbench.py
"""
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "aeris" / "data" / "processed"
DISTRICT_GRID = DATA / "district_grid"
GRID_FILE = DISTRICT_GRID / "nashik_district_grid.geojson"
PROFILES_FILE = DISTRICT_GRID / "nashik_district_grid_environmental_profiles.csv"
REPORT_FILE = DISTRICT_GRID / "district_grid_report.json"
DISTRICT_FILE = DATA / "spatial" / "nashik_district_boundary.geojson"

# Legacy experiment paths — never mixed with district-grid cells.
OPT = DATA / "spatial_optimization"
LEGACY_CANDIDATES = OPT / "rotation_optimizer_candidates.csv"
LEGACY_HISTORY = OPT / "optimization_generation_history.json"
LEGACY_REPORT = OPT / "rotation_optimizer_report.json"

DISTRICT_SUITABILITY = DISTRICT_GRID / "crop_suitability_district.csv"
DISTRICT_CANDIDATES = DISTRICT_GRID / "rotation_candidates_district.csv"
DISTRICT_HISTORY = DISTRICT_GRID / "ga_history_district.json"
DISTRICT_ENGINE_REPORT = DISTRICT_GRID / "district_crop_rotation_report.json"
RAINFALL_FORECAST = DATA / "rainfall_prediction" / "seasonal_rainfall_forecast.csv"
APPROVAL_FILE = DATA / "approved_recommendations.json"

BG, TEXT, MUTED = "#f0f0f0", "#202020", "#555555"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_id(feature):
    props = feature.get("properties") or {}
    for key in ("cell_id", "grid_id", "grid_cell", "target", "id", "name"):
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


class ScientistWorkbench:
    def __init__(self, root):
        self.root = root
        root.title("AERIS v1.0 — Scientist Workbench")
        root.geometry("1450x920")
        root.minsize(1150, 740)
        root.configure(bg=BG)

        self.profiles = pd.read_csv(PROFILES_FILE)
        geo = read_json(GRID_FILE)
        self.features = geo.get("features", []) if isinstance(geo, dict) else []
        self.district_features = []
        if DISTRICT_FILE.exists():
            raw = read_json(DISTRICT_FILE)
            self.district_features = raw.get("features", []) if isinstance(raw, dict) else []
        self.report = read_json(REPORT_FILE) if REPORT_FILE.exists() else {}
        self.suitability = pd.read_csv(DISTRICT_SUITABILITY) if DISTRICT_SUITABILITY.exists() else pd.DataFrame()
        self.candidates = pd.read_csv(DISTRICT_CANDIDATES) if DISTRICT_CANDIDATES.exists() else pd.DataFrame()
        self.ga_history = read_json(DISTRICT_HISTORY) if DISTRICT_HISTORY.exists() else []
        self.engine_report = read_json(DISTRICT_ENGINE_REPORT) if DISTRICT_ENGINE_REPORT.exists() else {}
        self.rainfall = pd.read_csv(RAINFALL_FORECAST) if RAINFALL_FORECAST.exists() else pd.DataFrame()
        self.selected_season = tk.StringVar(value="kharif")
        self.selected_generation = 1

        self.cell_col = next((c for c in
                              ("cell_id", "grid_id", "grid_cell", "target", "id")
                              if c in self.profiles.columns), None)
        if not self.cell_col:
            raise ValueError(
                "Could not identify the cell ID column in the district-grid profile CSV. "
                f"Available columns: {', '.join(self.profiles.columns)}"
            )
        self.profile_ids = self.profiles[self.cell_col].astype(str).tolist()
        self.feature_map = {}
        for feat in self.features:
            fid = feature_id(feat)
            if fid:
                self.feature_map[fid] = feat
        self.cells = [c for c in self.profile_ids if c in self.feature_map]
        if not self.cells:
            raise ValueError(
                "No matching cell IDs between the district grid GeoJSON and profile CSV. "
                "Check feature properties and the profile cell-ID column."
            )

        self.layers = self.find_layers()
        self.selected_cell = self.cells[0]
        self.selected_layer = tk.StringVar(value=self.layers[0] if self.layers else "")
        self.status = tk.StringVar(value=f"District grid loaded: {len(self.cells)} cells")

        self.setup_style()
        self.build_header()
        self.build_tabs()
        self.refresh_map()

    def find_layers(self):
        preferred = [
            "phh2o_0-5cm_mean", "phh2o_5-15cm_mean", "phh2o_15-30cm_mean",
            "soc_0-5cm_mean", "soc_5-15cm_mean", "soc_15-30cm_mean",
            "clay_0-5cm_mean", "clay_5-15cm_mean", "clay_15-30cm_mean",
            "sand_0-5cm_mean", "sand_5-15cm_mean", "sand_15-30cm_mean",
            "nitrogen_0-5cm_mean", "nitrogen_5-15cm_mean", "nitrogen_15-30cm_mean",
            "cec_0-5cm_mean", "cec_5-15cm_mean", "cec_15-30cm_mean",
            "elevation_m_mean", "slope_degrees_mean", "aspect_degrees_mean",
            "soil_mean_coverage_pct", "terrain_mean_coverage_pct",
        ]
        found = [c for c in preferred if c in self.profiles.columns]
        found.extend(c for c in self.profiles.columns
                     if c.endswith("_mean") and c not in found
                     and pd.api.types.is_numeric_dtype(self.profiles[c]))
        # Include any explicit coverage fields, even if named differently.
        found.extend(c for c in self.profiles.columns
                     if "coverage" in c.lower() and c not in found
                     and pd.api.types.is_numeric_dtype(self.profiles[c]))
        return found

    def setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 9))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), background=BG)
        style.configure("Sub.TLabel", foreground=MUTED, background=BG)
        style.configure("Treeview", rowheight=25, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def build_header(self):
        bar = ttk.Frame(self.root, padding=(12, 9))
        bar.pack(fill="x")
        ttk.Label(bar, text="AERIS v1.0", style="Header.TLabel").pack(side="left")
        ttk.Label(bar, text="  |  SCIENTIST WORKBENCH", style="Sub.TLabel").pack(side="left", pady=(5, 0))
        ttk.Label(bar, textvariable=self.status, style="Sub.TLabel").pack(side="right")
        ttk.Separator(self.root).pack(fill="x")

    def build_tabs(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.map_tab = ttk.Frame(self.tabs, padding=8)
        self.evidence_tab = ttk.Frame(self.tabs, padding=8)
        self.crop_tab = ttk.Frame(self.tabs, padding=8)
        self.rotation_tab = ttk.Frame(self.tabs, padding=8)
        self.rainfall_tab = ttk.Frame(self.tabs, padding=8)
        self.legacy_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.map_tab, text="District Spatial Workbench")
        self.tabs.add(self.evidence_tab, text="Cell Evidence")
        self.tabs.add(self.crop_tab, text="Crop Suitability")
        self.tabs.add(self.rotation_tab, text="Rotation & GA")
        self.tabs.add(self.rainfall_tab, text="Rainfall Outlook")
        self.tabs.add(self.legacy_tab, text="Legacy SRO-01 (8-cell)")
        self.build_map_tab()
        self.build_evidence_tab()
        self.build_crop_tab()
        self.build_rotation_tab()
        self.build_rainfall_tab()
        self.build_legacy_tab()
        self.refresh_crop_panel()
        self.refresh_rotation_panel()
        self.refresh_rainfall_panel()

    def build_map_tab(self):
        top = ttk.Frame(self.map_tab)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Environmental layer:").pack(side="left")
        self.layer_combo = ttk.Combobox(
            top, textvariable=self.selected_layer, values=self.layers,
            state="readonly", width=36
        )
        self.layer_combo.pack(side="left", padx=7)
        self.layer_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_map())
        ttk.Label(top, text="Cell polygons are from the district-wide grid, not the old sample grid.",
                  style="Sub.TLabel").pack(side="left", padx=12)

        body = ttk.Frame(self.map_tab)
        body.pack(fill="both", expand=True)
        plot_frame = ttk.Frame(body)
        plot_frame.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(body, width=330, padding=10, relief="groove")
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)

        self.fig, self.ax = plt.subplots(figsize=(8.6, 6.8), dpi=100)
        self.fig.subplots_adjust(left=.06, right=.94, top=.93, bottom=.08)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_map_click)

        ttk.Label(side, text="DISTRICT GRID CELLS", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.cell_list = tk.Listbox(side, height=11, exportselection=False)
        self.cell_list.pack(fill="x", pady=6)
        for cell in self.cells:
            self.cell_list.insert("end", cell)
        self.cell_list.bind("<<ListboxSelect>>", self.on_list_select)
        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="SELECTED CELL", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.cell_title = ttk.Label(side, text=self.selected_cell, font=("Segoe UI", 13, "bold"))
        self.cell_title.pack(anchor="w", pady=(5, 2))
        self.summary = tk.Text(side, height=16, wrap="word", relief="flat", bg="white")
        self.summary.pack(fill="both", expand=True, pady=5)
        self.summary.configure(state="disabled")
        ttk.Button(side, text="Open full evidence table",
                   command=lambda: self.tabs.select(self.evidence_tab)).pack(fill="x", pady=4)

    def build_evidence_tab(self):
        ttk.Label(self.evidence_tab, text="DISTRICT GRID ENVIRONMENTAL EVIDENCE",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(self.evidence_tab,
                  text="Values below are the zonal statistics written by the district-grid builder. "
                       "SoilGrids raw values are not converted to physical units here.",
                  style="Sub.TLabel", wraplength=1100).pack(anchor="w", pady=(4, 10))
        bar = ttk.Frame(self.evidence_tab)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Cell:").pack(side="left")
        self.evidence_cell = tk.StringVar(value=self.selected_cell)
        self.evidence_combo = ttk.Combobox(bar, textvariable=self.evidence_cell,
                                           values=self.cells, state="readonly", width=20)
        self.evidence_combo.pack(side="left", padx=6)
        self.evidence_combo.bind("<<ComboboxSelected>>", lambda _e: self.fill_evidence_table())
        cols = ("metric", "value")
        self.evidence_tree = ttk.Treeview(self.evidence_tab, columns=cols, show="headings")
        self.evidence_tree.heading("metric", text="Environmental metric")
        self.evidence_tree.heading("value", text="Cell statistic")
        self.evidence_tree.column("metric", width=430, anchor="w")
        self.evidence_tree.column("value", width=260, anchor="w")
        self.evidence_tree.pack(fill="both", expand=True)
        self.fill_evidence_table()

    def build_legacy_tab(self):
        ttk.Label(self.legacy_tab, text="ARCHIVED EXPERIMENT — NOT DISTRICT-GRID RESULTS",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            self.legacy_tab,
            text=(
                "The existing SRO-01 candidates and generation history were generated for the old "
                "sample-derived 8-cell grid (G101–G402). They do not correspond to the new "
                f"{len(self.cells)}-cell district grid. This workbench therefore does not display, "
                "transfer, or approve those rotations for district cells.\n\n"
                "The legacy files are left untouched. A new district-grid suitability run and a "
                "new district-grid optimizer run are required before candidate comparison, GA "
                "inspection, or farmer-facing approval can be enabled."
            ),
            wraplength=1100, justify="left"
        ).pack(anchor="w", pady=12)
        legacy_files = [
            ("Candidates", LEGACY_CANDIDATES),
            ("Generation history", LEGACY_HISTORY),
            ("Optimizer report", LEGACY_REPORT),
        ]
        for label, path in legacy_files:
            ttk.Label(self.legacy_tab, text=f"{label}: {path}").pack(anchor="w", pady=3)
            ttk.Label(self.legacy_tab, text=f"Present: {'Yes' if path.exists() else 'No'}",
                      style="Sub.TLabel").pack(anchor="w")


    # ---------------------------------------------------------
    # District-cell crop suitability
    # ---------------------------------------------------------
    def build_crop_tab(self):
        bar = ttk.Frame(self.crop_tab)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Cell:").pack(side="left")
        self.crop_cell = tk.StringVar(value=self.selected_cell)
        self.crop_cell_combo = ttk.Combobox(bar, textvariable=self.crop_cell, values=self.cells,
                                            state="readonly", width=16)
        self.crop_cell_combo.pack(side="left", padx=5)
        self.crop_cell_combo.bind("<<ComboboxSelected>>", lambda _e: self.select_cell(self.crop_cell.get()))
        ttk.Label(bar, text="Season:").pack(side="left", padx=(12, 0))
        self.crop_season = tk.StringVar(value="kharif")
        self.crop_season_combo = ttk.Combobox(bar, textvariable=self.crop_season,
                                              values=("kharif", "rabi", "summer", "perennial"),
                                              state="readonly", width=14)
        self.crop_season_combo.pack(side="left", padx=5)
        self.crop_season_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_crop_panel())
        self.crop_evidence_note = ttk.Label(bar, text="", style="Sub.TLabel")
        self.crop_evidence_note.pack(side="left", padx=12)

        body = ttk.Frame(self.crop_tab)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body, width=430, padding=8)
        right.pack(side="right", fill="both", padx=(8, 0))
        right.pack_propagate(False)
        cols = ("rank", "crop", "score", "evidence", "classification")
        self.crop_tree = ttk.Treeview(left, columns=cols, show="headings", height=18)
        for col, title, width in [
            ("rank","Rank",55),("crop","Crop",170),("score","Suitability /100",110),
            ("evidence","Spatial evidence",115),("classification","Evidence class",190)
        ]:
            self.crop_tree.heading(col, text=title)
            self.crop_tree.column(col, width=width, anchor="center" if col != "crop" else "w")
        self.crop_tree.pack(fill="both", expand=True)
        self.crop_tree.bind("<<TreeviewSelect>>", self.on_crop_select)

        self.crop_fig, self.crop_ax = plt.subplots(figsize=(5.2, 4.0), dpi=100)
        self.crop_fig.subplots_adjust(left=.28, right=.97, top=.92, bottom=.12)
        self.crop_canvas = FigureCanvasTkAgg(self.crop_fig, master=right)
        self.crop_canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(right, text="WHY THIS CROP?", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 3))
        self.crop_details = tk.Text(right, height=10, wrap="word", bg="white", relief="solid", bd=1)
        self.crop_details.pack(fill="both", expand=True)
        self.crop_details.insert("1.0", "Select a crop to inspect its supporting factors and warnings.")
        self.crop_details.configure(state="disabled")
        self.refresh_crop_panel()

    def refresh_crop_panel(self):
        if not hasattr(self, "crop_tree"):
            return
        self.crop_cell.set(self.selected_cell)
        for item in self.crop_tree.get_children():
            self.crop_tree.delete(item)
        self.crop_ax.clear()
        if self.suitability.empty:
            self.crop_ax.text(.5,.5,"Run DCR-01 to generate district-cell crop rankings.",ha="center",va="center",wrap=True)
            self.crop_canvas.draw_idle()
            self.crop_evidence_note.configure(text="No district suitability output yet")
            self.crop_details.configure(state="normal")
            self.crop_details.delete("1.0","end")
            self.crop_details.insert("1.0","District-cell crop suitability has not been generated. Run the AERIS district pipeline.")
            self.crop_details.configure(state="disabled")
            return
        season = self.crop_season.get()
        rows = self.suitability[
            (self.suitability.cell_id.astype(str) == self.selected_cell)
            & (self.suitability.season.astype(str) == season)
            & (self.suitability.classification.astype(str) != "out_of_season")
        ].copy()
        rows = rows.sort_values("suitability_score", ascending=False, na_position="last")
        for _, r in rows.iterrows():
            score = "—" if pd.isna(r.suitability_score) else f"{float(r.suitability_score):.1f}"
            self.crop_tree.insert("", "end", iid=str(r.crop_id), values=(
                "" if pd.isna(r.get("rank")) else int(r.get("rank")),
                r.crop_name, score, f"{float(r.evidence_coverage):.0%}",
                r.classification.replace("_"," ")
            ))
        plot = rows.dropna(subset=["suitability_score"]).head(10).sort_values("suitability_score")
        if not plot.empty:
            self.crop_ax.barh(plot.crop_name, plot.suitability_score)
            self.crop_ax.set_xlim(0,100)
            self.crop_ax.set_xlabel("Screening score / 100")
            self.crop_ax.set_title(f"{self.selected_cell} — {season} crop screening")
            self.crop_ax.grid(axis="x", alpha=.25)
        else:
            self.crop_ax.text(.5,.5,"No scored crops for this cell/season.",ha="center",va="center")
        self.crop_fig.tight_layout()
        self.crop_canvas.draw_idle()
        status = self.evidence_status(self.row_for(self.selected_cell))
        self.crop_evidence_note.configure(text=f"{status} | scores are screening scores, not yield predictions")
        self.crop_details.configure(state="normal")
        self.crop_details.delete("1.0","end")
        self.crop_details.insert("1.0","Select a crop row to inspect its reasons and limitations.")
        self.crop_details.configure(state="disabled")

    def on_crop_select(self, _event=None):
        selected = self.crop_tree.selection()
        if not selected or self.suitability.empty:
            return
        crop_id = selected[0]
        rows = self.suitability[
            (self.suitability.cell_id.astype(str)==self.selected_cell)
            & (self.suitability.season.astype(str)==self.crop_season.get())
            & (self.suitability.crop_id.astype(str)==crop_id)
        ]
        if rows.empty:
            return
        r = rows.iloc[0]
        text = (
            f"{r.crop_name} | {r.season}\n"
            f"Cell: {r.cell_id}\nSuitability screening score: {r.suitability_score}/100\n"
            f"Spatial evidence coverage: {float(r.evidence_coverage):.1%}\n"
            f"Evidence status: {r.evidence_status}\nClassification: {r.classification}\n\n"
            f"FACTOR VALUES\n{r.factor_scores_json}\n\n"
            f"SUPPORTING REASONS\n{r.reasons}\n\nWARNINGS\n{r.warnings}\n\n"
            f"RAINFALL CONTEXT\n{r.rainfall_context}"
        )
        self.crop_details.configure(state="normal")
        self.crop_details.delete("1.0","end")
        self.crop_details.insert("1.0",text)
        self.crop_details.configure(state="disabled")

    # ---------------------------------------------------------
    # Per-cell genetic algorithm analytics and candidate review
    # ---------------------------------------------------------
    def build_rotation_tab(self):
        bar = ttk.Frame(self.rotation_tab)
        bar.pack(fill="x", pady=(0,7))
        ttk.Label(bar,text="Management cell:").pack(side="left")
        self.rotation_cell = tk.StringVar(value=self.selected_cell)
        self.rotation_cell_combo = ttk.Combobox(bar,textvariable=self.rotation_cell,values=self.cells,
                                                state="readonly",width=16)
        self.rotation_cell_combo.pack(side="left",padx=6)
        self.rotation_cell_combo.bind("<<ComboboxSelected>>",lambda _e:self.refresh_rotation_panel())
        self.rotation_status = ttk.Label(bar,text="",style="Sub.TLabel")
        self.rotation_status.pack(side="left",padx=12)

        top = ttk.Frame(self.rotation_tab)
        top.pack(fill="both",expand=True)
        left = ttk.Frame(top)
        left.pack(side="left",fill="both",expand=True)
        right = ttk.Frame(top,width=490)
        right.pack(side="right",fill="both",padx=(8,0))
        right.pack_propagate(False)
        cols=("rank","rotation","fitness","suitability","compatibility")
        self.rotation_tree=ttk.Treeview(left,columns=cols,show="headings",height=12)
        widths={"rank":50,"rotation":310,"fitness":85,"suitability":95,"compatibility":105}
        for c in cols:
            self.rotation_tree.heading(c,text=c.replace("_"," ").title())
            self.rotation_tree.column(c,width=widths[c],anchor="w" if c=="rotation" else "center")
        self.rotation_tree.pack(fill="both",expand=True)
        self.rotation_tree.bind("<<TreeviewSelect>>",self.on_rotation_select)
        self.rotation_fig,self.rotation_ax=plt.subplots(figsize=(6.1,4.1),dpi=100)
        self.rotation_fig.subplots_adjust(left=.1,right=.97,top=.9,bottom=.13)
        self.rotation_canvas=FigureCanvasTkAgg(self.rotation_fig,master=right)
        self.rotation_canvas.get_tk_widget().pack(fill="both",expand=True)
        self.ga_generation_label=ttk.Label(right,text="Generation history",style="Sub.TLabel")
        self.ga_generation_label.pack(anchor="w")
        self.ga_generation_scale=tk.Scale(right,from_=1,to=60,orient="horizontal",command=self.inspect_generation)
        self.ga_generation_scale.pack(fill="x")
        self.rotation_details=tk.Text(self.rotation_tab,height=8,wrap="word",bg="white",relief="solid",bd=1)
        self.rotation_details.pack(fill="x",pady=(8,5))
        controls=ttk.Frame(self.rotation_tab)
        controls.pack(fill="x")
        ttk.Button(controls,text="Approve selected rotation",command=self.approve_district_rotation).pack(side="right")
        self.refresh_rotation_panel()

    def cell_history(self, cell):
        if not isinstance(self.ga_history,list): return pd.DataFrame()
        df=pd.DataFrame([r for r in self.ga_history if str(r.get("cell_id"))==str(cell)])
        return df.sort_values("generation") if not df.empty and "generation" in df else df

    def refresh_rotation_panel(self):
        if not hasattr(self,"rotation_tree"): return
        cell=self.rotation_cell.get() or self.selected_cell
        for item in self.rotation_tree.get_children(): self.rotation_tree.delete(item)
        self.rotation_ax.clear()
        if self.candidates.empty:
            self.rotation_status.configure(text="No district optimizer output yet")
            self.rotation_ax.text(.5,.5,"Run DCR-01 to generate per-cell rotations and GA history.",ha="center",va="center",wrap=True)
            self.rotation_canvas.draw_idle()
            self.rotation_details.delete("1.0","end")
            self.rotation_details.insert("1.0","No district rotation candidates exist yet.")
            return
        rows=self.candidates[self.candidates.cell_id.astype(str)==str(cell)].sort_values("rank")
        for idx,r in rows.iterrows():
            self.rotation_tree.insert("","end",iid=str(idx),values=(
                r.get("rank",""),r.get("rotation",""),r.get("fitness_score",""),
                r.get("average_suitability",""),r.get("rotation_compatibility","")
            ))
        hist=self.cell_history(cell)
        if not hist.empty:
            self.rotation_ax.plot(hist.generation,hist.best_fitness,label="Best fitness",linewidth=2)
            self.rotation_ax.plot(hist.generation,hist.mean_fitness,label="Mean population fitness",linewidth=1.5)
            self.rotation_ax.set_title(f"{cell} — genetic algorithm convergence")
            self.rotation_ax.set_xlabel("Generation")
            self.rotation_ax.set_ylabel("Fitness / 100")
            self.rotation_ax.set_ylim(0,100)
            self.rotation_ax.grid(True,alpha=.25)
            self.rotation_ax.legend()
            self.ga_generation_scale.configure(to=max(1,len(hist)))
            self.ga_generation_scale.set(1)
            self.inspect_generation("1")
            self.rotation_status.configure(text=f"{len(hist)} recorded generations | {len(rows)} ranked rotations")
        else:
            reason=""
            if not self.engine_report:
                reason="DCR-01 has not been run."
            else:
                st=self.engine_report.get("cell_status",{}).get(cell,{})
                reason=st.get("reason",st.get("status","No complete rotation available."))
            self.rotation_ax.text(.5,.5,reason,ha="center",va="center",wrap=True)
            self.rotation_status.configure(text=reason)
        self.rotation_canvas.draw_idle()
        self.rotation_details.delete("1.0","end")
        self.rotation_details.insert("1.0","Select a candidate to inspect its crop sequence and explanation.")

    def inspect_generation(self,value):
        if not hasattr(self,"ga_generation_label"): return
        cell=self.rotation_cell.get() or self.selected_cell
        hist=self.cell_history(cell)
        if hist.empty: return
        i=max(0,min(int(float(value))-1,len(hist)-1))
        r=hist.iloc[i]
        self.ga_generation_label.configure(text=(
            f"Generation {int(r.generation)} | best {float(r.best_fitness):.2f} | "
            f"mean {float(r.mean_fitness):.2f} | unique sequences {int(r.unique_sequences)}"
        ))

    def on_rotation_select(self,_event=None):
        selected=self.rotation_tree.selection()
        if not selected: return
        r=self.candidates.loc[int(selected[0])]
        self.rotation_details.delete("1.0","end")
        self.rotation_details.insert("1.0",(
            f"Cell: {r.cell_id} | Rank: {r['rank']}\nRotation: {r.rotation}\n"
            f"Fitness: {r.fitness_score}/100\nMean suitability: {r.average_suitability}/100\n"
            f"Rotation compatibility: {r.rotation_compatibility}/100\n"
            f"Rotation penalty: {r.total_rotation_penalty}\n\n{r.explanation}"
        ))

    def approve_district_rotation(self):
        selected=self.rotation_tree.selection()
        if not selected:
            messagebox.showinfo("Select rotation","Select a generated rotation first.")
            return
        r=self.candidates.loc[int(selected[0])]
        if not messagebox.askyesno(
            "Confirm approval",
            f"Approve this DCR-01 rotation for {r.cell_id}?\n"
            f"Evidence: {r.get('evidence_status', 'UNKNOWN')} "
            f"({float(r.get('evidence_coverage', 0)):.0%})\n\n"
            f"{r.rotation}\n\nPartial evidence should be treated as provisional."
        ):
            return
        existing={}
        if APPROVAL_FILE.exists():
            try: existing=read_json(APPROVAL_FILE)
            except Exception: existing={}
        if not isinstance(existing,dict): existing={}
        existing[str(r.cell_id)]={
            "project":"AERIS","approval_status":"approved","target":str(r.cell_id),
            "rank":int(r["rank"]),"rotation":str(r.rotation),
            "crop_ids":str(r.crop_ids),"crop_names":str(r.crop_names),
            "fitness_score":float(r.fitness_score),
            "suitability":float(r.average_suitability),
            "compatibility":float(r.rotation_compatibility),
            "evidence_status":str(self.row_for(str(r.cell_id)).get("evidence_status","UNKNOWN")),
            "explanation":str(r.explanation),"source":"DCR-01"
        }
        APPROVAL_FILE.parent.mkdir(parents=True,exist_ok=True)
        APPROVAL_FILE.write_text(json.dumps(existing,indent=2,ensure_ascii=False),encoding="utf-8")
        self.status.set(f"Approved {r.cell_id} — saved locally")
        messagebox.showinfo("Approved",f"Saved approved district-cell rotation:\n{APPROVAL_FILE}")

    # ---------------------------------------------------------
    # Rainfall outlook panel (only displays RIE-01 data if present)
    # ---------------------------------------------------------
    def build_rainfall_tab(self):
        ttk.Label(self.rainfall_tab,text="RIE-01 SEASONAL RAINFALL OUTLOOK",
                  font=("Segoe UI",12,"bold")).pack(anchor="w")
        ttk.Label(self.rainfall_tab,
                  text="Statistical outlook from the existing rainfall engine. This is a single Nashik climate point shared across all cells, not a cell-specific forecast or live weather feed.",
                  style="Sub.TLabel",wraplength=1100).pack(anchor="w",pady=(4,10))
        self.rainfall_note=ttk.Label(self.rainfall_tab,text="",style="Sub.TLabel")
        self.rainfall_note.pack(anchor="w")
        self.rainfall_fig,self.rainfall_ax=plt.subplots(figsize=(10,4.8),dpi=100)
        self.rainfall_fig.subplots_adjust(left=.1,right=.97,top=.9,bottom=.18)
        self.rainfall_canvas=FigureCanvasTkAgg(self.rainfall_fig,master=self.rainfall_tab)
        self.rainfall_canvas.get_tk_widget().pack(fill="both",expand=True)
        self.refresh_rainfall_panel()

    def refresh_rainfall_panel(self):
        if not hasattr(self,"rainfall_ax"): return
        self.rainfall_ax.clear()
        if self.rainfall.empty or not {"season","forecast_mm","historical_mean_mm"}.issubset(self.rainfall.columns):
            self.rainfall_note.configure(text=f"No RIE-01 forecast CSV found at: {RAINFALL_FORECAST}")
            self.rainfall_ax.text(.5,.5,"Run the existing RIE-01 rainfall prediction engine to populate this panel.",ha="center",va="center",wrap=True)
        else:
            d=self.rainfall.set_index("season")
            order=[s for s in ("kharif","post_monsoon","winter","summer") if s in d.index]
            x=np.arange(len(order))
            hist=[float(d.loc[s,"historical_mean_mm"]) for s in order]
            pred=[float(d.loc[s,"forecast_mm"]) for s in order]
            width=.36
            self.rainfall_ax.bar(x-width/2,hist,width,label="Historical seasonal mean")
            self.rainfall_ax.bar(x+width/2,pred,width,label="Forecast outlook")
            self.rainfall_ax.set_xticks(x,order)
            self.rainfall_ax.set_ylabel("Rainfall (mm)")
            self.rainfall_ax.set_title("Historical seasonal mean vs statistical outlook")
            self.rainfall_ax.legend()
            self.rainfall_ax.grid(axis="y",alpha=.25)
            self.rainfall_note.configure(text="RIE-01 data loaded. Forecast uncertainty intervals are retained in the source CSV; this chart shows point estimates.")
        self.rainfall_canvas.draw_idle()

    def row_for(self, cell):
        rows = self.profiles[self.profiles[self.cell_col].astype(str) == str(cell)]
        return None if rows.empty else rows.iloc[0]

    def numeric_value(self, row, col):
        if row is None or col not in row.index:
            return np.nan
        try:
            value = float(row[col])
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def geometry_shape(self, feature):
        from shapely.geometry import shape
        return shape(feature["geometry"])

    @staticmethod
    def polygon_parts(geom):
        if geom is None or geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            return [geom]
        if hasattr(geom, "geoms"):
            parts = []
            for g in geom.geoms:
                parts.extend(ScientistWorkbench.polygon_parts(g))
            return parts
        return []

    def draw_geometry(self, ax, geom, transformer, face, edge, width=1.0, alpha=1.0, z=2):
        for part in self.polygon_parts(geom):
            coords = [transformer.transform(x, y) for x, y in part.exterior.coords]
            ax.add_patch(MplPolygon(coords, closed=True, facecolor=face,
                                    edgecolor=edge, linewidth=width, alpha=alpha, zorder=z))
            for interior in part.interiors:
                hole = [transformer.transform(x, y) for x, y in interior.coords]
                ax.add_patch(MplPolygon(hole, closed=True, facecolor=BG,
                                        edgecolor=edge, linewidth=.5, zorder=z+0.1))

    def refresh_map(self):
        from pyproj import Transformer
        from shapely.geometry import shape
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        ax = self.ax
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

        # District boundary is drawn as the geographic frame.
        district_geoms = []
        for f in self.district_features:
            if f.get("geometry"):
                district_geoms.append(shape(f["geometry"]))
        if district_geoms:
            from shapely.ops import unary_union
            district = unary_union(district_geoms)
            self.draw_geometry(ax, district, transformer, "#f4f1e8", "#333333", 1.7, 1, 1)
            minx, miny, maxx, maxy = district.bounds
        else:
            # Fall back to grid extent only; never label this as district coverage.
            grid_geoms = [self.geometry_shape(f) for f in self.features if f.get("geometry")]
            if not grid_geoms:
                ax.text(.5, .5, "No district boundary or grid geometry available",
                        ha="center", va="center", transform=ax.transAxes)
                self.canvas.draw_idle()
                return
            from shapely.ops import unary_union
            district = unary_union(grid_geoms)
            minx, miny, maxx, maxy = district.bounds

        layer = self.selected_layer.get()
        vals = {cell: self.numeric_value(self.row_for(cell), layer) for cell in self.cells}
        valid = [v for v in vals.values() if pd.notna(v)]
        vmin, vmax = (min(valid), max(valid)) if valid else (0.0, 1.0)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")

        for cell in self.cells:
            feature = self.feature_map[cell]
            geom = shape(feature["geometry"])
            val = vals[cell]
            color = cmap(norm(val)) if pd.notna(val) else "#bdbdbd"
            self.draw_geometry(ax, geom, transformer, color, "#242424", 1.0, .86, 3)
            rp = geom.representative_point()
            x, y = transformer.transform(rp.x, rp.y)
            ax.text(x, y, cell, ha="center", va="center", fontsize=8,
                    weight="bold", color="#111111", zorder=5,
                    bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=1.7))

        # Set view limits using projected district bounds.
        corners = [transformer.transform(minx, miny), transformer.transform(minx, maxy),
                   transformer.transform(maxx, miny), transformer.transform(maxx, maxy)]
        xs, ys = zip(*corners)
        padx, pady = (max(xs)-min(xs))*.035, (max(ys)-min(ys))*.035
        ax.set_xlim(min(xs)-padx, max(xs)+padx)
        ax.set_ylim(min(ys)-pady, max(ys)+pady)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Nashik district | {layer}", fontsize=11)
        ax.set_xlabel("UTM Zone 43N easting (m)")
        ax.set_ylabel("UTM Zone 43N northing (m)")
        ax.grid(False)
        if valid:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            cb = self.fig.colorbar(sm, ax=ax, fraction=.035, pad=.02)
            cb.set_label(f"{layer} — zonal mean / raster statistic")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.update_cell_panel()
        self.fill_evidence_table()

    def on_map_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        from pyproj import Transformer
        from shapely.geometry import Point
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
        # Inverse transformation to WGS84 for GeoJSON point-in-polygon test.
        inverse = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
        lon, lat = inverse.transform(event.xdata, event.ydata)
        point = Point(lon, lat)
        for cell in self.cells:
            geom = self.geometry_shape(self.feature_map[cell])
            if geom.contains(point) or geom.touches(point):
                self.select_cell(cell)
                break

    def on_list_select(self, _event=None):
        selected = self.cell_list.curselection()
        if selected:
            self.select_cell(self.cell_list.get(selected[0]))

    def select_cell(self, cell):
        if cell not in self.cells:
            return
        self.selected_cell = cell
        self.cell_title.configure(text=cell)
        self.cell_list.selection_clear(0, "end")
        self.cell_list.selection_set(self.cells.index(cell))
        self.evidence_cell.set(cell)
        if hasattr(self, "crop_cell"): self.crop_cell.set(cell)
        if hasattr(self, "rotation_cell"): self.rotation_cell.set(cell)
        self.update_cell_panel()
        self.fill_evidence_table()
        self.refresh_crop_panel()
        self.refresh_rotation_panel()

    def evidence_status(self, row):
        if row is None:
            return "No profile row"
        for key in ("evidence_status", "status", "coverage_status"):
            if key in row.index and pd.notna(row[key]):
                return str(row[key])
        return "Status field not present in profile"

    def update_cell_panel(self):
        row = self.row_for(self.selected_cell)
        lines = [f"Cell: {self.selected_cell}",
                 f"Evidence status: {self.evidence_status(row)}", "",
                 "RASTER EVIDENCE"]
        if row is None:
            lines.append("No environmental profile row found.")
        else:
            coverage_cols = [c for c in row.index if "coverage" in c.lower()]
            for c in coverage_cols:
                val = row[c]
                lines.append(f"{c}: {val}")
            lines += ["", "SELECTED LAYER"]
            layer = self.selected_layer.get()
            value = self.numeric_value(row, layer)
            lines.append(f"{layer}: {value:.4f}" if pd.notna(value) else f"{layer}: no valid value")
            lines += ["", "INTERPRETATION NOTE",
                      "These are raster zonal statistics. SoilGrids scale factors and physical "
                      "units have not been applied by this viewer. Grey cells indicate missing "
                      "values for the selected layer, not zero-valued soil."]
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(lines))
        self.summary.configure(state="disabled")

    def fill_evidence_table(self):
        if not hasattr(self, "evidence_tree"):
            return
        for item in self.evidence_tree.get_children():
            self.evidence_tree.delete(item)
        cell = self.evidence_cell.get() if hasattr(self, "evidence_cell") else self.selected_cell
        row = self.row_for(cell)
        if row is None:
            return
        for col in self.profiles.columns:
            if col == self.cell_col:
                continue
            val = row[col]
            if pd.isna(val):
                shown = "No data"
            elif isinstance(val, (float, np.floating)):
                shown = f"{val:.6g}"
            else:
                shown = str(val)
            self.evidence_tree.insert("", "end", values=(col, shown))


def main():
    required = [GRID_FILE, PROFILES_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "AERIS district-grid data missing",
            "The district-grid workbench needs these existing outputs:\n\n" + "\n".join(missing)
            + "\n\nRun the district-grid builder first; this UI does not regenerate raster statistics."
        )
        root.destroy()
        return
    root = tk.Tk()
    try:
        ScientistWorkbench(root)
    except Exception as exc:
        messagebox.showerror("AERIS startup error", f"{type(exc).__name__}: {exc}")
        raise
    root.mainloop()


if __name__ == "__main__":
    main()