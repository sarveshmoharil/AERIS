"""
AERIS Scientist Workbench
Classic Tkinter interface for spatial crop-rotation analysis.
Reads existing AERIS outputs; does not fabricate or rerun optimization results.
Run from the E:\\TrialShit project root:
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
OPT = DATA / "spatial_optimization"
SPATIAL = DATA / "spatial"
APPROVAL_FILE = DATA / "approved_recommendations.json"

CANDIDATES_FILE = OPT / "rotation_optimizer_candidates.csv"
HISTORY_FILE = OPT / "optimization_generation_history.json"
REPORT_FILE = OPT / "rotation_optimizer_report.json"
PROFILES_FILE = SPATIAL / "grid_cell_environmental_profiles.csv"
ASSIGNMENTS_FILE = SPATIAL / "sample_grid_assignments.csv"
GRID_FILE = SPATIAL / "nashik_environmental_grid.geojson"
DISTRICT_FILE = SPATIAL / "nashik_district_boundary.geojson"

BG = "#f0f0f0"
PANEL = "#ffffff"
TEXT = "#202020"
MUTED = "#555555"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


class ScientistWorkbench:
    def __init__(self, root):
        self.root = root
        self.root.title("AERIS v1.0 — Scientist Workbench")
        self.root.geometry("1420x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg=BG)

        self.candidates = pd.read_csv(CANDIDATES_FILE)
        self.profiles = pd.read_csv(PROFILES_FILE)
        self.assignments = pd.read_csv(ASSIGNMENTS_FILE)
        self.history_raw = load_json(HISTORY_FILE)
        self.report = load_json(REPORT_FILE)
        self.history = self.normalise_history(self.history_raw)

        self.target_col = first_existing_column(self.candidates, ["target"])
        self.cell_col = first_existing_column(self.profiles, ["grid_cell", "target", "cell_id"])
        self.assignment_cell_col = first_existing_column(self.assignments, ["grid_cell"])
        self.layer_columns = self.find_environment_layers()
        self.geo_features = self.load_grid_geojson()
        self.district_features = self.load_district_geojson()
        self.clipped_grid = self.build_clipped_grid()
        self.selected_target = "G101"
        self.selected_layer = tk.StringVar(value=self.layer_columns[0] if self.layer_columns else "Sample count")
        self.selected_candidate = None

        self.setup_style()
        self.build_header()
        self.build_tabs()
        self.refresh_all()

    def normalise_history(self, raw):
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = None
            for key in ("history", "generation_history", "records", "data"):
                if isinstance(raw.get(key), list):
                    records = raw[key]
                    break
            if records is None:
                records = []
                for key, value in raw.items():
                    if isinstance(value, list):
                        records.extend(value)
        else:
            records = []
        df = pd.DataFrame(records)
        if df.empty:
            return df
        aliases = {
            "target": ["target", "target_id", "cell"],
            "generation": ["generation", "gen"],
            "best_fitness": ["best_fitness", "best_score"],
            "mean_fitness": ["mean_fitness", "average_fitness"],
        }
        for wanted, choices in aliases.items():
            if wanted not in df.columns:
                for c in choices:
                    if c in df.columns:
                        df[wanted] = df[c]
                        break
        return df

    def find_environment_layers(self):
        preferred = [
            "phh2o_0-5cm_mean", "phh2o_5-15cm_mean", "phh2o_15-30cm_mean",
            "soc_0-5cm_mean", "soc_5-15cm_mean", "soc_15-30cm_mean",
            "clay_0-5cm_mean", "sand_0-5cm_mean", "nitrogen_0-5cm_mean",
            "cec_0-5cm_mean", "rootzone_moisture_mean",
            "elevation_m_mean", "slope_degrees_mean",
        ]
        found = [c for c in preferred if c in self.profiles.columns]
        for c in self.profiles.columns:
            if c.endswith("_mean") and c not in found and pd.api.types.is_numeric_dtype(self.profiles[c]):
                found.append(c)
        return found

    def load_grid_geojson(self):
        if not GRID_FILE.exists():
            return []
        data = load_json(GRID_FILE)
        return data.get("features", []) if isinstance(data, dict) else []

    def load_district_geojson(self):
        if not DISTRICT_FILE.exists():
            return []
        data = load_json(DISTRICT_FILE)
        return data.get("features", []) if isinstance(data, dict) else []

    def build_clipped_grid(self):
        """Intersect each research grid cell with the official district polygon."""
        try:
            from shapely.geometry import shape
            from shapely.ops import unary_union
        except ImportError:
            return {}

        district_shapes = []
        for feature in self.district_features:
            geom = feature.get("geometry")
            if geom:
                try:
                    district_shapes.append(shape(geom))
                except Exception:
                    pass
        if not district_shapes:
            return {}
        district = unary_union(district_shapes)
        clipped = {}
        for feature in self.geo_features:
            cell = self.feature_cell_id(feature)
            geom = feature.get("geometry")
            if not cell or not geom:
                continue
            try:
                cell_shape = shape(geom)
                clipped_shape = cell_shape.intersection(district)
                if not clipped_shape.is_empty:
                    clipped[cell] = clipped_shape
            except Exception:
                continue
        return clipped

    @staticmethod
    def polygon_parts(geom):
        """Return polygon components from Polygon/MultiPolygon/GeometryCollection."""
        if geom is None or geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            return [geom]
        if hasattr(geom, "geoms"):
            parts = []
            for item in geom.geoms:
                parts.extend(ScientistWorkbench.polygon_parts(item))
            return parts
        return []

    def setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 9))
        style.configure("TFrame", background=BG)
        style.configure("White.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), background=BG)
        style.configure("Sub.TLabel", foreground=MUTED, background=BG)
        style.configure("TButton", padding=(8, 5))
        style.configure("Treeview", rowheight=25, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def build_header(self):
        header = ttk.Frame(self.root, padding=(12, 9))
        header.pack(fill="x")
        ttk.Label(header, text="AERIS v1.0", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="  |  SCIENTIST WORKBENCH", style="Sub.TLabel").pack(side="left", pady=(5, 0))
        self.status = tk.StringVar(value="Loaded optimizer outputs")
        ttk.Label(header, textvariable=self.status, style="Sub.TLabel").pack(side="right")
        ttk.Separator(self.root).pack(fill="x")

    def build_tabs(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.map_tab = ttk.Frame(self.tabs, padding=8)
        self.candidate_tab = ttk.Frame(self.tabs, padding=8)
        self.ga_tab = ttk.Frame(self.tabs, padding=8)
        self.approval_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.map_tab, text="Spatial Workbench")
        self.tabs.add(self.candidate_tab, text="Rotation Candidates")
        self.tabs.add(self.ga_tab, text="Genetic Algorithm")
        self.tabs.add(self.approval_tab, text="Review & Approval")
        self.build_map_tab()
        self.build_candidate_tab()
        self.build_ga_tab()
        self.build_approval_tab()

    def targets(self):
        return list(dict.fromkeys(self.candidates["target"].astype(str).tolist()))

    def build_map_tab(self):
        top = ttk.Frame(self.map_tab)
        top.pack(fill="x", pady=(0, 7))
        ttk.Label(top, text="Environmental layer:").pack(side="left")
        self.layer_combo = ttk.Combobox(
            top, textvariable=self.selected_layer,
            values=self.layer_columns, state="readonly", width=34
        )
        self.layer_combo.pack(side="left", padx=6)
        self.layer_combo.bind("<<ComboboxSelected>>", lambda e: self.draw_map())
        ttk.Label(top, text="Select a cell from the map or list.").pack(side="left", padx=12)

        body = ttk.Frame(self.map_tab)
        body.pack(fill="both", expand=True)
        map_frame = ttk.Frame(body)
        map_frame.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(body, width=310, padding=10, relief="groove")
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)

        self.map_fig, self.map_ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.map_fig.subplots_adjust(left=.06, right=.97, top=.94, bottom=.08)
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=map_frame)
        self.map_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.map_canvas.mpl_connect("button_press_event", self.on_map_click)

        ttk.Label(side, text="MANAGEMENT CELLS", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.cell_list = tk.Listbox(side, height=10, exportselection=False)
        self.cell_list.pack(fill="x", pady=6)
        for target in [f"G{r}0{c}" for r in range(1, 5) for c in range(1, 3)]:
            self.cell_list.insert("end", target)
        self.cell_list.bind("<<ListboxSelect>>", self.on_cell_list_select)
        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="SELECTED CELL", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.cell_title = ttk.Label(side, text="G101", font=("Segoe UI", 13, "bold"))
        self.cell_title.pack(anchor="w", pady=(5, 2))
        self.cell_summary = tk.Text(side, height=12, wrap="word", relief="flat", bg="white")
        self.cell_summary.pack(fill="both", expand=True, pady=5)
        self.cell_summary.configure(state="disabled")
        ttk.Button(side, text="View rotation candidates", command=lambda: self.tabs.select(self.candidate_tab)).pack(fill="x", pady=4)

    def profile_row(self, target):
        if self.cell_col and self.cell_col in self.profiles.columns:
            rows = self.profiles[self.profiles[self.cell_col].astype(str) == str(target)]
            if not rows.empty:
                return rows.iloc[0]
        return None

    def cell_value(self, target, layer):
        row = self.profile_row(target)
        if layer == "Sample count":
            if self.assignment_cell_col:
                return int((self.assignments[self.assignment_cell_col].astype(str) == target).sum())
            return np.nan
        if row is not None and layer in row.index:
            try:
                return float(row[layer])
            except (ValueError, TypeError):
                return np.nan
        return np.nan

    def feature_cell_id(self, feature):
        props = feature.get("properties", {})
        for k in ("grid_cell", "cell_id", "target", "id", "name"):
            if k in props and str(props[k]).startswith("G"):
                return str(props[k])
        return None

    def draw_map(self):
        from pyproj import Transformer
        ax = self.map_ax
        ax.clear()
        layer = self.selected_layer.get()
        cells = [f"G{r}0{c}" for r in range(1, 5) for c in range(1, 3)]
        values = {c: self.cell_value(c, layer) for c in cells}
        valid = [float(v) for v in values.values() if pd.notna(v)]
        vmin, vmax = (min(valid), max(valid)) if valid else (0.0, 1.0)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Draw the official district polygon as the geographic frame. The existing
        # 8-cell research grid is overlaid only where current samples exist.
        district_bounds = []
        for feature in self.district_features:
            geom = feature.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            polygons = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []
            for polygon in polygons:
                if not polygon:
                    continue
                ring = polygon[0]
                projected_ring = [transformer.transform(float(pt[0]), float(pt[1])) for pt in ring]
                district_bounds.extend(projected_ring)
                ax.add_patch(MplPolygon(projected_ring, closed=True, facecolor="#f2f0e6",
                                        edgecolor="#343434", linewidth=1.8, alpha=.9, zorder=1))
        if district_bounds:
            bx = [pt[0] for pt in district_bounds]
            by = [pt[1] for pt in district_bounds]
            padx = (max(bx) - min(bx)) * .04
            pady = (max(by) - min(by)) * .04
            x0, x1 = min(bx)-padx, max(bx)+padx
            y0, y1 = min(by)-pady, max(by)+pady
        else:
            # Fallback extent around current sample footprint; explicitly not district coverage.
            west, east, south, north = 73.30, 74.90, 19.25, 20.75
            x0, y0 = transformer.transform(west, south)
            x1, y1 = transformer.transform(east, north)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        # Draw cells only after clipping them to the district boundary.
        for cell, clipped_geom in self.clipped_grid.items():
            val = values.get(cell, np.nan)
            color = cmap(norm(float(val))) if pd.notna(val) else "#bdbdbd"
            parts = self.polygon_parts(clipped_geom)
            label_point = clipped_geom.representative_point()
            for part in parts:
                projected = [transformer.transform(float(x), float(y))
                             for x, y in part.exterior.coords]
                ax.add_patch(MplPolygon(
                    projected, closed=True, facecolor=color,
                    edgecolor="#202020", linewidth=1.5,
                    alpha=0.68, zorder=4
                ))
            lx, ly = transformer.transform(label_point.x, label_point.y)
            ax.text(lx, ly, cell, ha="center", va="center", fontsize=9,
                    weight="bold", color="#111111", zorder=5,
                    bbox=dict(facecolor="white", alpha=.78, edgecolor="none", pad=2))

        if {"longitude", "latitude"}.issubset(self.assignments.columns):
            px, py = transformer.transform(
                self.assignments["longitude"].astype(float).to_numpy(),
                self.assignments["latitude"].astype(float).to_numpy()
            )
            ax.scatter(px, py, s=5, c="#202020", alpha=.45, zorder=6, label="Environmental samples")

        self.map_basemap_status = (
            f"District boundary loaded | {len(self.clipped_grid)} clipped research cells"
            if self.district_features and self.clipped_grid else
            "District boundary loaded, but clipping failed — check GeoJSON CRS / Shapely"
            if self.district_features else
            "District boundary file missing — showing sample-grid extent only"
        )

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f"Nashik region | {layer}", fontsize=11)
        ax.set_xlabel("Web Mercator easting")
        ax.set_ylabel("Web Mercator northing")
        ax.grid(False)
        if valid:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            cb = self.map_fig.colorbar(sm, ax=ax, fraction=.028, pad=.015)
            cb.set_label(f"{layer} (aggregated mean)")
        ax.text(.01, .015, self.map_basemap_status, transform=ax.transAxes,
                fontsize=7, color="#333333",
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        self.map_canvas.draw_idle()
        self.update_cell_panel()

    def on_map_click(self, event):
        if event.inaxes != self.map_ax or event.xdata is None or event.ydata is None:
            return
        from pyproj import Transformer
        from shapely.geometry import Point
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        lon, lat = transformer.transform(event.xdata, event.ydata, direction="INVERSE")
        point = Point(lon, lat)
        for cell, geom in self.clipped_grid.items():
            if geom.contains(point) or geom.touches(point):
                self.select_target(cell)
                return

    def on_cell_list_select(self, event=None):
        selection = self.cell_list.curselection()
        if selection:
            self.select_target(self.cell_list.get(selection[0]))

    def select_target(self, target):
        self.selected_target = target
        cells = [self.cell_list.get(i) for i in range(self.cell_list.size())]
        if target in cells:
            idx = cells.index(target)
            self.cell_list.selection_clear(0, "end")
            self.cell_list.selection_set(idx)
        self.update_cell_panel()
        self.refresh_candidates()
        self.refresh_ga()

    def update_cell_panel(self):
        target = self.selected_target
        rows = self.candidates[self.candidates["target"].astype(str) == target].sort_values("rank")
        lines = [f"Target: {target}", f"Candidates available: {len(rows)}"]
        if not rows.empty:
            best = rows.iloc[0]
            lines += [
                "", "TOP ROTATION",
                str(best.get("rotation", "Not available")),
                f"Rotation score: {best.get('score', '—')}",
                f"Suitability: {best.get('suitability', '—')}",
                f"Compatibility: {best.get('compatibility', '—')}",
                f"Penalty: {best.get('penalty', '—')}",
                f"Evidence coverage: {best.get('evidence', '—')}",
                "", "EXPLANATION",
                str(best.get("explanation", "No explanation supplied in candidate output.")),
            ]
        profile = self.profile_row(target)
        if profile is not None:
            lines += ["", "ENVIRONMENTAL PROFILE"]
            for col in self.layer_columns[:6]:
                try:
                    lines.append(f"{col}: {float(profile[col]):.3f}")
                except Exception:
                    pass
        self.cell_title.configure(text=target)
        self.cell_summary.configure(state="normal")
        self.cell_summary.delete("1.0", "end")
        self.cell_summary.insert("1.0", "\n".join(lines))
        self.cell_summary.configure(state="disabled")

    def build_candidate_tab(self):
        bar = ttk.Frame(self.candidate_tab)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Optimization target:").pack(side="left")
        self.candidate_target = tk.StringVar(value=self.selected_target)
        self.candidate_combo = ttk.Combobox(bar, textvariable=self.candidate_target, values=self.targets(), state="readonly", width=20)
        self.candidate_combo.pack(side="left", padx=6)
        self.candidate_combo.bind("<<ComboboxSelected>>", lambda e: self.select_target(self.candidate_target.get()))
        cols = ("rank", "rotation", "score", "suitability", "compatibility", "penalty", "evidence")
        self.candidate_tree = ttk.Treeview(self.candidate_tab, columns=cols, show="headings", height=12)
        widths = {"rank": 55, "rotation": 390, "score": 85, "suitability": 95, "compatibility": 105, "penalty": 75, "evidence": 85}
        for c in cols:
            self.candidate_tree.heading(c, text=c.replace("_", " ").title())
            self.candidate_tree.column(c, width=widths[c], anchor="center" if c != "rotation" else "w")
        self.candidate_tree.pack(fill="x", expand=False)
        self.candidate_tree.bind("<<TreeviewSelect>>", self.on_candidate_select)
        self.candidate_details = tk.Text(self.candidate_tab, height=9, wrap="word", bg="white", relief="solid", bd=1)
        self.candidate_details.pack(fill="both", expand=True, pady=8)
        self.candidate_details.insert("1.0", "Select a candidate to inspect its explanation and crop sequence.")
        self.candidate_details.configure(state="disabled")
        ttk.Button(self.candidate_tab, text="Use selected candidate for approval", command=self.copy_candidate_to_approval).pack(anchor="e")

    def refresh_candidates(self):
        if not hasattr(self, "candidate_tree"):
            return
        self.candidate_target.set(self.selected_target)
        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)
        rows = self.candidates[self.candidates["target"].astype(str) == self.selected_target].sort_values("rank")
        for idx, r in rows.iterrows():
            self.candidate_tree.insert("", "end", iid=str(idx), values=(
                r.get("rank", ""), r.get("rotation", ""), r.get("score", ""),
                r.get("suitability", ""), r.get("compatibility", ""),
                r.get("penalty", ""), r.get("evidence", "")
            ))
        if len(rows) == 0:
            self.candidate_details.configure(state="normal")
            self.candidate_details.delete("1.0", "end")
            self.candidate_details.insert("1.0", "No candidates were generated for this target.")
            self.candidate_details.configure(state="disabled")

    def on_candidate_select(self, event=None):
        selected = self.candidate_tree.selection()
        if not selected:
            return
        row = self.candidates.loc[int(selected[0])]
        self.selected_candidate = row
        text = (
            f"Target: {row.get('target')}\nRank: {row.get('rank')}\n"
            f"Rotation: {row.get('rotation')}\nCrop IDs: {row.get('crop_ids')}\n"
            f"Seasonal suitability: {row.get('seasonal_suitability')}\n\n"
            f"Explanation:\n{row.get('explanation', 'No explanation provided.')}"
        )
        self.candidate_details.configure(state="normal")
        self.candidate_details.delete("1.0", "end")
        self.candidate_details.insert("1.0", text)
        self.candidate_details.configure(state="disabled")

    def build_ga_tab(self):
        bar = ttk.Frame(self.ga_tab)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="Target:").pack(side="left")
        self.ga_target = tk.StringVar(value=self.selected_target)
        self.ga_combo = ttk.Combobox(bar, textvariable=self.ga_target, values=self.targets(), state="readonly", width=20)
        self.ga_combo.pack(side="left", padx=6)
        self.ga_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_ga())
        self.ga_info = ttk.Label(bar, text="")
        self.ga_info.pack(side="left", padx=12)

        self.ga_fig, self.ga_ax = plt.subplots(figsize=(9, 4.7), dpi=100)
        self.ga_fig.subplots_adjust(left=.08, right=.97, top=.9, bottom=.13)
        self.ga_canvas = FigureCanvasTkAgg(self.ga_fig, master=self.ga_tab)
        self.ga_canvas.get_tk_widget().pack(fill="both", expand=True)
        controls = ttk.Frame(self.ga_tab)
        controls.pack(fill="x", pady=6)
        ttk.Button(controls, text="Play generations", command=self.play_generations).pack(side="left")
        ttk.Button(controls, text="Stop", command=self.stop_generations).pack(side="left", padx=5)
        self.gen_scale = tk.Scale(controls, from_=1, to=50, orient="horizontal", length=380, command=self.draw_generation)
        self.gen_scale.pack(side="left", padx=12)
        self.gen_label = ttk.Label(controls, text="Generation 1")
        self.gen_label.pack(side="left")
        self.playing = False
        self.play_after_id = None

    def target_history(self, target):
        if self.history.empty or "target" not in self.history.columns:
            return pd.DataFrame()
        df = self.history[self.history["target"].astype(str) == str(target)].copy()
        if "generation" in df:
            df = df.sort_values("generation")
        return df

    def refresh_ga(self):
        if not hasattr(self, "ga_ax"):
            return
        target = self.ga_target.get() or self.selected_target
        df = self.target_history(target)
        self.stop_generations()
        self.ga_ax.clear()
        if df.empty or not {"best_fitness", "mean_fitness"}.issubset(df.columns):
            self.ga_ax.text(.5, .5, "No generation history available for this target.", ha="center", va="center")
            self.ga_info.configure(text="No history")
            self.ga_canvas.draw_idle()
            return
        x = pd.to_numeric(df["generation"], errors="coerce")
        best = pd.to_numeric(df["best_fitness"], errors="coerce")
        mean = pd.to_numeric(df["mean_fitness"], errors="coerce")
        self.ga_ax.plot(x, best, label="Best fitness", linewidth=2)
        self.ga_ax.plot(x, mean, label="Mean population fitness", linewidth=1.5)
        self.ga_ax.set_title(f"Genetic algorithm evolution — {target}")
        self.ga_ax.set_xlabel("Generation")
        self.ga_ax.set_ylabel("Fitness score")
        self.ga_ax.grid(True, alpha=.3)
        self.ga_ax.legend()
        unique = df["unique_sequences"].iloc[-1] if "unique_sequences" in df.columns else "—"
        self.ga_info.configure(text=f"{len(df)} recorded generations | final unique sequences: {unique}")
        self.gen_scale.configure(to=max(1, len(df)))
        self.gen_scale.set(1)
        self.draw_generation("1")
        self.ga_canvas.draw_idle()

    def draw_generation(self, value):
        target = self.ga_target.get() or self.selected_target
        df = self.target_history(target)
        if df.empty:
            return
        gen = int(float(value))
        row = df.iloc[min(gen - 1, len(df) - 1)]
        self.gen_label.configure(text=f"Generation {int(row.get('generation', gen))}  |  Best: {row.get('best_fitness', '—')}  |  Mean: {row.get('mean_fitness', '—')}")
        self.ga_ax.set_title(f"{target} — generation {int(row.get('generation', gen))}: {row.get('best_sequence', 'sequence unavailable')}")
        self.ga_canvas.draw_idle()

    def play_generations(self):
        self.stop_generations()
        self.playing = True
        self.animate_step()

    def animate_step(self):
        if not self.playing:
            return
        current = int(float(self.gen_scale.get()))
        maximum = int(float(self.gen_scale.cget("to")))
        if current >= maximum:
            self.stop_generations()
            return
        self.gen_scale.set(current + 1)
        self.play_after_id = self.root.after(220, self.animate_step)

    def stop_generations(self):
        self.playing = False
        if self.play_after_id is not None:
            try:
                self.root.after_cancel(self.play_after_id)
            except tk.TclError:
                pass
            self.play_after_id = None

    def build_approval_tab(self):
        ttk.Label(self.approval_tab, text="SCIENTIST REVIEW", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            self.approval_tab,
            text="Select a target and a generated candidate. Approval is saved locally for the farmer portal.",
            style="Sub.TLabel"
        ).pack(anchor="w", pady=(3, 12))
        form = ttk.Frame(self.approval_tab)
        form.pack(fill="x")
        ttk.Label(form, text="Target:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.approval_target = tk.StringVar(value=self.selected_target)
        self.approval_target_combo = ttk.Combobox(form, textvariable=self.approval_target, values=self.targets(), state="readonly", width=25)
        self.approval_target_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.approval_target_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_approval_candidates())
        ttk.Label(form, text="Candidate:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.approval_candidate = tk.StringVar()
        self.approval_candidate_combo = ttk.Combobox(form, textvariable=self.approval_candidate, state="readonly", width=65)
        self.approval_candidate_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.approval_preview = tk.Text(self.approval_tab, height=12, wrap="word", bg="white", relief="solid", bd=1)
        self.approval_preview.pack(fill="both", expand=True, pady=10)
        self.approval_candidate_combo.bind("<<ComboboxSelected>>", lambda e: self.update_approval_preview())
        ttk.Button(self.approval_tab, text="APPROVE SELECTED ROTATION", command=self.approve_candidate).pack(anchor="e", pady=5)
        self.refresh_approval_candidates()

    def refresh_approval_candidates(self):
        if not hasattr(self, "approval_candidate_combo"):
            return
        target = self.approval_target.get()
        rows = self.candidates[self.candidates["target"].astype(str) == target].sort_values("rank")
        options = [f"Rank {r['rank']} — {r['rotation']} (score {r['score']})" for _, r in rows.iterrows()]
        self.approval_candidate_combo.configure(values=options)
        self.approval_candidate.set(options[0] if options else "")
        self.update_approval_preview()

    def approval_row(self):
        target = self.approval_target.get()
        rows = self.candidates[self.candidates["target"].astype(str) == target].sort_values("rank")
        choice = self.approval_candidate.get()
        for _, row in rows.iterrows():
            label = f"Rank {row['rank']} — {row['rotation']} (score {row['score']})"
            if label == choice:
                return row
        return None

    def update_approval_preview(self):
        row = self.approval_row()
        text = "No candidate selected." if row is None else (
            f"Target: {row['target']}\nRotation: {row['rotation']}\n"
            f"Overall rotation score: {row['score']}\nEnvironmental suitability: {row['suitability']}\n"
            f"Rotation compatibility: {row['compatibility']}\nRotation penalty: {row['penalty']}\n"
            f"Evidence coverage: {row['evidence']}\n\n"
            f"Explanation:\n{row.get('explanation', 'No explanation supplied.')}\n\n"
            "Approval publishes this selected sequence to the local farmer-facing recommendation store."
        )
        self.approval_preview.delete("1.0", "end")
        self.approval_preview.insert("1.0", text)

    def copy_candidate_to_approval(self):
        if self.selected_candidate is None:
            messagebox.showinfo("Select candidate", "Select a candidate row first.")
            return
        self.approval_target.set(str(self.selected_candidate["target"]))
        self.refresh_approval_candidates()
        row = self.selected_candidate
        options = list(self.approval_candidate_combo.cget("values"))
        desired = f"Rank {row['rank']} — {row['rotation']} (score {row['score']})"
        if desired in options:
            self.approval_candidate.set(desired)
        self.update_approval_preview()
        self.tabs.select(self.approval_tab)

    def approve_candidate(self):
        row = self.approval_row()
        if row is None:
            messagebox.showwarning("No candidate", "Choose a valid candidate before approving.")
            return
        if not messagebox.askyesno("Confirm approval", f"Approve this rotation for {row['target']}?\n\n{row['rotation']}"):
            return
        payload = {
            "project": "AERIS",
            "approval_status": "approved",
            "target": str(row["target"]),
            "rank": int(row["rank"]),
            "rotation": str(row["rotation"]),
            "crop_ids": str(row["crop_ids"]),
            "crop_names": str(row["crop_names"]),
            "score": float(row["score"]),
            "suitability": float(row["suitability"]),
            "compatibility": float(row["compatibility"]),
            "penalty": float(row["penalty"]),
            "evidence": float(row["evidence"]),
            "seasonal_suitability": str(row.get("seasonal_suitability", "")),
            "explanation": str(row.get("explanation", "")),
            "source": "SRO-01",
        }
        existing = {}
        if APPROVAL_FILE.exists():
            try:
                existing = load_json(APPROVAL_FILE)
            except Exception:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}
        existing[str(row["target"])] = payload
        APPROVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(APPROVAL_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        self.status.set(f"Approved {row['target']} — saved locally")
        messagebox.showinfo("Approved", f"Approved recommendation saved:\n{APPROVAL_FILE}")

    def refresh_all(self):
        self.draw_map()
        self.refresh_candidates()
        self.refresh_ga()
        self.refresh_approval_candidates()


def main():
    required = [CANDIDATES_FILE, HISTORY_FILE, REPORT_FILE, PROFILES_FILE, ASSIGNMENTS_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        messagebox.showerror("AERIS data missing", "Missing required files:\n\n" + "\n".join(missing))
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
    