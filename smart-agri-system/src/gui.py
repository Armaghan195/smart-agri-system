import os
import sys
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.join(os.path.dirname(__file__), '..')
MDL    = os.path.join(BASE, 'models')
RES    = os.path.join(BASE, 'results')

# ── Design tokens ──────────────────────────────────────────────────────────────
BG          = "#F7F8FC"
SIDEBAR_BG  = "#FFFFFF"
CARD_BG     = "#FFFFFF"
BORDER      = "#E4E6EF"
HDR_BG      = "#FFFFFF"

T1          = "#111827"
T2          = "#6B7280"
T3          = "#9CA3AF"

GREEN       = "#16A34A"
GREEN_LIGHT = "#DCFCE7"
GREEN_MID   = "#166534"
GREEN_PILL  = "#D1FAE5"

BLUE        = "#2563EB"
BLUE_LIGHT  = "#DBEAFE"
BLUE_MID    = "#1E40AF"
BLUE_PILL   = "#BFDBFE"

AMBER       = "#D97706"
AMBER_LIGHT = "#FEF3C7"
AMBER_MID   = "#92400E"
AMBER_PILL  = "#FDE68A"

FONT        = "Segoe UI"

ZONE_INFO = {
    0: ("Zone A", "Fertile & Balanced",
        "High nutrient content, optimal pH. Suitable for high-yield cash crops."),
    1: ("Zone B", "Moderate Fertility",
        "Adequate macro-nutrients. Good for cereal and legume cultivation."),
    2: ("Zone C", "Low Moisture",
        "Rainfall-deficient zone. Drought-tolerant varieties recommended."),
    3: ("Zone D", "Saline / High Potassium",
        "Elevated K and pH. Salt-tolerant crops advised; consider soil amendment."),
}


class AgriDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Agriculture Decision Support System")
        self.geometry("1300x820")
        self.minsize(1100, 720)
        self.configure(bg=BG)
        self._load_models()
        self._build_styles()
        self._build_layout()

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_models(self):
        try:
            self.clf     = joblib.load(os.path.join(MDL, 'decision_tree.pkl'))
            self.kmeans  = joblib.load(os.path.join(MDL, 'kmeans.pkl'))
            self.reg     = joblib.load(os.path.join(MDL, 'linear_regression.pkl'))
            self.scaler  = joblib.load(os.path.join(MDL, 'scaler.pkl'))
            self.encoder = joblib.load(os.path.join(MDL, 'label_encoder.pkl'))
            self.pca     = joblib.load(os.path.join(MDL, 'pca.pkl'))
        except FileNotFoundError as e:
            messagebox.showerror("Model Error",
                "Could not load model files.\nRun src/models.py first.\n\n" + str(e))
            sys.exit(1)

    # ── ttk styles ─────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure("TFrame",         background=BG)
        s.configure("Sidebar.TFrame", background=SIDEBAR_BG)
        s.configure("TNotebook",      background=BG, borderwidth=0, tabmargins=0)
        s.configure("TNotebook.Tab",  font=(FONT, 10), padding=(20, 10),
                    background=BG, foreground=T2, borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", CARD_BG)],
              foreground=[("selected", T1)])

    # ── Layout scaffold ────────────────────────────────────────────────────────
    def _build_layout(self):
        self._build_header()
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        self._build_main(body)
        self._build_statusbar()

    # ── Header ─────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=HDR_BG, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=HDR_BG)
        left.pack(side="left", padx=24, pady=12)

        dot = tk.Canvas(left, width=32, height=32, bg=HDR_BG, highlightthickness=0)
        dot.pack(side="left", padx=(0, 10))
        dot.create_oval(3, 3, 29, 29, fill=GREEN, outline="")
        dot.create_oval(10, 10, 22, 22, fill="#FFFFFF", outline="")

        tk.Label(left, text="AgriSense", font=(FONT, 16, "bold"),
                 bg=HDR_BG, fg=T1).pack(side="left")
        tk.Label(left, text="  Decision Support System", font=(FONT, 13),
                 bg=HDR_BG, fg=T2).pack(side="left")

        right = tk.Frame(hdr, bg=HDR_BG)
        right.pack(side="right", padx=24)
        pills = [("Decision Tree", GREEN, GREEN_PILL),
                 ("KMeans",        BLUE,  BLUE_PILL),
                 ("Linear Reg.",   AMBER, AMBER_PILL)]
        for txt, col, pill_bg in pills:
            pill = tk.Frame(right, bg=pill_bg, padx=10, pady=4)
            pill.pack(side="left", padx=4)
            tk.Label(pill, text=txt, font=(FONT, 8, "bold"),
                     bg=pill_bg, fg=col).pack()

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=SIDEBAR_BG, width=290)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Frame(sidebar, bg=BORDER, width=1).pack(side="right", fill="y")

        inner = tk.Frame(sidebar, bg=SIDEBAR_BG)
        inner.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(inner, text="Soil & Climate Parameters",
                 font=(FONT, 11, "bold"), bg=SIDEBAR_BG, fg=T1).pack(anchor="w")
        tk.Label(inner, text="Adjust sliders then click Analyze",
                 font=(FONT, 9), bg=SIDEBAR_BG, fg=T2).pack(anchor="w", pady=(2, 16))

        PARAMS = [
            ("N  — Nitrogen (kg/ha)",   "N",           0,   140, 50),
            ("P  — Phosphorus (kg/ha)", "P",           5,   145, 50),
            ("K  — Potassium (kg/ha)",  "K",           5,   205, 48),
            ("Temperature (C)",         "temperature", 8,    44, 25),
            ("Humidity (%)",            "humidity",   14,   100, 71),
            ("Soil pH",                 "ph",        3.5,  9.9, 6.5),
            ("Rainfall (mm)",           "rainfall",   20,  300, 103),
        ]

        self.slider_vars = {}
        for label, key, lo, hi, default in PARAMS:
            self._slider_row(inner, label, key, lo, hi, default)

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=18)

        tk.Button(inner, text="Analyze Farm Data",
                  font=(FONT, 11, "bold"),
                  bg=GREEN, fg="#FFFFFF",
                  activebackground=GREEN_MID, activeforeground="#FFFFFF",
                  bd=0, relief="flat", cursor="hand2", pady=12,
                  command=self._run_analysis).pack(fill="x")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=18)

        tk.Label(inner, text="Loaded Models", font=(FONT, 9, "bold"),
                 bg=SIDEBAR_BG, fg=T2).pack(anchor="w", pady=(0, 8))
        for name, score in [("Decision Tree", "Acc 98%"),
                             ("KMeans (k=4)", "Sil 0.25"),
                             ("Linear Reg.",  "R2 0.17")]:
            row = tk.Frame(inner, bg=SIDEBAR_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=name, font=(FONT, 9),
                     bg=SIDEBAR_BG, fg=T1).pack(side="left")
            tk.Label(row, text=score, font=(FONT, 9),
                     bg=SIDEBAR_BG, fg=T3).pack(side="right")

    def _slider_row(self, parent, label, key, lo, hi, default):
        frame = tk.Frame(parent, bg=SIDEBAR_BG)
        frame.pack(fill="x", pady=(0, 10))

        top = tk.Frame(frame, bg=SIDEBAR_BG)
        top.pack(fill="x")
        tk.Label(top, text=label, font=(FONT, 9),
                 bg=SIDEBAR_BG, fg=T1).pack(side="left")

        val_var = tk.StringVar(value=f"{default:.1f}")
        tk.Label(top, textvariable=val_var, font=(FONT, 9, "bold"),
                 bg=SIDEBAR_BG, fg=GREEN, width=6, anchor="e").pack(side="right")

        var = tk.DoubleVar(value=default)

        def on_change(v, vv=val_var):
            vv.set(f"{float(v):.1f}")

        tk.Scale(frame, from_=lo, to=hi, orient="horizontal",
                 variable=var, resolution=0.1,
                 bg=SIDEBAR_BG, fg=T1, troughcolor=BORDER,
                 highlightthickness=0, bd=0,
                 sliderrelief="flat", sliderlength=18,
                 activebackground=GREEN, showvalue=False,
                 command=on_change).pack(fill="x")

        self.slider_vars[key] = var

    # ── Main content ───────────────────────────────────────────────────────────
    def _build_main(self, parent):
        main = tk.Frame(parent, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)

        self._build_tab_results()
        self._build_tab_charts()

    # ── Tab 1: Results ─────────────────────────────────────────────────────────
    def _build_tab_results(self):
        self.tab_results = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.tab_results, text="  Analysis Results  ")

        self.placeholder = tk.Frame(self.tab_results, bg=BG)
        self.placeholder.pack(fill="both", expand=True)
        tk.Label(self.placeholder,
                 text="Adjust the parameters and click\n\"Analyze Farm Data\" to see results.",
                 font=(FONT, 14), bg=BG, fg=T3, justify="center").pack(expand=True)

        self.results_frame = tk.Frame(self.tab_results, bg=BG)

        # Three metric cards
        cards_row = tk.Frame(self.results_frame, bg=BG)
        cards_row.pack(fill="x", padx=24, pady=(24, 0))

        self.crop_card  = self._metric_card(cards_row, "Recommended Crop",  GREEN, 0)
        self.zone_card  = self._metric_card(cards_row, "Soil Zone",          BLUE,  1)
        self.yield_card = self._metric_card(cards_row, "Estimated Yield",    AMBER, 2)

        # Agronomic guidance
        desc_outer = tk.Frame(self.results_frame, bg=CARD_BG,
                              highlightbackground=BORDER, highlightthickness=1)
        desc_outer.pack(fill="x", padx=24, pady=16)
        desc_inner = tk.Frame(desc_outer, bg=CARD_BG)
        desc_inner.pack(fill="x", padx=20, pady=16)
        tk.Label(desc_inner, text="Agronomic Guidance",
                 font=(FONT, 11, "bold"), bg=CARD_BG, fg=T1).pack(anchor="w")
        self.guidance_var = tk.StringVar()
        tk.Label(desc_inner, textvariable=self.guidance_var,
                 font=(FONT, 10), bg=CARD_BG, fg=T2,
                 wraplength=800, justify="left").pack(anchor="w", pady=(6, 0))

        # Input echo
        echo_outer = tk.Frame(self.results_frame, bg=CARD_BG,
                              highlightbackground=BORDER, highlightthickness=1)
        echo_outer.pack(fill="x", padx=24, pady=(0, 24))
        echo_inner = tk.Frame(echo_outer, bg=CARD_BG)
        echo_inner.pack(fill="x", padx=20, pady=16)
        tk.Label(echo_inner, text="Input Parameters Used",
                 font=(FONT, 11, "bold"), bg=CARD_BG, fg=T1).pack(anchor="w", pady=(0, 10))
        self.echo_grid = tk.Frame(echo_inner, bg=CARD_BG)
        self.echo_grid.pack(fill="x")

    def _metric_card(self, parent, title, accent, col):
        pad_left = 0 if col == 0 else 10
        outer = tk.Frame(parent, bg=CARD_BG,
                         highlightbackground=BORDER, highlightthickness=1)
        outer.pack(side="left", fill="both", expand=True, padx=(pad_left, 0))

        tk.Frame(outer, bg=accent, height=4).pack(fill="x")

        inner = tk.Frame(outer, bg=CARD_BG)
        inner.pack(fill="both", padx=18, pady=16)

        tk.Label(inner, text=title, font=(FONT, 9),
                 bg=CARD_BG, fg=T2).pack(anchor="w")

        val_var = tk.StringVar(value="—")
        tk.Label(inner, textvariable=val_var,
                 font=(FONT, 20, "bold"), bg=CARD_BG, fg=accent).pack(anchor="w", pady=(4, 0))

        sub_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=sub_var,
                 font=(FONT, 9), bg=CARD_BG, fg=T3).pack(anchor="w")

        return val_var, sub_var

    # ── Tab 2: Charts ──────────────────────────────────────────────────────────
    def _build_tab_charts(self):
        tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(tab, text="  Visualizations  ")

        top_row = tk.Frame(tab, bg=BG)
        top_row.pack(fill="both", expand=True, pady=(0, 0))

        bot_row = tk.Frame(tab, bg=BG)
        bot_row.pack(fill="both", expand=True)

        charts_top = [
            ("feature_importance.png", "Feature Importance"),
            ("cluster_scatter.png",    "Soil Cluster Map"),
        ]
        charts_bot = [
            ("residual_plot.png", "Yield Residual Analysis"),
            ("elbow_plot.png",    "Elbow Method — Optimal k"),
        ]

        for fname, title in charts_top:
            self._chart_tile(top_row, fname, title)
        for fname, title in charts_bot:
            self._chart_tile(bot_row, fname, title)

    def _chart_tile(self, parent, fname, title):
        outer = tk.Frame(parent, bg=CARD_BG,
                         highlightbackground=BORDER, highlightthickness=1)
        outer.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        tk.Label(outer, text=title, font=(FONT, 10, "bold"),
                 bg=CARD_BG, fg=T1, anchor="w", pady=10,
                 padx=16).pack(fill="x")
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x")

        img_frame = tk.Frame(outer, bg=CARD_BG)
        img_frame.pack(fill="both", expand=True, padx=8, pady=8)

        path = os.path.join(RES, fname)
        if os.path.exists(path):
            lbl = tk.Label(img_frame, bg=CARD_BG)
            lbl.pack(fill="both", expand=True)
            img_frame.bind("<Configure>",
                           lambda e, p=path, l=lbl: self._resize_image(e, p, l))
        else:
            tk.Label(img_frame, text="Plot not found.\nRun models.py first.",
                     font=(FONT, 9), bg=CARD_BG, fg=T3).pack(expand=True)

    def _resize_image(self, event, path, label):
        w, h = max(event.width - 8, 1), max(event.height - 8, 1)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label.configure(image=photo)
            label.image = photo
        except Exception:
            pass

    # ── Status bar ─────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        bar = tk.Frame(self, bg=HDR_BG, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready  —  all models loaded successfully")
        tk.Label(bar, textvariable=self.status_var,
                 font=(FONT, 9), bg=HDR_BG, fg=T2).pack(side="left", padx=16)
        tk.Label(bar, text="Bahria University  |  BSE-6  |  AI Lab OEL",
                 font=(FONT, 9), bg=HDR_BG, fg=T3).pack(side="right", padx=16)

    # ── Analysis logic ─────────────────────────────────────────────────────────
    def _run_analysis(self):
        keys   = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        values = np.array([[self.slider_vars[k].get() for k in keys]])
        scaled = self.scaler.transform(values)

        crop_idx  = self.clf.predict(scaled)[0]
        crop_name = self.encoder.inverse_transform([crop_idx])[0].capitalize()

        zone_idx  = int(self.kmeans.predict(scaled)[0])
        zone_code, zone_name, zone_desc = ZONE_INFO[zone_idx]

        yield_val = float(self.reg.predict(scaled)[0])
        conf_lo, conf_hi = yield_val * 0.92, yield_val * 1.08

        # Show results
        self.placeholder.pack_forget()
        self.results_frame.pack(fill="both", expand=True)

        self.crop_card[0].set(crop_name)
        self.crop_card[1].set("Recommended crop")

        self.zone_card[0].set(zone_code)
        self.zone_card[1].set(zone_name)

        self.yield_card[0].set(f"{yield_val:,.0f} kg/ha")
        self.yield_card[1].set(f"Range: {conf_lo:,.0f} - {conf_hi:,.0f}")

        self.guidance_var.set(
            f"{zone_code} ({zone_name}): {zone_desc}   "
            f"Predicted best crop: {crop_name}.  "
            f"Expected yield {yield_val:,.0f} kg/ha "
            f"(confidence band {conf_lo:,.0f}-{conf_hi:,.0f} kg/ha)."
        )

        for w in self.echo_grid.winfo_children():
            w.destroy()

        labels = ["N (kg/ha)", "P (kg/ha)", "K (kg/ha)",
                  "Temp (C)", "Humidity (%)", "pH", "Rainfall (mm)"]
        for i, (lbl, val) in enumerate(zip(labels, values[0])):
            cell = tk.Frame(self.echo_grid, bg=BG, padx=12, pady=8)
            cell.grid(row=i // 4, column=i % 4, sticky="nsew", padx=4, pady=4)
            self.echo_grid.columnconfigure(i % 4, weight=1)
            tk.Label(cell, text=lbl, font=(FONT, 8),
                     bg=BG, fg=T2).pack(anchor="w")
            tk.Label(cell, text=f"{val:.1f}", font=(FONT, 11, "bold"),
                     bg=BG, fg=T1).pack(anchor="w")

        self.status_var.set(
            f"Analysis complete  |  Crop: {crop_name}  |  "
            f"Zone: {zone_code}  |  Yield: {yield_val:,.0f} kg/ha"
        )


if __name__ == '__main__':
    app = AgriDashboard()
    app.mainloop()
