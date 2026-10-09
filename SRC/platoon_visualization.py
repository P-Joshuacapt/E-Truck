"""
E-TRUCK / UGSR
Platoon Visualization V0.3  (visual prototype)

A dark, high-contrast dashboard for a truck platoon:

  1. OVERVIEW  - top-down, true-scale view of the whole platoon on a
                 textured two-lane road, with colour-coded bumper-to-bumper
                 gap dimensions and a ruler in metres.
  2. DETAIL    - zoomed view of the gap with the largest deviation: cab,
                 mirrors, lights, target-gap marker and a correction arrow.
  3. GAP CHART - gap error against the desired gap, with OK / WARN bands.
  4. KPI cards - platoon length, mean gap, worst deviation, gap health.

This is a visualization prototype, not a validated vehicle-dynamics or
communications simulation. All numbers are illustrative.

Usage
-----
    python platoon_v03.py
    python platoon_v03.py --gaps 10 10 14 10 --desired 10
    python platoon_v03.py --save platoon_v03.png --dpi 220 --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ----------------------------------------------------------------------
# Configuration (illustrative assumptions, metres)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    gaps: tuple = (10.0, 10.0, 14.0, 10.0)   # gap[i]: truck i -> truck i+1
    desired_gap: float = 10.0
    tol_ok: float = 2.0        # |error| <= tol_ok   -> OK   (green)
    tol_warn: float = 5.0      # |error| <= tol_warn -> WARN (amber), else OFF
    truck_length: float = 16.5
    truck_width: float = 2.55
    leader_front: float = 120.0


# Road geometry (two lanes, same direction, +x is forward)
LANE_W = 3.75
ROAD_Y0, ROAD_Y1 = -LANE_W, LANE_W
ROAD_H = ROAD_Y1 - ROAD_Y0
TRUCK_LANE_Y = LANE_W / 2          # trucks drive in the upper lane

# Truck anatomy (cab-over tractor + semi-trailer, metres)
CAB_LEN = 2.7
CAB_W = 2.5
COUPLING = 0.55                    # visible chassis between cab and trailer
TRAILER_NOSE = CAB_LEN + COUPLING

# Visible y-range of the two scenes (metres)
OVERVIEW_Y = (-11.0, 7.4)
DETAIL_Y = (-9.0, 6.4)

TH = {
    "fig": "#0a0e13", "panel": "#0f141b", "panel_edge": "#232b36",
    "grid": "#18202a", "text": "#e6edf3", "muted": "#8b949e",
    "faint": "#596574", "outline": "#05070a",
    "asphalt": (0.168, 0.188, 0.220), "curb": "#3b424d", "paint": "#e8ecf1",
    "leader": "#ff9f1c", "follower": "#3b9cff",
    "trailer": "#cfd8e3", "trailer_roof": "#e9eef4", "rib": "#aeb9c7",
    "glass": "#0f1b28", "tyre": "#07090c", "light": "#fff2b3",
    "tail": "#ff3b3b",
    "ok": "#3ddc84", "warn": "#ffb020", "off": "#ff5252",
}


@dataclass
class Truck:
    index: int
    front: float   # front bumper, road coordinate (m)
    rear: float    # rear bumper, road coordinate (m)

    @property
    def name(self) -> str:
        return f"TRUCK {self.index + 1}"

    @property
    def role(self) -> str:
        return "LEADER" if self.index == 0 else f"FOLLOWER {self.index}"

    @property
    def length(self) -> float:
        return self.front - self.rear


def build_platoon(cfg: Config) -> list:
    """Place trucks front-to-back using the bumper-to-bumper gaps."""
    trucks, front = [], cfg.leader_front
    for i in range(len(cfg.gaps) + 1):
        rear = front - cfg.truck_length
        trucks.append(Truck(i, front, rear))
        if i < len(cfg.gaps):
            front = rear - cfg.gaps[i]
    return trucks


def gap_status(gap: float, cfg: Config):
    """Return (status key, colour) for a gap value."""
    err = abs(gap - cfg.desired_gap)
    if err <= cfg.tol_ok:
        return "ok", TH["ok"]
    if err <= cfg.tol_warn:
        return "warn", TH["warn"]
    return "off", TH["off"]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def blend(colour, other="white", amount=0.2):
    a, b = np.array(to_rgb(colour)), np.array(to_rgb(other))
    return tuple(a * (1 - amount) + b * amount)


def _smooth(a, k):
    """Cheap separable box blur (no SciPy needed)."""
    k = int(max(k, 1))
    if k == 1:
        return a
    ker = np.ones(k) / k
    a = np.apply_along_axis(np.convolve, 1, a, ker, "same")
    return np.apply_along_axis(np.convolve, 0, a, ker, "same")


# ----------------------------------------------------------------------
# Road
# ----------------------------------------------------------------------


def draw_road(ax, x0, x1, ppm, seed=7):
    """Textured asphalt, curbs, edge lines and dashed lane divider."""
    rng = np.random.default_rng(seed)
    nx = max(int((x1 - x0) * ppm), 2)
    ny = max(int(ROAD_H * ppm), 2)

    fine = rng.normal(size=(ny, nx))
    blot = _smooth(rng.normal(size=(ny, nx)), int(ppm * 0.6))
    blot /= blot.std() + 1e-9
    tex = 0.012 * fine + 0.010 * blot

    # Tyre-wear stripes (darker wheel paths), strongest in the truck lane
    ys = np.linspace(ROAD_Y0, ROAD_Y1, ny)
    wear = np.zeros(ny)
    for c in (TRUCK_LANE_Y - 0.95, TRUCK_LANE_Y + 0.95):
        wear += np.exp(-0.5 * ((ys - c) / 0.32) ** 2)
    for c in (-TRUCK_LANE_Y - 0.95, -TRUCK_LANE_Y + 0.95):
        wear += 0.4 * np.exp(-0.5 * ((ys - c) / 0.32) ** 2)
    tex = tex - 0.022 * wear[:, None]

    img = np.clip(np.array(TH["asphalt"])[None, None, :] + tex[..., None], 0, 1)
    ax.imshow(img, extent=(x0, x1, ROAD_Y0, ROAD_Y1), origin="lower",
              aspect="auto", interpolation="bilinear", zorder=1)

    for y in (ROAD_Y1, ROAD_Y0 - 0.35):                       # curbs
        ax.add_patch(Rectangle((x0, y), x1 - x0, 0.35, fc=TH["curb"],
                               ec="none", zorder=1.2))
    for y in (ROAD_Y1 - 0.45, ROAD_Y0 + 0.25):                # edge lines
        ax.add_patch(Rectangle((x0, y), x1 - x0, 0.20, fc=TH["paint"],
                               ec="none", alpha=0.85, zorder=1.5))
    for x in np.arange(np.floor(x0 / 12) * 12, x1, 12):       # lane dashes
        ax.add_patch(Rectangle((x, -0.08), 3.0, 0.16, fc=TH["paint"],
                               ec="none", alpha=0.8, zorder=1.5))


# ----------------------------------------------------------------------
# Truck
# ----------------------------------------------------------------------


def draw_truck(ax, t: Truck, cfg: Config, detail=False):
    """Top-down cab-over tractor with a semi-trailer, drawn at true scale."""
    W, yc = cfg.truck_width, TRUCK_LANE_Y
    accent = TH["leader"] if t.index == 0 else TH["follower"]
    edge = TH["outline"]
    lw = 1.4 if detail else 0.8
    nose = t.front - TRAILER_NOSE            # trailer front edge

    # Soft drop shadow
    ax.add_patch(FancyBboxPatch(
        (t.rear + 0.18, yc - W / 2 - 0.20), t.length, W,
        boxstyle="round,pad=0,rounding_size=0.4",
        fc="black", ec="none", alpha=0.40, zorder=2.5))

    # Headlight beams (leader only, kept inside its lane/road)
    if t.index == 0:
        for s in (-1, 1):
            hy = yc + s * 0.85
            for length, alpha in ((20, 0.045), (13, 0.06), (6.5, 0.08)):
                ax.add_patch(Polygon(
                    [(t.front, hy + s * 0.2),
                     (t.front + length, hy + s * (0.2 + 0.03 * length)),
                     (t.front + length, hy - s * (0.2 + 0.10 * length)),
                     (t.front, hy - s * 0.2)],
                    closed=True, fc=TH["light"], ec="none",
                    alpha=alpha, zorder=2.3))

    # Tail-light glow
    for s in (-1, 1):
        for r, a in ((0.75, 0.09), (0.38, 0.16)):
            ax.add_patch(Circle((t.rear - 0.1, yc + s * (W / 2 - 0.4)), r,
                                fc=TH["tail"], ec="none", alpha=a, zorder=2.6))

    # Tyres peeking out of the body sides
    axles = [t.front - 1.1, t.front - 4.4, t.front - 5.7,
             t.rear + 3.1, t.rear + 1.8]
    for axle_x in axles:
        for s in (-1, 1):
            ax.add_patch(Rectangle(
                (axle_x - 0.5, yc + s * (W / 2 - 0.12) - 0.17), 1.0, 0.34,
                fc=TH["tyre"], ec="#2a313a", lw=0.5, zorder=3))

    # Tractor chassis (visible in the coupling gap)
    ax.add_patch(Rectangle((nose - 0.4, yc - 0.95), TRAILER_NOSE - CAB_LEN + 0.6,
                           1.9, fc="#39414c", ec=edge, lw=lw * 0.6, zorder=3.2))

    # Trailer: body, roof panel, ribs, nose bulkhead
    trailer_len = nose - t.rear
    ax.add_patch(FancyBboxPatch(
        (t.rear, yc - W / 2), trailer_len, W,
        boxstyle="round,pad=0,rounding_size=0.25",
        fc=TH["trailer"], ec=edge, lw=lw, zorder=3.5))
    ax.add_patch(FancyBboxPatch(
        (t.rear + 0.18, yc - W / 2 + 0.18), trailer_len - 0.36, W - 0.36,
        boxstyle="round,pad=0,rounding_size=0.18",
        fc=TH["trailer_roof"], ec="none", zorder=3.6))
    step = 1.2 if detail else 2.4
    for x in np.arange(t.rear + 1.2, nose - 0.6, step):
        ax.plot([x, x], [yc - W / 2 + 0.3, yc + W / 2 - 0.3],
                color=TH["rib"], lw=0.5 if detail else 0.35, zorder=3.7)
    ax.add_patch(Rectangle((nose - 0.28, yc - W / 2 + 0.1), 0.28, W - 0.2,
                           fc="#8d99a8", ec="none", zorder=3.75))

    # Rear bumper bar and tail lights
    ax.add_patch(Rectangle((t.rear, yc - W / 2 + 0.2), 0.16, W - 0.4,
                           fc="#0b0e12", ec="none", zorder=3.8))
    for s in (-1, 1):
        ax.add_patch(Rectangle((t.rear, yc + s * (W / 2 - 0.4) - 0.21), 0.12,
                               0.42, fc=TH["tail"], ec="none", zorder=3.9))

    # Number badge on the trailer roof
    badge_x = t.rear + trailer_len * 0.5
    ax.add_patch(Circle((badge_x, yc), 0.85, fc=accent, ec="white",
                        lw=1.0 if detail else 0.6, zorder=3.95))
    ax.text(badge_x, yc, str(t.index + 1), ha="center", va="center",
            fontsize=12 if detail else 7, fontweight="bold",
            color="#05070a", zorder=4.0)

    # Cab: body, roof, windshield
    ax.add_patch(FancyBboxPatch(
        (t.front - CAB_LEN, yc - CAB_W / 2), CAB_LEN, CAB_W,
        boxstyle="round,pad=0,rounding_size=0.5",
        fc=accent, ec=edge, lw=lw, zorder=4))
    ax.add_patch(FancyBboxPatch(
        (t.front - CAB_LEN + 0.22, yc - CAB_W / 2 + 0.22),
        CAB_LEN - 1.25, CAB_W - 0.44,
        boxstyle="round,pad=0,rounding_size=0.3",
        fc=blend(accent, "white", 0.22), ec="none", zorder=4.1))
    ax.add_patch(Polygon(
        [(t.front - 0.30, yc - CAB_W / 2 + 0.35),
         (t.front - 0.30, yc + CAB_W / 2 - 0.35),
         (t.front - 1.00, yc + CAB_W / 2 - 0.20),
         (t.front - 1.00, yc - CAB_W / 2 + 0.20)],
        closed=True, fc=TH["glass"], ec="#7f93a8", lw=0.6, zorder=4.2))
    if detail:   # roof vent
        ax.add_patch(Rectangle((t.front - 2.1, yc - 0.35), 0.6, 0.7,
                               fc=blend(accent, "black", 0.35), ec="none",
                               zorder=4.15))

    # Front bumper, headlights, mirrors
    ax.add_patch(Rectangle((t.front - 0.20, yc - W / 2 + 0.15), 0.20, W - 0.3,
                           fc="#0b0e12", ec="none", zorder=4.3))
    for s in (-1, 1):
        ax.add_patch(Rectangle((t.front - 0.34, yc + s * 0.85 - 0.2), 0.30,
                               0.40, fc=TH["light"], ec="none", zorder=4.4))
        y0 = yc + CAB_W / 2 - 0.03 if s > 0 else yc - CAB_W / 2 + 0.03 - 0.4
        ax.add_patch(Rectangle((t.front - 1.7, y0), 0.25, 0.4,
                               fc="#0b0e12", ec="none", zorder=4.5))


def draw_label(ax, t: Truck, fs, lo=None, hi=None):
    """Role pill above the road, centred on the visible part of the truck."""
    a, b = t.rear, t.front
    if lo is not None:
        a, b = max(a, lo), min(b, hi)
        if b - a < 3.0:
            return
    accent = TH["leader"] if t.index == 0 else TH["follower"]
    ax.text((a + b) / 2, ROAD_Y1 + 1.2, f"{t.name}  ·  {t.role}",
            ha="center", va="bottom", fontsize=fs, fontweight="bold",
            color=TH["text"], zorder=8,
            bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.8",
                      fc=blend(accent, "black", 0.82), ec=accent, lw=0.9))


# ----------------------------------------------------------------------
# Gap annotation
# ----------------------------------------------------------------------


def annotate_gap(ax, leader: Truck, follower: Truck, gap, cfg: Config,
                 detail=False):
    """Dimension line, shaded gap and value pill for one gap."""
    left, right = follower.front, leader.rear
    _, colour = gap_status(gap, cfg)
    err = gap - cfg.desired_gap
    W, yc = cfg.truck_width, TRUCK_LANE_Y

    if detail:
        y_dim, val_fs, sub_fs, ms, lw = ROAD_Y0 - 1.9, 15, 9, 14, 2.0
        val_dy, sub_dy = 0.65, 2.05
    else:
        y_dim, val_fs, sub_fs, ms, lw = ROAD_Y0 - 2.4, 9.5, 7.5, 9, 1.4
        val_dy, sub_dy = 0.75, 3.1

    # Shaded gap region in the truck lane
    ax.add_patch(Rectangle((left, yc - LANE_W / 2), gap, LANE_W, fc=colour,
                           ec="none", alpha=0.15, zorder=2.2))

    # Extension lines from the bumpers down to the dimension line
    for x in (left, right):
        ax.plot([x, x], [yc - W / 2 - 0.1, y_dim - 0.35], color=colour,
                lw=0.8, ls=(0, (2, 2)), alpha=0.65, zorder=6)
    ax.annotate("", xy=(left, y_dim), xytext=(right, y_dim), zorder=7,
                arrowprops=dict(arrowstyle="<|-|>", color=colour, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=ms))

    mid = (left + right) / 2
    ax.text(mid, y_dim - val_dy, f"{gap:.1f} m", ha="center", va="top",
            fontsize=val_fs, fontweight="bold", color=colour, zorder=8,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.7",
                      fc=blend(colour, "black", 0.85), ec=colour, lw=0.9))
    delta = "on target" if abs(err) < 0.05 else f"Δ {err:+.1f} m"
    sub = (f"target {cfg.desired_gap:.1f} m  ·  {delta}" if detail
           else delta)
    ax.text(mid, y_dim - sub_dy, sub, ha="center", va="top",
            fontsize=sub_fs, color=TH["muted"], zorder=8)

    if not detail:
        return

    # Target-gap marker and correction arrow (detail view only)
    x_t = right - cfg.desired_gap
    ax.plot([x_t, x_t], [yc - W / 2 - 0.5, ROAD_Y1 + 0.4],
            color="white", lw=1.3, ls=(0, (4, 3)), zorder=6)
    ax.text(x_t, ROAD_Y1 + 0.5, "TARGET", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="white", zorder=8)
    if abs(err) > 0.5:
        ax.annotate("", xy=(x_t, yc), xytext=(left, yc), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2,
                                    shrinkA=0, shrinkB=0, mutation_scale=16))
        verb = "close" if err > 0 else "open"
        ax.text((left + x_t) / 2, yc + 0.4, f"{verb} {abs(err):.1f} m",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=colour, zorder=8)


# ----------------------------------------------------------------------
# Chart, cards, panels
# ----------------------------------------------------------------------


def draw_chart(ax, gaps, cfg: Config):
    errs = np.array(gaps) - cfg.desired_gap
    n = len(gaps)
    lim = max(cfg.tol_warn * 1.7, np.abs(errs).max() * 1.5)

    ax.set_facecolor("none")
    ax.axhspan(-cfg.tol_warn, cfg.tol_warn, fc=TH["warn"], ec="none",
               alpha=0.07, zorder=0)
    ax.axhspan(-cfg.tol_ok, cfg.tol_ok, fc=TH["ok"], ec="none",
               alpha=0.13, zorder=0)
    ax.axhline(0, color=TH["text"], lw=1, ls=(0, (4, 3)), alpha=0.6, zorder=1)

    for i, (g, e) in enumerate(zip(gaps, errs)):
        _, c = gap_status(g, cfg)
        ax.plot([i, i], [0, e], color=c, lw=5, solid_capstyle="round", zorder=2)
        ax.scatter([i], [e], s=170, color=c, edgecolor=TH["panel"], lw=2.5,
                   zorder=3)
        ax.annotate(f"{g:.1f} m", (i, e), xytext=(0, 16 if e >= 0 else -16),
                    textcoords="offset points", ha="center",
                    va="bottom" if e >= 0 else "top", fontsize=10,
                    fontweight="bold", color=c)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-lim, lim)
    ticks = sorted({-cfg.tol_warn, -cfg.tol_ok, 0.0, cfg.tol_ok, cfg.tol_warn})
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "target" if abs(v) < 1e-9 else f"{v:+g}"))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"GAP {i + 1}\nT{i + 1} → T{i + 2}" for i in range(n)])
    ax.set_ylabel("Δ gap vs target (m)", fontsize=8, color=TH["muted"])
    ax.grid(axis="y", color=TH["grid"], lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TH["faint"])
    ax.tick_params(colors=TH["muted"], labelsize=8, length=3, color=TH["faint"])


def kpi_card(ov, x, y, w, h, label, value, colour):
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.09",
                 fc=TH["panel"], ec=TH["panel_edge"], lw=1))
    ov.add_patch(FancyBboxPatch((x + 0.02, y + 0.1), 0.05, h - 0.2,
                 boxstyle="round,pad=0,rounding_size=0.02",
                 fc=colour, ec="none"))
    ov.text(x + 0.2, y + h - 0.17, label, fontsize=7, color=TH["muted"],
            fontweight="bold", va="center")
    ov.text(x + 0.2, y + 0.2, value, fontsize=13, color=TH["text"],
            fontweight="bold", va="center")


def panel(ov, x, y, w, h, title):
    ov.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.12",
                 fc=TH["panel"], ec=TH["panel_edge"], lw=1, zorder=0))
    ov.text(x + 0.2, y + h - 0.27, title, fontsize=8, color=TH["muted"],
            fontweight="bold", va="center")


def style_scene_axis(ax, major, minor):
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TH["faint"])
    ax.tick_params(axis="x", colors=TH["muted"], labelsize=8, length=4,
                   color=TH["faint"])
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(axis="x", which="minor", length=2, color=TH["faint"])
    ax.grid(axis="x", color=TH["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Longitudinal position along the road (m)", fontsize=8,
                  color=TH["muted"], labelpad=3)


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------


def render(cfg: Config):
    trucks = build_platoon(cfg)
    gaps = list(cfg.gaps)
    errs = [g - cfg.desired_gap for g in gaps]
    worst = int(np.argmax(np.abs(errs)))
    states = [gap_status(g, cfg)[0] for g in gaps]

    # --- scene extents -------------------------------------------------
    x_min = np.floor((trucks[-1].rear - 12) / 10) * 10
    x_max = np.ceil((trucks[0].front + 22) / 10) * 10

    # --- layout in inches ---------------------------------------------
    FIG_W, M, PAD = 16.0, 0.45, 0.20
    CW = FIG_W - 2 * M
    scene_w = CW - 2 * PAD
    scene_h = scene_w * (OVERVIEW_Y[1] - OVERVIEW_Y[0]) / (x_max - x_min)
    scene_ph = 0.45 + scene_h + 0.55

    zoom_pw, zoom_h = 9.3, 3.3
    zoom_w = zoom_pw - 2 * PAD
    row2_h = 0.45 + zoom_h + 0.55
    FIG_H = 0.4 + 1.0 + 0.25 + scene_ph + 0.25 + row2_h + 0.8

    scene_top = FIG_H - 0.4 - 1.0 - 0.25
    scene_bot = scene_top - scene_ph
    row2_top = scene_bot - 0.25
    row2_bot = row2_top - row2_h

    def rect(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=TH["fig"])
    ov = fig.add_axes([0, 0, 1, 1], zorder=0)
    ov.set_xlim(0, FIG_W)
    ov.set_ylim(0, FIG_H)
    ov.axis("off")

    # --- header --------------------------------------------------------
    ov.text(M, FIG_H - 0.47, "E-TRUCK  ·  UGSR", fontsize=9,
            color=TH["leader"], fontweight="bold", va="top")
    ov.text(M, FIG_H - 0.70, "Platoon Visualization", fontsize=24,
            color=TH["text"], fontweight="bold", va="top")
    ov.text(M, FIG_H - 1.20,
            f"Top-down  ·  true scale  ·  {len(trucks)} trucks  ·  "
            f"gaps vs {cfg.desired_gap:g} m target  ·  V0.3",
            fontsize=9.5, color=TH["muted"], va="top")

    n_ok, n_warn, n_off = (states.count(k) for k in ("ok", "warn", "off"))
    health = " · ".join(p for p in (f"{n_ok} OK" if n_ok else "",
                                    f"{n_warn} WARN" if n_warn else "",
                                    f"{n_off} OFF" if n_off else "") if p)
    worst_col = gap_status(gaps[worst], cfg)[1]
    health_col = TH["off"] if n_off else TH["warn"] if n_warn else TH["ok"]
    platoon_len = trucks[0].front - trucks[-1].rear
    cards = [
        ("TRUCKS", f"{len(trucks)}", TH["follower"]),
        ("PLATOON LENGTH", f"{platoon_len:.1f} m", TH["follower"]),
        ("MEAN GAP", f"{np.mean(gaps):.1f} m", TH["follower"]),
        ("WORST DEVIATION", f"{errs[worst]:+.1f} m", worst_col),
        ("GAP HEALTH", health, health_col),
    ]
    cw, ch, cg = 1.8, 0.65, 0.12
    cx = FIG_W - M - (len(cards) * cw + (len(cards) - 1) * cg)
    for label, value, colour in cards:
        kpi_card(ov, cx, FIG_H - 1.25, cw, ch, label, value, colour)
        cx += cw + cg

    # --- overview ------------------------------------------------------
    panel(ov, M, scene_bot, CW, scene_ph, "OVERVIEW  ·  TRUE SCALE")
    ov.text(M + 2.55, scene_top - 0.27, "dashed box = detail view below",
            fontsize=7.5, color=TH["faint"], va="center")
    ov.text(M + CW - PAD - 0.7, scene_top - 0.27, "DIRECTION OF TRAVEL",
            fontsize=7.5, color=TH["muted"], fontweight="bold",
            va="center", ha="right")
    ov.annotate("", xy=(M + CW - PAD, scene_top - 0.27),
                xytext=(M + CW - PAD - 0.6, scene_top - 0.27),
                arrowprops=dict(arrowstyle="-|>", color=TH["text"], lw=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=12))

    ax = fig.add_axes(rect(M + PAD, scene_bot + 0.55, scene_w, scene_h),
                      zorder=2)
    ax.set_facecolor(TH["panel"])
    draw_road(ax, x_min, x_max, ppm=14)
    for t in trucks:
        draw_truck(ax, t, cfg, detail=False)
        draw_label(ax, t, fs=7.5)
    for i, g in enumerate(gaps):
        annotate_gap(ax, trucks[i], trucks[i + 1], g, cfg, detail=False)

    # Detail window (centre of the worst gap)
    zoom_xr = zoom_w * (DETAIL_Y[1] - DETAIL_Y[0]) / zoom_h
    zc = (trucks[worst + 1].front + trucks[worst].rear) / 2
    zlo, zhi = zc - zoom_xr / 2, zc + zoom_xr / 2
    ax.add_patch(Rectangle((zlo, ROAD_Y0 - 0.6), zoom_xr, ROAD_H + 1.2,
                           fill=False, ec="#c9d1d9", lw=1.0,
                           ls=(0, (4, 3)), alpha=0.6, zorder=9))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*OVERVIEW_Y)
    ax.set_aspect("equal", adjustable="box")
    style_scene_axis(ax, 10, 5)

    # --- detail --------------------------------------------------------
    panel(ov, M, row2_bot, zoom_pw, row2_h,
          f"DETAIL VIEW  ·  GAP {worst + 1}  "
          f"(TRUCK {worst + 1} → TRUCK {worst + 2})  ·  "
          f"{'LARGEST DEVIATION' if max(map(abs, errs)) > 0 else 'FIRST GAP'}")
    az = fig.add_axes(rect(M + PAD, row2_bot + 0.55, zoom_w, zoom_h), zorder=2)
    az.set_facecolor(TH["panel"])
    draw_road(az, zlo - 1, zhi + 1, ppm=40)
    for t in trucks:
        if t.front > zlo - 5 and t.rear < zhi + 5:
            draw_truck(az, t, cfg, detail=True)
            draw_label(az, t, fs=9.5, lo=zlo, hi=zhi)
    annotate_gap(az, trucks[worst], trucks[worst + 1], gaps[worst], cfg,
                 detail=True)
    az.set_xlim(zlo, zhi)
    az.set_ylim(*DETAIL_Y)
    az.set_aspect("equal", adjustable="box")
    style_scene_axis(az, 5, 1)

    # --- gap chart -----------------------------------------------------
    cx0 = M + zoom_pw + 0.25
    cpw = CW - zoom_pw - 0.25
    panel(ov, cx0, row2_bot, cpw, row2_h, "GAP ERROR VS TARGET")
    ac = fig.add_axes(rect(cx0 + 0.85, row2_bot + 0.8, cpw - 0.85 - 0.3,
                           row2_h - 0.45 - 0.8 - 0.15), zorder=2)
    draw_chart(ac, gaps, cfg)

    # --- footer --------------------------------------------------------
    x = M
    for colour, label in (
            (TH["ok"], f"OK  ·  within ±{cfg.tol_ok:g} m of target"),
            (TH["warn"], f"WARN  ·  within ±{cfg.tol_warn:g} m"),
            (TH["off"], "OFF  ·  further off")):
        ov.plot([x], [0.52], "o", color=colour, ms=7, mec="none")
        ov.text(x + 0.14, 0.52, label, fontsize=8.5, color=TH["text"],
                va="center")
        x += 0.14 + len(label) * 0.056 + 0.3
    ov.text(FIG_W - M, 0.52,
            "Visualization prototype  ·  not a validated vehicle-dynamics or "
            "communications model  ·  all values illustrative",
            fontsize=8, color=TH["faint"], va="center", ha="right")

    ov.set_xlim(0, FIG_W)
    ov.set_ylim(0, FIG_H)
    return fig, trucks


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="E-TRUCK platoon visualization")
    p.add_argument("--gaps", type=float, nargs="+",
                   default=list(Config.gaps),
                   help="bumper-to-bumper gaps in metres, front to back")
    p.add_argument("--desired", type=float, default=Config.desired_gap)
    p.add_argument("--tol-ok", type=float, default=Config.tol_ok)
    p.add_argument("--tol-warn", type=float, default=Config.tol_warn)
    p.add_argument("--save", metavar="PNG", help="export a PNG")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-show", action="store_true",
                   help="do not open a window")
    return p.parse_args()


def main():
    args = parse_args()
    if any(g <= 0 for g in args.gaps):
        raise SystemExit("All gaps must be positive.")
    cfg = Config(gaps=tuple(args.gaps), desired_gap=args.desired,
                 tol_ok=args.tol_ok, tol_warn=args.tol_warn)

    fig, trucks = render(cfg)

    print("E-TRUCK Platoon Visualization V0.3")
    for t in trucks:
        print(f"  {t.name:<8} {t.role:<11} front {t.front:7.1f} m   "
              f"rear {t.rear:7.1f} m")
    print("  Gaps (m):", [round(g, 2) for g in cfg.gaps])

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, facecolor=fig.get_facecolor())
        print(f"  Saved {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
