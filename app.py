import math
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy import ndimage

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ----------------------------------------------------------------------------
# Field / image constants
# ----------------------------------------------------------------------------
IMG_H = 360
IMG_W = 640
PX_PER_M = 20.0
FIELD_W_M = IMG_W / PX_PER_M
FIELD_H_M = IMG_H / PX_PER_M

GRID_COLS = 8
GRID_ROWS = 6

SOIL_COLOR = np.array([125, 98, 72], dtype=float)
CROP_COLOR = np.array([45, 125, 55], dtype=float)

SPECIES_COLORS = {
    "broadleaf": np.array([190, 75, 110], dtype=float),
    "grass": np.array([205, 170, 60], dtype=float),
    "vine": np.array([150, 60, 160], dtype=float),
    "emerging": np.array([220, 110, 90], dtype=float),
}
SPECIES_NAMES = list(SPECIES_COLORS.keys())

Yg, Xg = np.mgrid[0:IMG_H, 0:IMG_W].astype(float)

SURVEY_DAY_PRESETS = {
    "Quick demo": [0, 5, 10, 16, 23, 31, 40],
    "Standard": [0, 4, 8, 13, 18, 24, 31, 39, 48, 58],
    "Detailed": [0, 3, 6, 10, 14, 19, 24, 30, 37, 44, 52, 61],
}

DEFAULT_CONFIG = {
    "seed": 42,
    "initial_clusters": 14,
    "survey_days": SURVEY_DAY_PRESETS["Standard"],
    "auto_treat": True,
    "treat_area_threshold": 0.45,
    "treatment_intensity": 0.78,
    "max_treatments_per_survey": 2,
    "forecast_days": 21,
    "min_area_px": 28,
    "match_tolerance_m": 0.55,
    "track_max_gap_days": 24,
    "persistence_days": 18,
    "removal_confirm_days": 8,
    "regrowth_horizon_days": 35,
    "new_zone_distance_m": 3.8,
    "lost_revive_days": 12,
    "hotspot_count": 6,
    "extra_treatments": [],
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def sigmoid(x):
    x = float(np.clip(x, -25, 25))
    return 1.0 / (1.0 + math.exp(-x))


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def mean_safe(values):
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isnan(fv):
            vals.append(fv)
    return float(np.mean(vals)) if vals else 0.0


def safe_binary_metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "n": 0,
        }

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "n": int(len(y_true)),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = None

    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    except Exception:
        cm = np.zeros((2, 2), dtype=int)
    out["confusion_matrix"] = cm.tolist()
    return out


# ----------------------------------------------------------------------------
# Synthetic weed cluster generation
# ----------------------------------------------------------------------------
def make_cluster(rng, uid, x_m, y_m, radius_m, species=None, born_day=0.0):
    if species is None:
        species = str(rng.choice(SPECIES_NAMES))

    n_plants = int(max(3, np.pi * radius_m * radius_m / 0.10))
    plant_offsets = []
    for _ in range(n_plants):
        u = rng.random()
        theta = rng.uniform(0, 2 * math.pi)
        rr = math.sqrt(u) * 0.85
        dx = rr * math.cos(theta)
        dy = rr * math.sin(theta)
        rad_factor = rng.uniform(0.22, 0.42)
        plant_offsets.append((float(dx), float(dy), float(rad_factor)))

    return {
        "uid": int(uid),
        "x_m": float(np.clip(x_m, 0.6, FIELD_W_M - 0.6)),
        "y_m": float(np.clip(y_m, 0.6, FIELD_H_M - 0.6)),
        "radius_m": float(radius_m),
        "max_radius_m": float(min(1.65, radius_m * rng.uniform(1.9, 3.2))),
        "growth_m_day": float(rng.uniform(0.008, 0.035)),
        "species": species,
        "plant_offsets": plant_offsets,
        "born_day": float(born_day),
        "alive": True,
        "dead": False,
        "killed_until": None,
        "regrow_prob": float(rng.uniform(0.22, 0.76)),
        "regrow_delay_days": float(rng.uniform(12.0, 35.0)),
        "susceptibility": float(rng.uniform(0.55, 0.96)),
        "regrowth_count": 0,
    }


def generate_initial_clusters(rng, n_clusters):
    clusters = []
    uid = 1
    n_hubs = max(2, n_clusters // 3)

    for _ in range(n_hubs):
        hx = rng.uniform(1.2, FIELD_W_M - 1.2)
        hy = rng.uniform(1.2, FIELD_H_M - 1.2)
        group_size = int(rng.integers(1, max(2, n_clusters // n_hubs + 1)))
        for _ in range(group_size):
            x = hx + rng.normal(0, 0.95)
            y = hy + rng.normal(0, 0.95)
            radius = rng.uniform(0.28, 0.88)
            clusters.append(make_cluster(rng, uid, x, y, radius, born_day=0.0))
            uid += 1
            if len(clusters) >= n_clusters:
                break
        if len(clusters) >= n_clusters:
            break

    while len(clusters) < n_clusters:
        x = rng.uniform(1.0, FIELD_W_M - 1.0)
        y = rng.uniform(1.0, FIELD_H_M - 1.0)
        radius = rng.uniform(0.25, 0.75)
        clusters.append(make_cluster(rng, uid, x, y, radius, born_day=0.0))
        uid += 1

    return clusters


def spawn_new_clusters(clusters, rng, day, count):
    if count <= 0:
        return

    alive_candidates = [c for c in clusters if c["alive"] and not c.get("dead", False)]
    next_uid = max([c["uid"] for c in clusters], default=0) + 1

    for _ in range(count):
        x = y = None
        for _attempt in range(30):
            if alive_candidates and rng.random() < 0.68:
                base = alive_candidates[int(rng.integers(0, len(alive_candidates)))]
                x = base["x_m"] + rng.normal(0, 1.25)
                y = base["y_m"] + rng.normal(0, 1.25)
            else:
                x = rng.uniform(1.0, FIELD_W_M - 1.0)
                y = rng.uniform(1.0, FIELD_H_M - 1.0)

            x = float(np.clip(x, 0.8, FIELD_W_M - 0.8))
            y = float(np.clip(y, 0.8, FIELD_H_M - 0.8))

            ok = True
            for c in clusters:
                if math.hypot(x - c["x_m"], y - c["y_m"]) < 1.8:
                    ok = False
                    break
            if ok:
                break

        radius = rng.uniform(0.18, 0.46)
        clusters.append(make_cluster(rng, next_uid, x, y, radius, born_day=day))
        next_uid += 1


# ----------------------------------------------------------------------------
# Image rendering and segmentation
# ----------------------------------------------------------------------------
def render_field(clusters, day, rng, config):
    img = np.tile(SOIL_COLOR, (IMG_H, IMG_W, 1)).astype(float)
    img += rng.normal(0, 3.2, img.shape)

    band_px = int(1.15 * PX_PER_M)
    crop_1d = ((np.arange(IMG_W) // band_px) % 2 == 0).astype(float)
    crop_band = np.repeat(crop_1d[None, :], IMG_H, axis=0)
    crop_float = ndimage.gaussian_filter(crop_band, sigma=1.8)

    img = img * (1.0 - crop_float[..., None]) + CROP_COLOR * crop_float[..., None]
    img += rng.normal(0, 2.2, img.shape) * crop_float[..., None]

    true_map = np.zeros((IMG_H, IMG_W), dtype=np.int32)
    alive_clusters = []

    for c in clusters:
        if c.get("dead", False) or not c["alive"]:
            continue
        alive_clusters.append(c)

        color = SPECIES_COLORS[c["species"]]
        cx_m = c["x_m"]
        cy_m = c["y_m"]
        rad_m = c["radius_m"]

        for dxn, dyn, rf in c["plant_offsets"]:
            cx = cx_m * PX_PER_M + dxn * rad_m * PX_PER_M
            cy = cy_m * PX_PER_M + dyn * rad_m * PX_PER_M
            r = max(1.2, rf * rad_m * PX_PER_M)

            x0 = max(0, int(cx - r - 2))
            x1 = min(IMG_W, int(cx + r + 3))
            y0 = max(0, int(cy - r - 2))
            y1 = min(IMG_H, int(cy + r + 3))
            if x1 <= x0 or y1 <= y0:
                continue

            yy = Yg[y0:y1, x0:x1]
            xx = Xg[y0:y1, x0:x1]
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)
            if not mask.any():
                continue

            n = int(mask.sum())
            patch = img[y0:y1, x0:x1]
            patch[mask] = color + rng.normal(0, 7.0, (n, 3))
            img[y0:y1, x0:x1] = patch

            tview = true_map[y0:y1, x0:x1]
            tview[mask] = c["uid"]

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, true_map, alive_clusters


def segment_image(img, true_map, config):
    imgf = img.astype(float)

    dist_soil = np.sqrt(np.sum((imgf - SOIL_COLOR) ** 2, axis=2))
    dist_crop = np.sqrt(np.sum((imgf - CROP_COLOR) ** 2, axis=2))

    weed_colors = np.vstack([SPECIES_COLORS[s] for s in SPECIES_NAMES])
    dist_weed = np.full((IMG_H, IMG_W), np.inf, dtype=float)
    weed_idx = np.zeros((IMG_H, IMG_W), dtype=int)

    for k, color in enumerate(weed_colors):
        d = np.sqrt(np.sum((imgf - color) ** 2, axis=2))
        update = d < dist_weed
        dist_weed[update] = d[update]
        weed_idx[update] = k

    weed_mask = (dist_weed < dist_soil) & (dist_weed < dist_crop)
    weed_mask = ndimage.binary_opening(weed_mask, iterations=1)
    weed_mask = ndimage.binary_closing(weed_mask, iterations=2)

    labels, n_comp = ndimage.label(weed_mask)
    detections = []
    if n_comp == 0:
        return detections

    idx_array = np.arange(1, n_comp + 1)
    areas = ndimage.sum(weed_mask, labels, index=idx_array)
    coms = ndimage.center_of_mass(weed_mask, labels, index=idx_array)
    objects = ndimage.find_objects(labels)

    for i, sl in enumerate(objects, start=1):
        area_px = int(areas[i - 1])
        if area_px < config["min_area_px"]:
            continue

        cy, cx = coms[i - 1]
        sub_lab = labels[sl] == i
        sub_true = true_map[sl]

        vals, counts = np.unique(sub_true[sub_lab], return_counts=True)
        valid = vals != 0
        if valid.any():
            vals_v = vals[valid]
            counts_v = counts[valid]
            gt_uid = int(vals_v[np.argmax(counts_v)])
            overlap_count = int(counts_v.max())
        else:
            gt_uid = 0
            overlap_count = 0

        margin = np.minimum(dist_soil[sl], dist_crop[sl]) - dist_weed[sl]
        if sub_lab.any():
            mean_margin = float(np.mean(margin[sub_lab]))
        else:
            mean_margin = 0.0
        confidence = clip01(mean_margin / 70.0)
        confidence = float(np.clip(confidence, 0.05, 0.99))

        idx_vals = weed_idx[sl][sub_lab]
        if len(idx_vals) > 0:
            species_guess = SPECIES_NAMES[int(np.bincount(idx_vals).argmax())]
        else:
            species_guess = "unknown"

        x_m = float(cx / PX_PER_M)
        y_m = float(cy / PX_PER_M)
        area_m2 = float(area_px / (PX_PER_M * PX_PER_M))
        radius_m = float(math.sqrt(area_m2 / math.pi))

        detections.append(
            {
                "x_m": x_m,
                "y_m": y_m,
                "area_m2": area_m2,
                "radius_m": radius_m,
                "area_px": area_px,
                "confidence": confidence,
                "gt_uid": gt_uid,
                "overlap_count": overlap_count,
                "species_guess": species_guess,
                "matched_track_id": None,
                "track_spatial_id": None,
                "track_status": "unmatched",
            }
        )

    return detections


# ----------------------------------------------------------------------------
# Lifecycle, treatment, tracking
# ----------------------------------------------------------------------------
def advance_clusters(clusters, prev_day, day, rng, config):
    for c in clusters:
        if c.get("dead", False):
            continue

        if (
            (not c["alive"])
            and c.get("killed_until") is not None
            and day >= c["killed_until"]
        ):
            c["alive"] = True
            c["radius_m"] = max(0.15, c["radius_m"] * 0.35)
            c["regrowth_count"] = int(c.get("regrowth_count", 0)) + 1

        if c["alive"] and prev_day is not None and day > prev_day and c["born_day"] < day:
            c["radius_m"] = min(
                c["max_radius_m"],
                c["radius_m"] + c["growth_m_day"] * (day - prev_day),
            )


def apply_treatment_to_clusters(clusters, x_m, y_m, radius_m, intensity, day, rng):
    affected = []
    for c in clusters:
        if c.get("dead", False):
            continue

        d = math.hypot(c["x_m"] - x_m, c["y_m"] - y_m)
        if d > radius_m + c["radius_m"]:
            continue

        affected.append(c["uid"])
        overshoot = max(0.0, d - radius_m)
        sigma = max(0.25, c["radius_m"])
        eff = float(intensity) * math.exp(-(overshoot * overshoot) / (2.0 * sigma * sigma))
        eff = float(np.clip(eff, 0.0, 1.0))

        if c["alive"]:
            kill_prob = eff * c["susceptibility"]
            if rng.random() < kill_prob:
                c["alive"] = False
                if rng.random() < c["regrow_prob"]:
                    c["killed_until"] = day + c["regrow_delay_days"] * rng.uniform(0.75, 1.35)
                else:
                    c["killed_until"] = None
                    c["dead"] = True
            else:
                c["radius_m"] = max(0.12, c["radius_m"] * (1.0 - 0.60 * eff))
        else:
            if c.get("killed_until") is not None:
                c["killed_until"] += eff * 6.0

    return affected


def heuristic_regrowth_risk(features):
    area_norm = clip01(features.get("pre_area_m2", 0.0) / 1.4)
    vel_norm = clip01(features.get("pre_velocity_m_day", 0.0) / 0.08)
    pressure_norm = clip01(features.get("local_pressure_m2", 0.0) / 3.0)
    intensity = clip01(features.get("treatment_intensity", 0.0))
    days_norm = clip01(features.get("days_since_treatment", 0.0) / 30.0)
    prior_norm = clip01(features.get("regrowth_count_prior", 0) / 2.0)

    z = (
        -1.75
        + 1.85 * area_norm
        + 1.05 * vel_norm
        + 1.55 * pressure_norm
        - 2.45 * intensity
        + 0.70 * days_norm
        + 0.90 * prior_norm
    )

    sp = features.get("species_guess", "unknown")
    if sp == "grass":
        z += 0.18
    elif sp == "vine":
        z += 0.36
    elif sp == "broadleaf":
        z += 0.10

    return float(sigmoid(z))


def create_regrowth_episode(tr, day, config, tracks):
    treated_day = tr.get("treated_day")
    if treated_day is None:
        treated_day = tr["last_seen_day"]

    pre_hist = None
    for h in tr["history"]:
        if h["day"] <= treated_day:
            pre_hist = h
    if pre_hist is None:
        pre_hist = tr["history"][-1]

    pre_area = float(pre_hist["area_m2"])
    pre_vel = float(tr.get("velocity_m_day", 0.0))
    intensity = (
        float(tr["treatments"][-1]["intensity"])
        if tr.get("treatments")
        else float(config["treatment_intensity"])
    )

    local_pressure = 0.0
    nearest_active = 1e9
    for tid2, tr2 in tracks.items():
        if tid2 == tr["track_id"]:
            continue
        if tr2["status"] not in ("active", "regrown", "treated"):
            continue
        d = math.hypot(tr2["x_m"] - tr["x_m"], tr2["y_m"] - tr["y_m"])
        if d <= 4.0:
            local_pressure += float(tr2["area_m2"])
        nearest_active = min(nearest_active, d)

    if nearest_active >= 1e8:
        nearest_active = 10.0

    days_since_treatment = max(0.0, day - treated_day)

    features = {
        "pre_area_m2": pre_area,
        "pre_velocity_m_day": pre_vel,
        "treatment_intensity": intensity,
        "local_pressure_m2": local_pressure,
        "nearest_active_m": nearest_active,
        "days_since_treatment": days_since_treatment,
        "regrowth_count_prior": int(tr.get("regrowth_count", 0)),
        "species_guess": tr.get("species_guess", "unknown"),
    }
    risk = heuristic_regrowth_risk(features)

    tr["pre_removal_area"] = pre_area
    tr["last_regrowth_risk"] = risk

    ep = {
        "track_id": tr["track_id"],
        "spatial_id": tr["spatial_id"],
        "removed_day": int(day),
        "treated_day": float(treated_day),
        "days_since_treatment": float(days_since_treatment),
        "pre_area_m2": float(pre_area),
        "pre_velocity_m_day": float(pre_vel),
        "treatment_intensity": float(intensity),
        "local_pressure_m2": float(local_pressure),
        "nearest_active_m": float(nearest_active),
        "regrowth_count_prior": int(tr.get("regrowth_count", 0)),
        "species_guess": tr.get("species_guess", "unknown"),
        "predicted_risk": float(risk),
        "outcome": None,
        "outcome_day": None,
        "days_to_regrowth": None,
    }
    return ep


def mark_tracks_treated(tracks, x_m, y_m, radius_m, intensity, day, affected_uids, exclude=None):
    exclude = exclude or set()
    treated_ids = []
    for tid, tr in tracks.items():
        if tid in exclude:
            continue
        if tr["status"] not in ("active", "regrown"):
            continue
        d = math.hypot(tr["x_m"] - x_m, tr["y_m"] - y_m)
        if d <= radius_m + tr["radius_m"] + 0.35:
            tr["status"] = "treated"
            tr["treated_day"] = float(day)
            tr["treatments"].append(
                {
                    "day": float(day),
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "radius_m": float(radius_m),
                    "intensity": float(intensity),
                    "affected_gt_uids": list(affected_uids),
                }
            )
            treated_ids.append(tid)
    return treated_ids


def match_and_update(
    tracks,
    cell_counters,
    detections,
    day,
    config,
    episodes,
    new_zone_points,
    prev_gt_uids,
):
    pairs = []

    for di, det in enumerate(detections):
        for tid, tr in tracks.items():
            gap = day - tr["last_seen_day"]

            if tr["status"] == "lost":
                if gap > config["lost_revive_days"]:
                    continue
            elif tr["status"] == "removed":
                if gap > config["regrowth_horizon_days"] + 15:
                    continue
            elif tr["status"] == "archived":
                continue
            else:
                if gap > config["track_max_gap_days"]:
                    continue

            dist = math.hypot(det["x_m"] - tr["x_m"], det["y_m"] - tr["y_m"])
            thresh = float(config["match_tolerance_m"]) + max(det["radius_m"], tr["radius_m"])
            if tr["status"] == "removed":
                thresh += 0.45

            if dist <= thresh:
                pairs.append((dist, di, tid))

    pairs.sort(key=lambda t: t[0])

    assigned_det = {}
    assigned_track = {}
    for dist, di, tid in pairs:
        if di in assigned_det or tid in assigned_track:
            continue
        assigned_det[di] = tid
        assigned_track[tid] = di

    new_tracks = []
    removed_tracks = []
    regrown_tracks = []
    new_zones_today = []

    # Matched detections update existing tracks
    for di, tid in assigned_det.items():
        det = detections[di]
        tr = tracks[tid]
        prev = tr["history"][-1]

        dt = max(float(day - prev["day"]), 1e-6)
        cd = math.hypot(det["x_m"] - prev["x_m"], det["y_m"] - prev["y_m"])
        centroid_vel = cd / dt
        radial_vel = (det["radius_m"] - prev["radius_m"]) / dt

        tr["centroid_velocity_m_day"] = float(centroid_vel)
        tr["radial_velocity_m_day"] = float(radial_vel)
        tr["velocity_m_day"] = float(max(0.0, radial_vel) + 0.35 * centroid_vel)

        hist_entry = {
            "day": float(day),
            "x_m": float(det["x_m"]),
            "y_m": float(det["y_m"]),
            "area_m2": float(det["area_m2"]),
            "radius_m": float(det["radius_m"]),
            "confidence": float(det["confidence"]),
            "gt_uid": int(det["gt_uid"]),
            "species_guess": det["species_guess"],
        }
        tr["history"].append(hist_entry)

        tr["last_seen_day"] = float(day)
        tr["x_m"] = float(det["x_m"])
        tr["y_m"] = float(det["y_m"])
        tr["area_m2"] = float(det["area_m2"])
        tr["radius_m"] = float(det["radius_m"])
        tr["confidence"] = float(det["confidence"])

        if det["gt_uid"] and (
            not tr.get("dominant_gt_uid")
            or det["overlap_count"] > tr.get("dominant_gt_overlap", 0)
        ):
            tr["dominant_gt_uid"] = int(det["gt_uid"])
            tr["dominant_gt_overlap"] = int(det["overlap_count"])
            tr["species_guess"] = det["species_guess"]

        old_status = tr["status"]
        if old_status == "removed":
            tr["status"] = "regrown"
            tr["regrowth_count"] = int(tr.get("regrowth_count", 0)) + 1
            tr["last_regrowth_day"] = float(day)
            regrown_tracks.append(tid)

            for ep in reversed(episodes):
                if (
                    ep["track_id"] == tid
                    and ep["outcome"] is None
                    and day - ep["removed_day"] <= config["regrowth_horizon_days"]
                ):
                    ep["outcome"] = 1
                    ep["outcome_day"] = float(day)
                    ep["days_to_regrowth"] = float(day - ep["removed_day"])
                    break

        elif old_status == "treated":
            tr["status"] = "active"
        else:
            tr["status"] = "active"

        det["matched_track_id"] = tid
        det["track_spatial_id"] = tr["spatial_id"]
        det["track_status"] = tr["status"]

    # Unmatched tracks: confirm removal / lose persistence
    for tid, tr in tracks.items():
        if tid in assigned_track:
            continue

        if tr["status"] == "treated":
            treated_day = tr.get("treated_day")
            if treated_day is None:
                treated_day = tr["last_seen_day"]

            if day - treated_day >= config["removal_confirm_days"]:
                tr["status"] = "removed"
                tr["removed_day"] = float(day)
                ep = create_regrowth_episode(tr, day, config, tracks)
                episodes.append(ep)
                removed_tracks.append(tid)

        elif tr["status"] in ("active", "regrown"):
            if day - tr["last_seen_day"] >= config["persistence_days"]:
                tr["status"] = "lost"

        elif tr["status"] == "removed":
            removed_day = tr.get("removed_day")
            if removed_day is None:
                removed_day = tr["last_seen_day"]
            if day - removed_day > config["regrowth_horizon_days"] + 25:
                tr["status"] = "archived"

    # Unmatched detections become new spatial IDs
    for di, det in enumerate(detections):
        if di in assigned_det:
            continue

        min_prior = 1e9
        for tr in tracks.values():
            for h in tr["history"]:
                if h["day"] >= day:
                    continue
                d = math.hypot(det["x_m"] - h["x_m"], det["y_m"] - h["y_m"])
                min_prior = min(min_prior, d)

        for nz in new_zone_points:
            d = math.hypot(det["x_m"] - nz["x_m"], det["y_m"] - nz["y_m"])
            min_prior = min(min_prior, d)

        is_new_zone = (
            min_prior > config["new_zone_distance_m"]
            or (
                det["gt_uid"]
                and det["gt_uid"] not in prev_gt_uids
                and min_prior > config["new_zone_distance_m"] * 0.65
            )
        )

        tid = f"W{len(tracks) + 1:03d}"

        col = int(np.clip(det["x_m"] // (FIELD_W_M / GRID_COLS), 0, GRID_COLS - 1))
        row = int(np.clip(det["y_m"] // (FIELD_H_M / GRID_ROWS), 0, GRID_ROWS - 1))
        cell_key = f"{chr(65 + col)}{row + 1}"
        cell_counters[cell_key] = cell_counters.get(cell_key, 0) + 1
        spatial_id = f"{cell_key}-{cell_counters[cell_key]:02d}"

        hist_entry = {
            "day": float(day),
            "x_m": float(det["x_m"]),
            "y_m": float(det["y_m"]),
            "area_m2": float(det["area_m2"]),
            "radius_m": float(det["radius_m"]),
            "confidence": float(det["confidence"]),
            "gt_uid": int(det["gt_uid"]),
            "species_guess": det["species_guess"],
        }

        tr = {
            "track_id": tid,
            "spatial_id": spatial_id,
            "first_seen_day": float(day),
            "last_seen_day": float(day),
            "status": "active",
            "x_m": float(det["x_m"]),
            "y_m": float(det["y_m"]),
            "area_m2": float(det["area_m2"]),
            "radius_m": float(det["radius_m"]),
            "confidence": float(det["confidence"]),
            "velocity_m_day": 0.0,
            "centroid_velocity_m_day": 0.0,
            "radial_velocity_m_day": 0.0,
            "regrowth_count": 0,
            "treated_day": None,
            "removed_day": None,
            "last_regrowth_day": None,
            "treatments": [],
            "history": [hist_entry],
            "dominant_gt_uid": int(det["gt_uid"]),
            "dominant_gt_overlap": int(det["overlap_count"]),
            "species_guess": det["species_guess"],
            "is_new_zone": bool(is_new_zone),
            "pre_removal_area": float(det["area_m2"]),
            "last_regrowth_risk": 0.0,
        }

        tracks[tid] = tr
        new_tracks.append(tid)

        det["matched_track_id"] = tid
        det["track_spatial_id"] = spatial_id
        det["track_status"] = "new"

        if is_new_zone:
            nz = {
                "day": float(day),
                "track_id": tid,
                "spatial_id": spatial_id,
                "x_m": float(det["x_m"]),
                "y_m": float(det["y_m"]),
                "area_m2": float(det["area_m2"]),
                "distance_from_prior_m": None if min_prior >= 1e8 else float(min_prior),
            }
            new_zones_today.append(nz)
            new_zone_points.append(nz)

    return tracks, detections, new_tracks, removed_tracks, regrown_tracks, new_zones_today


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def detection_metrics(true_alive_clusters, detections, config):
    true_ids = set(c["uid"] for c in true_alive_clusters)
    true_by_id = {c["uid"]: c for c in true_alive_clusters}

    dets_sorted = sorted(detections, key=lambda d: d["area_m2"], reverse=True)
    matched_true = set()
    tp = 0

    for det in dets_sorted:
        matched_uid = None

        if det["gt_uid"] in true_ids and det["gt_uid"] not in matched_true:
            matched_uid = det["gt_uid"]
        else:
            best = None
            best_d = 1e9
            for uid, c in true_by_id.items():
                if uid in matched_true:
                    continue
                d = math.hypot(det["x_m"] - c["x_m"], det["y_m"] - c["y_m"])
                if d <= c["radius_m"] + config["match_tolerance_m"] and d < best_d:
                    best = uid
                    best_d = d
            matched_uid = best

        if matched_uid is not None:
            tp += 1
            matched_true.add(matched_uid)

    fp = len(detections) - tp
    fn = len(true_ids) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0

    return {
        "true_clusters": len(true_ids),
        "detections": len(detections),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_uid_track_maps(detections):
    uid_best = {}
    uid_tracks = {}

    for det in detections:
        uid = det.get("gt_uid", 0)
        tid = det.get("matched_track_id")
        if not uid or not tid:
            continue

        uid_tracks.setdefault(uid, []).append(tid)

        if uid not in uid_best or det["overlap_count"] > uid_best[uid][1]:
            uid_best[uid] = (tid, det["overlap_count"])

    uid_to_track = {u: v[0] for u, v in uid_best.items()}
    return uid_to_track, uid_tracks


def tracking_metrics(prev_uid_to_track, curr_uid_to_track, curr_uid_tracks):
    common = set(prev_uid_to_track) & set(curr_uid_to_track)
    stable = sum(1 for u in common if prev_uid_to_track[u] == curr_uid_to_track[u])
    switches = len(common) - stable

    fragmentation = sum(1 for u, tids in curr_uid_tracks.items() if len(set(tids)) > 1)

    track_to_uids = {}
    for u, tids in curr_uid_tracks.items():
        for t in tids:
            track_to_uids.setdefault(t, set()).add(u)
    merge_errors = sum(1 for t, uids in track_to_uids.items() if len(uids) > 1)

    stability = stable / len(common) if common else float("nan")

    return {
        "common_true_clusters": len(common),
        "stable_ids": int(stable),
        "id_switches": int(switches),
        "id_stability": float(stability),
        "fragmentation_events": int(fragmentation),
        "merge_errors": int(merge_errors),
    }


# ----------------------------------------------------------------------------
# Heatmaps, hotspots, treatment coordinates
# ----------------------------------------------------------------------------
def add_kernel(grid, x_m, y_m, weight, sigma):
    if weight <= 0:
        return
    cell_w = FIELD_W_M / GRID_COLS
    cell_h = FIELD_H_M / GRID_ROWS
    for rr in range(GRID_ROWS):
        cy = (rr + 0.5) * cell_h
        for cc in range(GRID_COLS):
            cx = (cc + 0.5) * cell_w
            d = math.hypot(cx - x_m, cy - y_m)
            if d <= sigma * 2.5:
                grid[rr, cc] += weight * math.exp(-(d * d) / (2.0 * sigma * sigma))


def compute_pressure(tracks, day, config, new_zone_points):
    obs = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
    pred = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)

    cell_w = FIELD_W_M / GRID_COLS
    cell_h = FIELD_H_M / GRID_ROWS

    for tr in tracks.values():
        col = int(np.clip(tr["x_m"] / cell_w, 0, GRID_COLS - 1))
        row = int(np.clip(tr["y_m"] / cell_h, 0, GRID_ROWS - 1))

        if tr["status"] in ("active", "regrown", "treated"):
            recency = math.exp(-max(0.0, day - tr["last_seen_day"]) / 15.0)
            treat_penalty = 0.55 if tr["status"] == "treated" else 1.0
            w = (
                float(tr["area_m2"])
                * (1.0 + 8.0 * float(tr.get("velocity_m_day", 0.0)))
                * recency
                * treat_penalty
            )
            obs[row, col] += w

            sigma = max(
                0.65,
                float(tr["radius_m"])
                + float(tr.get("velocity_m_day", 0.0)) * float(config["forecast_days"])
                + 0.35,
            )
            add_kernel(pred, tr["x_m"], tr["y_m"], w, sigma)

        elif tr["status"] == "removed":
            risk = float(tr.get("last_regrowth_risk", 0.0))
            if risk > 0.05:
                removed_day = tr.get("removed_day")
                if removed_day is None:
                    removed_day = tr["last_seen_day"]
                age = max(0.0, day - float(removed_day))
                w = float(tr.get("pre_removal_area", 0.2)) * risk * math.exp(-age / 25.0)
                add_kernel(pred, tr["x_m"], tr["y_m"], w, 1.35)

    for nz in new_zone_points:
        age = max(0.0, day - float(nz["day"]))
        w = 0.12 * math.exp(-age / 40.0)
        add_kernel(pred, nz["x_m"], nz["y_m"], w, 1.1)

    max_obs = float(obs.max()) if obs.size else 0.0
    if max_obs > 0:
        obs = obs / max_obs

    max_pred = float(pred.max()) if pred.size else 0.0
    if max_pred > 0:
        pred = pred / max_pred

    hotspots = []
    if max_pred > 0:
        flat_order = np.argsort(pred.ravel())[::-1]
        for flat_idx in flat_order[: int(config["hotspot_count"])]:
            val = float(pred.flat[flat_idx])
            if val < 0.12:
                break
            rr, cc = np.unravel_index(flat_idx, pred.shape)
            hotspots.append(
                {
                    "row": int(rr),
                    "col": int(cc),
                    "cell": f"{chr(65 + cc)}{rr + 1}",
                    "x_m": float((cc + 0.5) * cell_w),
                    "y_m": float((rr + 0.5) * cell_h),
                    "predicted_pressure": val,
                    "observed_pressure": float(obs[rr, cc]),
                }
            )

    return obs, pred, hotspots


def generate_treatment_coordinates(tracks, predicted_grid, day, config):
    rows = []
    cell_w = FIELD_W_M / GRID_COLS
    cell_h = FIELD_H_M / GRID_ROWS

    for tr in tracks.values():
        if tr["status"] not in ("active", "regrown"):
            continue

        col = int(np.clip(tr["x_m"] / cell_w, 0, GRID_COLS - 1))
        row = int(np.clip(tr["y_m"] / cell_h, 0, GRID_ROWS - 1))
        pressure = float(predicted_grid[row, col]) if predicted_grid.size else 0.0

        priority = (
            float(tr["area_m2"])
            * (1.0 + 8.0 * float(tr.get("velocity_m_day", 0.0)))
            * (1.0 + 0.45 * float(tr.get("regrowth_count", 0)))
            * (1.0 + 0.55 * pressure)
        )

        rec_radius = max(
            0.45,
            float(tr["radius_m"]) * 1.35 + 0.20 + 0.10 * float(tr.get("regrowth_count", 0)),
        )
        rec_intensity = float(
            np.clip(
                0.55
                + 0.25 * clip01(float(tr.get("velocity_m_day", 0.0)) / 0.08)
                + 0.12 * float(tr.get("regrowth_count", 0))
                + 0.10 * pressure,
                0.45,
                0.95,
            )
        )

        rows.append(
            {
                "survey_day": float(day),
                "spatial_id": tr["spatial_id"],
                "track_id": tr["track_id"],
                "status": tr["status"],
                "x_m": float(tr["x_m"]),
                "y_m": float(tr["y_m"]),
                "x_px": float(tr["x_m"] * PX_PER_M),
                "y_px": float(tr["y_m"] * PX_PER_M),
                "area_m2": float(tr["area_m2"]),
                "radius_m": float(tr["radius_m"]),
                "expansion_velocity_m_day": float(tr.get("velocity_m_day", 0.0)),
                "regrowth_count": int(tr.get("regrowth_count", 0)),
                "local_predicted_pressure": pressure,
                "priority_score": float(priority),
                "recommended_radius_m": float(rec_radius),
                "recommended_intensity": float(rec_intensity),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Snapshot copying
# ----------------------------------------------------------------------------
def copy_track(tr):
    ct = {k: v for k, v in tr.items() if k not in ("history", "treatments")}
    ct["history"] = [h.copy() for h in tr.get("history", [])]
    ct["treatments"] = [t.copy() for t in tr.get("treatments", [])]
    return ct


# ----------------------------------------------------------------------------
# Regrowth model validation
# ----------------------------------------------------------------------------
def validate_regrowth_prediction(episodes, config):
    known = [e for e in episodes if e.get("outcome") in (0, 1)]
    out = {
        "n_episodes": len(episodes),
        "n_known": len(known),
        "n_positive": int(sum(1 for e in known if e["outcome"] == 1)),
        "n_negative": int(sum(1 for e in known if e["outcome"] == 0)),
        "status": "ok",
    }

    if len(known) == 0:
        out["status"] = "insufficient_data"
        return out

    y = np.array([e["outcome"] for e in known], dtype=int)
    heuristic_probs = np.array([e["predicted_risk"] for e in known], dtype=float)
    heuristic_pred = (heuristic_probs >= 0.5).astype(int)

    out["heuristic"] = safe_binary_metrics(y, heuristic_pred, heuristic_probs)

    feature_names = [
        "pre_area_m2",
        "pre_velocity_m_day",
        "treatment_intensity",
        "local_pressure_m2",
        "nearest_active_m",
        "days_since_treatment",
        "regrowth_count_prior",
    ]

    X = np.array([[float(e.get(fn, 0.0)) for fn in feature_names] for e in known], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))

    if len(known) >= 8 and pos >= 2 and neg >= 2:
        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            k = min(5, pos, neg)
            if k >= 2:
                model = LogisticRegression(max_iter=2000)
                cv_probs = cross_val_predict(
                    model,
                    Xs,
                    y,
                    cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=42),
                    method="predict_proba",
                )[:, 1]
                cv_pred = (cv_probs >= 0.5).astype(int)
                out["model_cross_validation"] = safe_binary_metrics(y, cv_pred, cv_probs)
                out["model_cross_validation"]["folds"] = k

            final_model = LogisticRegression(max_iter=2000)
            final_model.fit(Xs, y)
            out["coefficients"] = {fn: float(c) for fn, c in zip(feature_names, final_model.coef_[0])}
            out["intercept"] = float(final_model.intercept_[0])
        except Exception as exc:
            out["model_error"] = str(exc)

    return out


# ----------------------------------------------------------------------------
# Main simulation runner
# ----------------------------------------------------------------------------
def run_simulation(config):
    cfg = {**DEFAULT_CONFIG, **config}
    rng = np.random.default_rng(int(cfg["seed"]))

    clusters = generate_initial_clusters(rng, int(cfg["initial_clusters"]))
    tracks = {}
    cell_counters = {}
    episodes = []
    treatment_log = []
    new_zone_points = []
    surveys = []

    prev_day = None
    prev_gt_uids = set()
    prev_uid_to_track = {}

    survey_days = list(cfg["survey_days"])

    for i, day in enumerate(survey_days):
        day = float(day)

        # Stochastic new infestation waves
        if i > 0 and i % 3 == 2:
            count = int(rng.integers(1, 4))
            spawn_new_clusters(clusters, rng, day, count)

        advance_clusters(clusters, prev_day, day, rng, cfg)

        img, true_map, alive_clusters = render_field(clusters, day, rng, cfg)
        detections = segment_image(img, true_map, cfg)

        (
            tracks,
            detections,
            new_tracks,
            removed_tracks,
            regrown_tracks,
            new_zones_today,
        ) = match_and_update(
            tracks=tracks,
            cell_counters=cell_counters,
            detections=detections,
            day=day,
            config=cfg,
            episodes=episodes,
            new_zone_points=new_zone_points,
            prev_gt_uids=prev_gt_uids,
        )

        det_met = detection_metrics(alive_clusters, detections, cfg)
        curr_uid_to_track, curr_uid_tracks = build_uid_track_maps(detections)
        track_met = tracking_metrics(prev_uid_to_track, curr_uid_to_track, curr_uid_tracks)
        prev_uid_to_track = curr_uid_to_track

        for det in detections:
            if det["gt_uid"]:
                prev_gt_uids.add(int(det["gt_uid"]))

        treatments_today = []
        treated_today = set()

        # Manual / what-if treatments scheduled for this survey day
        for et in cfg.get("extra_treatments", []):
            if abs(float(et["day"]) - day) < 0.5:
                affected_uids = apply_treatment_to_clusters(
                    clusters,
                    float(et["x_m"]),
                    float(et["y_m"]),
                    float(et["radius_m"]),
                    float(et["intensity"]),
                    day,
                    rng,
                )
                tids = mark_tracks_treated(
                    tracks,
                    float(et["x_m"]),
                    float(et["y_m"]),
                    float(et["radius_m"]),
                    float(et["intensity"]),
                    day,
                    affected_uids,
                    exclude=treated_today,
                )
                treated_today.update(tids)
                event = {
                    "day": day,
                    "x_m": float(et["x_m"]),
                    "y_m": float(et["y_m"]),
                    "radius_m": float(et["radius_m"]),
                    "intensity": float(et["intensity"]),
                    "source": "manual/what-if",
                    "affected_tracks": tids,
                    "affected_gt_uids": affected_uids,
                }
                treatments_today.append(event)
                treatment_log.append(event)

        # Automatic precision treatment policy
        if cfg.get("auto_treat", True):
            candidates = [
                tr
                for tr in tracks.values()
                if tr["track_id"] not in treated_today
                and tr["status"] in ("active", "regrown")
                and float(tr["area_m2"]) >= float(cfg["treat_area_threshold"])
            ]
            candidates.sort(
                key=lambda t: float(t["area_m2"])
                * (1.0 + 8.0 * float(t.get("velocity_m_day", 0.0)))
                * (1.0 + 0.40 * float(t.get("regrowth_count", 0))),
                reverse=True,
            )

            for tr in candidates[: int(cfg["max_treatments_per_survey"])]:
                radius = max(0.55, float(tr["radius_m"]) * 1.35 + 0.20)
                intensity = float(cfg["treatment_intensity"])
                affected_uids = apply_treatment_to_clusters(
                    clusters, tr["x_m"], tr["y_m"], radius, intensity, day, rng
                )
                tids = mark_tracks_treated(
                    tracks,
                    tr["x_m"],
                    tr["y_m"],
                    radius,
                    intensity,
                    day,
                    affected_uids,
                    exclude=treated_today,
                )
                treated_today.update(tids)
                event = {
                    "day": day,
                    "x_m": float(tr["x_m"]),
                    "y_m": float(tr["y_m"]),
                    "radius_m": float(radius),
                    "intensity": float(intensity),
                    "source": "auto-policy",
                    "affected_tracks": tids,
                    "affected_gt_uids": affected_uids,
                }
                treatments_today.append(event)
                treatment_log.append(event)

        pressure_obs, pressure_pred, hotspots = compute_pressure(tracks, day, cfg, new_zone_points)
        treatment_coords = generate_treatment_coordinates(tracks, pressure_pred, day, cfg)

        active_statuses = ("active", "regrown")
        active_tracks = [tr for tr in tracks.values() if tr["status"] in active_statuses]
        treated_pending = [tr for tr in tracks.values() if tr["status"] == "treated"]
        removed_now = [tr for tr in tracks.values() if tr["status"] == "removed"]

        velocities = [float(tr.get("velocity_m_day", 0.0)) for tr in active_tracks]
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0
        max_velocity = float(np.max(velocities)) if velocities else 0.0

        survey = {
            "idx": i,
            "day": day,
            "img": img,
            "detections": detections,
            "tracks_snapshot": [copy_track(tr) for tr in tracks.values()],
            "new_tracks": new_tracks,
            "removed_tracks": removed_tracks,
            "regrown_tracks": regrown_tracks,
            "new_zones_today": new_zones_today,
            "treatments_today": treatments_today,
            "detection_metrics": det_met,
            "tracking_metrics": track_met,
            "pressure_grid": pressure_obs,
            "predicted_grid": pressure_pred,
            "hotspots": hotspots,
            "treatment_coords": treatment_coords,
            "active_count": len(active_tracks),
            "treated_pending_count": len(treated_pending),
            "removed_count": len(removed_now),
            "mean_velocity": mean_velocity,
            "max_velocity": max_velocity,
        }
        surveys.append(survey)

        prev_day = day

    last_day = float(survey_days[-1]) if survey_days else 0.0
    for ep in episodes:
        if ep["outcome"] is None and last_day >= ep["removed_day"] + cfg["regrowth_horizon_days"]:
            ep["outcome"] = 0

    # Hotspot hit-rate validation
    hotspot_eval = []
    for i, s in enumerate(surveys[:-1]):
        if not s["hotspots"]:
            continue
        target_day = s["day"] + float(cfg["forecast_days"])
        future = None
        for fu in surveys[i + 1 :]:
            if fu["day"] >= target_day:
                future = fu
                break
        if future is None:
            future = surveys[-1]

        future_cells = set()
        cell_w = FIELD_W_M / GRID_COLS
        cell_h = FIELD_H_M / GRID_ROWS
        for det in future["detections"]:
            col = int(np.clip(det["x_m"] / cell_w, 0, GRID_COLS - 1))
            row = int(np.clip(det["y_m"] / cell_h, 0, GRID_ROWS - 1))
            future_cells.add((row, col))

        hits = sum(1 for h in s["hotspots"] if (h["row"], h["col"]) in future_cells)
        hotspot_eval.append(
            {
                "day": s["day"],
                "forecast_day": future["day"],
                "hotspots": len(s["hotspots"]),
                "hits": hits,
                "precision": hits / len(s["hotspots"]) if s["hotspots"] else 0.0,
            }
        )

    det_df = pd.DataFrame([s["detection_metrics"] for s in surveys])
    track_df = pd.DataFrame([s["tracking_metrics"] for s in surveys])

    validation = {
        "detection_summary": {
            "mean_precision": mean_safe(det_df["precision"].tolist()) if not det_df.empty else 0.0,
            "mean_recall": mean_safe(det_df["recall"].tolist()) if not det_df.empty else 0.0,
            "mean_f1": mean_safe(det_df["f1"].tolist()) if not det_df.empty else 0.0,
        },
        "tracking_summary": {
            "mean_id_stability": mean_safe(track_df["id_stability"].tolist()) if not track_df.empty else 0.0,
            "total_id_switches": int(track_df["id_switches"].sum()) if not track_df.empty else 0,
            "total_fragmentation": int(track_df["fragmentation_events"].sum()) if not track_df.empty else 0,
            "total_merge_errors": int(track_df["merge_errors"].sum()) if not track_df.empty else 0,
        },
        "regrowth": validate_regrowth_prediction(episodes, cfg),
        "hotspot_eval": hotspot_eval,
        "hotspot_precision_mean": mean_safe([h["precision"] for h in hotspot_eval]) if hotspot_eval else None,
    }

    return {
        "config": cfg,
        "surveys": surveys,
        "final_tracks": [copy_track(tr) for tr in tracks.values()],
        "episodes": episodes,
        "treatment_log": treatment_log,
        "new_zone_log": new_zone_points,
        "validation": validation,
    }


# ----------------------------------------------------------------------------
# Overlay rendering
# ----------------------------------------------------------------------------
STATUS_COLORS = {
    "active": [255, 80, 60],
    "regrown": [255, 140, 0],
    "treated": [255, 220, 0],
    "new": [190, 60, 220],
    "removed": [60, 160, 255],
    "unmatched": [200, 200, 200],
}


def blend_disk(arr, cx, cy, r, color, alpha, outline=False):
    H, W = arr.shape[:2]
    x0 = max(0, int(cx - r - 3))
    x1 = min(W, int(cx + r + 4))
    y0 = max(0, int(cy - r - 3))
    y1 = min(H, int(cy + r + 4))
    if x1 <= x0 or y1 <= y0:
        return

    yy = Yg[y0:y1, x0:x1]
    xx = Xg[y0:y1, x0:x1]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    if outline:
        thickness = 2.0
        inner = max(0.0, r - thickness)
        outer = r + thickness
        mask = (dist2 <= outer * outer) & (dist2 >= inner * inner)
    else:
        mask = dist2 <= r * r

    if not mask.any():
        return

    region = arr[y0:y1, x0:x1]
    a = float(alpha)
    col = np.array(color, dtype=float)
    region[mask] = (1.0 - a) * region[mask] + a * col
    arr[y0:y1, x0:x1] = region


def make_overlay(img, survey, show_predictions=False):
    arr = img.astype(float)

    for det in survey["detections"]:
        status = det.get("track_status", "unmatched")
        color = STATUS_COLORS.get(status, STATUS_COLORS["unmatched"])
        cx = det["x_m"] * PX_PER_M
        cy = det["y_m"] * PX_PER_M
        r = max(3.0, det["radius_m"] * PX_PER_M)
        blend_disk(arr, cx, cy, r, color, alpha=0.30, outline=False)

        ix = int(round(cx))
        iy = int(round(cy))
        if 0 <= ix < IMG_W and 0 <= iy < IMG_H:
            arr[iy, ix] = [255, 255, 255]

    for event in survey.get("treatments_today", []):
        cx = event["x_m"] * PX_PER_M
        cy = event["y_m"] * PX_PER_M
        r = max(4.0, event["radius_m"] * PX_PER_M)
        blend_disk(arr, cx, cy, r, [40, 220, 120], alpha=0.55, outline=True)

    for nz in survey.get("new_zones_today", []):
        cx = nz["x_m"] * PX_PER_M
        cy = nz["y_m"] * PX_PER_M
        blend_disk(arr, cx, cy, 8.0, [230, 70, 230], alpha=0.55, outline=True)

    if show_predictions:
        for h in survey.get("hotspots", []):
            cx = h["x_m"] * PX_PER_M
            cy = h["y_m"] * PX_PER_M
            alpha = 0.10 + 0.25 * float(h.get("predicted_pressure", 0.0))
            blend_disk(arr, cx, cy, 28.0, [255, 170, 0], alpha=alpha, outline=False)

    return np.clip(arr, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="WeedCycle Temporal Weed Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌿 WeedCycle Temporal Weed Intelligence Platform")
st.caption(
    "Synthetic live demo: repeated field capture → crop/weed segmentation → spatial IDs → lifecycle tracking → "
    "treatment removal/regrowth analysis → expansion velocity → hotspot prediction → precision coordinates → validation."
)

if "extra_treatments" not in st.session_state:
    st.session_state.extra_treatments = []

if "config" not in st.session_state:
    st.session_state.config = dict(DEFAULT_CONFIG)
    st.session_state.config["extra_treatments"] = st.session_state.extra_treatments
    st.session_state.result = run_simulation(st.session_state.config)
    st.session_state.selected_survey = len(st.session_state.result["surveys"]) - 1

if "result" not in st.session_state:
    st.session_state.result = run_simulation(st.session_state.config)
    st.session_state.selected_survey = len(st.session_state.result["surveys"]) - 1

result = st.session_state.result
surveys = result["surveys"]

if not surveys:
    st.error("Simulation produced no surveys. Try changing the scenario controls.")
    st.stop()

st.session_state.selected_survey = int(
    np.clip(st.session_state.selected_survey, 0, len(surveys) - 1)
)


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.header("Scenario controls")

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=int(st.session_state.config.get("seed", 42)),
)
initial_clusters = st.sidebar.slider(
    "Initial weed clusters",
    6,
    30,
    int(st.session_state.config.get("initial_clusters", 14)),
)
cadence = st.sidebar.selectbox(
    "Survey cadence",
    list(SURVEY_DAY_PRESETS.keys()),
    index=1,
)
auto_treat = st.sidebar.checkbox(
    "Automatic precision treatments",
    bool(st.session_state.config.get("auto_treat", True)),
)
treat_threshold = st.sidebar.slider(
    "Auto-treat area threshold (m²)",
    0.15,
    1.50,
    float(st.session_state.config.get("treat_area_threshold", 0.45)),
    0.05,
)
treat_intensity = st.sidebar.slider(
    "Treatment intensity",
    0.30,
    1.00,
    float(st.session_state.config.get("treatment_intensity", 0.78)),
    0.02,
)
forecast_days = st.sidebar.slider(
    "Forecast horizon (days)",
    7,
    45,
    int(st.session_state.config.get("forecast_days", 21)),
    1,
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**What-if treatments queued:** {len(st.session_state.extra_treatments)}")

if st.sidebar.button("Re-run simulation", use_container_width=True):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "seed": int(seed),
            "initial_clusters": int(initial_clusters),
            "survey_days": SURVEY_DAY_PRESETS[cadence],
            "auto_treat": bool(auto_treat),
            "treat_area_threshold": float(treat_threshold),
            "treatment_intensity": float(treat_intensity),
            "forecast_days": int(forecast_days),
            "extra_treatments": st.session_state.extra_treatments,
        }
    )
    st.session_state.config = cfg
    st.session_state.result = run_simulation(cfg)
    st.session_state.selected_survey = len(st.session_state.result["surveys"]) - 1
    st.rerun()

if st.sidebar.button("Clear what-if treatments", use_container_width=True):
    st.session_state.extra_treatments = []
    cfg = dict(st.session_state.config)
    cfg.update(
        {
            "seed": int(seed),
            "initial_clusters": int(initial_clusters),
            "survey_days": SURVEY_DAY_PRESETS[cadence],
            "auto_treat": bool(auto_treat),
            "treat_area_threshold": float(treat_threshold),
            "treatment_intensity": float(treat_intensity),
            "forecast_days": int(forecast_days),
            "extra_treatments": [],
        }
    )
    st.session_state.config = cfg
    st.session_state.result = run_simulation(cfg)
    st.session_state.selected_survey = len(st.session_state.result["surveys"]) - 1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.success("No API key required. All demo data is generated locally.")


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_live, tab_tracker, tab_treatment, tab_prediction, tab_validation = st.tabs(
    ["Live Field", "Temporal Tracker", "Treatment & Regrowth", "Prediction & Heatmaps", "Validation"]
)


# ----------------------------------------------------------------------------
# Live Field tab
# ----------------------------------------------------------------------------
with tab_live:
    idx = st.slider(
        "Survey timeline",
        min_value=0,
        max_value=len(surveys) - 1,
        value=int(st.session_state.selected_survey),
    )
    st.session_state.selected_survey = idx
    survey = surveys[idx]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Survey day", f"{int(survey['day'])} d")
    c2.metric("Active clusters", survey["active_count"])
    c3.metric("Detections", len(survey["detections"]))
    c4.metric("New zones", len(survey["new_zones_today"]))
    c5.metric("Treatments applied", len(survey["treatments_today"]))
    c6.metric("Mean expansion", f"{survey['mean_velocity']:.3f} m/d")

    show_pred_overlay = st.checkbox("Show predicted hotspots on image", value=False)

    col_img, col_map = st.columns([2, 1])
    with col_img:
        overlay = make_overlay(survey["img"], survey, show_predictions=show_pred_overlay)
        st.image(overlay, use_container_width=True, caption="Segmented weed clusters, treatment footprints, and new zones")

    with col_map:
        if survey["detections"]:
            df_det = pd.DataFrame(survey["detections"])
            fig = px.scatter(
                df_det,
                x="x_m",
                y="y_m",
                color="track_status",
                size="area_m2",
                hover_data=["track_spatial_id", "area_m2", "confidence", "species_guess"],
                title="Geospatial detection map",
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))

            if show_pred_overlay and survey["hotspots"]:
                df_hs = pd.DataFrame(survey["hotspots"])
                fig.add_trace(
                    go.Scatter(
                        x=df_hs["x_m"],
                        y=df_hs["y_m"],
                        mode="markers",
                        marker=dict(symbol="circle-open", size=18, color="orange", line=dict(width=2)),
                        name="Predicted hotspot",
                        hovertext=df_hs["cell"],
                    )
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No weed detections in this survey.")

    if st.button("▶ Play replay through all surveys"):
        img_ph = st.empty()
        cap_ph = st.empty()
        for j in range(len(surveys)):
            s = surveys[j]
            ov = make_overlay(s["img"], s, show_predictions=False)
            img_ph.image(ov, use_container_width=True)
            cap_ph.markdown(
                f"**Replaying:** Day {int(s['day'])} | detections={len(s['detections'])} | "
                f"active={s['active_count']} | new zones={len(s['new_zones_today'])}"
            )
            time.sleep(0.45)

    st.subheader("Current detection table")
    if survey["detections"]:
        df_show = pd.DataFrame(survey["detections"])[
            [
                "track_spatial_id",
                "matched_track_id",
                "track_status",
                "x_m",
                "y_m",
                "area_m2",
                "radius_m",
                "confidence",
                "species_guess",
            ]
        ].rename(
            columns={
                "track_spatial_id": "Spatial ID",
                "matched_track_id": "Track ID",
                "track_status": "Status",
                "x_m": "X (m)",
                "y_m": "Y (m)",
                "area_m2": "Area (m²)",
                "radius_m": "Radius (m)",
                "confidence": "Confidence",
                "species_guess": "Weed type guess",
            }
        )
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.download_button(
            "Download current detections CSV",
            df_show.to_csv(index=False).encode("utf-8"),
            file_name=f"detections_day_{int(survey['day'])}.csv",
            mime="text/csv",
        )
    else:
        st.info("No detections to display.")


# ----------------------------------------------------------------------------
# Temporal Tracker tab
# ----------------------------------------------------------------------------
with tab_tracker:
    st.subheader("Weed / cluster lifecycle tracks")
    tracks_df = pd.DataFrame(survey["tracks_snapshot"])
    if not tracks_df.empty:
        show_cols = [
            "spatial_id",
            "track_id",
            "status",
            "first_seen_day",
            "last_seen_day",
            "x_m",
            "y_m",
            "area_m2",
            "radius_m",
            "velocity_m_day",
            "regrowth_count",
            "species_guess",
            "is_new_zone",
        ]
        tracks_view = tracks_df[show_cols].rename(
            columns={
                "spatial_id": "Spatial ID",
                "track_id": "Track ID",
                "status": "Lifecycle status",
                "first_seen_day": "First seen (d)",
                "last_seen_day": "Last seen (d)",
                "x_m": "X (m)",
                "y_m": "Y (m)",
                "area_m2": "Area (m²)",
                "radius_m": "Radius (m)",
                "velocity_m_day": "Expansion velocity (m/d)",
                "regrowth_count": "Regrowths",
                "species_guess": "Type guess",
                "is_new_zone": "New zone?",
            }
        )
        st.dataframe(tracks_view, use_container_width=True, hide_index=True)
    else:
        st.info("No tracks yet.")

    st.subheader("Temporal area and velocity history")
    final_tracks = result["final_tracks"]
    if final_tracks:
        track_options = [f"{t['spatial_id']} | {t['track_id']}" for t in final_tracks]
        default_selected = track_options[: min(4, len(track_options))]
        selected = st.multiselect("Select tracks to plot", track_options, default=default_selected)

        rows = []
        for opt in selected:
            tid = opt.split("|")[1].strip()
            tr = next((t for t in final_tracks if t["track_id"] == tid), None)
            if not tr:
                continue
            for h in tr["history"]:
                rows.append(
                    {
                        "Spatial ID": tr["spatial_id"],
                        "Track ID": tr["track_id"],
                        "Day": h["day"],
                        "Area (m²)": h["area_m2"],
                        "Radius (m)": h["radius_m"],
                        "Confidence": h["confidence"],
                    }
                )

        if rows:
            hist_df = pd.DataFrame(rows)
            colA, colB = st.columns(2)
            with colA:
                fig_area = px.line(
                    hist_df,
                    x="Day",
                    y="Area (m²)",
                    color="Spatial ID",
                    markers=True,
                    title="Cluster area over time",
                )
                fig_area.update_layout(height=360)
                st.plotly_chart(fig_area, use_container_width=True)
            with colB:
                fig_rad = px.line(
                    hist_df,
                    x="Day",
                    y="Radius (m)",
                    color="Spatial ID",
                    markers=True,
                    title="Cluster radius over time",
                )
                fig_rad.update_layout(height=360)
                st.plotly_chart(fig_rad, use_container_width=True)
        else:
            st.info("Select at least one track.")

        st.subheader("Lifecycle event map")
        event_rows = []
        for tr in final_tracks:
            event_rows.append(
                {
                    "Spatial ID": tr["spatial_id"],
                    "Day": tr["first_seen_day"],
                    "Event": "first detection",
                    "Status": "active",
                }
            )
            for treat in tr.get("treatments", []):
                event_rows.append(
                    {
                        "Spatial ID": tr["spatial_id"],
                        "Day": treat["day"],
                        "Event": "treatment",
                        "Status": "treated",
                    }
                )
            if tr.get("removed_day") is not None:
                event_rows.append(
                    {
                        "Spatial ID": tr["spatial_id"],
                        "Day": tr["removed_day"],
                        "Event": "removal confirmed",
                        "Status": "removed",
                    }
                )
            if tr.get("last_regrowth_day") is not None:
                event_rows.append(
                    {
                        "Spatial ID": tr["spatial_id"],
                        "Day": tr["last_regrowth_day"],
                        "Event": "regrowth",
                        "Status": "regrown",
                    }
                )

        if event_rows:
            ev_df = pd.DataFrame(event_rows)
            fig_ev = px.scatter(
                ev_df,
                x="Day",
                y="Spatial ID",
                color="Status",
                symbol="Event",
                hover_data=["Event"],
                title="First detection, treatment, removal, and regrowth events",
            )
            fig_ev.update_layout(height=max(360, 28 * ev_df["Spatial ID"].nunique()))
            st.plotly_chart(fig_ev, use_container_width=True)

    st.subheader("New infestation zones")
    nz_df = pd.DataFrame(result["new_zone_log"])
    if not nz_df.empty:
        st.dataframe(
            nz_df.rename(
                columns={
                    "day": "Detected day",
                    "track_id": "Track ID",
                    "spatial_id": "Spatial ID",
                    "x_m": "X (m)",
                    "y_m": "Y (m)",
                    "area_m2": "Area (m²)",
                    "distance_from_prior_m": "Distance from prior weed (m)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No new infestation zones detected in this scenario.")


# ----------------------------------------------------------------------------
# Treatment & Regrowth tab
# ----------------------------------------------------------------------------
with tab_treatment:
    st.subheader("Treatment history")
    treat_df = pd.DataFrame(result["treatment_log"])
    if not treat_df.empty:
        treat_show = treat_df.copy()
        treat_show["affected_tracks"] = treat_show["affected_tracks"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
        treat_show["affected_gt_uids"] = treat_show["affected_gt_uids"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else ""
        )
        st.dataframe(
            treat_show.rename(
                columns={
                    "day": "Day",
                    "x_m": "X (m)",
                    "y_m": "Y (m)",
                    "radius_m": "Radius (m)",
                    "intensity": "Intensity",
                    "source": "Source",
                    "affected_tracks": "Affected tracks",
                    "affected_gt_uids": "Affected ground-truth clusters",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No treatments recorded.")

    st.markdown("---")
    st.subheader("What-if precision treatment")
    st.caption("Add a treatment at a selected survey day. The full temporal simulation is re-run with this intervention.")

    # FIXED: use a direct label-to-index mapping instead of parsing "#6)"
    day_label_to_idx = {
        f"Day {int(s['day'])} (survey #{s['idx']})": int(s["idx"])
        for s in surveys
    }
    day_choices = list(day_label_to_idx.keys())

    day_choice = st.selectbox(
        "Survey day for what-if treatment",
        day_choices,
        index=len(day_choices) - 1,
    )
    day_idx = day_label_to_idx[day_choice]
    target_survey = surveys[day_idx]

    track_opts = ["Custom coordinates"]
    for tr in target_survey["tracks_snapshot"]:
        if tr["status"] in ("active", "regrown", "treated"):
            track_opts.append(f"{tr['spatial_id']} | {tr['track_id']} | {tr['status']}")

    target_choice = st.selectbox("Target weed cluster", track_opts)

    default_x = FIELD_W_M / 2.0
    default_y = FIELD_H_M / 2.0
    default_r = 0.85
    if target_choice != "Custom coordinates":
        tid = target_choice.split("|")[1].strip()
        tr = next((t for t in target_survey["tracks_snapshot"] if t["track_id"] == tid), None)
        if tr:
            default_x = float(tr["x_m"])
            default_y = float(tr["y_m"])
            default_r = max(0.55, float(tr["radius_m"]) * 1.35 + 0.20)

    cwx, cwy, cwrad, cwint = st.columns(4)
    with cwx:
        whatif_x = st.number_input("X (m)", min_value=0.0, max_value=FIELD_W_M, value=float(default_x), step=0.1)
    with cwy:
        whatif_y = st.number_input("Y (m)", min_value=0.0, max_value=FIELD_H_M, value=float(default_y), step=0.1)
    with cwrad:
        whatif_r = st.number_input("Radius (m)", min_value=0.20, max_value=5.00, value=float(default_r), step=0.05)
    with cwint:
        whatif_i = st.number_input("Intensity", min_value=0.10, max_value=1.00, value=0.82, step=0.02)

    if st.button("Apply what-if and re-run simulation", use_container_width=True):
        et = {
            "day": float(target_survey["day"]),
            "x_m": float(whatif_x),
            "y_m": float(whatif_y),
            "radius_m": float(whatif_r),
            "intensity": float(whatif_i),
        }
        st.session_state.extra_treatments = st.session_state.extra_treatments + [et]

        cfg = dict(st.session_state.config)
        cfg.update(
            {
                "seed": int(seed),
                "initial_clusters": int(initial_clusters),
                "survey_days": SURVEY_DAY_PRESETS[cadence],
                "auto_treat": bool(auto_treat),
                "treat_area_threshold": float(treat_threshold),
                "treatment_intensity": float(treat_intensity),
                "forecast_days": int(forecast_days),
                "extra_treatments": st.session_state.extra_treatments,
            }
        )

        st.session_state.config = cfg
        st.session_state.result = run_simulation(cfg)
        st.session_state.selected_survey = day_idx
        st.rerun()

    st.markdown("---")
    st.subheader("Regrowth episodes: treatment history vs later emergence")
    ep_df = pd.DataFrame(result["episodes"])
    if not ep_df.empty:
        ep_view = ep_df.copy()
        ep_view["Observed"] = ep_view["outcome"].map({1: "regrown", 0: "no regrowth", None: "pending"})
        show_ep_cols = [
            "spatial_id",
            "track_id",
            "treated_day",
            "removed_day",
            "days_since_treatment",
            "pre_area_m2",
            "pre_velocity_m_day",
            "treatment_intensity",
            "local_pressure_m2",
            "predicted_risk",
            "Observed",
            "days_to_regrowth",
        ]
        st.dataframe(
            ep_view[show_ep_cols].rename(
                columns={
                    "spatial_id": "Spatial ID",
                    "track_id": "Track ID",
                    "treated_day": "Treated day",
                    "removed_day": "Removal confirmed day",
                    "days_since_treatment": "Days to removal",
                    "pre_area_m2": "Pre-treatment area (m²)",
                    "pre_velocity_m_day": "Pre-treatment velocity (m/d)",
                    "treatment_intensity": "Intensity",
                    "local_pressure_m2": "Local pressure (m²)",
                    "predicted_risk": "Predicted regrowth risk",
                    "days_to_regrowth": "Days to regrowth",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        known = ep_df[ep_df["outcome"].isin([0, 1])]
        if not known.empty:
            fig_reg = px.scatter(
                known,
                x="treated_day",
                y="days_to_regrowth",
                color="outcome",
                hover_data=["spatial_id", "predicted_risk", "treatment_intensity"],
                title="Time-to-regrowth after treatment",
            )
            fig_reg.update_layout(height=340)
            st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("No removal episodes yet. Increase survey duration or lower the auto-treatment threshold.")


# ----------------------------------------------------------------------------
# Prediction & Heatmaps tab
# ----------------------------------------------------------------------------
with tab_prediction:
    survey = surveys[int(st.session_state.selected_survey)]

    mode = st.radio(
        "Heatmap mode",
        ["Observed weed pressure", "Predicted future hotspots"],
        horizontal=True,
    )
    grid = survey["pressure_grid"] if mode.startswith("Observed") else survey["predicted_grid"]

    x_centers = (np.arange(GRID_COLS) + 0.5) * (FIELD_W_M / GRID_COLS)
    y_centers = (np.arange(GRID_ROWS) + 0.5) * (FIELD_H_M / GRID_ROWS)

    fig_heat = go.Figure(
        go.Heatmap(
            z=grid,
            x=x_centers,
            y=y_centers,
            colorscale="Viridis" if mode.startswith("Observed") else "Hot",
            hovertemplate="X=%{x:.2f} m<br>Y=%{y:.2f} m<br>Pressure=%{z:.3f}<extra></extra>",
        )
    )
    fig_heat.update_yaxes(autorange="reversed")
    fig_heat.update_layout(
        title=f"{mode} — Day {int(survey['day'])}",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=390,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Predicted hotspot cells")
        hs_df = pd.DataFrame(survey["hotspots"])
        if not hs_df.empty:
            st.dataframe(
                hs_df.rename(
                    columns={
                        "cell": "Cell",
                        "x_m": "X (m)",
                        "y_m": "Y (m)",
                        "predicted_pressure": "Predicted pressure",
                        "observed_pressure": "Observed pressure",
                    }
                )[["Cell", "X (m)", "Y (m)", "Predicted pressure", "Observed pressure"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No hotspot cells above threshold.")

    with c2:
        st.subheader("Expansion velocity vs area")
        tr_df = pd.DataFrame(survey["tracks_snapshot"])
        active_df = (
            tr_df[tr_df["status"].isin(["active", "regrown"])]
            if not tr_df.empty
            else pd.DataFrame()
        )
        if not active_df.empty:
            fig_vel = px.scatter(
                active_df,
                x="area_m2",
                y="velocity_m_day",
                color="status",
                size="regrowth_count",
                hover_data=["spatial_id", "radius_m", "confidence"],
                title="Active cluster dynamics",
            )
            fig_vel.update_layout(height=340)
            st.plotly_chart(fig_vel, use_container_width=True)
        else:
            st.info("No active clusters in this survey.")

    st.subheader("Precision treatment coordinates")
    tx_df = survey["treatment_coords"]
    if not tx_df.empty:
        st.dataframe(tx_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download precision treatment coordinates CSV",
            tx_df.to_csv(index=False).encode("utf-8"),
            file_name=f"precision_treatment_day_{int(survey['day'])}.csv",
            mime="text/csv",
        )
    else:
        st.info("No active clusters requiring treatment coordinates in this survey.")


# ----------------------------------------------------------------------------
# Validation tab
# ----------------------------------------------------------------------------
with tab_validation:
    st.subheader("Temporal detection and tracking accuracy")

    det_df = pd.DataFrame([{**s["detection_metrics"], "day": s["day"]} for s in surveys])
    track_df = pd.DataFrame([{**s["tracking_metrics"], "day": s["day"]} for s in surveys])

    if not det_df.empty:
        det_long = det_df.melt(
            id_vars="day",
            value_vars=["precision", "recall", "f1"],
            var_name="Metric",
            value_name="Score",
        )
        fig_det = px.line(
            det_long,
            x="day",
            y="Score",
            color="Metric",
            markers=True,
            title="Weed detection accuracy by survey",
        )
        fig_det.update_layout(height=340)
        st.plotly_chart(fig_det, use_container_width=True)

    if not track_df.empty:
        track_plot = track_df.dropna(subset=["id_stability"])
        if not track_plot.empty:
            fig_track = px.line(
                track_plot,
                x="day",
                y="id_stability",
                title="Spatial ID stability across repeated surveys",
            )
            fig_track.update_layout(height=320, yaxis_range=[0, 1.05])
            st.plotly_chart(fig_track, use_container_width=True)

        colm1, colm2, colm3, colm4 = st.columns(4)
        colm1.metric(
            "Mean ID stability",
            f"{result['validation']['tracking_summary']['mean_id_stability']:.3f}",
        )
        colm2.metric("ID switches", result["validation"]["tracking_summary"]["total_id_switches"])
        colm3.metric("Fragmentation events", result["validation"]["tracking_summary"]["total_fragmentation"])
        colm4.metric("Merge errors", result["validation"]["tracking_summary"]["total_merge_errors"])

    st.markdown("---")
    st.subheader("Regrowth prediction validation")

    val_reg = result["validation"]["regrowth"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes", val_reg.get("n_episodes", 0))
    c2.metric("Known outcomes", val_reg.get("n_known", 0))
    c3.metric("Regrowth positives", val_reg.get("n_positive", 0))
    c4.metric("Non-regrowth negatives", val_reg.get("n_negative", 0))

    if val_reg.get("status") == "insufficient_data":
        st.info(
            "Not enough completed removal episodes to validate regrowth prediction. "
            "Re-run with a longer cadence or more initial clusters."
        )
    else:
        heur = val_reg.get("heuristic", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Heuristic accuracy", f"{heur.get('accuracy', 0):.3f}")
        c2.metric("Heuristic precision", f"{heur.get('precision', 0):.3f}")
        c3.metric("Heuristic recall", f"{heur.get('recall', 0):.3f}")
        c4.metric("Heuristic F1", f"{heur.get('f1', 0):.3f}")
        c5.metric(
            "Heuristic ROC-AUC",
            f"{heur.get('roc_auc'):.3f}" if heur.get("roc_auc") is not None else "n/a",
        )

        cm = np.array(heur.get("confusion_matrix", [[0, 0], [0, 0]]))
        cm_labels = ["No regrowth", "Regrowth"]
        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            labels=dict(color="Count"),
            color_continuous_scale="Blues",
            title="Heuristic regrowth confusion matrix",
        )
        fig_cm.update_layout(height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

        if "model_cross_validation" in val_reg:
            mc = val_reg["model_cross_validation"]
            st.markdown("#### Logistic regression cross-validation")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("CV accuracy", f"{mc.get('accuracy', 0):.3f}")
            c2.metric("CV precision", f"{mc.get('precision', 0):.3f}")
            c3.metric("CV recall", f"{mc.get('recall', 0):.3f}")
            c4.metric("CV F1", f"{mc.get('f1', 0):.3f}")
            c5.metric(
                "CV ROC-AUC",
                f"{mc.get('roc_auc'):.3f}" if mc.get("roc_auc") is not None else "n/a",
            )

        if "coefficients" in val_reg:
            coef_df = pd.DataFrame(list(val_reg["coefficients"].items()), columns=["Feature", "Coefficient"])
            coef_df = coef_df.sort_values("Coefficient", ascending=False)
            fig_coef = px.bar(
                coef_df,
                x="Feature",
                y="Coefficient",
                orientation="h",
                title="Regrowth risk feature weights",
            )
            fig_coef.update_layout(height=320)
            st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")
    st.subheader("Future hotspot hit-rate")
    hs_eval = result["validation"].get("hotspot_eval", [])
    if hs_eval:
        hs_df = pd.DataFrame(hs_eval)
        st.dataframe(hs_df, use_container_width=True, hide_index=True)
        st.metric(
            "Mean hotspot precision",
            f"{result['validation'].get('hotspot_precision_mean', 0):.3f}",
        )
    else:
        st.info("No hotspot forecasts available for validation in this scenario.")

    st.markdown("---")
    st.markdown(
        """
        ### What these metrics mean
        - **Detection precision/recall/F1:** how well the image segmentation finds true weed clusters without false alarms.
        - **ID stability:** whether the same weed cluster keeps the same spatial ID across repeated surveys.
        - **ID switches / fragmentation / merge errors:** temporal tracking failures where one cluster changes ID, splits, or multiple clusters merge.
        - **Regrowth prediction:** evaluates whether removed clusters are correctly flagged as likely to return.
        - **Hotspot precision:** checks whether predicted future high-pressure cells actually contain weeds in the forecast survey.
        """
    )