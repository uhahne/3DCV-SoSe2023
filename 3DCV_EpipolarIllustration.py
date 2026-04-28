import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt


def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def camera_frame(yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    forward = normalize(np.array([np.sin(yaw), 0.0, np.cos(yaw)]))
    up_world = np.array([0.0, 1.0, 0.0])
    right = normalize(np.cross(up_world, forward))
    up = normalize(np.cross(forward, right))
    return right, up, forward


def plane_corners(center, right, up, w, h):
    hw, hh = w / 2.0, h / 2.0
    return np.array([
        center - hw * right - hh * up,
        center + hw * right - hh * up,
        center + hw * right + hh * up,
        center - hw * right + hh * up,
    ])


def project_to_image_plane(C, fwd, X, d):
    ray = X - C
    denom = np.dot(ray, fwd)
    if abs(denom) < 1e-9:
        return None
    t = d / denom
    return C + t * ray


def clip_line_to_image_rect(p, dvec, O, r, u, w, h):
    # 2D coordinates in image plane basis
    a0 = np.dot(p - O, r)
    b0 = np.dot(p - O, u)
    da = np.dot(dvec, r)
    db = np.dot(dvec, u)

    t_min, t_max = -np.inf, np.inf
    bounds = [(-w / 2, w / 2, a0, da), (-h / 2, h / 2, b0, db)]

    for lo, hi, x0, dx in bounds:
        if abs(dx) < 1e-12:
            if x0 < lo or x0 > hi:
                return None
            continue
        t1 = (lo - x0) / dx
        t2 = (hi - x0) / dx
        t_lo, t_hi = sorted((t1, t2))
        t_min = max(t_min, t_lo)
        t_max = min(t_max, t_hi)
        if t_min > t_max:
            return None

    p1 = p + t_min * dvec
    p2 = p + t_max * dvec
    return p1, p2


def draw_scene(ax, yaw2_deg, X):
    # Camera setup
    C1 = np.array([0.0, 0.0, 0.0])
    C2 = np.array([2.0, 0.0, 0.0])

    r1, u1, f1 = camera_frame(0.0)
    r2, u2, f2 = camera_frame(yaw2_deg)

    # Image plane params
    d = 1.2
    w, h = 2.0, 1.5

    O1 = C1 + d * f1
    O2 = C2 + d * f2

    img1 = plane_corners(O1, r1, u1, w, h)
    img2 = plane_corners(O2, r2, u2, w, h)

    # Clear & axes setup
    ax.cla()
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 8)
    ax.set_box_aspect((8, 6, 8))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    ax.set_title("Epipolar Geometry (3D)")

    # Draw image planes
    ax.add_collection3d(
        Poly3DCollection([img1], facecolor="tab:blue", alpha=0.15, edgecolor="tab:blue")
    )
    ax.add_collection3d(
        Poly3DCollection([img2], facecolor="tab:orange", alpha=0.15, edgecolor="tab:orange")
    )

    # Camera centers
    ax.scatter(*C1, c="tab:blue", s=60, label="Camera 1")
    ax.scatter(*C2, c="tab:orange", s=60, label="Camera 2")
    ax.plot([C1[0], C1[0] + 0.8 * f1[0]], [C1[1], C1[1] + 0.8 * f1[1]], [C1[2], C1[2] + 0.8 * f1[2]], c="tab:blue")
    ax.plot([C2[0], C2[0] + 0.8 * f2[0]], [C2[1], C2[1] + 0.8 * f2[1]], [C2[2], C2[2] + 0.8 * f2[2]], c="tab:orange")

    # Baseline
    ax.plot([C1[0], C2[0]], [C1[1], C2[1]], [C1[2], C2[2]], "k--", lw=1, label="Baseline")

    # 3D point
    ax.scatter(*X, c="crimson", s=70, label="3D point X")

    # Projection rays
    ax.plot([C1[0], X[0]], [C1[1], X[1]], [C1[2], X[2]], c="tab:blue", lw=1.5)
    ax.plot([C2[0], X[0]], [C2[1], X[1]], [C2[2], X[2]], c="tab:orange", lw=1.5)

    # Epipolar plane
    epi_plane = Poly3DCollection([[C1, C2, X]], facecolor="green", alpha=0.10, edgecolor="green")
    ax.add_collection3d(epi_plane)

    # Image projections of X
    x1 = project_to_image_plane(C1, f1, X, d)
    x2 = project_to_image_plane(C2, f2, X, d)
    if x1 is not None:
        ax.scatter(*x1, c="tab:blue", marker="x", s=80, label="x1")
    if x2 is not None:
        ax.scatter(*x2, c="tab:orange", marker="x", s=80, label="x2")

    # Epipoles (projection of opposite camera center)
    e1 = project_to_image_plane(C1, f1, C2, d)
    e2 = project_to_image_plane(C2, f2, C1, d)
    if e1 is not None:
        ax.scatter(*e1, c="tab:blue", marker="*", s=120, label="e1")
    if e2 is not None:
        ax.scatter(*e2, c="tab:orange", marker="*", s=120, label="e2")

    # Epipolar lines = intersection(epipolar plane, image plane)
    n_epi = np.cross(C2 - C1, X - C1)
    if np.linalg.norm(n_epi) > 1e-10 and x1 is not None and x2 is not None:
        # In image 1
        d1 = np.cross(n_epi, f1)
        seg1 = clip_line_to_image_rect(x1, d1, O1, r1, u1, w, h)
        if seg1 is not None:
            p1, p2 = seg1
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="tab:blue", lw=2, label="epiline 1")

        # In image 2
        d2 = np.cross(n_epi, f2)
        seg2 = clip_line_to_image_rect(x2, d2, O2, r2, u2, w, h)
        if seg2 is not None:
            p1, p2 = seg2
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="tab:orange", lw=2, label="epiline 2")

    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=8)


# ---- UI ----
fig = plt.figure(figsize=(11, 8))
ax3d = fig.add_axes([0.05, 0.28, 0.90, 0.68], projection="3d")

ax_yaw = fig.add_axes([0.12, 0.20, 0.76, 0.03])
ax_px = fig.add_axes([0.12, 0.15, 0.76, 0.03])
ax_py = fig.add_axes([0.12, 0.10, 0.76, 0.03])
ax_pz = fig.add_axes([0.12, 0.05, 0.76, 0.03])

s_yaw = Slider(ax_yaw, "Cam2 yaw (deg)", -70.0, 70.0, valinit=15.0, valstep=1.0)
s_px = Slider(ax_px, "Point X", -2.5, 4.5, valinit=0.8)
s_py = Slider(ax_py, "Point Y", -2.5, 2.5, valinit=0.3)
s_pz = Slider(ax_pz, "Point Z", 1.0, 7.5, valinit=4.0)


def update(_=None):
    X = np.array([s_px.val, s_py.val, s_pz.val], dtype=float)
    draw_scene(ax3d, s_yaw.val, X)
    fig.canvas.draw_idle()


s_yaw.on_changed(update)
s_px.on_changed(update)
s_py.on_changed(update)
s_pz.on_changed(update)

update()
plt.show()