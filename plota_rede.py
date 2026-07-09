import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def PlotaRede(conec, Xno, p, q, factor_units=0.001):

    edges = conec
    coord = Xno
    nv = np.max(np.max(conec))+1
    nc = conec.shape[0]

    # Internal: get edge and midpoint coordinates
    segs = []
    mids = []
    for (i, j) in edges:
      x1, y1 = coord[i,0], coord[i,1]
      x2, y2 = coord[j,0], coord[j,1]
      segs.append(((x1, y1), (x2, y2)))
      mids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    segs = np.array(segs)
    mids = np.array(mids)

    fig_size=(10, 10)
    cmap_name="coolwarm"
    node_size=50
    show_flux_labels=False
    arrow_scale=0.0005
    text_scale=0.6
    save_path=None
    show_pressure_values=False

    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')

    # ---- Pressure colormap ----
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = float(p.min()), float(p.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    xs, ys = [], []
    for i in range(nv):
      xs.append(coord[i,0])
      ys.append(coord[i,1])
    colors = [cmap(norm(pi)) for pi in p]
    ax.scatter(xs, ys, s=node_size, c=colors, zorder=3, edgecolors="black")

    # ---- Draw black edges and arrows ----
    #segs, mids = edge_coords()
    for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.5, zorder=1)

        xm, ym = mids[idx]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0:
          continue
        dxn, dyn = dx / L, dy / L
        nx, ny = -dyn, dxn  # normal vector

        # --- Flux arrow (black) ---
        p1, p2 = p[edges[idx,0]], p[edges[idx,1]]
        q_dir = 1 if p1 > p2 else -1

        ax.annotate(
              "",
              xy=(xm + q_dir * 0.5 * arrow_scale * dxn, ym + q_dir * 0.5 * arrow_scale * dyn),
              xytext=(xm - q_dir * 0.5 * arrow_scale * dxn, ym - q_dir * 0.5 * arrow_scale * dyn),
              arrowprops=dict(
              arrowstyle="-|>",
              color="black",
              lw=0.5,
              mutation_scale=5 * text_scale * 3,  # scales arrowhead size
              ),
              zorder=5,
              )

        # --- Flux label ---
        if show_flux_labels:
          label_offset = 0.0725*factor_units
          ax.text(
              xm + nx * label_offset,
              ym + ny * label_offset,
              f"q={q[idx]:.1e}",
              ha="center",
              va="center",
              fontsize=12 * text_scale,
              zorder=6,
              bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0)
              )

    # ---- Node labels ----
    for node, (x, y) in enumerate(coord):
      if show_pressure_values:
          ax.text(x, y, str(node),
                  ha="center", va="center", fontsize=11 * text_scale, zorder=4)
        
          ax.text(x - 0.075*factor_units, y - 0.075*factor_units, f"p={p[node]:.1e}",
          ha="right", va="bottom", fontsize=12 * text_scale,
          color="black", zorder=5)

    # ---- Final adjustments ----
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Set limits with small margin ----
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range, y_range = x_max - x_min, y_max - y_min
    ax.set_xlim(x_min - 0.5*factor_units, x_max + 0.5*factor_units)
    ax.set_ylim(y_min - 0.5*factor_units, y_max + 0.5*factor_units)

    # Optionally adapt figure size to graph geometry
    aspect_ratio = x_range / y_range if y_range != 0 else 1.0
    base_size = 8  # base figure size
    fig.set_size_inches(base_size * aspect_ratio, base_size)

    # ---- Colorbar ----
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, label="Pressure, p", fraction=0.0225, pad=0.025)
    cbar.ax.tick_params(labelsize=10 * text_scale)
    cbar.set_label("Pressure, p", fontsize=12 * text_scale)

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches="tight")
      plt.show()

    return fig, ax
