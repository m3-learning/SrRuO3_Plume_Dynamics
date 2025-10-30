# sro_sto_plume/rsm_helpers.py
import numpy as np

def intensity_weighted_centroid(Qx, Qz, I):
    """Return intensity-weighted centroid (qx_c, qz_c) for arrays Qx, Qz, I with same shape."""
    Ipos = np.clip(I, 0, None)
    w = Ipos.sum()
    if w == 0:
        r, c = np.unravel_index(np.argmax(I), I.shape)
        return float(Qx[r, c]), float(Qz[r, c])
    qx_c = float((Qx * Ipos).sum() / w)
    qz_c = float((Qz * Ipos).sum() / w)
    return qx_c, qz_c

def parse_plane_slope(plane: str):
    """
    Minimal parser for plane like '103' -> slope m = l/h (z = m*(x - qx0) + qz0).
    Returns None for h=0 (vertical line).
    """
    s = plane.strip().replace("(", "").replace(")", "").replace(",", "")
    parts, num = [], ""
    for ch in s:
        if ch in "+-":
            if num: parts.append(num)
            num = ch
        else:
            num += ch
    if num: parts.append(num)
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0]); l = int(parts[-1])
    except ValueError:
        return None
    if h == 0:
        return None
    return l / h

def clip_line_to_axes(ax, qx0, qz0, m):
    """
    Build line segment within current axes for z = m*(x - qx0) + qz0.
    Returns (x1,z1,x2,z2) or None.
    """
    x_min, x_max = ax.get_xlim()
    z_min, z_max = ax.get_ylim()

    xs = [x_min, x_max]
    zs = [m * (x_min - qx0) + qz0, m * (x_max - qx0) + qz0]

    if m != 0:
        for z_edge in (z_min, z_max):
            x_int = qx0 + (z_edge - qz0) / m
            xs.append(x_int); zs.append(z_edge)

    pts = [(x, z) for x, z in zip(xs, zs)
           if (x_min - 1e-12) <= x <= (x_max + 1e-12)
           and (z_min - 1e-12) <= z <= (z_max + 1e-12)]
    if len(pts) < 2:
        return None

    uniq = []
    for p in pts:
        if not any(np.allclose(p, u, atol=1e-9) for u in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None

    dmax, pair = -1.0, None
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            d = (uniq[i][0]-uniq[j][0])**2 + (uniq[i][1]-uniq[j][1])**2
            if d > dmax:
                dmax, pair = d, (uniq[i], uniq[j])
    (x1, z1), (x2, z2) = pair
    return x1, z1, x2, z2
