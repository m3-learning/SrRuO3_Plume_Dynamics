"""Lightweight helpers for Zygo/NewView-style `.datx` files and processed maps.

Example:
    >>> from sro_sto_plume import microscopy
    >>> height_map, geom = microscopy.load_processed_surface("G1.datx", "G1.txt")
    >>> stats = microscopy.compute_roughness(height_map, geometry=geom, unit="um")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Sequence
import math
import zipfile
import numpy as np

try:
    import h5py  # Most .datx files are HDF5 containers
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False


# -----------------------------
# Core HDF5 helpers
# -----------------------------
def _h5_collect_datasets(h5: "h5py.File") -> Dict[str, np.ndarray]:
    """Collect all datasets as NumPy arrays from an HDF5 file handle."""
    out: Dict[str, np.ndarray] = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                out[name] = np.asarray(obj[()])
            except Exception:
                # skip unreadable datasets
                pass

    h5.visititems(visitor)
    return out


def _pick_primary(datasets: Dict[str, np.ndarray],
                  prefer_2d: bool = True) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Choose a 'primary' dataset. By default, pick the largest 2D array.
    If no 2D arrays exist, pick the largest by total size.
    """
    primary_name: Optional[str] = None
    primary_arr: Optional[np.ndarray] = None
    best_size = -1

    if prefer_2d:
        # First pass: only 2D arrays
        for k, v in datasets.items():
            if v.ndim == 2:
                size = v.size
                if size > best_size:
                    best_size = size
                    primary_name, primary_arr = k, v

    if primary_arr is None:
        # Fallback: any shape
        for k, v in datasets.items():
            size = v.size
            if size > best_size:
                best_size = size
                primary_name, primary_arr = k, v

    return primary_name, primary_arr


# -----------------------------
# Public API
# -----------------------------
def load_datx(path: str,
              name_contains: Optional[str] = None,
              prefer_2d: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a .datx file.

    Returns:
        data_dict: {
            "primary": np.ndarray | None,
            "all": Dict[str, np.ndarray],   # all datasets (if HDF5)
            "xml": Dict[str, str]           # any XML/TXT/JSON (if ZIP-style)
        }
        meta: {
            "format": "datx",
            "primary_dataset": str | None,
            "h5_datasets": List[str] (if HDF5),
            "zip_members": List[str] (if ZIP),
            "notes": str (optional diagnostic)
        }

    Args:
        path: path to .datx
        name_contains: if provided, try to pick primary by partial name match
                       (e.g., "Height", "Phase", "Intensity")
        prefer_2d: if True, prefer the largest 2D array as primary
    """
    meta: Dict[str, Any] = {"format": "datx"}
    data_dict: Dict[str, Any] = {"primary": None, "all": {}, "xml": {}}

    # 1) Try HDF5 (typical for Zygo/NewView)
    if _HAS_H5PY:
        try:
            with h5py.File(path, "r") as h5:
                datasets = _h5_collect_datasets(h5)
                data_dict["all"] = datasets
                meta["h5_datasets"] = list(datasets.keys())

                primary_name = None
                primary = None

                # If user asked for a dataset name substring, try that first
                if name_contains:
                    name_lower = name_contains.lower()
                    for k, v in datasets.items():
                        if name_lower in k.lower():
                            primary_name, primary = k, v
                            break

                # Otherwise pick a reasonable default primary
                if primary is None:
                    primary_name, primary = _pick_primary(datasets, prefer_2d=prefer_2d)

                data_dict["primary"] = primary
                meta["primary_dataset"] = primary_name

                return data_dict, meta
        except Exception as e:
            meta["notes"] = f"HDF5 read failed: {e!r}"

    # 2) ZIP-style (some vendors embed XML + binary)
    #    We'll at least expose XML/TXT/JSON as metadata.
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            meta["zip_members"] = names
            for nm in names:
                if nm.lower().endswith((".xml", ".json", ".txt")):
                    with zf.open(nm) as fh:
                        try:
                            data_dict["xml"][nm] = fh.read().decode("utf-8", errors="ignore")
                        except Exception:
                            pass
            meta["primary_dataset"] = None
            return data_dict, meta
    except Exception as e:
        meta["notes"] = meta.get("notes", "") + f" ZIP read failed: {e!r}"

    # 3) Nothing worked
    raise RuntimeError(
        "Unsupported .datx structure. "
        "Install h5py for HDF5-based .datx or provide a sample for a custom reader."
    )


def list_dataset_shapes(data_dict: Dict[str, Any]) -> List[Tuple[str, tuple, np.dtype]]:
    """
    Utility to list dataset names, shapes, and dtypes from the loaded .datx.
    Returns a list of (name, shape, dtype).
    """
    items: List[Tuple[str, tuple, np.dtype]] = []
    for k, v in data_dict.get("all", {}).items():
        if isinstance(v, np.ndarray):
            items.append((k, v.shape, v.dtype))
    return items


def select_dataset(data_dict: Dict[str, Any], name_contains: str) -> Optional[np.ndarray]:
    """
    Return the first dataset whose path contains the given substring (case-insensitive).
    """
    target = name_contains.lower()
    for k, v in data_dict.get("all", {}).items():
        if target in k.lower():
            return v
    return None


# -----------------------------
# Processed-surface utilities
# -----------------------------
_UNIT_TO_METERS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
_DATX_HEIGHT_UNIT_TO_METERS = {
    "meters": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "micrometers": 1e-6,
    "micrometer": 1e-6,
    "micrometres": 1e-6,
    "micrometre": 1e-6,
    "microns": 1e-6,
    "micron": 1e-6,
    "nanometers": 1e-9,
    "nanometer": 1e-9,
    "nanometres": 1e-9,
    "nanometre": 1e-9,
    "millimeters": 1e-3,
    "millimeter": 1e-3,
    "millimetres": 1e-3,
    "millimetre": 1e-3,
}


def _require_h5py():
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required for this operation")


def _unit_scale(unit: str) -> float:
    try:
        return _UNIT_TO_METERS[unit]
    except KeyError as exc:  # pragma: no cover - minimal mapping
        raise ValueError(f"Unsupported unit '{unit}'. Valid options: {sorted(_UNIT_TO_METERS)}") from exc


def _iter_datasets(group: "h5py.Group"):
    for obj in group.values():
        if isinstance(obj, h5py.Dataset):
            yield obj
        elif isinstance(obj, h5py.Group):
            yield from _iter_datasets(obj)


def _polygon_mask(geometry: "ScanGeometry", polygon_m: np.ndarray) -> np.ndarray:
    """Return a boolean mask for points inside the polygon (in meters)."""

    from matplotlib.path import Path

    height_px = geometry.height_px
    width_px = geometry.width_px

    x_centers = (np.arange(width_px, dtype=float) + 0.5) * geometry.pixel_size_x_m
    y_centers = (np.arange(height_px, dtype=float) + 0.5) * geometry.pixel_size_y_m
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    path = Path(polygon_m)
    mask_flat = path.contains_points(points, radius=-1e-12)
    return mask_flat.reshape(height_px, width_px)


def _extract_step_from_converter(attr: Any) -> Optional[float]:
    if attr is None:
        return None
    try:
        params = attr["Parameters"]
    except Exception:
        try:
            params = attr[0][-1]
        except Exception:
            return None
        else:
            params = np.asarray(params)
            return float(params[1]) if params.size >= 2 else None

    if not isinstance(params, np.ndarray) or params.size == 0:
        return None

    arr = np.asarray(params[0])
    if arr.size >= 2:
        return float(arr[1])
    return None


def _search_attribute_value(root: "h5py.Group", needle: str) -> Optional[float]:
    if root is None:
        return None

    needle_lower = needle.lower()
    queue: List[Any] = [root]

    while queue:
        obj = queue.pop()
        for name, value in getattr(obj, "attrs", {}).items():
            if needle_lower in name.lower():
                try:
                    arr = np.asarray(value)
                    if arr.size:
                        return float(arr.flat[0])
                except Exception:
                    continue
        if isinstance(obj, h5py.Group):
            queue.extend(obj.values())

    return None


@dataclass(frozen=True)
class ScanGeometry:
    width_px: int
    height_px: int
    pixel_size_x_m: float
    pixel_size_y_m: float

    def size(self, unit: str = "m") -> Tuple[float, float]:
        scale = _unit_scale(unit)
        return (
            self.width_px * self.pixel_size_x_m / scale,
            self.height_px * self.pixel_size_y_m / scale,
        )

    def pixel_size(self, unit: str = "m") -> Tuple[float, float]:
        scale = _unit_scale(unit)
        return (
            self.pixel_size_x_m / scale,
            self.pixel_size_y_m / scale,
        )

    def extent(self, unit: str = "m") -> Tuple[float, float, float, float]:
        size_x, size_y = self.size(unit)
        return (0.0, size_x, 0.0, size_y)


def extract_scan_geometry(datx_path: str,
                          dataset_keywords: Sequence[str] = ("Surface", "Height")) -> ScanGeometry:
    """Infer lateral pixel spacing from a .datx file."""

    _require_h5py()
    path = Path(datx_path)
    if not path.exists():
        raise FileNotFoundError(f"No such .datx file: {path}")

    with h5py.File(path, "r") as h5:
        data_group = h5.get("Data")
        if data_group is None:
            raise RuntimeError("Unexpected .datx structure: missing 'Data' group")

        datasets = list(_iter_datasets(data_group))
        if not datasets:
            raise RuntimeError("No datasets found inside the 'Data' group")

        chosen = None
        keywords_lower = [kw.lower() for kw in dataset_keywords]
        for ds in datasets:
            name_lower = ds.name.lower()
            if any(kw in name_lower for kw in keywords_lower):
                chosen = ds
                break

        if chosen is None:
            chosen = datasets[0]

        height_px, width_px = chosen.shape

        step_x = _extract_step_from_converter(chosen.attrs.get("X Converter"))
        step_y = _extract_step_from_converter(chosen.attrs.get("Y Converter"))

        if step_x is None or step_y is None:
            attr_group = h5.get("Attributes")
            lateral = _search_attribute_value(attr_group, "Surface Data Context.Lateral Resolution:Value")
            if lateral is not None:
                step_x = step_x or lateral
                step_y = step_y or lateral

        if step_x is None or step_y is None:
            raise RuntimeError("Could not infer lateral resolution from the .datx file")

        return ScanGeometry(width_px=width_px,
                             height_px=height_px,
                             pixel_size_x_m=float(step_x),
                             pixel_size_y_m=float(step_y))


def load_height_txt(path: str,
                    scale: float = 1.0,
                    delimiter: Optional[str] = None,
                    dtype: Any = float) -> np.ndarray:
    """Load a flattened height map exported as a plain-text grid."""

    arr = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
    if scale != 1.0:
        arr = arr * scale
    return arr


def load_processed_surface(datx_path: str, txt_path: str,
                           **txt_kwargs: Any) -> Tuple[np.ndarray, ScanGeometry]:
    """Convenience helper: txt-derived height map + lateral geometry from .datx."""

    geometry = extract_scan_geometry(datx_path)
    height_map = load_height_txt(txt_path, **txt_kwargs)
    if height_map.shape != (geometry.height_px, geometry.width_px):
        raise ValueError(
            "Height map shape does not match .datx metadata: "
            f"got {height_map.shape}, expected {(geometry.height_px, geometry.width_px)}"
        )
    return height_map, geometry


def _normalize_corners(corners: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    if not corners:
        raise ValueError("Corners must contain at least two coordinate pairs")

    arr = np.asarray(corners, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Corners should be an iterable of (x, y) pairs")

    x_min = float(np.min(arr[:, 0]))
    x_max = float(np.max(arr[:, 0]))
    y_min = float(np.min(arr[:, 1]))
    y_max = float(np.max(arr[:, 1]))
    if not math.isfinite(x_min) or not math.isfinite(x_max) or not math.isfinite(y_min) or not math.isfinite(y_max):
        raise ValueError("Corner coordinates must be finite numbers")
    if x_min == x_max or y_min == y_max:
        raise ValueError("Corner coordinates span zero area")

    return x_min, x_max, y_min, y_max


def _region_slices_from_corners(geometry: ScanGeometry,
                                corners: Sequence[Sequence[float]],
                                unit: str) -> Tuple[slice, slice]:
    x_min, x_max, y_min, y_max = _normalize_corners(corners)
    scale = _unit_scale(unit)

    px_size_x = geometry.pixel_size_x_m
    px_size_y = geometry.pixel_size_y_m

    def clamp(val: float, limit: int) -> int:
        return int(min(max(val, 0), limit))

    col_start = clamp(math.floor((min(x_min, x_max) * scale) / px_size_x), geometry.width_px)
    col_stop = clamp(math.ceil((max(x_min, x_max) * scale) / px_size_x), geometry.width_px)
    row_start = clamp(math.floor((min(y_min, y_max) * scale) / px_size_y), geometry.height_px)
    row_stop = clamp(math.ceil((max(y_min, y_max) * scale) / px_size_y), geometry.height_px)

    if col_start == col_stop or row_start == row_stop:
        raise ValueError("Corner selection collapses to an empty pixel region")

    return slice(row_start, row_stop), slice(col_start, col_stop)


def compute_roughness(height_map: np.ndarray,
                      geometry: Optional[ScanGeometry] = None,
                      *,
                      corners: Optional[Sequence[Sequence[float]]] = None,
                      pixel_region: Optional[Tuple[slice, slice]] = None,
                      unit: str = "um",
                      detrend: bool = False) -> Dict[str, Any]:
    """Compute basic surface roughness metrics (Ra, Rq, Rz).

    Args:
        height_map: 2D array of heights in meters.
        geometry: Optional lateral geometry (required if `corners` is given).
        corners: Iterable of (x, y) pairs describing a rectangular region in `unit` coordinates.
        pixel_region: Alternate way to specify a region using pixel slices (row_slice, col_slice).
        unit: Desired output unit for the statistics (default: micrometers).
        detrend: If ``True``, remove a best-fit plane from the selected region
            before computing roughness metrics.
    """

    if corners is not None and geometry is None:
        raise ValueError("Geometry must be provided when selecting by coordinates")

    roi = np.asarray(height_map, dtype=float)

    region_slices: Optional[Tuple[slice, slice]] = None
    if corners is not None:
        region_slices = _region_slices_from_corners(geometry, corners, unit)
    elif pixel_region is not None:
        region_slices = pixel_region

    if region_slices is not None:
        row_slice, col_slice = region_slices
        roi = roi[row_slice, col_slice]

    if detrend:
        roi = _detrend_plane(roi)

    finite = roi[np.isfinite(roi)]
    if finite.size == 0:
        raise ValueError("Selected region does not contain finite data")

    mean_m = float(np.mean(finite))
    centered = finite - mean_m
    ra_m = float(np.mean(np.abs(centered)))
    rq_m = float(np.sqrt(np.mean(centered ** 2)))
    rz_m = float(np.max(finite) - np.min(finite))

    scale = _unit_scale(unit)

    result: Dict[str, Any] = {
        "unit": unit,
        "count": int(finite.size),
        "mean": mean_m / scale,
        "Ra": ra_m / scale,
        "Rq": rq_m / scale,
        "Rz": rz_m / scale,
    }

    if region_slices is not None and geometry is not None:
        row_slice, col_slice = region_slices
        size_x_m = (col_slice.stop - col_slice.start) * geometry.pixel_size_x_m
        size_y_m = (row_slice.stop - row_slice.start) * geometry.pixel_size_y_m
        result["area"] = (
            size_x_m / scale,
            size_y_m / scale,
        )

    return result


def compute_roughness_polygon(height_map: np.ndarray,
                              geometry: ScanGeometry,
                              polygon: Sequence[Sequence[float]],
                              *,
                              polygon_unit: str = "um",
                              unit: str = "um",
                              detrend: bool = False) -> Dict[str, Any]:
    """Compute roughness metrics within an arbitrary polygonal region.

    Args:
        height_map: Full 2D height map in meters.
        geometry: Scan metadata describing pixel spacing.
        polygon: ROI described by (x, y) coordinate pairs.
        polygon_unit: Unit of the polygon coordinates.
        unit: Desired output unit for the statistics.
        detrend: If ``True``, subtract a best-fit plane from the polygonal
            region prior to computing the metrics.
    """

    if geometry is None:
        raise ValueError("Geometry is required for polygon-based roughness")

    coords = np.asarray(polygon, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 3:
        raise ValueError("Polygon must be an iterable of at least three (x, y) pairs")

    poly_scale = _unit_scale(polygon_unit)
    polygon_m = coords * poly_scale

    mask = _polygon_mask(geometry, polygon_m)
    if not np.any(mask):
        raise ValueError("Polygon selection does not overlap the scan area")

    data = np.asarray(height_map, dtype=float)
    if detrend:
        data = _detrend_plane(data, mask=mask)

    roi = data[mask]
    finite = roi[np.isfinite(roi)]
    if finite.size == 0:
        raise ValueError("Polygon region does not contain finite data")

    mean_m = float(np.mean(finite))
    centered = finite - mean_m
    ra_m = float(np.mean(np.abs(centered)))
    rq_m = float(np.sqrt(np.mean(centered ** 2)))
    rz_m = float(np.max(finite) - np.min(finite))

    scale = _unit_scale(unit)
    area_m2 = mask.sum() * geometry.pixel_size_x_m * geometry.pixel_size_y_m
    poly_scale_sq = _unit_scale(polygon_unit) ** 2

    return {
        "unit": unit,
        "polygon_unit": polygon_unit,
        "count": int(finite.size),
        "mask_pixels": int(mask.sum()),
        "mean": mean_m / scale,
        "Ra": ra_m / scale,
        "Rq": rq_m / scale,
        "Rz": rz_m / scale,
        "area": area_m2 / poly_scale_sq,
        "polygon": coords.tolist(),
    }


def plot_height_map(height_map: np.ndarray,
                     geometry: ScanGeometry,
                     *,
                     xy_unit: str = "um",
                     z_unit: str = "um",
                     cmap: str = "viridis",
                     clip_percentile: Optional[float] = None,
                     ax: Any = None,
                     polygons: Optional[Sequence[Sequence[Sequence[float]]]] = None,
                     polygon_unit: Optional[str] = None,
                     polygon_style: Optional[Dict[str, Any]] = None,
                     polygon_fill_alpha: Optional[float] = 0.15):
    """Visualize a height map with physical scaling."""

    import matplotlib.pyplot as plt

    roi = np.asarray(height_map, dtype=float)
    z_scale = _unit_scale(z_unit)
    display = roi / z_scale

    extent = geometry.extent(xy_unit)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:  # pragma: no cover - plotting convenience
        fig = ax.figure

    im = ax.imshow(display, extent=extent, origin="lower", cmap=cmap, aspect="auto")

    if clip_percentile is not None:
        lower, upper = np.nanpercentile(display, [clip_percentile, 100 - clip_percentile])
        im.set_clim(lower, upper)

    cbar = fig.colorbar(im, ax=ax)
    ax.set_xlabel(f"x ({xy_unit})")
    ax.set_ylabel(f"y ({xy_unit})")
    cbar.set_label(f"Height ({z_unit})")
    ax.set_title("Surface Topography")

    if polygons:
        style = {"color": "red", "linewidth": 2}
        if polygon_style:
            style.update(polygon_style)

        display_scale = _unit_scale(xy_unit)
        if polygon_unit is None:
            polygon_scale = display_scale
        else:
            polygon_scale = _unit_scale(polygon_unit)

        for poly in polygons:
            coords = np.asarray(poly, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("Each polygon must be an iterable of (x, y) pairs")

            coords_disp = coords * (polygon_scale / display_scale)
            xs = np.append(coords_disp[:, 0], coords_disp[0, 0])
            ys = np.append(coords_disp[:, 1], coords_disp[0, 1])
            ax.plot(xs, ys, **style)

            if polygon_fill_alpha and polygon_fill_alpha > 0:
                facecolor = style.get("color", "red")
                ax.fill(coords_disp[:, 0], coords_disp[:, 1],
                        facecolor=facecolor, alpha=polygon_fill_alpha, linewidth=0)

    return ax


# -----------------------------
# High-level batch helpers
# -----------------------------
def analyze_surfaces(
    surfaces: Sequence[Dict[str, Any]],
    *,
    polygon_unit: str = "um",
    roughness_unit: str = "um",
    xy_unit: str = "um",
    z_unit: str = "um",
    cmap: str = "viridis",
    clip_percentile: Optional[float] = None,
    max_cols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    detrend: bool = False,
) -> Tuple[List[Dict[str, Any]], Any]:
    """Process multiple surfaces and plot them as subplots with polygon overlays."""

    import matplotlib.pyplot as plt

    if not surfaces:
        raise ValueError("No surfaces provided for analysis")
    if max_cols <= 0:
        raise ValueError("max_cols must be a positive integer")

    results: List[Dict[str, Any]] = []
    prepared: List[Tuple[np.ndarray, ScanGeometry, Sequence[Sequence[float]], Dict[str, Any]]] = []

    for item in surfaces:
        datx_path = Path(item["datx"]).expanduser()
        txt_path = Path(item["txt"]).expanduser()
        polygon = item.get("polygon")
        if polygon is None:
            raise ValueError("Each surface dict must include a 'polygon' entry")

        label = item.get("label")
        if label is None:
            label = datx_path.stem

        height_map, geom = load_processed_surface(str(datx_path), str(txt_path))
        stats = compute_roughness_polygon(
            height_map,
            geometry=geom,
            polygon=polygon,
            polygon_unit=polygon_unit,
            unit=roughness_unit,
            detrend=detrend,
        )
        stats.update({
            "label": label,
            "detrended": detrend,
            # "datx": str(datx_path),
            # "txt": str(txt_path),
        })
        results.append(stats)
        prepared.append((height_map, geom, polygon, stats))

    n = len(prepared)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    if figsize is None:
        figsize = (cols * 4.5, rows * 4.0)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for ax, (height_map, geom, polygon, stats) in zip(axes_arr, prepared):
        plot_height_map(
            height_map,
            geom,
            ax=ax,
            xy_unit=xy_unit,
            z_unit=z_unit,
            cmap=cmap,
            clip_percentile=clip_percentile,
            polygons=[polygon],
            polygon_unit=polygon_unit,
            polygon_style={"color": "red", "linewidth": 1.5},
            polygon_fill_alpha=0.05,
        )
        ax.set_title(
            f"{stats['label']}\nRq = {stats['Rq']:.3f} {roughness_unit}",
            fontsize=10,
        )

    for ax in axes_arr[n:]:
        ax.axis("off")
        ax.tick_params(direction="in")

    fig.tight_layout()
    return results, fig


def build_surfaces(sample_map: Dict[str, Sequence],
                   area: Optional[str] = None,
                   data_root: Path | str = "data") -> List[Dict[str, Any]]:
    """Convert a mapping of ROI definitions into ``analyze_surfaces`` inputs.

    Args:
        sample_map: mapping of scan/area names to iterables of ROIs. Each ROI
            may be provided as ``(polygon, label)`` tuple, a mapping with
            ``"polygon"``/``"label"`` keys, or directly as a polygon iterable.
        area: optional area key to restrict the output.
        data_root: base directory containing the ``.datx`` and ``.txt`` files
            for each area.
    """

    if not sample_map:
        raise ValueError("sample_map must contain at least one area definition")

    if area is not None and area not in sample_map:
        raise KeyError(f"Area '{area}' not found in sample_map")

    root_path = Path(data_root)
    items = [(area, sample_map[area])] if area else sample_map.items()
    surfaces: List[Dict[str, Any]] = []

    for area_name, rois in items:
        for idx, entry in enumerate(rois, start=1):
            polygon: Any
            label: str

            if isinstance(entry, tuple):
                polygon, label = entry
            elif isinstance(entry, dict):
                polygon = entry.get("polygon")
                if polygon is None:
                    raise ValueError(f"ROI entry for '{area_name}' is missing 'polygon'")
                label = entry.get("label", f"{area_name}-ROI{idx}")
            else:
                polygon, label = entry, f"{area_name}-ROI{idx}"

            if not label.startswith(area_name):
                label = f"{area_name}-{label}"

            surfaces.append({
                "area": area_name,
                "label": label,
                "datx": str(root_path / f"{area_name}.datx"),
                "txt": str(root_path / f"{area_name}.txt"),
                "polygon": polygon,
            })

    return surfaces


# -----------------------------
# Surface preprocessing helpers
# -----------------------------
def _detrend_plane(data: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Subtract the best-fit plane from ``data``.

    Args:
        data: 2D height map.
        mask: optional boolean mask selecting the region to use for the fit.

    Returns:
        Copy of ``data`` with the fitted plane removed. Values outside ``mask``
        (if provided) are returned as ``np.nan``.
    """

    arr = np.asarray(data, dtype=float)
    result = arr.copy()

    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask

    if not np.any(valid):
        if mask is not None:
            result[~mask] = np.nan
        return result

    yy, xx = np.indices(arr.shape)
    A = np.column_stack((xx[valid], yy[valid], np.ones(valid.sum())))
    b = arr[valid]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    result = arr - plane

    if mask is not None:
        result = np.where(mask, result, np.nan)

    return result


# -----------------------------
# Batch roughness helpers
# -----------------------------
def load_datx_height_map(path: Path | str,
                         dataset_keyword: str = "Surface") -> np.ndarray:
    """Load a 2D height map from a .datx file and return it in meters."""

    if not _HAS_H5PY:
        raise RuntimeError("h5py is required to read datx files")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such datx file: {path}")

    with h5py.File(path, "r") as h5:
        matches: List[Tuple[str, int]] = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                score = obj.size
                if dataset_keyword.lower() in name.lower():
                    score += obj.size  # prefer keyword matches
                matches.append((name, score))

        h5.visititems(visitor)

        if not matches:
            raise ValueError(f"No 2D datasets found in {path}")

        matches.sort(key=lambda item: item[1], reverse=True)
        dataset_name = matches[0][0]
        ds = h5[dataset_name]
        data = np.asarray(ds[()], dtype=float)

        unit_attr = ds.attrs.get("Unit")
        unit_str = _decode_attribute_string(unit_attr)
        scale = None
        if unit_str:
            scale = _DATX_HEIGHT_UNIT_TO_METERS.get(unit_str.lower())

        if scale is None:
            z_converter = ds.attrs.get("Z Converter")
            if z_converter is not None:
                try:
                    params = z_converter[0][-1]
                    params = np.asarray(params)
                    if params.size >= 2:
                        offset, factor = params[:2]
                        data = offset + data * factor
                        scale = 1.0
                except Exception:
                    pass

        if scale is None:
            raise ValueError(f"Unsupported or missing vertical unit for dataset '{dataset_name}' in {path}")

        return data * scale


def _decode_attribute_string(attr: Any) -> Optional[str]:
    if attr is None:
        return None
    if isinstance(attr, bytes):
        return attr.decode("utf-8", errors="ignore")
    if isinstance(attr, str):
        return attr
    if isinstance(attr, np.ndarray) and attr.size > 0:
        value = attr.flat[0]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value
    return None


def _roughness_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute mean, Ra, Rq, Rz for a 1D array of height samples."""

    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Input contains no finite samples")

    mean_val = float(np.mean(finite))
    centered = finite - mean_val
    ra_val = float(np.mean(np.abs(centered)))
    rq_val = float(np.sqrt(np.mean(centered ** 2)))
    rz_val = float(np.max(finite) - np.min(finite))

    return {
        "count": int(finite.size),
        "mean": mean_val,
        "Ra": ra_val,
        "Rq": rq_val,
        "Rz": rz_val,
    }


def batch_roughness_from_maps(height_maps: Sequence[np.ndarray],
                              *,
                              paths: Optional[Sequence[str]] = None,
                              detrend: bool = False,
                              zero_center: bool = False) -> Dict[str, Any]:
    """Aggregate roughness metrics across multiple height maps.

    Args:
        height_maps: Sequence of 2D arrays in meters.
        paths: Optional list of source paths (for reporting).
        detrend: Remove a best-fit plane from each map.
        zero_center: Subtract the mean of each map (after detrending) so focus
            offsets do not inflate pooled roughness.
    """

    if not height_maps:
        raise ValueError("height_maps must contain at least one array")

    if paths is not None and len(paths) != len(height_maps):
        raise ValueError("Length of 'paths' must match number of height maps")

    per_image: List[Dict[str, Any]] = []
    flattened: List[np.ndarray] = []
    image_means: List[float] = []

    for idx, arr in enumerate(height_maps):
        data = np.asarray(arr, dtype=float)
            
        # Sanitize now so everything downstream only sees finite numbers
        data = data.copy()
        data[~np.isfinite(data)] = np.nan

        if detrend:
            data = _detrend_plane(data)

        raw_mean = np.nanmean(data)

        if zero_center and np.isfinite(raw_mean):
            data = data - raw_mean

        stats = _roughness_stats(data)
        stats["detrended"] = detrend
        stats["zero_center"] = zero_center
        if paths is not None:
            stats["source"] = paths[idx]
        per_image.append(stats)

        flattened.append(data.ravel())
        if np.isfinite(raw_mean):
            image_means.append(raw_mean)

    all_pixels = np.concatenate([vals[np.isfinite(vals)] for vals in flattened])
    global_stats = _roughness_stats(all_pixels)
    global_stats["detrended"] = detrend
    global_stats["zero_center"] = zero_center

    if image_means:
        image_means_stats = _roughness_stats(np.asarray(image_means, dtype=float))
        image_means_stats["detrended"] = detrend
        image_means_stats["zero_center"] = zero_center
    else:
        image_means_stats = {
            "count": 0,
            "mean": float("nan"),
            "Ra": float("nan"),
            "Rq": float("nan"),
            "Rz": float("nan"),
            "detrended": detrend,
            "zero_center": zero_center,
        }

    return {
        "global_pixels": global_stats,
        "per_image": per_image,
        "image_mean_variation": image_means_stats,
    }


def batch_roughness_from_txt(paths: Sequence[Path | str],
                             *,
                             detrend: bool = False,
                             zero_center: bool = False,
                             delimiter: Optional[str] = None,
                             dtype: Any = float) -> Dict[str, Any]:
    """Load multiple text-based height maps and compute aggregate roughness."""

    if not paths:
        raise ValueError("paths must contain at least one file")

    maps: List[np.ndarray] = []
    str_paths: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        maps.append(load_height_txt(str(path), delimiter=delimiter, dtype=dtype))
        str_paths.append(str(path))

    return batch_roughness_from_maps(maps, paths=str_paths,
                                     detrend=detrend, zero_center=zero_center)


def batch_roughness_from_datx(paths: Sequence[Path | str],
                              *,
                              dataset_keyword: str = "Surface",
                              detrend: bool = False,
                              zero_center: bool = False) -> Dict[str, Any]:
    """Load multiple .datx files and compute aggregate roughness."""

    if not paths:
        raise ValueError("paths must contain at least one datx file")

    maps: List[np.ndarray] = []
    str_paths: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        maps.append(load_datx_height_map(path, dataset_keyword=dataset_keyword))
        str_paths.append(str(path))

    return batch_roughness_from_maps(maps, paths=str_paths,
                                     detrend=detrend, zero_center=zero_center)


# -----------------------------
# Visualization helpers
# -----------------------------
def imshow_2d(arr: np.ndarray, title: str = "Height/Intensity Map"):
    """
    Show a 2D array with matplotlib.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("imshow_2d expects a 2D array")

    plt.figure()
    plt.imshow(arr)
    plt.title(title)
    plt.colorbar(label="value")
    plt.xlabel("x (col)")
    plt.ylabel("y (row)")
    plt.show()
