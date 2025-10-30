# sro_sto_plume/crystallography.py
import math, re

def parse_hkl(plane: str):
    """
    Parse HKL from many formats ('103','(103)','1,0,3','h=1,k=0,l=3', etc.) -> (h,k,l) ints.
    """
    s = plane.strip().lower()
    s = re.sub(r'[\[\]\(\)\{\}]', ' ', s)      # remove brackets
    s = re.sub(r'[hklxyzabc]\s*=?', ' ', s)    # drop labels like h=, k=, l=
    s = s.replace(',', ' ')
    nums = re.findall(r'[+-]?\d+', s)
    if len(nums) == 3:
        return tuple(map(int, nums))
    compact = re.findall(r'[+-]?\d', s)
    if len(compact) == 3:
        return tuple(map(int, compact))
    raise ValueError(f"Could not parse HKL from '{plane}'. Provide e.g. '103' or '1,0,3'.")

def ideal_q_h0l(plane: str, a_bulk: float = 3.93):
    """
    Ideal (Qx, Qz) [Å⁻¹] for fully relaxed pseudo-cubic film in a Qy≈0 cut (h0l).
    k is ignored for the cut. (00l) -> Qx=0.
    """
    h, k, l = parse_hkl(plane)
    if h == 0:
        return 0.0, 2*math.pi*l/a_bulk
    return 2*math.pi*h/a_bulk, 2*math.pi*l/a_bulk
