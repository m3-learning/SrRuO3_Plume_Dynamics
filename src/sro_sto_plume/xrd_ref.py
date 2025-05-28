from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import matplotlib.pyplot as plt
import numpy as np

def calculate_2theta(hkl, abc, wavelength=1.5406):
    h, k, l = hkl
    a, b, c = abc
    
    # Calculate d-spacing for orthorhombic crystal
    d_hkl = 1 / np.sqrt((h**2 / a**2) + (k**2 / b**2) + (l**2 / c**2))
    
    # Apply Bragg's Law to find theta
    theta_rad = np.arcsin(wavelength / (2 * d_hkl))
    theta_deg = np.degrees(theta_rad)
    
    # Calculate 2theta
    two_theta = 2 * theta_deg
    return two_theta


def plot_ref_xrd(api_key: str, material_id: str, label: str):

    with MPRester(api_key=api_key) as mpr:
        # first retrieve the relevant structure
        structure = mpr.get_structure_by_material_id(material_id)

    # important to use the conventional structure to ensure
    # that peaks are labelled with the conventional Miller indices
    sga = SpacegroupAnalyzer(structure)
    conventional_structure = sga.get_conventional_standard_structure()

    # this example shows how to obtain an XRD diffraction pattern
    # these patterns are calculated on-the-fly from the structure
    xrd_calculator = XRDCalculator(wavelength="CuKa")

    # pattern = xrd_calculator.get_pattern(conventional_structure)
    # Generate the full XRD pattern for the structure
    pattern = xrd_calculator.get_pattern(conventional_structure, two_theta_range=(0, 90))
    # Extract data: two-theta values, intensities, and peaks
    two_theta = pattern.x  # Full 2θ scan values
    intensities = pattern.y  # Corresponding intensities
    peaks = pattern.hkls  # Peak positions with hkl information

    # Plot the full XRD pattern
    plt.figure(figsize=(8, 4))
    plt.plot(two_theta, intensities, label=label, linewidth=1.5)

    # Highlight individual peaks
    for peak, x, y in zip(peaks, pattern.x, pattern.y):
        hkl = peak[0]['hkl']  # Extract hkl values
        plt.text(x, y + 2, f"{hkl}\n{x:.2f}", ha='center', fontsize=8, rotation=90)  # Label peaks

    # Set axis labels and title
    plt.xlabel('2θ (degrees)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title('Simulated XRD Pattern with Peak Labels', fontsize=14)

    # Add legend and grid
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.tight_layout()
    plt.show()