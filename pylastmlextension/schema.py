""" Pre-defined schema for pylast datalevels"""

SCHEMA_DICT = {
    # DL2 fields
    "rec_impact_parameter": {
        "level": "dl2",
        "description": "Reconstructed impact distance from telescope to shower axis",
    },
    "tel_rec_energy": {"level": "dl2", "description": "Telescope-level reconstructed energy"},
    "rec_energy": {
        "level": "dl2",
        "description": "Event-level reconstructed energy from ML energy reconstructor",
    },
    # Simulation fields - Hillas parameters
    "hillas_length": {
        "level": "simulation",
        "description": "Length of the shower ellipse (major axis)",
    },
    "hillas_width": {
        "level": "simulation",
        "description": "Width of the shower ellipse (minor axis)",
    },
    "hillas_shape": {
        "level": "simulation",
        "description": "Shape parameter: ratio of width to length",
    },
    "hillas_psi": {"level": "simulation", "description": "Orientation angle of the shower ellipse"},
    "hillas_x": {
        "level": "simulation",
        "description": "X coordinate of the shower center of gravity",
    },
    "hillas_y": {
        "level": "simulation",
        "description": "Y coordinate of the shower center of gravity",
    },
    "hillas_skewness": {
        "level": "simulation",
        "description": "Skewness of the charge distribution along the major axis",
    },
    "hillas_kurtosis": {
        "level": "simulation",
        "description": "Kurtosis of the charge distribution along the major axis",
    },
    "hillas_intensity": {
        "level": "simulation",
        "description": "Total integrated charge in the shower image",
    },
    "log_intensity": {
        "level": "simulation",
        "description": "Logarithm of the total integrated charge in the shower image",
    },
    "hillas_r": {
        "level": "simulation",
        "description": "Radial distance from camera center to shower CoG",
    },
    "hillas_phi": {
        "level": "simulation",
        "description": "Azimuthal angle of shower CoG in camera coordinates",
    },
    # Simulation fields - Leakage parameters
    "leakage_pixels_width_1": {
        "level": "simulation",
        "description": "Number of pixels in first border ring",
    },
    "leakage_pixels_width_2": {
        "level": "simulation",
        "description": "Number of pixels in second border ring",
    },
    "leakage_intensity_width_1": {
        "level": "simulation",
        "description": "Fraction of intensity in first border ring",
    },
    "leakage_intensity_width_2": {
        "level": "simulation",
        "description": "Fraction of intensity in second border ring",
    },
    # Simulation fields - Concentration parameters
    "concentration_cog": {
        "level": "simulation",
        "description": "Concentration around center of gravity",
    },
    "concentration_core": {"level": "simulation", "description": "Concentration in core pixels"},
    "concentration_pixel": {
        "level": "simulation",
        "description": "Concentration in brightest pixel",
    },
    # Simulation fields - Morphology parameters
    "morphology_n_pixels": {
        "level": "simulation",
        "description": "Total number of pixels in the image",
    },
    "morphology_n_islands": {
        "level": "simulation",
        "description": "Total number of islands (connected pixel groups)",
    },
    "morphology_n_small_islands": {"level": "simulation", "description": "Number of small islands"},
    "morphology_n_medium_islands": {
        "level": "simulation",
        "description": "Number of medium-sized islands",
    },
    "morphology_n_large_islands": {"level": "simulation", "description": "Number of large islands"},
    # Simulation fields - Intensity statistics
    "intensity_max": {"level": "simulation", "description": "Maximum pixel intensity in the image"},
    "intensity_mean": {"level": "simulation", "description": "Mean pixel intensity in the image"},
    "intensity_std": {
        "level": "simulation",
        "description": "Standard deviation of pixel intensities",
    },
    "intensity_skewness": {
        "level": "simulation",
        "description": "Skewness of pixel intensity distribution",
    },
    "intensity_kurtosis": {
        "level": "simulation",
        "description": "Kurtosis of pixel intensity distribution",
    },
}
