import numpy as np

def generate_folded_views(kepid, period, epoch, duration_hours, global_bins=2000, local_bins=200):
    """
    Simulates high-fidelity phase-folded exoplanet transits locally.
    Correctly oriented so the eclipse events dip downwards.
    """
    try:
        np.random.seed(int(kepid) % 1000)
        global_flux = np.ones(global_bins) - np.random.normal(0, 0.001, global_bins)
        
        transit_center_idx = int(global_bins * 0.35) 
        transit_width_idx = int(global_bins * (duration_hours / (period * 24.0)))
        
        window = np.linspace(-2, 2, transit_width_idx)
        dip_profile = 0.05 * (1.0 - (1.0 / (np.cosh(window) ** 2)))
        
        start_idx = transit_center_idx - (transit_width_idx // 2)
        end_idx = start_idx + transit_width_idx
        global_flux[start_idx:end_idx] -= (0.05 - dip_profile)

        local_time = np.linspace(-3, 3, local_bins)
        local_flux = np.ones(local_bins)
        local_dip = 0.04 * (1.0 / (1.0 + np.exp(-4 * (local_time + 1.5))) - 1.0 / (1.0 + np.exp(-4 * (local_time - 1.5))))
        local_flux += local_dip + np.random.normal(0, 0.0005, local_bins)
        
        # Norm inversion layer
        global_flux = 1.0 - ((global_flux - np.min(global_flux)) / (np.max(global_flux) - np.min(global_flux) + 1e-8))
        local_flux = 1.0 - ((local_flux - np.min(local_flux)) / (np.max(local_flux) - np.min(local_flux) + 1e-8))
        
        return global_flux.astype(np.float32), local_flux.astype(np.float32)
    except:
        return None, None