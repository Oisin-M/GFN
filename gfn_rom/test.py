import numpy as np
import torch
from gfn_rom.preprocessing import undo_scaling

def print_results(Z, Z_net, x_enc, x_map):
    error_abs_list = list()
    norm_z_list = list()
    latents_error = list()

    for snap in range(Z.shape[1]):
        error_abs = np.linalg.norm(abs(Z[:, snap] - Z_net[:, snap]))
        norm_z = np.linalg.norm(Z[:, snap], 2)
        error_abs_list.append(error_abs)
        norm_z_list.append(norm_z)
        lat_err = np.linalg.norm(x_enc[snap] - x_map[snap])/np.linalg.norm(x_enc[snap])
        latents_error.append(lat_err)

    latents_error = np.array(latents_error)
    print("\nMaximum relative error for latent  = ", max(latents_error))
    print("Mean relative error for latent = ", sum(latents_error)/len(latents_error))
    print("Minimum relative error for latent = ", min(latents_error))

    error = np.array(error_abs_list)
    norm = np.array(norm_z_list)
    rel_error = error/norm
    print("\nMaximum absolute error for field "+" = ", max(error))
    print("Mean absolute error for field "+" = ", sum(error)/len(error))
    print("Minimum absolute error for field "+" = ", min(error))
    print("\nMaximum relative error for field "+" = ", max(rel_error))
    print("Mean relative error for field "+" = ", sum(rel_error)/len(rel_error))
    print("Minimum relative error for field "+" = ", min(rel_error))
    return error, rel_error

def evaluate_results(model, df_large, U_large, scale, params):
    with torch.no_grad():
        x_recon, x_enc, x_map, _, _  = model(U_large, df_large, params)
        x_rom = model.decoder(x_map)
        x_rom = x_rom@model.GFN.Wd_n + model.GFN.bd_n
        Z = undo_scaling(U_large, scale)
        Z_net = undo_scaling(x_rom, scale)
    return Z, Z_net, x_enc, x_map