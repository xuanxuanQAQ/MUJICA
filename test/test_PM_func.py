import simulation

htt, V, M_scale, P_scale, cover_size, second, Num, patch_size, mesh_size, deta_x = simulation.PM3D(V=4, P_scale=4, M_scale=4, fps=200)
simulation.save_data(htt, V, M_scale, P_scale, cover_size, second, Num, patch_size, mesh_size, deta_x)

print("Simulation completed successfully.")