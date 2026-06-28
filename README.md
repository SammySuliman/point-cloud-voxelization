To add void ratio file to an existing vtk:
python .\voxelization.py .\name_of_original_vtk.vtk -n [number of desired voxels]

Example:
python .\voxelization.py .\prestress2800000.vtk -n 10

Additional arguments: \\
--void-field \
Name of the raw void-ratio output field. Default: void_ratio \\
--smoothed-void-field \
Name of the Gaussian-smoothed void-ratio output field. Default: void_ratio_smoothed \\
--smooth-sigma \
Gaussian smoothing sigma in voxel units. Default: 1.0. Use 0 to effectively disable smoothing. \\
--negative-n-search-limit \
If negative raw void ratios occur, search integer N values above current N up to this limit and report the first one with no negatives. Default: 100. Use 0 to disable the search. \\
--binary \
Write binary legacy VTK. Only works with --vtk-writer. \\
--allow-other-vtk \
Allow running with VTK versions other than 7.1. \\
--vtk-writer \
Use VTK reader/writer instead of the default legacy ASCII append mode. I do not recommend this for your current file because VTK 7.1 was not loading all arrays reliably.
