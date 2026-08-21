# Voxelization Void Ratio Workflow

This folder contains scripts for computing voxel-based void ratio fields from a legacy ASCII VTK particle point cloud.

## Main Script

Use `voxelization.py` to read the particle positions and `radius` field, voxelize the point cloud, compute raw and face-reconstructed void ratio fields, and write them back to VTK.

Typical command:

```powershell
python .\voxelization.py .\prestress2800000.vtk -n 10 --sphere-clip-order 4
```

This creates the default particle-output file:

```text
prestress2800000_void_ratio.vtk
```

That file remains a `POLYDATA` point-cloud file. It stores the computed values as `POINT_DATA`, so ParaView colors the original particles rather than actual voxel cells.

## Voxel Cell Output

To also create a file that displays actual voxel cells in ParaView, add the `--voxel-output` parameter:

```powershell
python .\voxelization.py .\prestress2800000.vtk -n 10 --sphere-clip-order 4 --voxel-output .\prestress2800000_voxel_cells.vtk
```

The `--voxel-output` file is written as a legacy VTK `UNSTRUCTURED_GRID` with one `VTK_VOXEL` cell per occupied voxel. The fields are written as `CELL_DATA`:

```text
void_ratio
void_ratio_smoothed
particle_count
solid_volume
measurement_volume
```

Open the voxel-cell file in ParaView and use:

```text
Representation: Surface With Edges
Coloring: void_ratio_smoothed
```

This makes the voxel grid directly visible, which is useful for checking whether apparent spacing or banding is coming from the voxel field itself rather than from plotting point data on the original particles.

## Important Parameters

`-n`, `--particles-per-voxel`

Target average number of particles per occupied voxel. Larger values create larger voxels.

`--sphere-clip-order`

Gauss-Legendre quadrature order for estimating sphere/voxel intersection volumes. Default is `4`. Higher values are more accurate but slower.

`--void-field`

Name of the raw void-ratio field written to the output. Default:

```text
void_ratio
```

`--smoothed-void-field`

Name of the face-reconstructed Gauss field written to the output. Default:

```text
void_ratio_smoothed
```

`--voxel-output`

Optional path for an additional VTK file containing actual voxel cells. Use this when you want to inspect voxel geometry in ParaView.

`--allow-other-vtk`

Allows the script to run with VTK versions other than 7.1. Avoid `--vtk-writer` unless you specifically need VTK reader/writer behavior.

## Plotting Helper

Use `plot_void_ratio_contours.py` to generate a side-by-side PNG from a particle-output VTK file:

```powershell
python .\plot_void_ratio_contours.py .\prestress2800000_void_ratio.vtk
```

This plots `void_ratio` and `void_ratio_smoothed` from the point-cloud output. It is a convenient preview, but it is not a true voxel-cell visualization.
