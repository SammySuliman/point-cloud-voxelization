from pathlib import Path

source = Path(
    r"C:\Users\sammy\Desktop\voxelization"
    r"\prestress2800000_void_ratio.vtk"
)

output = source.with_name("prestress2800000_void_ratio_fixed.vtk")

with source.open("r", encoding="ascii") as src, output.open(
    "w",
    encoding="ascii",
    newline="\n",
) as dst:
    for line in src:
        if line.strip() == "VECTORS i float":
            dst.write("VECTORS inertia float\n")
        else:
            dst.write(line)

print(f"Fixed file written to:\n{output}")

import pyvista as pv

file_path = (
    r"C:\Users\sammy\Desktop\voxelization"
    r"\prestress2800000_void_ratio_fixed.vtk"
)

mesh = pv.read(file_path)

print(mesh)
print("Point-data fields:", list(mesh.point_data.keys()))
print("Cell-data fields:", list(mesh.cell_data.keys()))

mesh.plot(
    scalars="void_ratio",
    cmap="viridis",
    point_size=3,
    render_points_as_spheres=False,
)