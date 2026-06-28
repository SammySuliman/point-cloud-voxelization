#!/usr/bin/env python
"""
Voxelize a legacy VTK point cloud and write a per-particle void-ratio field.

This script intentionally uses the classic VTK API style that works with
VTK 7.1. It reads the particle positions and radius scalar array from a
POLYDATA file, bins particle centers into cubic voxels, computes one void
ratio per occupied voxel, and assigns that value to all particles in the voxel.

Void ratio is computed as:

    e = (V_voxel - sum(V_sphere_particles_in_voxel)) / sum(V_sphere)

Sphere volumes are assigned to the voxel containing each particle center.
Interior voxels use the full cubic voxel volume. Boundary voxels are shortened
on their outside face(s) to the local sphere extent of the particles assigned to
that boundary voxel, reducing artificial empty space outside the particle cloud.
A second point-data field stores a Gaussian-smoothed void ratio over occupied
neighboring voxels.
"""

from __future__ import print_function

import argparse
import shutil
import math
import os
import sys


def require_vtk_71(allow_other_vtk):
    try:
        import vtk
    except ImportError:
        print(
            "ERROR: Python cannot import vtk. Run this with your VTK 7.1 Python "
            "environment.",
            file=sys.stderr,
        )
        raise

    version = vtk.vtkVersion()
    major = version.GetVTKMajorVersion()
    minor = version.GetVTKMinorVersion()
    version_text = version.GetVTKVersion()
    if not allow_other_vtk and (major, minor) != (7, 1):
        raise RuntimeError(
            "This script is pinned to VTK 7.1 because newer VTK versions are "
            "known to mishandle this file in your workflow. Found VTK {0}. "
            "Use --allow-other-vtk to override.".format(version_text)
        )
    return vtk, version_text


def vtk_array_to_numpy(vtk_array):
    try:
        from vtk.util.numpy_support import vtk_to_numpy

        return vtk_to_numpy(vtk_array)
    except Exception:
        return None


def read_polydata(vtk, input_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_path)

    # VTK's legacy readers do not always load every point-data array unless
    # explicitly asked. This matters here because radius appears after several
    # vector arrays in the file.
    for method_name in (
        "ReadAllScalarsOn",
        "ReadAllVectorsOn",
        "ReadAllTensorsOn",
        "ReadAllNormalsOn",
        "ReadAllColorScalarsOn",
    ):
        method = getattr(reader, method_name, None)
        if method is not None:
            method()

    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise RuntimeError("No points were read from {0}".format(input_path))
    return polydata


def get_radius_array(polydata, radius_field):
    point_data = polydata.GetPointData()
    radius_array = point_data.GetArray(radius_field)
    if radius_array is None:
        names = []
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            names.append(array.GetName() if array is not None else "<unnamed>")
        raise RuntimeError(
            "Radius field {0!r} was not found. Available point-data arrays: {1}".format(
                radius_field, ", ".join(names)
            )
        )
    if radius_array.GetNumberOfComponents() != 1:
        raise RuntimeError(
            "Radius field {0!r} must have one component, found {1}".format(
                radius_field, radius_array.GetNumberOfComponents()
            )
        )
    return radius_array


def load_points_and_radii(polydata, radius_array):
    points = polydata.GetPoints()
    count = polydata.GetNumberOfPoints()

    radius_np = vtk_array_to_numpy(radius_array)
    if radius_np is not None:
        try:
            import numpy as np

            coords = np.empty((count, 3), dtype=np.float64)
            for i in range(count):
                coords[i, :] = points.GetPoint(i)
            return coords, radius_np.astype(np.float64, copy=False)
        except Exception:
            pass

    coords = []
    radii = []
    for i in range(count):
        coords.append(points.GetPoint(i))
        radii.append(radius_array.GetTuple1(i))
    return coords, radii


def gaussian_smooth_occupied_numpy(voxel_indices, values_by_voxel, dims, sigma):
    import numpy as np

    if sigma <= 0.0:
        return values_by_voxel.copy()

    radius = int(math.ceil(3.0 * sigma))
    shape = tuple(int(v) for v in dims)
    values_grid = np.zeros(shape, dtype=np.float64)
    occupied_grid = np.zeros(shape, dtype=np.float64)
    coords = tuple(voxel_indices[:, axis] for axis in range(3))
    values_grid[coords] = values_by_voxel
    occupied_grid[coords] = 1.0

    weighted_sum = np.zeros(shape, dtype=np.float64)
    weight_sum = np.zeros(shape, dtype=np.float64)

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                distance_sq = dx * dx + dy * dy + dz * dz
                weight = math.exp(-distance_sq / (2.0 * sigma * sigma))

                target_slices = []
                source_slices = []
                for offset, size in ((dx, shape[0]), (dy, shape[1]), (dz, shape[2])):
                    if offset >= 0:
                        target_slices.append(slice(offset, size))
                        source_slices.append(slice(0, size - offset))
                    else:
                        target_slices.append(slice(0, size + offset))
                        source_slices.append(slice(-offset, size))

                target = tuple(target_slices)
                source = tuple(source_slices)
                weighted_sum[target] += weight * values_grid[source] * occupied_grid[source]
                weight_sum[target] += weight * occupied_grid[source]

    smoothed_grid = np.empty(shape, dtype=np.float64)
    np.divide(weighted_sum, weight_sum, out=smoothed_grid, where=weight_sum > 0.0)
    smoothed_grid[weight_sum == 0.0] = values_grid[weight_sum == 0.0]
    return smoothed_grid[coords]


def gaussian_smooth_occupied_plain(voxel_keys, values_by_voxel, sigma):
    if sigma <= 0.0:
        return dict(values_by_voxel)

    radius = int(math.ceil(3.0 * sigma))
    smoothed = {}
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                distance_sq = dx * dx + dy * dy + dz * dz
                offsets.append((dx, dy, dz, math.exp(-distance_sq / (2.0 * sigma * sigma))))

    for key in voxel_keys:
        weighted_sum = 0.0
        weight_sum = 0.0
        for dx, dy, dz, weight in offsets:
            neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
            if neighbor in values_by_voxel:
                weighted_sum += weight * values_by_voxel[neighbor]
                weight_sum += weight
        smoothed[key] = weighted_sum / weight_sum if weight_sum > 0.0 else values_by_voxel[key]
    return smoothed


def choose_voxel_size_numpy(points, target_particles_per_voxel):
    import numpy as np

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    max_extent = float(extents.max())

    if max_extent <= 0.0:
        raise RuntimeError("Point cloud extents are degenerate.")

    lo = 0.0
    hi = max_extent
    best = None
    for _ in range(35):
        size = (lo + hi) / 2.0
        dims = np.maximum(np.ceil(extents / size).astype(np.int64), 1)
        indices = np.floor((points - mins) / size).astype(np.int64)
        indices = np.minimum(indices, dims - 1)
        linear = indices[:, 0] + dims[0] * (indices[:, 1] + dims[1] * indices[:, 2])
        occupied = np.unique(linear).shape[0]
        average = float(points.shape[0]) / float(occupied)
        best = (size, dims, occupied, average)
        if average > target_particles_per_voxel:
            hi = size
        else:
            lo = size

    return mins, maxs, best


def compute_void_ratio_numpy(points, radii, target_particles_per_voxel, smooth_sigma):
    import numpy as np

    mins, maxs, best = choose_voxel_size_numpy(points, target_particles_per_voxel)
    voxel_size, dims, occupied, average = best

    indices = np.floor((points - mins) / voxel_size).astype(np.int64)
    indices = np.minimum(indices, dims - 1)
    linear = indices[:, 0] + dims[0] * (indices[:, 1] + dims[1] * indices[:, 2])

    unique_linear, unique_first, inverse = np.unique(
        linear, return_index=True, return_inverse=True
    )
    voxel_indices = indices[unique_first]
    sphere_volumes = (4.0 / 3.0) * math.pi * np.power(radii, 3)
    solid_by_voxel = np.bincount(inverse, weights=sphere_volumes)
    counts_by_voxel = np.bincount(inverse)

    nominal_voxel_volume = voxel_size ** 3
    lower_sphere_extent = points - radii[:, None]
    upper_sphere_extent = points + radii[:, None]
    local_min = np.full((solid_by_voxel.shape[0], 3), np.inf, dtype=np.float64)
    local_max = np.full((solid_by_voxel.shape[0], 3), -np.inf, dtype=np.float64)
    for axis in range(3):
        np.minimum.at(local_min[:, axis], inverse, lower_sphere_extent[:, axis])
        np.maximum.at(local_max[:, axis], inverse, upper_sphere_extent[:, axis])

    lengths = np.full((solid_by_voxel.shape[0], 3), voxel_size, dtype=np.float64)
    for axis in range(3):
        nominal_low = mins[axis] + voxel_indices[:, axis] * voxel_size
        nominal_high = nominal_low + voxel_size
        adjusted_low = nominal_low.copy()
        adjusted_high = nominal_high.copy()

        low_boundary = voxel_indices[:, axis] == 0
        high_boundary = voxel_indices[:, axis] == dims[axis] - 1
        adjusted_low[low_boundary] = np.maximum(
            adjusted_low[low_boundary], local_min[low_boundary, axis]
        )
        adjusted_high[high_boundary] = np.minimum(
            adjusted_high[high_boundary], local_max[high_boundary, axis]
        )
        lengths[:, axis] = np.maximum(adjusted_high - adjusted_low, 0.0)

    measurement_volume_by_voxel = np.prod(lengths, axis=1)
    adjusted_boundary_voxels = int(
        np.count_nonzero(measurement_volume_by_voxel < nominal_voxel_volume)
    )

    void_by_voxel = (measurement_volume_by_voxel - solid_by_voxel) / solid_by_voxel
    smoothed_void_by_voxel = gaussian_smooth_occupied_numpy(
        voxel_indices, void_by_voxel, dims, smooth_sigma
    )
    void_by_particle = void_by_voxel[inverse]
    smoothed_void_by_particle = smoothed_void_by_voxel[inverse]

    stats = {
        "point_count": int(points.shape[0]),
        "bounds_min": [float(v) for v in mins],
        "bounds_max": [float(v) for v in maxs],
        "extents": [float(v) for v in (maxs - mins)],
        "target_particles_per_voxel": float(target_particles_per_voxel),
        "voxel_size": float(voxel_size),
        "nominal_voxel_volume": float(nominal_voxel_volume),
        "min_measurement_volume": float(measurement_volume_by_voxel.min()),
        "median_measurement_volume": float(np.median(measurement_volume_by_voxel)),
        "max_measurement_volume": float(measurement_volume_by_voxel.max()),
        "adjusted_boundary_voxels": adjusted_boundary_voxels,
        "dims": [int(v) for v in dims],
        "total_voxels": int(dims[0] * dims[1] * dims[2]),
        "occupied_voxels": int(occupied),
        "average_particles_per_occupied_voxel": float(average),
        "min_particles_per_occupied_voxel": int(counts_by_voxel.min()),
        "max_particles_per_occupied_voxel": int(counts_by_voxel.max()),
        "min_void_ratio": float(void_by_voxel.min()),
        "median_void_ratio": float(np.median(void_by_voxel)),
        "max_void_ratio": float(void_by_voxel.max()),
        "negative_void_ratio_voxels": int(np.count_nonzero(void_by_voxel < 0.0)),
        "smooth_sigma": float(smooth_sigma),
        "min_smoothed_void_ratio": float(smoothed_void_by_voxel.min()),
        "median_smoothed_void_ratio": float(np.median(smoothed_void_by_voxel)),
        "max_smoothed_void_ratio": float(smoothed_void_by_voxel.max()),
        "unique_linear_voxels": unique_linear,
    }
    return void_by_particle, smoothed_void_by_particle, stats


def compute_void_ratio_plain(points, radii, target_particles_per_voxel, smooth_sigma):
    mins = [min(p[axis] for p in points) for axis in range(3)]
    maxs = [max(p[axis] for p in points) for axis in range(3)]
    extents = [maxs[i] - mins[i] for i in range(3)]
    max_extent = max(extents)

    if max_extent <= 0.0:
        raise RuntimeError("Point cloud extents are degenerate.")

    lo = 0.0
    hi = max_extent
    best = None
    for _ in range(45):
        size = (lo + hi) / 2.0
        dims = [max(int(math.ceil(extent / size)), 1) for extent in extents]
        occupied = set()
        for point in points:
            ix = min(int((point[0] - mins[0]) / size), dims[0] - 1)
            iy = min(int((point[1] - mins[1]) / size), dims[1] - 1)
            iz = min(int((point[2] - mins[2]) / size), dims[2] - 1)
            occupied.add((ix, iy, iz))
        average = float(len(points)) / float(len(occupied))
        best = (size, dims, len(occupied), average)
        if average > target_particles_per_voxel:
            hi = size
        else:
            lo = size

    voxel_size, dims, occupied_count, average = best
    nominal_voxel_volume = voxel_size ** 3
    solid_by_voxel = {}
    count_by_voxel = {}
    local_min_by_voxel = {}
    local_max_by_voxel = {}
    particle_keys = []

    for point, radius in zip(points, radii):
        key = (
            min(int((point[0] - mins[0]) / voxel_size), dims[0] - 1),
            min(int((point[1] - mins[1]) / voxel_size), dims[1] - 1),
            min(int((point[2] - mins[2]) / voxel_size), dims[2] - 1),
        )
        particle_keys.append(key)
        solid_by_voxel[key] = solid_by_voxel.get(key, 0.0) + (4.0 / 3.0) * math.pi * radius ** 3
        count_by_voxel[key] = count_by_voxel.get(key, 0) + 1
        lower = (point[0] - radius, point[1] - radius, point[2] - radius)
        upper = (point[0] + radius, point[1] + radius, point[2] + radius)
        if key not in local_min_by_voxel:
            local_min_by_voxel[key] = list(lower)
            local_max_by_voxel[key] = list(upper)
        else:
            for axis in range(3):
                local_min_by_voxel[key][axis] = min(local_min_by_voxel[key][axis], lower[axis])
                local_max_by_voxel[key][axis] = max(local_max_by_voxel[key][axis], upper[axis])

    void_by_voxel = {}
    measurement_volumes = []
    adjusted_boundary_voxels = 0
    for key, solid in solid_by_voxel.items():
        lengths = []
        for axis in range(3):
            nominal_low = mins[axis] + key[axis] * voxel_size
            nominal_high = nominal_low + voxel_size
            adjusted_low = nominal_low
            adjusted_high = nominal_high
            if key[axis] == 0:
                adjusted_low = max(adjusted_low, local_min_by_voxel[key][axis])
            if key[axis] == dims[axis] - 1:
                adjusted_high = min(adjusted_high, local_max_by_voxel[key][axis])
            lengths.append(max(adjusted_high - adjusted_low, 0.0))
        measurement_volume = lengths[0] * lengths[1] * lengths[2]
        if measurement_volume < nominal_voxel_volume:
            adjusted_boundary_voxels += 1
        measurement_volumes.append(measurement_volume)
        void_by_voxel[key] = (measurement_volume - solid) / solid

    smoothed_void_by_voxel = gaussian_smooth_occupied_plain(
        solid_by_voxel.keys(), void_by_voxel, smooth_sigma
    )
    void_by_particle = [void_by_voxel[key] for key in particle_keys]
    smoothed_void_by_particle = [smoothed_void_by_voxel[key] for key in particle_keys]
    void_values = list(void_by_voxel.values())
    smoothed_void_values = list(smoothed_void_by_voxel.values())
    counts = list(count_by_voxel.values())
    void_sorted = sorted(void_values)
    smoothed_void_sorted = sorted(smoothed_void_values)
    measurement_sorted = sorted(measurement_volumes)

    stats = {
        "point_count": len(points),
        "bounds_min": mins,
        "bounds_max": maxs,
        "extents": extents,
        "target_particles_per_voxel": float(target_particles_per_voxel),
        "voxel_size": voxel_size,
        "nominal_voxel_volume": nominal_voxel_volume,
        "min_measurement_volume": min(measurement_volumes),
        "median_measurement_volume": measurement_sorted[len(measurement_sorted) // 2],
        "max_measurement_volume": max(measurement_volumes),
        "adjusted_boundary_voxels": adjusted_boundary_voxels,
        "dims": dims,
        "total_voxels": dims[0] * dims[1] * dims[2],
        "occupied_voxels": occupied_count,
        "average_particles_per_occupied_voxel": average,
        "min_particles_per_occupied_voxel": min(counts),
        "max_particles_per_occupied_voxel": max(counts),
        "min_void_ratio": min(void_values),
        "median_void_ratio": void_sorted[len(void_sorted) // 2],
        "max_void_ratio": max(void_values),
        "negative_void_ratio_voxels": sum(1 for value in void_values if value < 0.0),
        "smooth_sigma": float(smooth_sigma),
        "min_smoothed_void_ratio": min(smoothed_void_values),
        "median_smoothed_void_ratio": smoothed_void_sorted[len(smoothed_void_sorted) // 2],
        "max_smoothed_void_ratio": max(smoothed_void_values),
    }
    return void_by_particle, smoothed_void_by_particle, stats


def find_smallest_n_without_negative_void_ratio(
    points, radii, search_start_n, search_limit_n
):
    if search_limit_n < search_start_n:
        return None, None

    use_numpy = hasattr(points, "shape")
    for candidate_n in range(search_start_n, search_limit_n + 1):
        if use_numpy:
            _, _, candidate_stats = compute_void_ratio_numpy(
                points, radii, float(candidate_n), 0.0
            )
        else:
            _, _, candidate_stats = compute_void_ratio_plain(
                points, radii, float(candidate_n), 0.0
            )
        if candidate_stats["negative_void_ratio_voxels"] == 0:
            return candidate_n, candidate_stats
    return None, None


def add_negative_void_ratio_recommendation(points, radii, stats, search_limit_n):
    stats["negative_n_search_limit"] = int(search_limit_n)
    stats["nonnegative_recommended_n"] = None
    stats["nonnegative_recommended_dims"] = None
    stats["nonnegative_recommended_min_void_ratio"] = None

    if stats["negative_void_ratio_voxels"] == 0 or search_limit_n <= 0:
        return

    current_n = stats["target_particles_per_voxel"]
    search_start_n = max(1, int(math.ceil(current_n)))
    if abs(float(search_start_n) - float(current_n)) < 1.0e-12:
        search_start_n += 1

    recommended_n, recommended_stats = find_smallest_n_without_negative_void_ratio(
        points, radii, search_start_n, int(search_limit_n)
    )
    if recommended_n is None:
        return

    stats["nonnegative_recommended_n"] = int(recommended_n)
    stats["nonnegative_recommended_dims"] = recommended_stats["dims"]
    stats["nonnegative_recommended_min_void_ratio"] = recommended_stats[
        "min_void_ratio"
    ]


def add_point_scalar_array(vtk, polydata, field_name, values):
    array = vtk.vtkDoubleArray()
    array.SetName(field_name)
    array.SetNumberOfComponents(1)
    array.SetNumberOfTuples(polydata.GetNumberOfPoints())

    for i, value in enumerate(values):
        array.SetTuple1(i, float(value))

    polydata.GetPointData().AddArray(array)
    polydata.GetPointData().SetActiveScalars(field_name)


def write_polydata(vtk, polydata, output_path, binary):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    if writer.Write() != 1:
        raise RuntimeError("Failed to write {0}".format(output_path))


def read_float_values(file_obj, needed):
    values = []
    while len(values) < needed:
        line = file_obj.readline()
        if line == "":
            raise RuntimeError("Unexpected end of file while reading numeric values.")
        values.extend(float(part) for part in line.split())
    if len(values) > needed:
        raise RuntimeError(
            "This parser expected exactly {0} values but found extra values on "
            "the final data line.".format(needed)
        )
    return values


def read_legacy_ascii_points_and_scalar(input_path, scalar_name):
    try:
        import numpy as np
    except ImportError:
        np = None

    with open(input_path, "r") as file_obj:
        first = file_obj.readline()
        if not first.startswith("# vtk DataFile"):
            raise RuntimeError("Input does not look like a legacy VTK file.")

        mode = None
        point_count = None
        points = None
        scalar_values = None

        while True:
            line = file_obj.readline()
            if line == "":
                break
            stripped = line.strip()
            if stripped == "ASCII":
                mode = "ASCII"
            elif stripped == "BINARY":
                mode = "BINARY"
            elif stripped.startswith("POINTS "):
                if mode != "ASCII":
                    raise RuntimeError(
                        "The direct fallback parser only supports legacy ASCII VTK."
                    )
                parts = stripped.split()
                point_count = int(parts[1])
                raw = read_float_values(file_obj, point_count * 3)
                if np is not None:
                    points = np.asarray(raw, dtype=np.float64).reshape((point_count, 3))
                else:
                    points = [
                        (raw[i], raw[i + 1], raw[i + 2])
                        for i in range(0, len(raw), 3)
                    ]
            elif stripped.startswith("SCALARS "):
                parts = stripped.split()
                name = parts[1]
                components = int(parts[3]) if len(parts) >= 4 else 1
                lookup_line = file_obj.readline()
                if not lookup_line.strip().startswith("LOOKUP_TABLE"):
                    raise RuntimeError(
                        "Expected LOOKUP_TABLE after SCALARS {0}.".format(name)
                    )
                if point_count is None:
                    raise RuntimeError(
                        "Found scalar data before reading the POINTS section."
                    )
                raw = read_float_values(file_obj, point_count * components)
                if name == scalar_name:
                    if components != 1:
                        raise RuntimeError(
                            "Scalar field {0!r} has {1} components; expected 1.".format(
                                scalar_name, components
                            )
                        )
                    if np is not None:
                        scalar_values = np.asarray(raw, dtype=np.float64)
                    else:
                        scalar_values = raw
                    return points, scalar_values

        if points is None:
            raise RuntimeError("POINTS section was not found.")
        if scalar_values is None:
            raise RuntimeError("SCALARS {0!r} was not found.".format(scalar_name))
        return points, scalar_values


def append_scalar_to_legacy_ascii(input_path, output_path, field_name, values):
    if os.path.abspath(input_path) != os.path.abspath(output_path):
        shutil.copyfile(input_path, output_path)

    with open(output_path, "a") as file_obj:
        if os.path.getsize(output_path) > 0:
            file_obj.write("\n")
        file_obj.write("SCALARS {0} double 1\n".format(field_name))
        file_obj.write("LOOKUP_TABLE default\n")
        for value in values:
            file_obj.write("{0:.17g}\n".format(float(value)))


def print_stats(stats, output_path, version_text):
    print("VTK version: {0}".format(version_text))
    print("Output: {0}".format(output_path))
    print("Points: {0}".format(stats["point_count"]))
    print("Bounds min: {0}".format(stats["bounds_min"]))
    print("Bounds max: {0}".format(stats["bounds_max"]))
    print("Extents: {0}".format(stats["extents"]))
    print("Target particles per occupied voxel: {0:g}".format(stats["target_particles_per_voxel"]))
    print("Voxel size: {0:.12g}".format(stats["voxel_size"]))
    print("Nominal full voxel volume: {0:.12g}".format(stats["nominal_voxel_volume"]))
    print(
        "Measurement volume min/median/max: {0:.6g} / {1:.6g} / {2:.6g}".format(
            stats["min_measurement_volume"],
            stats["median_measurement_volume"],
            stats["max_measurement_volume"],
        )
    )
    print("Boundary voxels adjusted: {0}".format(stats["adjusted_boundary_voxels"]))
    print("Voxel dimensions: {0}".format(stats["dims"]))
    print("Total voxels in extent grid: {0}".format(stats["total_voxels"]))
    print("Occupied voxels: {0}".format(stats["occupied_voxels"]))
    print(
        "Average particles per occupied voxel: {0:.6g}".format(
            stats["average_particles_per_occupied_voxel"]
        )
    )
    print(
        "Particles per occupied voxel min/max: {0}/{1}".format(
            stats["min_particles_per_occupied_voxel"],
            stats["max_particles_per_occupied_voxel"],
        )
    )
    print(
        "Void ratio min/median/max: {0:.6g} / {1:.6g} / {2:.6g}".format(
            stats["min_void_ratio"],
            stats["median_void_ratio"],
            stats["max_void_ratio"],
        )
    )
    print(
        "Occupied voxels with negative void ratio: {0}".format(
            stats["negative_void_ratio_voxels"]
        )
    )
    if stats["negative_void_ratio_voxels"] > 0:
        print(
            "WARNING: Negative raw void ratios occurred for the current N. "
            "This means assigned sphere volume exceeded adjusted voxel volume "
            "in those voxels."
        )
        if stats.get("nonnegative_recommended_n") is not None:
            print(
                "Smallest integer N above the current N with no negative raw "
                "void ratios: {0}".format(stats["nonnegative_recommended_n"])
            )
            print(
                "Recommended N voxel dimensions: {0}".format(
                    stats["nonnegative_recommended_dims"]
                )
            )
            print(
                "Recommended N minimum void ratio: {0:.6g}".format(
                    stats["nonnegative_recommended_min_void_ratio"]
                )
            )
        elif stats.get("negative_n_search_limit", 0) > 0:
            print(
                "No integer N up to {0} eliminated negative raw void ratios.".format(
                    stats["negative_n_search_limit"]
                )
            )
    print("Gaussian smoothing sigma: {0:g} voxel(s)".format(stats["smooth_sigma"]))
    print(
        "Smoothed void ratio min/median/max: {0:.6g} / {1:.6g} / {2:.6g}".format(
            stats["min_smoothed_void_ratio"],
            stats["median_smoothed_void_ratio"],
            stats["max_smoothed_void_ratio"],
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voxelize a VTK point cloud and add a per-particle void_ratio field."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="prestress2800000.vtk",
        help="Input legacy VTK POLYDATA file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output VTK file. Defaults to INPUT with _void_ratio before .vtk.",
    )
    parser.add_argument(
        "-n",
        "--particles-per-voxel",
        type=float,
        default=10.0,
        help="Target average particles per occupied voxel. Try 10 first; 8 is finer.",
    )
    parser.add_argument(
        "--radius-field",
        default="radius",
        help="Name of the point-data scalar array containing particle radius.",
    )
    parser.add_argument(
        "--void-field",
        default="void_ratio",
        help="Name of the raw point-data scalar array to write.",
    )
    parser.add_argument(
        "--smoothed-void-field",
        default="void_ratio_smoothed",
        help="Name of the Gaussian-smoothed point-data scalar array to write.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in voxel units for the smoothed field.",
    )
    parser.add_argument(
        "--negative-n-search-limit",
        type=int,
        default=100,
        help=(
            "If raw negative void ratios occur, search integer N values above "
            "the current N through this limit and report the first with none. "
            "Use 0 to disable the search."
        ),
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write binary legacy VTK instead of ASCII.",
    )
    parser.add_argument(
        "--allow-other-vtk",
        action="store_true",
        help="Allow running with VTK versions other than 7.1.",
    )
    parser.add_argument(
        "--vtk-writer",
        action="store_true",
        help=(
            "Use vtkPolyDataReader/vtkPolyDataWriter for IO. By default the "
            "script preserves the original legacy ASCII file and appends the "
            "new scalar field, because VTK 7.1 may not load all arrays here."
        ),
    )
    return parser.parse_args()


def default_output_path(input_path):
    root, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".vtk"
    return root + "_void_ratio" + ext


def main():
    args = parse_args()
    if args.particles_per_voxel <= 0.0:
        raise RuntimeError("--particles-per-voxel must be positive.")
    if args.smooth_sigma < 0.0:
        raise RuntimeError("--smooth-sigma must be non-negative.")
    if args.negative_n_search_limit < 0:
        raise RuntimeError("--negative-n-search-limit must be non-negative.")

    vtk, version_text = require_vtk_71(args.allow_other_vtk)
    output_path = args.output or default_output_path(args.input)

    if args.vtk_writer:
        polydata = read_polydata(vtk, args.input)
        radius_array = get_radius_array(polydata, args.radius_field)
        points, radii = load_points_and_radii(polydata, radius_array)
    else:
        polydata = None
        points, radii = read_legacy_ascii_points_and_scalar(
            args.input, args.radius_field
        )

    if hasattr(points, "shape"):
        void_by_particle, smoothed_void_by_particle, stats = compute_void_ratio_numpy(
            points, radii, args.particles_per_voxel, args.smooth_sigma
        )
    else:
        void_by_particle, smoothed_void_by_particle, stats = compute_void_ratio_plain(
            points, radii, args.particles_per_voxel, args.smooth_sigma
        )

    add_negative_void_ratio_recommendation(
        points, radii, stats, args.negative_n_search_limit
    )

    if args.vtk_writer:
        add_point_scalar_array(vtk, polydata, args.void_field, void_by_particle)
        add_point_scalar_array(
            vtk, polydata, args.smoothed_void_field, smoothed_void_by_particle
        )
        write_polydata(vtk, polydata, output_path, args.binary)
    else:
        if args.binary:
            raise RuntimeError("--binary requires --vtk-writer.")
        append_scalar_to_legacy_ascii(args.input, output_path, args.void_field, void_by_particle)
        append_scalar_to_legacy_ascii(
            output_path, output_path, args.smoothed_void_field, smoothed_void_by_particle
        )

    print_stats(stats, output_path, version_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        sys.exit(1)
