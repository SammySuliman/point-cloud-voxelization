#!/usr/bin/env python
"""
Voxelize a legacy VTK point cloud and write a per-particle void-ratio field.

This script intentionally uses the classic VTK API style that works with
VTK 7.1. It reads the particle positions and radius scalar array from a
POLYDATA file, bins particle centers into cubic voxels, computes one void
ratio per occupied voxel, and assigns that value to all particles in the voxel.

Void ratio is computed as:

    e = (V_voxel - sum(V_sphere_particles_in_voxel)) / sum(V_sphere)

Particle spheres are clipped across voxel boundaries when accumulating solid
volume, so a sphere contributes only the portion of its volume that lies inside
each voxel it overlaps. Interior voxels use the full cubic voxel volume. Boundary
voxels are shortened on their outside face(s) to the local sphere extent of the
particles assigned to that boundary voxel. A second point-data field stores a
face-reconstructed value obtained from the voxel value, six neighbor-averaged
face values, and two-point Gauss quadrature inside each voxel.
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


def face_reconstructed_gauss_smooth_numpy(voxel_indices, values_by_voxel, dims):
    import numpy as np

    shape = tuple(int(v) for v in dims)
    compact_grid = np.full(shape, -1, dtype=np.int64)
    coords = tuple(voxel_indices[:, axis] for axis in range(3))
    compact_grid[coords] = np.arange(values_by_voxel.shape[0], dtype=np.int64)

    face_values = np.empty((values_by_voxel.shape[0], 6), dtype=np.float64)
    directions = [
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 1),
        (2, -1),
        (2, 1),
    ]

    for face_index, (axis, step) in enumerate(directions):
        neighbor_indices = voxel_indices.copy()
        neighbor_indices[:, axis] += step
        valid = (neighbor_indices[:, axis] >= 0) & (neighbor_indices[:, axis] < dims[axis])
        neighbor_compact = np.full(values_by_voxel.shape[0], -1, dtype=np.int64)
        valid_rows = np.where(valid)[0]
        if valid_rows.size:
            lookup = tuple(neighbor_indices[valid_rows, dim] for dim in range(3))
            neighbor_compact[valid_rows] = compact_grid[lookup]
        has_neighbor = neighbor_compact >= 0
        face_values[:, face_index] = values_by_voxel
        face_values[has_neighbor, face_index] = 0.5 * (
            values_by_voxel[has_neighbor] + values_by_voxel[neighbor_compact[has_neighbor]]
        )

    return evaluate_face_reconstruction_at_gauss_numpy(values_by_voxel, face_values)


def evaluate_face_reconstruction_at_gauss_numpy(center_values, face_values):
    import numpy as np

    e0 = center_values
    ex_minus = face_values[:, 0]
    ex_plus = face_values[:, 1]
    ey_minus = face_values[:, 2]
    ey_plus = face_values[:, 3]
    ez_minus = face_values[:, 4]
    ez_plus = face_values[:, 5]

    ax = 0.5 * (ex_plus - ex_minus)
    bx = 0.5 * (ex_plus + ex_minus) - e0
    ay = 0.5 * (ey_plus - ey_minus)
    by = 0.5 * (ey_plus + ey_minus) - e0
    az = 0.5 * (ez_plus - ez_minus)
    bz = 0.5 * (ez_plus + ez_minus) - e0

    gp = 1.0 / math.sqrt(3.0)
    total = np.zeros_like(e0)
    count = 0
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            for zeta in (-gp, gp):
                total += (
                    e0
                    + ax * xi + bx * xi * xi
                    + ay * eta + by * eta * eta
                    + az * zeta + bz * zeta * zeta
                )
                count += 1
    return total / float(count)


def face_reconstructed_gauss_smooth_plain(voxel_keys, values_by_voxel):
    voxel_keys = list(voxel_keys)
    occupied = set(voxel_keys)
    smoothed = {}

    for key in voxel_keys:
        e0 = values_by_voxel[key]
        face_values = []
        for axis, step in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
            neighbor = list(key)
            neighbor[axis] += step
            neighbor = tuple(neighbor)
            if neighbor in occupied:
                face_values.append(0.5 * (e0 + values_by_voxel[neighbor]))
            else:
                face_values.append(e0)
        smoothed[key] = evaluate_face_reconstruction_at_gauss_plain(e0, face_values)
    return smoothed


def evaluate_face_reconstruction_at_gauss_plain(e0, face_values):
    ex_minus, ex_plus, ey_minus, ey_plus, ez_minus, ez_plus = face_values
    ax = 0.5 * (ex_plus - ex_minus)
    bx = 0.5 * (ex_plus + ex_minus) - e0
    ay = 0.5 * (ey_plus - ey_minus)
    by = 0.5 * (ey_plus + ey_minus) - e0
    az = 0.5 * (ez_plus - ez_minus)
    bz = 0.5 * (ez_plus + ez_minus) - e0

    gp = 1.0 / math.sqrt(3.0)
    total = 0.0
    count = 0
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            for zeta in (-gp, gp):
                total += (
                    e0
                    + ax * xi + bx * xi * xi
                    + ay * eta + by * eta * eta
                    + az * zeta + bz * zeta * zeta
                )
                count += 1
    return total / float(count)


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


def make_sphere_clip_quadrature(order):
    try:
        import numpy as np
    except ImportError:
        return None, None
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes.astype(np.float64), weights.astype(np.float64)


def sphere_box_intersection_volume_numpy(center, radius, box_min, box_max, nodes, weights):
    import numpy as np

    sphere_min = center - radius
    sphere_max = center + radius
    clipped_min = np.maximum(box_min, sphere_min)
    clipped_max = np.minimum(box_max, sphere_max)
    if np.any(clipped_max <= clipped_min):
        return 0.0

    if np.all(box_min <= sphere_min) and np.all(box_max >= sphere_max):
        return (4.0 / 3.0) * math.pi * radius ** 3

    x0, y0, z0 = clipped_min - center
    x1, y1, z1 = clipped_max - center
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    x_half = 0.5 * (x1 - x0)
    y_half = 0.5 * (y1 - y0)

    x = x_mid + x_half * nodes
    y = y_mid + y_half * nodes
    xx = x[:, None]
    yy = y[None, :]
    remaining = radius * radius - xx * xx - yy * yy
    valid = remaining > 0.0
    z_radius = np.zeros_like(remaining)
    z_radius[valid] = np.sqrt(remaining[valid])
    z_low = np.maximum(z0, -z_radius)
    z_high = np.minimum(z1, z_radius)
    height = np.maximum(z_high - z_low, 0.0)
    return float(x_half * y_half * np.sum((weights[:, None] * weights[None, :]) * height))


def sphere_box_intersection_volume_plain(center, radius, box_min, box_max, nodes, weights):
    sphere_min = [center[i] - radius for i in range(3)]
    sphere_max = [center[i] + radius for i in range(3)]
    clipped_min = [max(box_min[i], sphere_min[i]) for i in range(3)]
    clipped_max = [min(box_max[i], sphere_max[i]) for i in range(3)]
    if any(clipped_max[i] <= clipped_min[i] for i in range(3)):
        return 0.0

    if all(box_min[i] <= sphere_min[i] and box_max[i] >= sphere_max[i] for i in range(3)):
        return (4.0 / 3.0) * math.pi * radius ** 3

    x0, y0, z0 = [clipped_min[i] - center[i] for i in range(3)]
    x1, y1, z1 = [clipped_max[i] - center[i] for i in range(3)]
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    x_half = 0.5 * (x1 - x0)
    y_half = 0.5 * (y1 - y0)
    total = 0.0
    for node_x, weight_x in zip(nodes, weights):
        x = x_mid + x_half * node_x
        for node_y, weight_y in zip(nodes, weights):
            y = y_mid + y_half * node_y
            remaining = radius * radius - x * x - y * y
            if remaining <= 0.0:
                continue
            z_radius = math.sqrt(remaining)
            height = min(z1, z_radius) - max(z0, -z_radius)
            if height > 0.0:
                total += weight_x * weight_y * height
    return x_half * y_half * total


def accumulate_clipped_solid_numpy(
    points, radii, mins, voxel_size, dims, unique_linear, box_min, box_max, order
):
    import numpy as np

    nodes, weights = make_sphere_clip_quadrature(order)
    linear_to_compact = {int(linear): i for i, linear in enumerate(unique_linear)}
    solid_by_voxel = np.zeros(len(unique_linear), dtype=np.float64)

    for center, radius in zip(points, radii):
        low = np.floor((center - radius - mins) / voxel_size).astype(np.int64)
        high = np.floor((center + radius - mins) / voxel_size).astype(np.int64)
        low = np.maximum(low, 0)
        high = np.minimum(high, dims - 1)
        for ix in range(int(low[0]), int(high[0]) + 1):
            for iy in range(int(low[1]), int(high[1]) + 1):
                for iz in range(int(low[2]), int(high[2]) + 1):
                    linear = ix + dims[0] * (iy + dims[1] * iz)
                    compact = linear_to_compact.get(int(linear))
                    if compact is None:
                        continue
                    solid_by_voxel[compact] += sphere_box_intersection_volume_numpy(
                        center, float(radius), box_min[compact], box_max[compact], nodes, weights
                    )
    return solid_by_voxel


def accumulate_clipped_solid_plain(
    points, radii, mins, voxel_size, dims, occupied_keys, box_bounds_by_key, order
):
    nodes, weights = make_sphere_clip_quadrature(order)
    if nodes is None:
        nodes, weights = legendre_gauss_plain(order)

    solid_by_voxel = {key: 0.0 for key in occupied_keys}
    occupied = set(occupied_keys)
    for center, radius in zip(points, radii):
        low = [max(int(math.floor((center[i] - radius - mins[i]) / voxel_size)), 0) for i in range(3)]
        high = [min(int(math.floor((center[i] + radius - mins[i]) / voxel_size)), dims[i] - 1) for i in range(3)]
        for ix in range(low[0], high[0] + 1):
            for iy in range(low[1], high[1] + 1):
                for iz in range(low[2], high[2] + 1):
                    key = (ix, iy, iz)
                    if key not in occupied:
                        continue
                    box_min, box_max = box_bounds_by_key[key]
                    solid_by_voxel[key] += sphere_box_intersection_volume_plain(
                        center, radius, box_min, box_max, nodes, weights
                    )
    return solid_by_voxel


def legendre_gauss_plain(order):
    # Fallback tables for environments without NumPy in the plain path.
    if order == 2:
        nodes = [
            -0.5773502691896257,
            0.5773502691896257,
        ]
        weights = [
            1.0,
            1.0,
        ]
    elif order == 4:
        nodes = [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ]
        weights = [
            0.3478548451374538,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374538,
        ]
    elif order == 8:
        nodes = [
            -0.9602898564975363,
            -0.7966664774136267,
            -0.5255324099163290,
            -0.1834346424956498,
            0.1834346424956498,
            0.5255324099163290,
            0.7966664774136267,
            0.9602898564975363,
        ]
        weights = [
            0.1012285362903763,
            0.2223810344533745,
            0.3137066458778873,
            0.3626837833783620,
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ]
    else:
        raise RuntimeError("Plain quadrature fallback only supports orders 2, 4, or 8.")
    return nodes, weights


def compute_void_ratio_numpy(points, radii, target_particles_per_voxel, sphere_clip_order):
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
    counts_by_voxel = np.bincount(inverse)

    nominal_voxel_volume = voxel_size ** 3
    lower_sphere_extent = points - radii[:, None]
    upper_sphere_extent = points + radii[:, None]
    local_min = np.full((len(unique_linear), 3), np.inf, dtype=np.float64)
    local_max = np.full((len(unique_linear), 3), -np.inf, dtype=np.float64)
    for axis in range(3):
        np.minimum.at(local_min[:, axis], inverse, lower_sphere_extent[:, axis])
        np.maximum.at(local_max[:, axis], inverse, upper_sphere_extent[:, axis])

    lengths = np.full((len(unique_linear), 3), voxel_size, dtype=np.float64)
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

    box_min = np.empty((len(unique_linear), 3), dtype=np.float64)
    box_max = np.empty((len(unique_linear), 3), dtype=np.float64)
    for axis in range(3):
        nominal_low = mins[axis] + voxel_indices[:, axis] * voxel_size
        nominal_high = nominal_low + voxel_size
        low_boundary = voxel_indices[:, axis] == 0
        high_boundary = voxel_indices[:, axis] == dims[axis] - 1
        box_min[:, axis] = nominal_low
        box_max[:, axis] = nominal_high
        box_min[low_boundary, axis] = np.maximum(
            box_min[low_boundary, axis], local_min[low_boundary, axis]
        )
        box_max[high_boundary, axis] = np.minimum(
            box_max[high_boundary, axis], local_max[high_boundary, axis]
        )

    measurement_volume_by_voxel = np.prod(box_max - box_min, axis=1)
    adjusted_boundary_voxels = int(
        np.count_nonzero(measurement_volume_by_voxel < nominal_voxel_volume)
    )
    solid_by_voxel = accumulate_clipped_solid_numpy(
        points, radii, mins, voxel_size, dims, unique_linear, box_min, box_max, sphere_clip_order
    )

    void_by_voxel = (measurement_volume_by_voxel - solid_by_voxel) / solid_by_voxel
    smoothed_void_by_voxel = face_reconstructed_gauss_smooth_numpy(
        voxel_indices, void_by_voxel, dims
    )
    void_by_particle = void_by_voxel[inverse]
    smoothed_void_by_particle = smoothed_void_by_voxel[inverse]
    voxel_data = {
        "box_min": box_min,
        "box_max": box_max,
        "void_ratio": void_by_voxel,
        "void_ratio_smoothed": smoothed_void_by_voxel,
        "particle_count": counts_by_voxel,
        "solid_volume": solid_by_voxel,
        "measurement_volume": measurement_volume_by_voxel,
        "voxel_indices": voxel_indices,
    }

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
        "smooth_method": "face_reconstructed_two_point_gauss",
        "gauss_point_count": 8,
        "sphere_clip_order": int(sphere_clip_order),
        "min_smoothed_void_ratio": float(smoothed_void_by_voxel.min()),
        "median_smoothed_void_ratio": float(np.median(smoothed_void_by_voxel)),
        "max_smoothed_void_ratio": float(smoothed_void_by_voxel.max()),
        "unique_linear_voxels": unique_linear,
    }
    return void_by_particle, smoothed_void_by_particle, stats, voxel_data


def compute_void_ratio_plain(points, radii, target_particles_per_voxel, sphere_clip_order):
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
        solid_by_voxel[key] = 0.0
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

    box_bounds_by_key = {}
    measurement_volume_by_key = {}
    measurement_volumes = []
    adjusted_boundary_voxels = 0
    for key in solid_by_voxel:
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
        measurement_volume_by_key[key] = measurement_volume
        box_bounds_by_key[key] = (
            [
                mins[axis] + key[axis] * voxel_size
                if key[axis] != 0
                else max(mins[axis] + key[axis] * voxel_size, local_min_by_voxel[key][axis])
                for axis in range(3)
            ],
            [
                mins[axis] + (key[axis] + 1) * voxel_size
                if key[axis] != dims[axis] - 1
                else min(mins[axis] + (key[axis] + 1) * voxel_size, local_max_by_voxel[key][axis])
                for axis in range(3)
            ],
        )

    solid_by_voxel = accumulate_clipped_solid_plain(
        points, radii, mins, voxel_size, dims, solid_by_voxel.keys(), box_bounds_by_key, sphere_clip_order
    )
    void_by_voxel = {}
    for key, solid in solid_by_voxel.items():
        void_by_voxel[key] = (measurement_volume_by_key[key] - solid) / solid

    smoothed_void_by_voxel = face_reconstructed_gauss_smooth_plain(
        solid_by_voxel.keys(), void_by_voxel
    )
    void_by_particle = [void_by_voxel[key] for key in particle_keys]
    smoothed_void_by_particle = [smoothed_void_by_voxel[key] for key in particle_keys]
    void_values = list(void_by_voxel.values())
    smoothed_void_values = list(smoothed_void_by_voxel.values())
    counts = list(count_by_voxel.values())
    voxel_keys = list(solid_by_voxel.keys())
    void_sorted = sorted(void_values)
    smoothed_void_sorted = sorted(smoothed_void_values)
    measurement_sorted = sorted(measurement_volumes)
    voxel_data = {
        "box_min": [box_bounds_by_key[key][0] for key in voxel_keys],
        "box_max": [box_bounds_by_key[key][1] for key in voxel_keys],
        "void_ratio": [void_by_voxel[key] for key in voxel_keys],
        "void_ratio_smoothed": [smoothed_void_by_voxel[key] for key in voxel_keys],
        "particle_count": [count_by_voxel[key] for key in voxel_keys],
        "solid_volume": [solid_by_voxel[key] for key in voxel_keys],
        "measurement_volume": [measurement_volume_by_key[key] for key in voxel_keys],
        "voxel_indices": voxel_keys,
    }

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
        "smooth_method": "face_reconstructed_two_point_gauss",
        "gauss_point_count": 8,
        "sphere_clip_order": int(sphere_clip_order),
        "min_smoothed_void_ratio": min(smoothed_void_values),
        "median_smoothed_void_ratio": smoothed_void_sorted[len(smoothed_void_sorted) // 2],
        "max_smoothed_void_ratio": max(smoothed_void_values),
    }
    return void_by_particle, smoothed_void_by_particle, stats, voxel_data


def write_voxel_cells_legacy_ascii(output_path, voxel_data, raw_field_name, smoothed_field_name):
    box_min = voxel_data["box_min"]
    box_max = voxel_data["box_max"]
    raw_values = voxel_data["void_ratio"]
    smoothed_values = voxel_data["void_ratio_smoothed"]
    particle_counts = voxel_data["particle_count"]
    solid_volumes = voxel_data["solid_volume"]
    measurement_volumes = voxel_data["measurement_volume"]
    cell_count = len(raw_values)
    point_count = cell_count * 8

    def point_tuple(bounds_min, bounds_max, local_index):
        xmin, ymin, zmin = [float(v) for v in bounds_min]
        xmax, ymax, zmax = [float(v) for v in bounds_max]
        points = (
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmin, ymax, zmin),
            (xmax, ymax, zmin),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
            (xmin, ymax, zmax),
            (xmax, ymax, zmax),
        )
        return points[local_index]

    with open(output_path, "w") as file_obj:
        file_obj.write("# vtk DataFile Version 2.0\n")
        file_obj.write("Voxelized void ratio cells\n")
        file_obj.write("ASCII\n")
        file_obj.write("DATASET UNSTRUCTURED_GRID\n")
        file_obj.write("POINTS {0} float\n".format(point_count))
        for cell_index in range(cell_count):
            for local_index in range(8):
                x, y, z = point_tuple(box_min[cell_index], box_max[cell_index], local_index)
                file_obj.write("{0:.17g} {1:.17g} {2:.17g}\n".format(x, y, z))

        file_obj.write("CELLS {0} {1}\n".format(cell_count, cell_count * 9))
        for cell_index in range(cell_count):
            start = cell_index * 8
            file_obj.write(
                "8 {0} {1} {2} {3} {4} {5} {6} {7}\n".format(
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                )
            )

        file_obj.write("CELL_TYPES {0}\n".format(cell_count))
        for _ in range(cell_count):
            file_obj.write("11\n")

        file_obj.write("CELL_DATA {0}\n".format(cell_count))
        write_cell_scalar(file_obj, raw_field_name, "double", raw_values)
        write_cell_scalar(file_obj, smoothed_field_name, "double", smoothed_values)
        write_cell_scalar(file_obj, "particle_count", "int", particle_counts)
        write_cell_scalar(file_obj, "solid_volume", "double", solid_volumes)
        write_cell_scalar(file_obj, "measurement_volume", "double", measurement_volumes)


def write_cell_scalar(file_obj, name, vtk_type, values):
    file_obj.write("SCALARS {0} {1} 1\n".format(name, vtk_type))
    file_obj.write("LOOKUP_TABLE default\n")
    if vtk_type == "int":
        for value in values:
            file_obj.write("{0}\n".format(int(value)))
    else:
        for value in values:
            file_obj.write("{0:.17g}\n".format(float(value)))


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
            "WARNING: Negative raw void ratios occurred. This means clipped "
            "solid volume still exceeded adjusted voxel volume in those voxels. "
            "Try a higher --sphere-clip-order for a more accurate sphere/voxel "
            "intersection estimate."
        )
    print("Sphere clipping quadrature order: {0}".format(stats["sphere_clip_order"]))
    print("Smoothing method: {0}".format(stats["smooth_method"]))
    print("Gauss points per voxel reconstruction: {0}".format(stats["gauss_point_count"]))
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
        help="Name of the face-reconstructed Gauss point-data scalar array to write.",
    )
    parser.add_argument(
        "--voxel-output",
        default=None,
        help=(
            "Optional output VTK file containing one VTK_VOXEL cell per occupied "
            "voxel with void-ratio values written as CELL_DATA."
        ),
    )
    parser.add_argument(
        "--sphere-clip-order",
        type=int,
        default=4,
        help=(
            "Gauss-Legendre quadrature order used for sphere-box clipping. "
            "Higher values are more accurate but slower."
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
    if args.sphere_clip_order <= 0:
        raise RuntimeError("--sphere-clip-order must be positive.")

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
        void_by_particle, smoothed_void_by_particle, stats, voxel_data = compute_void_ratio_numpy(
            points,
            radii,
            args.particles_per_voxel,
            args.sphere_clip_order,
        )
    else:
        void_by_particle, smoothed_void_by_particle, stats, voxel_data = compute_void_ratio_plain(
            points,
            radii,
            args.particles_per_voxel,
            args.sphere_clip_order,
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

    if args.voxel_output:
        write_voxel_cells_legacy_ascii(
            args.voxel_output,
            voxel_data,
            args.void_field,
            args.smoothed_void_field,
        )
        print("Voxel-cell output: {0}".format(args.voxel_output))

    print_stats(stats, output_path, version_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        sys.exit(1)
