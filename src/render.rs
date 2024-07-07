use std::cmp::max;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_6};

use enterpolation::{linear::ConstEquidistantLinear, Curve};
use glam::{IVec3, Quat, Vec3};
use ndarray::{Array3, ArrayBase, Dimension, OwnedRepr};
use palette::{LinSrgba, Srgba};

use crate::noise::{Perlin1, Perlin2, Perlin3};
use crate::tile_map::{rotate, TileMap};
use crate::vector::{Dim3, Vector3};

/// Generate a full color spectrum heatmap gradient.
fn heatmap_gradient(gradient_size: usize) -> Vec<LinSrgba> {
    let gradient = ConstEquidistantLinear::<f32, _, 7>::equidistant_unchecked([
        Srgba::new(0.0, 0.0, 0.0, 255.0).into_linear(),
        Srgba::new(0.0, 0.0, 1.0, 255.0).into_linear(),
        Srgba::new(0.0, 1.0, 1.0, 255.0).into_linear(),
        Srgba::new(0.0, 1.0, 0.0, 255.0).into_linear(),
        Srgba::new(1.0, 1.0, 0.0, 255.0).into_linear(),
        Srgba::new(1.0, 0.0, 0.0, 255.0).into_linear(),
        Srgba::new(1.0, 1.0, 1.0, 255.0).into_linear(),
    ]);
    gradient.take(gradient_size).collect()
}

/// Generate a binary heatmap gradient black to white.
pub fn binary_heatmap_gradient(gradient_size: usize) -> Vec<LinSrgba> {
    let gradient = ConstEquidistantLinear::<f32, _, 2>::equidistant_unchecked([
        Srgba::new(0.0, 0.0, 0.0, 255.0).into_linear(),
        Srgba::new(1.0, 1.0, 1.0, 255.0).into_linear(),
    ]);
    gradient.take(gradient_size).collect()
}

/// Generate a line between two points using Bresemham's line algorithm.
fn line(p0: IVec3, p1: IVec3) -> Vec<IVec3> {
    let delta_x = (p1.x - p0.x).abs();
    let delta_y = (p1.y - p0.y).abs();
    let delta_z = (p1.z - p0.z).abs();
    let delta_max = max(max(delta_x, delta_y), delta_z);
    let step_x = if p0.x < p1.x { 1 } else { -1 };
    let step_y = if p0.y < p1.y { 1 } else { -1 };
    let step_z = if p0.z < p1.z { 1 } else { -1 };
    let mut error = delta_max;
    let mut error_x = delta_max / 2;
    let mut error_y = delta_max / 2;
    let mut error_z = delta_max / 2;
    let (mut x, mut y, mut z) = (p0.x, p0.y, p0.z);
    let mut line = Vec::new();
    while error >= 0 {
        line.push(IVec3::new(x, y, z));
        error_x -= delta_x;
        if error_x < 0 {
            error_x += delta_max;
            x += step_x;
        }
        error_y -= delta_y;
        if error_y < 0 {
            error_y += delta_max;
            y += step_y;
        }
        error_z -= delta_z;
        if error_z < 0 {
            error_z += delta_max;
            z += step_z;
        }
        error -= 1;
    }
    line.push(IVec3::new(x, y, z));
    line
}

/// A 3D array of RGBA 1D Perlin heatmap voxels.
pub struct PerlinHeatmap1 {
    pub values: Array3<[u8; 4]>,
}
impl PerlinHeatmap1 {
    /// Generate a 3D array of RGBA 1D Perlin heatmap voxels.
    pub fn gen(noise: &mut Perlin1) -> PerlinHeatmap1 {
        let gradient_size = 16;
        let gradient = binary_heatmap_gradient(gradient_size);
        let scalar = gradient_size as f32 - 1.0;

        let voxels_size = (noise.values.dim(), 1, 1);
        let mut values = Array3::from_elem(voxels_size, [0; 4]);

        noise.normalize();

        for (x, n) in noise.values.indexed_iter() {
            let index = (n * scalar) as usize;
            let linsrgba = gradient[index];
            values[[x, 0, 0]] = Srgba::from(linsrgba).into()
        }

        PerlinHeatmap1 { values }
    }
}

/// A 3D array of RGBA 2D Perlin heatmap voxels.
pub struct PerlinHeatmap2 {
    pub values: Array3<[u8; 4]>,
}
impl PerlinHeatmap2 {
    /// Generate a 3D array of RGBA 2D Perlin heatmap voxels.
    pub fn gen(noise: &mut Perlin2) -> PerlinHeatmap2 {
        let gradient_size = 16;
        let gradient = binary_heatmap_gradient(gradient_size);
        let scalar = gradient_size as f32 - 1.0;

        let d = noise.values.dim();
        let voxels_size = (d.0, d.1, 1);
        let mut values = Array3::from_elem(voxels_size, [0; 4]);

        noise.normalize();

        for ((x, y), n) in noise.values.indexed_iter() {
            let index = (n * scalar) as usize;
            let linsrgba = gradient[index];
            values[[x, y, 0]] = Srgba::from(linsrgba).into()
        }

        PerlinHeatmap2 { values }
    }
}

/// A 3D array of RGBA 3D Perlin heatmap voxels.
pub struct PerlinHeatmap3 {
    pub values: Array3<[u8; 4]>,
}
impl PerlinHeatmap3 {
    /// Generate a 3D array of RGBA 3D Perlin heatmap voxels.
    pub fn gen(noise: &mut Perlin3) -> PerlinHeatmap3 {
        let gradient_size = 16;
        let gradient = binary_heatmap_gradient(gradient_size);
        let scalar = gradient_size as f32 - 1.0;

        let d = noise.values.dim();
        let mut values = Array3::from_elem(d, [0; 4]);

        noise.normalize();

        for ((x, y, z), n) in noise.values.indexed_iter() {
            let index = (n * scalar) as usize;
            let linsrgba = gradient[index];
            values[[x, y, z]] = Srgba::from(linsrgba).into()
        }

        PerlinHeatmap3 { values }
    }
}

/// A 3D array of RGBA 3D voxel lines.
pub struct LineMap {
    pub values: Array3<[u8; 4]>,
}
impl LineMap {
    /// Generate a 3D test array of 3D voxel lines.
    pub fn gen() -> LineMap {
        let gradient_size = 31;
        let gradient = heatmap_gradient(gradient_size);
        let mut g_i = 0;

        let y_axis_rotations = [0.0, FRAC_PI_6, FRAC_PI_3, 2.0 * FRAC_PI_3, 5.0 * FRAC_PI_6];
        let z_axis_rotations = [
            0.0,
            FRAC_PI_6,
            FRAC_PI_3,
            FRAC_PI_2,
            2.0 * FRAC_PI_3,
            5.0 * FRAC_PI_6,
        ];

        let size = (64, 64, 64);
        let len = 16;
        let size_v = size.into_ivec3();
        let center = size_v / 2;
        let base = Vec3::new(1.0, 0.0, 0.0);
        let y_axis = Vec3::new(0.0, 1.0, 0.0);
        let z_axis = Vec3::new(0.0, 0.0, 1.0);
        let mut values = Array3::from_elem((64, 64, 64), [0; 4]);
        for y_axis_rotation in y_axis_rotations.iter() {
            for z_axis_rotation in z_axis_rotations.iter() {
                let r_y = Quat::from_axis_angle(y_axis, *y_axis_rotation);
                let r_z = Quat::from_axis_angle(z_axis, *z_axis_rotation);
                let v = r_z * (r_y * base);
                let p0 = center - (v * len as f32).as_ivec3();
                let p1 = center + (v * len as f32).as_ivec3();

                if *y_axis_rotation == FRAC_PI_3 && *z_axis_rotation == FRAC_PI_2 {
                    let p0 = center - (z_axis * len as f32).as_ivec3();
                    let p1 = center + (z_axis * len as f32).as_ivec3();
                    let line = line(p0, p1);
                    for pos in line.iter() {
                        let p = pos.as_uvec3().into_size3();
                        let linsrgba = gradient[g_i];
                        values[p] = Srgba::from(linsrgba).into();
                    }
                    g_i += 1;
                }
                let line = line(p0, p1);
                for pos in line.iter() {
                    let p = pos.as_uvec3().into_size3();
                    let linsrgba = gradient[g_i];
                    values[p] = Srgba::from(linsrgba).into();
                }
                g_i += 1;
            }
        }

        LineMap { values }
    }
}

/// A 3D render of a voxel TileMap.
pub struct Render {
    pub values: Array3<[u8; 4]>,
}
impl Render {
    /// Render a voxel TileMap.
    pub fn gen(tile_map: TileMap) -> Render {
        let size = tile_map.values.dim().into_uvec3() * tile_map.tileset.tile_size as u32;
        let mut values = Array3::from_elem(size.into_size3(), [0; 4]);

        for ((xyz, tile_id), r) in tile_map
            .values
            .indexed_iter()
            .zip(tile_map.rotations.iter())
        {
            let tile = tile_map.tileset.get(*tile_id as usize).unwrap();
            let tile_r = rotate(tile, *r);
            let tile_vector = xyz.into_uvec3() * tile_map.tileset.tile_size as u32;
            for (xyz, rgba) in tile_r.indexed_iter() {
                let vector = tile_vector + xyz.into_uvec3();
                values[vector.into_size3()] = *rgba;
            }
        }

        Render { values }
    }
}
