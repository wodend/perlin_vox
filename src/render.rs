use enterpolation::{linear::ConstEquidistantLinear, Curve};
use ndarray::{Array3, s};
use palette::{LinSrgba, Srgba};
use glam::{IVec3, Vec2Swizzles};
use std::cmp::max;

use crate::vector::{Pos2, Pos3, Vector3};
use crate::noise::Noise;
use crate::tile_map::{Rotation, TileMap, normalize};

// Geneate a heatmap gradient.
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
    gradient.take(gradient_size + 1).collect()
}

pub fn scalar_normalize(x: f32, min: f32, max: f32) -> f32 {
    (x - min) / (max - min)
}

// Generate a 3D ndarray of RGBA voxels from a 3D noise ndarray.
pub fn varray3(narray: &mut Array3<f32>) -> Array3<[u8; 4]> {
    let gradient_size = 32;
    let gradient = heatmap_gradient(gradient_size);
    let scalar = gradient_size as f32;

    normalize(narray);

    // .2, .4, .6

    let varray = narray.mapv(|noise| {
        let linsrgba = if noise > 0.5 {
            Srgba::new(0.0, 0.0, 0.0, 0.0).into_linear()
        } else {
            let n = scalar_normalize(noise, 0.0, 0.5);
            let index = (n * scalar) as usize;
            gradient[index]
        };
        Srgba::from(linsrgba).into()
    });

    varray
}

// Bresenham's line algorithm.
pub fn line(p0: IVec3, p1: IVec3) -> Vec<IVec3> {
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

// Generate a 3D ndarray of RGBA voxels from a 2D noise ndarray.
pub fn varray2(narray: &mut Noise) -> Array3<[u8; 4]> {
    let s = narray.values.dim();
    let gradient_size = s.0;
    let gradient = heatmap_gradient(gradient_size);
    let scalar = gradient_size as f32 / 2.0;

    let vs = s.into_uvec2().xyx().into_pos();
    let mut varray = Array3::from_elem(vs, [0; 4]);

    normalize(&mut narray.values);

    for ((xy, n), _) in narray.values.indexed_iter().zip(narray.derivatives.iter()) {
        let surface = (scalar - 1.0 + n * scalar) as usize;
        let linsrgba = gradient[surface];
        for (z, voxel) in varray.slice_mut(s![xy.0, xy.1, ..]).indexed_iter_mut() {
            // Uncomment to fill
            // if z < index {
            //     *voxel = Srgba::from(linsrgba).into();
            // }
            if z == surface {
                *voxel = Srgba::from(linsrgba).into();
            }
        }
    }

    varray
}

fn rotate<T: Copy>(array: &Array3<T>, rotation: Rotation) -> Array3<T> {
    let mut rotated = array.clone();
    match rotation {
        Rotation::D0 => rotated,
        Rotation::D90 => {
            rotated.invert_axis(ndarray::Axis(0));
            rotated.permuted_axes([1, 0, 2])
        }
        Rotation::D180 => {
            rotated.invert_axis(ndarray::Axis(0));
            rotated.invert_axis(ndarray::Axis(1));
            rotated
        }
        Rotation::D270 => {
            rotated.invert_axis(ndarray::Axis(1));
            rotated.permuted_axes([1, 0, 2])
        }
    }
}

pub fn tile_array(tile_map: TileMap) -> Array3<[u8; 4]> {
    let size = tile_map.values.dim().into_uvec3() * tile_map.tiles.tile_size as u32;
    let mut varray = Array3::from_elem(size.into_pos(), [0; 4]);

    for ((xyz, tile_id), r) in tile_map.values.indexed_iter().zip(tile_map.rotations.iter()) {
        let tile = tile_map.tiles.get(*tile_id as usize).unwrap();
        let tile_r = rotate(tile, *r);
        let tile_vector = xyz.into_uvec3() * tile_map.tiles.tile_size as u32;
        for (xyz, rgba) in tile_r.indexed_iter() {
            let vector = tile_vector + xyz.into_uvec3();
            varray[vector.into_pos()] = *rgba;
        }
    }
    varray
}