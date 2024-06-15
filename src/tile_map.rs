use std::vec;

use glam::{Vec2, Vec3};
use ndarray::{s, Array3, ArrayBase, Dimension, OwnedRepr};

use crate::vox::Vox;
use crate::tile_set::TileSet;
use crate::noise::{narray1, narray2};

#[derive(Clone, Copy, Debug)]
pub enum Rotation {
    D0,
    D90,
    D180,
    D270,
}

/// Normalize an ndarray of f32.
pub fn normalize<S>(narray: &mut ArrayBase<OwnedRepr<f32>, S>)
where
    S: Dimension,
{
    let min = *narray
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max = *narray
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    narray.mapv_inplace(|x| (x - min) / (max - min));
}

// TODO:
// - Move to a new file named tile_map.rs
// - Add a new method for perlin_tile_bridge
// - Use initial perlin noise as the locations for buildings in a waterway
// - Generate a graph between the building nodes
// - Use the graph to generate a road network
// https://cdnb.artstation.com/p/assets/images/images/039/417/121/4k/mari-k-ruin-lowrez.jpg?1625837233
pub struct TileMap {
    pub values: Array3<usize>,
    pub rotations: Array3<Rotation>,
    pub tiles: TileSet,
}

fn tile_set() -> TileSet {
    let mut tiles = TileSet::new(3);
    let file_names = vec![
        "tiles/debug_none.vox",
        "tiles/debug_full.vox",
        "tiles/rot_098.vox",
        "tiles/rot_079.vox",
        "tiles/rot_058.vox",
    ];
    for file_name in file_names {
        let varray = Vox::open(file_name).unwrap().into_varray();
        tiles.insert(varray).unwrap();
    }
    tiles
}

/// Generate a 2D tile array of Perlin noise.
pub fn tile_map() -> TileMap {
    let tiles = tile_set();
    let max_tile_id = tiles.len();
    let size = (4, max_tile_id, 1);
    let mut values = Array3::from_elem(size, 0);
    let mut rotations = Array3::from_elem(size, Rotation::D0);

    let mut tile_id = 0;
    for (xyz, v) in values.indexed_iter_mut() {
        *v = tile_id;
        tile_id += 1;
        if tile_id >= max_tile_id {
            tile_id = 0;
        }

        if xyz.0 == 1 {
            rotations[xyz] = Rotation::D90;
        } else if xyz.0 == 2 {
            rotations[xyz] = Rotation::D180;
        } else if xyz.0 == 3 {
            rotations[xyz] = Rotation::D270;
        }
    }

    TileMap { values, rotations, tiles }
}

pub fn perlin_tile() -> TileMap {
    let length = 8;
    let size = (length, length, length);
    let mut values = Array3::from_elem(size, 0);
    let mut rotations = Array3::from_elem(size, Rotation::D0);
    let tiles = tile_set();

    let mut noise = narray2((size.0, size.1), Vec2::new(0.1, 0.1));
    normalize(&mut noise.values);

    let map_scale = length as f32 / 4.0;
    let z = Vec3::new(0.0, 0.0, 1.0);
    for ((xy, n), d) in noise.values.indexed_iter().zip(noise.derivatives.iter()) {
        let surface = (map_scale - 1.0 + n * map_scale) as usize;
        let pos = (xy.0 as usize, xy.1 as usize, surface);
        for (z, voxel) in values.slice_mut(s![xy.0, xy.1, ..]).indexed_iter_mut() {
            if z < surface {
                *voxel = 1;
            }
        }
        let normal = Vec3::new(-d.x, -d.y, 1.0).normalize();
        let z_angle = z.angle_between(normal);

        let tile_id = if z_angle < 0.58 {
            1
        } else if z_angle < 0.79 {
            2
        } else if z_angle < 0.98 {
            3
        } else if z_angle < 3.14 {
            4
        } else {
            1
        };
        let x_lt_y = normal.x.abs() < normal.y.abs();
        let rotation = if x_lt_y && normal.y >= 0.0 {
            Rotation::D180
        } else if !x_lt_y && normal.x >= 0.0 {
            Rotation::D270
        } else if !x_lt_y && normal.y < 0.0 {
            Rotation::D90
        } else {
            Rotation::D0
        };
        values[pos] = tile_id;
        rotations[pos] = rotation;
    }

    TileMap { values, rotations, tiles }
}

pub fn test_1d_perlin() -> TileMap {
    let length = 8;
    let size = (length * 10, length, length);
    let mut values = Array3::from_elem(size, 0);
    let mut rotations = Array3::from_elem(size, Rotation::D0);
    let tiles = tile_set();

    let mut noise = narray1(size.0, 0.1);
    normalize(&mut noise);

    let map_scale = length as f32 - 1.0;
    let z = Vec3::new(0.0, 0.0, 1.0);
    for (x, n) in noise.indexed_iter() {
        let y = (n * map_scale) as usize;
        let pos = (x, y, 0);
        values[pos] = 1;
    }

    TileMap { values, rotations, tiles }
}

pub fn test2_1d_perlin() -> TileMap {
    let length = 8;
    let size = (length * 10, length, length);
    let mut values = Array3::from_elem(size, 0);
    let mut rotations = Array3::from_elem(size, Rotation::D0);
    let tiles = tile_set();

    let mut noise = narray1(size.0, 0.1);
    normalize(&mut noise);

    let map_scale = length as f32 - 1.0;
    let z = Vec3::new(0.0, 0.0, 1.0);
    for (x, n) in noise.indexed_iter() {
        let y = (n * map_scale) as usize;
        let pos = (x, y, 0);
        values[pos] = 1;
    }

    // for (x, t) in values.indexed_iter() {
    // }
    println!("{:?}", values);

    TileMap { values, rotations, tiles }
}
