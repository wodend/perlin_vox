use std::vec;

use ndarray::Array3;

use crate::tile_set::TileSet;
use crate::vox::Vox;

/// A 3D tile rotation.
#[derive(Clone, Copy, Debug)]
pub enum Rotation {
    D0,
    D90,
    D180,
    D270,
}

/// Rotate a 3D tile.
pub fn rotate<T: Copy>(array: &Array3<T>, rotation: Rotation) -> Array3<T> {
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

/// A 3D tile map testing rotations.
pub struct TileMap {
    pub values: Array3<usize>,
    pub rotations: Array3<Rotation>,
    pub tileset: TileSet,
}

impl TileMap {
    /// Generate a 3D test tile map.
    pub fn gen() -> TileMap {
        let tileset = Self::tile_set();
        let rs = [Rotation::D0, Rotation::D90, Rotation::D180, Rotation::D270];
        let max_tile_id = tileset.len();
        let size = (max_tile_id, rs.len(), 1);
        let mut values = Array3::from_elem(size, 0);
        let mut rotations = Array3::from_elem(size, Rotation::D0);

        let mut x = 0;
        let mut y = 0;
        for t in 0..tileset.len() {
            for r in rs.iter() {
                values[(x, y, 0)] = t;
                rotations[(x, y, 0)] = *r;
                y += 1;
            }
            x += 1;
        }

        TileMap {
            values,
            rotations,
            tileset,
        }
    }

    /// Generate a 3D test tile set.
    fn tile_set() -> TileSet {
        let mut tiles = TileSet::new(3);
        let file_names = vec!["tiles/road_turn_left.vox"];
        for file_name in file_names {
            let varray = Vox::open(file_name).unwrap().into_varray();
            tiles.insert(varray).unwrap();
        }
        tiles
    }
}

