use std::vec;

use glam::{IVec3, UVec3};
use ndarray::Array3;

use crate::noise::Perlin1;
use crate::vector::{Pos3, Vector3};

/// A cubic honeycomb for Cells.
pub struct Honeycomb {
    pub cells: Array3<Cell>,
    pub path: Vec<(usize, usize, usize)>,
}
impl Honeycomb {
    pub fn gen_perlin(seed: u32) -> Honeycomb {
        // Initialize Honeycomb array.
        let size = UVec3::new(32, 32, 4);
        let center = size / 2;
        let mut cells = Array3::from_elem(size.into_pos(), Cell::empty());
        let mut path = Vec::new();
        let mut sprawl_path = Vec::new();

        // Generate 1D Perlin noise path.
        let perlin_scale = 0.08;
        let mut noise_path = Perlin1::gen(size.x as usize, seed, perlin_scale);
        noise_path.normalize();

        // Generate 1D Perlin noise for the width of sprawl from the path.
        let mut noise_sprawl = Perlin1::gen(size.x as usize, seed + 1, perlin_scale);
        noise_sprawl.normalize();

        // Define voxel space scaling.
        let path_scale = 7.0;
        let sprawl_scale = 5.0;
        let mut prev_y = None;
        for ((x, n_p), n_s) in noise_path.values.indexed_iter().zip(noise_sprawl.values) {
            // Scale path noise value.
            let y = center.y + (n_p * path_scale) as u32;
            // Scale sprawl noise value.
            let s = ((n_s + 1.0) * sprawl_scale) as i32;
            // Set sprawl to Ground Cells.
            for i in 1..s {
                let p0 = (x as usize, (y as i32 - i) as usize, 0);
                let p1 = (x as usize, (y as i32 + i) as usize, 0);
                cells[p0] = Cell::ground();
                cells[p1] = Cell::ground();
                sprawl_path.push(p0);
                sprawl_path.push(p1);
            }
            // Set path to Path Cells.
            let p = (x as usize, y as usize, 0);
            // Handle cases where y values are not adjacent.
            if let Some(p_y) = prev_y {
                for i in 1..(y as i32 - p_y) {
                    let p = (x as usize, (p_y + i) as usize, 0);
                    cells[p] = Cell::path();
                    path.push(p);
                }
                cells[p] = Cell::path();
                path.push(p);
            } else {
                cells[p] = Cell::path();
                path.push(p);
            }
            prev_y = Some(y as i32);
        }
        path.append(&mut sprawl_path);

        Honeycomb {
            cells,
            path
        }
    }

    fn size(&self) -> UVec3 {
        self.cells.dim().into_uvec3()
    }

    pub fn debug_render(&self) -> Array3<[u8; 4]> {
        let size = self.size() * Cell::DEBUG_VOXELS_SIZE as u32;
        let mut voxels = Array3::from_elem(size.into_pos(), [0; 4]);
        for (xyz, cell) in self.cells.indexed_iter() {
            let cell_voxels = cell.debug_voxels();
            let cell_vector = xyz.into_uvec3() * Cell::DEBUG_VOXELS_SIZE as u32;
            for (xyz, rgba) in cell_voxels.indexed_iter() {
                let vector = cell_vector + xyz.into_uvec3();
                voxels[vector.into_pos()] = rgba.0;
            }
        }
        voxels
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A cubic honeycomb cell.
/// 
/// The cell is defined by the CellPointType values for several CellPoints its top face.
/// One in the center, four for the cardinal directions in the center of each
/// side, and four for the intercardinal directions in each corner.
pub struct Cell {
    pub center: CellPointType,
    pub n: CellPointType,
    pub ne: CellPointType,
    pub e: CellPointType,
    pub se: CellPointType,
    pub s: CellPointType,
    pub sw: CellPointType,
    pub w: CellPointType,
    pub nw: CellPointType,
}
impl Cell {
    const DEBUG_VOXELS_SIZE: usize = 3;
    const DIRECTION_SPEC_SIZE: usize = 8;

    /// Create an Empty Cell.
    pub fn empty() -> Cell {
        Cell {
            center: CellPointType::Empty,
            n: CellPointType::Empty,
            ne: CellPointType::Empty,
            e: CellPointType::Empty,
            se: CellPointType::Empty,
            s: CellPointType::Empty,
            sw: CellPointType::Empty,
            w: CellPointType::Empty,
            nw: CellPointType::Empty,
        }
    }

    /// Create a Ground Cell.
    pub fn ground() -> Cell {
        Cell {
            center: CellPointType::Ground,
            n: CellPointType::Ground,
            ne: CellPointType::Ground,
            e: CellPointType::Ground,
            se: CellPointType::Ground,
            s: CellPointType::Ground,
            sw: CellPointType::Ground,
            w: CellPointType::Ground,
            nw: CellPointType::Ground,
        }
    }

    /// Create a Path Cell.
    pub fn path() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Path,
            ne: CellPointType::Path,
            e: CellPointType::Path,
            se: CellPointType::Path,
            s: CellPointType::Path,
            sw: CellPointType::Path,
            w: CellPointType::Path,
            nw: CellPointType::Path,
        }
    }

    /// Create a straight Path Cell.
    fn path_straight() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Ground,
            ne: CellPointType::Ground,
            e: CellPointType::Path,
            se: CellPointType::Ground,
            s: CellPointType::Ground,
            sw: CellPointType::Ground,
            w: CellPointType::Path,
            nw: CellPointType::Ground,
        }
    }

    /// Create a three-way turn Path Cell.
    fn path_3way() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Ground,
            ne: CellPointType::Ground,
            e: CellPointType::Path,
            se: CellPointType::Ground,
            s: CellPointType::Path,
            sw: CellPointType::Ground,
            w: CellPointType::Path,
            nw: CellPointType::Ground,
        }
    }

    /// Create a three-way turn Path Cell.
    fn path_4way() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Path,
            ne: CellPointType::Ground,
            e: CellPointType::Path,
            se: CellPointType::Ground,
            s: CellPointType::Path,
            sw: CellPointType::Ground,
            w: CellPointType::Path,
            nw: CellPointType::Ground,
        }
    }

    /// Create a left turn Path Cell.
    fn path_left() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Ground,
            ne: CellPointType::Path,
            e: CellPointType::Ground,
            se: CellPointType::Ground,
            s: CellPointType::Ground,
            sw: CellPointType::Ground,
            w: CellPointType::Path,
            nw: CellPointType::Ground,
        }
    }

    /// Create a 3 way left turn Path Cell.
    fn path_3left() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Ground,
            ne: CellPointType::Path,
            e: CellPointType::Ground,
            se: CellPointType::Ground,
            s: CellPointType::Path,
            sw: CellPointType::Ground,
            w: CellPointType::Path,
            nw: CellPointType::Ground,
        }
    }

    /// Create a right turn Path Cell.
    fn path_right() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Ground,
            ne: CellPointType::Ground,
            e: CellPointType::Ground,
            se: CellPointType::Path,
            s: CellPointType::Ground,
            sw: CellPointType::Ground,
            w: CellPointType::Path,
            nw: CellPointType::Ground,
        }
    }

    /// Create a 3 way right turn Path Cell.
    fn path_3right() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Path,
            ne: CellPointType::Ground,
            e: CellPointType::Ground,
            se: CellPointType::Path,
            s: CellPointType::Ground,
            sw: CellPointType::Ground,
            w: CellPointType::Path,
            nw: CellPointType::Ground,
        }
    }


    /// Create a u-turn Path Cell.
    fn path_uturn() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Ground,
            ne: CellPointType::Ground,
            e: CellPointType::Ground,
            se: CellPointType::Path,
            s: CellPointType::Ground,
            sw: CellPointType::Path,
            w: CellPointType::Ground,
            nw: CellPointType::Ground,
        }
    }

    /// Create a dead end Path Cell.
    fn path_end() -> Cell {
        Cell {
            center: CellPointType::Path,
            n: CellPointType::Ground,
            ne: CellPointType::Ground,
            e: CellPointType::Path,
            se: CellPointType::Ground,
            s: CellPointType::Ground,
            sw: CellPointType::Ground,
            w: CellPointType::Ground,
            nw: CellPointType::Ground,
        }
    }

    /// Create a new Cell from a direction spec.
    fn from_direction_spec(center: CellPointType, array: [CellPointType; Self::DIRECTION_SPEC_SIZE]) -> Cell {
        Cell {
            center: center,
            n: array[0],
            ne: array[1],
            e: array[2],
            se: array[3],
            s: array[4],
            sw: array[5],
            w: array[6],
            nw: array[7],
        }
    }

    /// Create an direction spec from a Cell.
    fn into_direction_spec(self) -> [CellPointType; Self::DIRECTION_SPEC_SIZE] {
        [
            self.n,
            self.ne,
            self.e,
            self.se,
            self.s,
            self.sw,
            self.w,
            self.nw,
        ]
    }

    /// Create a Vec of all Path Cells.
    pub fn path_cells() -> Vec<Cell> {
        let mut cells = Vec::new();
        cells.append(&mut Cell::path_4way().rotations(1, 1));
        cells.append(&mut Cell::path_straight().rotations(4, 1));
        cells.append(&mut Cell::path_left().rotations(4, 2));
        cells.append(&mut Cell::path_right().rotations(4, 2));
        cells.append(&mut Cell::path_uturn().rotations(4, 2));
        cells.append(&mut Cell::path_end().rotations(4, 2));
        cells.append(&mut Cell::path_3left().rotations(4, 2));
        cells.append(&mut Cell::path_3right().rotations(4, 2));
        cells.append(&mut Cell::path_3way().rotations(6, 1));
        cells
    }

    /// Creates a Vec of the first four n rotations of a Cell.
    /// 
    /// The minimum rotation is 45 degrees, so r=1 rotates 45 degrees, r=2 rotates 90 degrees, etc.
    fn rotations(&mut self, n: usize, r: usize) -> Vec<Cell> {
        let mut cells = Vec::new();
        for i in 0..n {
            let mut ds = self.into_direction_spec();
            ds.rotate_left(i * r);
            let t = Cell::from_direction_spec(self.center, ds);
            cells.push(t);
        }
        cells
    }

    /// Get all subsets of a set of Points.
    pub fn subsets(nums: Vec<CellPoint>) -> Vec<Vec<CellPoint>> {
        let mut result = vec![vec![]];

        for num in nums {
            let len = result.len();
            for i in 0..len {
                let mut subset = result[i].clone();
                subset.push(num);
                result.push(subset);
            }
        }

        result
    }

    /// Get the CellPointType at a specific CellPoint.
    pub fn get(&self, p: CellPoint) -> CellPointType {
        match p {
            CellPoint::Center => self.center,
            CellPoint::N => self.n,
            CellPoint::NE => self.ne,
            CellPoint::E => self.e,
            CellPoint::SE => self.se,
            CellPoint::S => self.s,
            CellPoint::SW => self.sw,
            CellPoint::W => self.w,
            CellPoint::NW => self.nw,
        }
    }

    /// Create the debug voxels for a Cell.
    fn debug_voxels(&self) -> Array3<Rgba> {
        let size = UVec3::splat(Self::DEBUG_VOXELS_SIZE as u32);
        let base = self.center.rgba();
        let mut voxels = Array3::from_elem(size.into_pos(), base);
        let center = IVec3::new(1, 1, 2);
        voxels[center.as_uvec3().into_pos()] = base;
        for d in CellPoint::directions() {
            let p = (center + d.voxel_offset()).as_uvec3().into_pos();
            voxels[p] = self.get(d).rgba();
        }
        voxels
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A point type.
pub enum CellPointType {
    Empty,
    Path,
    Ground,
    Building,
}
impl CellPointType {
    fn rgba(&self) -> Rgba {
        match *self {
            CellPointType::Empty => Rgba([0; 4]),
            CellPointType::Path => Rgba([51, 51, 51, 255]),
            CellPointType::Ground => Rgba([120, 159, 138, 255]),
            CellPointType::Building => Rgba([103, 127, 163, 255]),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A standard RGBA color.
struct Rgba([u8; 4]);

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A Cell CellPoint.
pub enum CellPoint {
    Center,
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}
impl CellPoint {
    /// Get the directional points.
    fn directions() -> Vec<CellPoint> {
        vec![
            CellPoint::N,
            CellPoint::NE,
            CellPoint::E,
            CellPoint::SE,
            CellPoint::S,
            CellPoint::SW,
            CellPoint::W,
            CellPoint::NW,
        ]
    }

    /// Get the adjacent directional points, including self.
    pub fn adjacent(&self) -> Vec<CellPoint> {
        match self {
            CellPoint::N => vec![CellPoint::N, CellPoint::NE, CellPoint::NW],
            CellPoint::E => vec![CellPoint::E, CellPoint::NE, CellPoint::SE],
            CellPoint::S => vec![CellPoint::S, CellPoint::SE, CellPoint::SW],
            CellPoint::W => vec![CellPoint::W, CellPoint::SW, CellPoint::NW],
            e => vec![*e],
        }
    }

    /// Get the voxel space offset.
    fn voxel_offset(&self) -> IVec3 {
        match *self {
            CellPoint::Center => IVec3::new(0, 0, 0),
            CellPoint::N => IVec3::new(0, 1, 0),
            CellPoint::NE => IVec3::new(1, 1, 0),
            CellPoint::E => IVec3::new(1, 0, 0),
            CellPoint::SE => IVec3::new(1, -1, 0),
            CellPoint::S => IVec3::new(0, -1, 0),
            CellPoint::SW => IVec3::new(-1, -1, 0),
            CellPoint::W => IVec3::new(-1, 0, 0),
            CellPoint::NW => IVec3::new(-1, 1, 0),
        }
    }


    
    pub fn opposite(&self) -> CellPoint {
        match self {
            CellPoint::Center => CellPoint::Center,
            CellPoint::N => CellPoint::S,
            CellPoint::NE => CellPoint::SW,
            CellPoint::E => CellPoint::W,
            CellPoint::SE => CellPoint::NW,
            CellPoint::S => CellPoint::N,
            CellPoint::SW => CellPoint::NE,
            CellPoint::W => CellPoint::E,
            CellPoint::NW => CellPoint::SE,
        }
    }
}
