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
            cells[p] = Cell::path();
            path.push(p);
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
/// The cell is defined by the PointType values for several Points its top face.
/// One in the center, four for the cardinal directions in the center of each
/// side, and four for the intercardinal directions in each corner.
pub struct Cell {
    pub center: PointType,
    pub n: PointType,
    pub ne: PointType,
    pub e: PointType,
    pub se: PointType,
    pub s: PointType,
    pub sw: PointType,
    pub w: PointType,
    pub nw: PointType,
}
impl Cell {
    const DEBUG_VOXELS_SIZE: usize = 3;
    const DIRECTION_SPEC_SIZE: usize = 8;

    /// Create an Empty Cell.
    pub fn empty() -> Cell {
        Cell {
            center: PointType::Empty,
            n: PointType::Empty,
            ne: PointType::Empty,
            e: PointType::Empty,
            se: PointType::Empty,
            s: PointType::Empty,
            sw: PointType::Empty,
            w: PointType::Empty,
            nw: PointType::Empty,
        }
    }

    /// Create a Ground Cell.
    pub fn ground() -> Cell {
        Cell {
            center: PointType::Ground,
            n: PointType::Ground,
            ne: PointType::Ground,
            e: PointType::Ground,
            se: PointType::Ground,
            s: PointType::Ground,
            sw: PointType::Ground,
            w: PointType::Ground,
            nw: PointType::Ground,
        }
    }

    /// Create a Path Cell.
    pub fn path() -> Cell {
        Cell {
            center: PointType::Path,
            n: PointType::Path,
            ne: PointType::Path,
            e: PointType::Path,
            se: PointType::Path,
            s: PointType::Path,
            sw: PointType::Path,
            w: PointType::Path,
            nw: PointType::Path,
        }
    }

    /// Create a straight Path Cell.
    fn path_straight() -> Cell {
        Cell {
            center: PointType::Path,
            n: PointType::Ground,
            ne: PointType::Ground,
            e: PointType::Path,
            se: PointType::Ground,
            s: PointType::Ground,
            sw: PointType::Ground,
            w: PointType::Path,
            nw: PointType::Ground,
        }
    }

    /// Create a three-way turn Path Cell.
    fn path_3way() -> Cell {
        Cell {
            center: PointType::Path,
            n: PointType::Ground,
            ne: PointType::Ground,
            e: PointType::Path,
            se: PointType::Ground,
            s: PointType::Path,
            sw: PointType::Ground,
            w: PointType::Path,
            nw: PointType::Ground,
        }
    }

    /// Create a left turn Path Cell.
    fn path_left() -> Cell {
        Cell {
            center: PointType::Path,
            n: PointType::Ground,
            ne: PointType::Path,
            e: PointType::Ground,
            se: PointType::Ground,
            s: PointType::Ground,
            sw: PointType::Ground,
            w: PointType::Path,
            nw: PointType::Ground,
        }
    }

    /// Create a right turn Path Cell.
    fn path_right() -> Cell {
        Cell {
            center: PointType::Path,
            n: PointType::Ground,
            ne: PointType::Ground,
            e: PointType::Ground,
            se: PointType::Path,
            s: PointType::Ground,
            sw: PointType::Ground,
            w: PointType::Path,
            nw: PointType::Ground,
        }
    }

    /// Create a u-turn Path Cell.
    fn path_uturn() -> Cell {
        Cell {
            center: PointType::Path,
            n: PointType::Ground,
            ne: PointType::Ground,
            e: PointType::Ground,
            se: PointType::Path,
            s: PointType::Ground,
            sw: PointType::Path,
            w: PointType::Ground,
            nw: PointType::Ground,
        }
    }

    /// Create a dead end Path Cell.
    fn path_end() -> Cell {
        Cell {
            center: PointType::Path,
            n: PointType::Ground,
            ne: PointType::Ground,
            e: PointType::Path,
            se: PointType::Ground,
            s: PointType::Ground,
            sw: PointType::Ground,
            w: PointType::Ground,
            nw: PointType::Ground,
        }
    }

    /// Create a new Cell from a direction spec.
    fn from_direction_spec(center: PointType, array: [PointType; Self::DIRECTION_SPEC_SIZE]) -> Cell {
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
    fn into_direction_spec(self) -> [PointType; Self::DIRECTION_SPEC_SIZE] {
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

    /// Get the PointType at a specific Point.
    fn point_type(&self, point: Point) -> PointType {
        match point {
            Point::Center => self.center,
            Point::N => self.n,
            Point::NE => self.ne,
            Point::E => self.e,
            Point::SE => self.se,
            Point::S => self.s,
            Point::SW => self.sw,
            Point::W => self.w,
            Point::NW => self.nw,
        }
    }

    /// Create a Vec of all Path Cells.
    pub fn path_cells() -> Vec<Cell> {
        let mut cells = Vec::new();
        cells.append(&mut Cell::path_straight().rotations(1));
        cells.append(&mut Cell::path_left().rotations(2));
        cells.append(&mut Cell::path_right().rotations(2));
        cells.append(&mut Cell::path_uturn().rotations(2));
        cells.append(&mut Cell::path_end().rotations(2));
        cells.append(&mut Cell::path_3way().rotations(2));
        cells
    }

    /// Creates a Vec of the first four n rotations of a Cell.
    /// 
    /// The minimum rotation is 45 degrees, so n=1 rotates 45 degrees, n=2 rotates 90 degrees, etc.
    fn rotations(&mut self, n: usize) -> Vec<Cell> {
        let mut cells = Vec::new();
        for i in 0..4 {
            let mut ds = self.into_direction_spec();
            ds.rotate_left(i * n);
            let t = Cell::from_direction_spec(self.center, ds);
            cells.push(t);
        }
        cells
    }

    /// Get the PointType at a specific Point.
    pub fn get(&self, p: Point) -> PointType {
        match p {
            Point::Center => self.center,
            Point::N => self.n,
            Point::NE => self.ne,
            Point::E => self.e,
            Point::SE => self.se,
            Point::S => self.s,
            Point::SW => self.sw,
            Point::W => self.w,
            Point::NW => self.nw,
        }
    }

    /// Create the debug voxels for a Cell.
    fn debug_voxels(&self) -> Array3<Rgba> {
        let size = UVec3::splat(Self::DEBUG_VOXELS_SIZE as u32);
        let base = self.center.rgba();
        let mut voxels = Array3::from_elem(size.into_pos(), base);
        let center = IVec3::new(1, 1, 2);
        voxels[center.as_uvec3().into_pos()] = base;
        for d in Point::directions() {
            let p = (center + d.voxel_offset()).as_uvec3().into_pos();
            voxels[p] = self.point_type(d).rgba();
        }
        voxels
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A point type.
pub enum PointType {
    Empty,
    Path,
    Ground,
}
impl PointType {
    fn rgba(&self) -> Rgba {
        match *self {
            PointType::Empty => Rgba([0; 4]),
            PointType::Path => Rgba([51, 51, 51, 255]),
            PointType::Ground => Rgba([120, 159, 138, 255]),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A standard RGBA color.
struct Rgba([u8; 4]);

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A Cell Point.
pub enum Point {
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
impl Point {
    /// Get the directional points.
    fn directions() -> Vec<Point> {
        vec![
            Point::N,
            Point::NE,
            Point::E,
            Point::SE,
            Point::S,
            Point::SW,
            Point::W,
            Point::NW,
        ]
    }

    /// Get the voxel space offset.
    fn voxel_offset(&self) -> IVec3 {
        match *self {
            Point::Center => IVec3::new(0, 0, 0),
            Point::N => IVec3::new(0, 1, 0),
            Point::NE => IVec3::new(1, 1, 0),
            Point::E => IVec3::new(1, 0, 0),
            Point::SE => IVec3::new(1, -1, 0),
            Point::S => IVec3::new(0, -1, 0),
            Point::SW => IVec3::new(-1, -1, 0),
            Point::W => IVec3::new(-1, 0, 0),
            Point::NW => IVec3::new(-1, 1, 0),
        }
    }


    
    pub fn opposite(&self) -> Point {
        match self {
            Point::Center => Point::Center,
            Point::N => Point::S,
            Point::NE => Point::SW,
            Point::E => Point::W,
            Point::SE => Point::NW,
            Point::S => Point::N,
            Point::SW => Point::NE,
            Point::W => Point::E,
            Point::NW => Point::SE,
        }
    }
}
