use std::vec;

use glam::{IVec3, UVec3};
use ndarray::{s, Array1, Array3, Array4, ArrayView1, Axis, SliceInfo, SliceInfoElem, Dim};
use ndarray::Zip;

use crate::noise::Perlin1;
use crate::vector::{Size3, Dim3, Vector3, Vector4};

pub struct Honeycomb2 {
    size: UVec3,
    point_types: Array3<PointType>,
    points: Array3<PointType>,
}

impl Honeycomb2 {
    const CHUNK_SIZE: UVec3 = Cell2::SIZE;

    pub fn new(size: UVec3) -> Honeycomb2 {
        let s = size * Self::CHUNK_SIZE;
        Honeycomb2 {
            size: size,
            point_types: Array3::from_elem(size.into_size3(), PointType::Empty),
            points: Array3::from_elem(s.into_size3(), PointType::Empty),
        }
    }

    pub fn from(cellset: Cellset2) -> Honeycomb2 {
        let size = UVec3::new(80, 80, 1);
        let mut honeycomb = Honeycomb2::new(size);
        for id in 0..cellset.len() {
            if id < 1600 {
                let cell = cellset.get(CellId(id));
                let x = id as u32 / 40;
                let y = id as u32 % 40;
                let xyz = UVec3::new(x * 2, y * 2, 0);
                honeycomb.insert(xyz, cell);
            }
        }
        honeycomb
    }

    pub fn size(&self) -> UVec3 {
        self.size
    }

    pub fn insert(&mut self, xyz: UVec3, cell: Cell2) {
        self.point_types[xyz.into_size3()] = cell.point_type;
        self.points.slice_mut(Self::slice(xyz)).assign(&cell.points);
    }

    pub fn get(&mut self, xyz: UVec3) -> Option<Cell2> {
        if xyz.cmplt(self.size).all() {
            Some(self.get_unchecked(xyz))
        } else {
            None
        }
    }

    fn get_unchecked(&self, xyz: UVec3) -> Cell2 {
        Cell2 {
            point_type: self.point_types[xyz.into_size3()],
            points: self.points.slice(Self::slice(xyz)).to_owned(),
        }
    }

    fn slice(xyz: UVec3) -> SliceInfo<[SliceInfoElem; 3], Dim<[usize; 3]>, Dim<[usize; 3]>> {
        let xyz_0 = xyz * Self::CHUNK_SIZE;
        let xyz_1 = xyz_0 + Self::CHUNK_SIZE;
        s![
            xyz_0.x as usize..xyz_1.x as usize,
            xyz_0.y as usize..xyz_1.y as usize,
            xyz_0.z as usize..xyz_1.z as usize,
        ]
    }                       

    pub fn debug_render(&self) -> Array3<[u8; 4]> {
        let voxels = self.points.mapv(|t| t.rgba().0);
        voxels
    }
}
/// A set of Cell2s.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Cellset2 {
    point_types: Vec<PointType>,
    points: Array4<PointType>,
}

impl Cellset2 {
    /// Create a new empty Cellset2.
    pub fn new() -> Cellset2 {
        Cellset2 {
            point_types: Vec::new(),
            points: Array4::from_elem(Cell2::SIZE.extend(0).into_size4(), PointType::Empty),
        }
    }

    /// Push a Cell2 into the Cellset2.
    pub fn push(&mut self, cell: Cell2) {
        self.point_types.push(cell.point_type);
        self.points.push(Axis(3), cell.points.view()).unwrap();
    }

    /// Append a Vec of Cell2s into the Cellset2.
    pub fn append(&mut self, cells: Vec<Cell2>) {
        for c in cells {
            self.point_types.push(c.point_type);
            self.points.push(Axis(3), c.points.view()).unwrap();
        }
    }

    /// Get a Cell2 from the Cellset2 by CellId.
    pub fn get(&self, cell_id: CellId) -> Cell2 {
        Cell2 {
            point_type: self.point_types[cell_id.0],
            points: self.points.index_axis(Axis(3), cell_id.0).to_owned(),
        }
    }

    /// Get the number of Cell2s in the set.
    pub fn len(&self) -> usize {
        self.points.len_of(Axis(3))
    }

    /// Get all cell PointTypes.
    pub fn point_types(&self) -> &Vec<PointType> {
        &self.point_types
    }

    /// Get all PointTypes for a given `xyz`.
    pub fn points(&self, xyz: Size3) -> ArrayView1<PointType> {
        self.points.slice(s![xyz.0, xyz.1, xyz.2, ..])
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct CellId(pub usize);

/// A cubic honeycomb cell.
/// 
/// A cell is defined by 27 points in a 3x3x3 grid and a PointType.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Cell2 {
    point_type: PointType,
    points: Array3<PointType>,
}

impl Cell2 {
    const SIZE: UVec3 = UVec3::new(3, 3, 3);
    /// Create a new empty Cell2.
    pub fn empty() -> Cell2 {
        Cell2 {
            point_type: PointType::Empty,
            points: Array3::from_elem(Self::SIZE.into_size3(), PointType::Empty),
        }
    }

    /// Create a new Cell2 with all elements set to `point_type`.
    pub fn splat(point_type: PointType) -> Cell2 {
        Cell2 {
            point_type: point_type,
            points: Array3::from_elem(Self::SIZE.into_size3(), point_type),
        }
    }

    /// Get the PointType at `xyz`.
    pub fn get(&self, xyz: Size3) -> PointType {
        self.points[xyz]
    }

    /// Get a mutable reference to the PointType at `xyz`.
    pub fn get_mut(&mut self, xyz: Size3) -> &mut PointType {
        &mut self.points[xyz]
    }

    /// Get the PointType for treating the Cell2 as a single point.
    pub fn get_point_type(&self) -> PointType {
        self.point_type
    }

    /// Get the PointType for treating the Cell2 as a single point.
    pub fn get_point_type_mut(&mut self) -> &mut PointType {
        &mut self.point_type
    }

    /// Get the PointType for treating the Cell2 as a single point.
    pub fn get_points(&self) -> &Array3<PointType> {
        &self.points
    }

    /// Create a Vec of `cell` rotated `i` degrees `n` times.
    /// 
    /// The minimum rotation is 45 degrees, so `i=1` is 45 degrees, `i=2` is 90, etc.
    pub fn rotated_z(&self, n: usize, i: usize) -> Vec<Cell2> {
        let mut rotations = Vec::new();
        for r in 0..n {
            let mut rotated = self.clone();
            for _ in 0..r * i {
                rotated = Self::rotate_z_45(&rotated);
            }
            rotations.push(rotated);
        }
        rotations
    }

    /// Rotate a Cell2 45 degrees around the z-axis.
    fn rotate_z_45(&self) -> Cell2 {
        let mut rotated = self.clone();
        for (xyz, p) in self.points.indexed_iter() {
            let xyz_rot = Self::rotate_z_45_xyz(xyz);
            *rotated.get_mut(xyz_rot) = *p;
        }
        rotated
    }

    /// Rotate an `xyz` 45 degrees around the z-axis.
    fn rotate_z_45_xyz(xyz: Size3) -> Size3 {
        let center = IVec3::new(1, 1, 1);
        let v = xyz.into_ivec3();
        let o = v - center;
        // First quadrant.
        if o.x >= 0 && o.y > 0 {
            let r = IVec3::new(-1, 0, 0);
            (v + r).as_uvec3().into_size3()
        // Second quadrant.
        } else if o.x < 0 && o.y >= 0 {
            let r = IVec3::new(0, -1, 0);
            (v + r).as_uvec3().into_size3()
        // Third quadrant.
        } else if o.x <= 0 && o.y < 0 {
            let r = IVec3::new(1, 0, 0);
            (v + r).as_uvec3().into_size3()
        // Fourth quadrant.
        } else {
            let r = IVec3::new(0, 1, 0);
            (v + r).as_uvec3().into_size3()
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct CellPoint2(pub Size3);

impl CellPoint2 {
    pub fn adjacent(&self) -> Vec<CellPoint2> {
        // TODO: merge this with neighbor code somehow, both should be in the same file with honeycomb logic.
        match self {
            CellPoint2((1, 2, 2)) => vec![CellPoint2((1, 2, 2)), CellPoint2((2, 2, 2)), CellPoint2((0, 2, 2))],
            CellPoint2((2, 1, 2)) => vec![CellPoint2((2, 1, 2)), CellPoint2((2, 2, 2)), CellPoint2((2, 0, 2))],
            CellPoint2((2, 0, 2)) => vec![CellPoint2((2, 0, 2)), CellPoint2((1, 0, 2)), CellPoint2((0, 0, 2))],
            CellPoint2((0, 1, 2)) => vec![CellPoint2((0, 1, 2)), CellPoint2((0, 0, 2)), CellPoint2((0, 2, 2))],
            e => vec![*e],
        }
    }
}

/// A cubic cell point type.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum PointType {
    Empty,
    Path,
    Ground,
    Building,
    BuildingBig,
}
impl PointType {
    fn rgba(&self) -> Rgba {
        match *self {
            PointType::Empty => Rgba([0; 4]),
            PointType::Path => Rgba([51, 51, 51, 255]),
            PointType::Ground => Rgba([120, 159, 138, 255]),
            PointType::Building => Rgba([103, 127, 163, 255]),
            PointType::BuildingBig => Rgba([109, 103, 163, 255]),
            // PointType::Edge => Rgba([157, 163, 103, 255]),
        }
    }
}

/// A cubic honeycomb for Cells.
pub struct Honeycomb {
    pub cells: Array3<Cell>,
    pub path: Vec<(usize, usize, usize)>,
}
impl Honeycomb {
    pub fn gen_perlin(seed: u32) -> Honeycomb {
        // Path noise settings voxel space scaling.
        let size = UVec3::new(16, 16, 4);
        let center = size / 2;
        let path_scale = 2.0;
        let sprawl_scale = 3.0;
        let perlin_scale = 0.08;

        // Initialize Honeycomb array.
        let mut cells = Array3::from_elem(size.into_size3(), Cell::empty());
        let mut path = Vec::new();
        let mut sprawl_path = Vec::new();

        // Generate 1D Perlin noise path.
        let mut noise_path = Perlin1::gen(size.x as usize, seed, perlin_scale);
        noise_path.normalize();

        // Generate 1D Perlin noise for the width of sprawl from the path.
        let mut noise_sprawl = Perlin1::gen(size.x as usize, seed + 1, perlin_scale);
        noise_sprawl.normalize();

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
        let mut voxels = Array3::from_elem(size.into_size3(), [0; 4]);
        for (xyz, cell) in self.cells.indexed_iter() {
            let cell_voxels = cell.debug_voxels();
            let cell_vector = xyz.into_uvec3() * Cell::DEBUG_VOXELS_SIZE as u32;
            for (xyz, rgba) in cell_voxels.indexed_iter() {
                let vector = cell_vector + xyz.into_uvec3();
                voxels[vector.into_size3()] = rgba.0;
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

    pub fn splat(cell_point: CellPointType) -> Cell {
        Cell {
            center: cell_point,
            n: cell_point,
            ne: cell_point,
            e: cell_point,
            se: cell_point,
            s: cell_point,
            sw: cell_point,
            w: cell_point,
            nw: cell_point,
        }
    }

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

    /// Create a edge Cell.
    fn edge() -> Cell {
        Cell {
            center: CellPointType::Edge,
            n: CellPointType::Edge,
            ne: CellPointType::Edge,
            e: CellPointType::Edge,
            se: CellPointType::Edge,
            s: CellPointType::Edge,
            sw: CellPointType::Edge,
            w: CellPointType::Edge,
            nw: CellPointType::Edge,
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
    pub fn paths() -> Vec<Cell> {
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

    fn building(cell_points: Vec<CellPoint>) -> Cell {
        let mut cell = Self::empty();
        for c in cell_points {
            match c {
                CellPoint::Center => cell.center = CellPointType::Building,
                CellPoint::N => cell.n = CellPointType::Building,
                CellPoint::NE => cell.ne = CellPointType::Building,
                CellPoint::E => cell.e = CellPointType::Building,
                CellPoint::SE => cell.se = CellPointType::Building,
                CellPoint::S => cell.s = CellPointType::Building,
                CellPoint::SW => cell.sw = CellPointType::Building,
                CellPoint::W => cell.w = CellPointType::Building,
                CellPoint::NW => cell.nw = CellPointType::Building,
            }
        }
        cell
    }

    fn buildingbig() -> Cell {
        Cell {
            center: CellPointType::BuildingBig,
            n: CellPointType::BuildingBig,
            ne: CellPointType::BuildingBig,
            e: CellPointType::BuildingBig,
            se: CellPointType::BuildingBig,
            s: CellPointType::BuildingBig,
            sw: CellPointType::BuildingBig,
            w: CellPointType::BuildingBig,
            nw: CellPointType::BuildingBig,
        }
    }

    fn buildings() -> Vec<Cell> {
        let cell_points = CellPoint::directions();
        let subsets = Self::subsets(cell_points);
        let mut cells = Vec::new();
        for subset in subsets {
            cells.push(Self::building(subset));
        }
        cells
    }

    pub fn paths_buildings() -> Vec<Cell> {
        let mut cells = Vec::new();
        let paths = Self::paths();
        let buildings = Self::buildings();
        for path in paths {
            for building in buildings.iter() {
                for p in CellPoint::directions() {
                    if building.get(p) == CellPointType::Building && path.get(p) == CellPointType::Ground {
                        let mut n = path.clone();
                        *n.get_mut(p) = CellPointType::Building;
                        cells.push(n);
                    }
                }
            }
        }
        cells.push(Self::ground());
        cells.push(Self::buildingbig());
        cells.push(Self::edge());
        cells
    }

    /// Get all subsets of a set of Points.
    fn subsets(nums: Vec<CellPoint>) -> Vec<Vec<CellPoint>> {
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

    /// Get the CellPointType at a specific CellPoint.
    pub fn get_mut(&mut self, p: CellPoint) -> &mut CellPointType {
        match p {
            CellPoint::Center => &mut self.center,
            CellPoint::N => &mut self.n,
            CellPoint::NE => &mut self.ne,
            CellPoint::E => &mut self.e,
            CellPoint::SE => &mut self.se,
            CellPoint::S => &mut self.s,
            CellPoint::SW => &mut self.sw,
            CellPoint::W => &mut self.w,
            CellPoint::NW => &mut self.nw,
        }
    }

    /// Create the debug voxels for a Cell.
    fn debug_voxels(&self) -> Array3<Rgba> {
        let size = UVec3::splat(Self::DEBUG_VOXELS_SIZE as u32);
        let base = self.center.rgba();
        let mut voxels = Array3::from_elem(size.into_size3(), base);
        let center = IVec3::new(1, 1, 2);
        voxels[center.as_uvec3().into_size3()] = base;
        for d in CellPoint::directions() {
            let p = (center + d.voxel_offset()).as_uvec3().into_size3();
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
    BuildingBig,
    Edge,
}
impl CellPointType {
    fn rgba(&self) -> Rgba {
        match *self {
            CellPointType::Empty => Rgba([0; 4]),
            CellPointType::Path => Rgba([51, 51, 51, 255]),
            CellPointType::Ground => Rgba([120, 159, 138, 255]),
            CellPointType::Building => Rgba([103, 127, 163, 255]),
            CellPointType::BuildingBig => Rgba([109, 103, 163, 255]),
            CellPointType::Edge => Rgba([157, 163, 103, 255]),
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

    /// Get the neighbors of a CellPoint.
    pub fn neighbors() -> Vec<(CellPoint, IVec3)> {
        vec![
            (CellPoint::N, IVec3::new(0, 1, 0)),
            (CellPoint::NE, IVec3::new(1, 1, 0)),
            (CellPoint::E, IVec3::new(1, 0, 0)),
            (CellPoint::SE, IVec3::new(1, -1, 0)),
            (CellPoint::S, IVec3::new(0, -1, 0)),
            (CellPoint::SW, IVec3::new(-1, -1, 0)),
            (CellPoint::W, IVec3::new(-1, 0, 0)),
            (CellPoint::NW, IVec3::new(-1, 1, 0)),
        ]
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
