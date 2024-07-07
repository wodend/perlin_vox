use std::{cell, path};

use glam::{IVec3, UVec3};
use ndarray::{Array1, Array3, Array4, Axis, Zip, s};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rand::seq::IteratorRandom;

use crate::noise::Perlin1;
use crate::vector::{Dim3, Vector3, Size3, Size2};
use crate::wfc::honeycomb::{Cell, CellPoint, CellPoint2, CellPointType, Honeycomb, Honeycomb2, Cell2, PointType, Cellset2, CellId};

pub struct WfcOptions {
    size: UVec3,
    seed: u32,
    scale_path: f32,
    scale_sprawl: f32,
    scale_perlin: f32,
}

impl WfcOptions {
    /// Get the default WFC options.
    pub fn default() -> WfcOptions {
        WfcOptions {
            size: UVec3::new(16, 16, 4),
            seed: 0,
            scale_path: 2.0,
            scale_sprawl: 3.0,
            scale_perlin: 0.08,
        }
    }

    /// Get the default WFC options with the specified `seed`.
    pub fn default_with_seed(seed: u32) -> WfcOptions {
        WfcOptions {
            size: UVec3::new(16, 16, 4),
            seed: seed,
            scale_path: 2.0,
            scale_sprawl: 3.0,
            scale_perlin: 0.08,
        }
    }
}

/// wfc on a honeycomb.
pub struct WfcMap2 {
    options: WfcOptions,
    honeycomb: Honeycomb2,
    path: Vec<UVec3>,
}

impl WfcMap2 {
    pub fn new(options: WfcOptions) -> WfcMap2 {
        let honeycomb = Honeycomb2::new(options.size);
        let path = Vec::new();
        WfcMap2 {
            options,
            honeycomb,
            path,
        }
    }

    pub fn gen(&mut self) -> &Honeycomb2 {
        self.perlin_base();
        let mut wfc_cells = self.initialize();
        self.wfc_2d(&mut wfc_cells);
        &self.honeycomb
    }

    fn perlin_base(&mut self) {
        // Generate 1D Perlin noise path.
        let mut noise_path = Perlin1::gen(self.options.size.x as usize, self.options.seed, self.options.scale_perlin);
        noise_path.normalize();

        // Generate 1D Perlin noise for the width of sprawl from the path.
        let mut noise_sprawl = Perlin1::gen(self.options.size.x as usize, self.options.seed + 1, self.options.scale_perlin);
        noise_sprawl.normalize();

        let center = self.options.size / 2;
        let mut path_sprawl = Vec::new();
        let mut prev_y = None;
        for ((x, n_p), n_s) in noise_path.values.indexed_iter().zip(noise_sprawl.values) {
            // Scale path noise value.
            let y = center.y + (n_p * self.options.scale_path) as u32;
            // Scale sprawl noise value.
            let s = ((n_s + 1.0) * self.options.scale_sprawl) as i32;
            // Set sprawl to Ground Cells.
            for i in 1..s {
                let p0 = UVec3::new(x as u32, (y as i32 - i) as u32, 0);
                let p1 = UVec3::new(x as u32, (y as i32 + i) as u32, 0);
                self.honeycomb.insert(p0, Cell2::splat(PointType::Ground));
                self.honeycomb.insert(p1, Cell2::splat(PointType::Ground));
                path_sprawl.push(p0);
                path_sprawl.push(p1);
            }
            // Set path to Path Cells.
            let p = UVec3::new(x as u32, y, 0);
            // Handle cases where y values are not adjacent.
            if let Some(p_y) = prev_y {
                for i in 1..(y as i32 - p_y) {
                    let p = UVec3::new(x as u32, (p_y + i) as u32, 0);
                    self.honeycomb.insert(p, Cell2::splat(PointType::Path));
                    self.path.push(p);
                }
            }
            self.honeycomb.insert(p, Cell2::splat(PointType::Path));
            self.path.push(p);
            prev_y = Some(y as i32);
        }
        self.path.append(&mut path_sprawl);
    }

    /// Initialize WfcCells.
    fn initialize(&mut self) -> Array3<WfcCell2> {
        let cellset = CellsetLayer0::new();
        let mut wfc_cells = Array3::from_elem(self.honeycomb.size().into_size3(), WfcCell2::from(cellset.0));
        for p in self.path.iter() {
            let w = wfc_cells.get_mut(p.into_size3()).expect("Path out of bounds!");
            let c = self.honeycomb.get(*p).unwrap();
            w.init(c.get_point_type());
        }
        wfc_cells
    }


    fn wfc_2d(&mut self, wfc_cells: &mut Array3<WfcCell2>) {
        // Wave Function Collapse.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let neighbors = Neighbor::neighbors_2d();
        let empty = Cell2::splat(PointType::Empty);
        for p in self.path.iter() {
            let w = wfc_cells.get_mut(p.into_size3()).expect("Path out of bounds!");
            let v = p.as_ivec3();
            // Get updates from neighbors before collapsing.
            for n in &neighbors {
                let n_xyz = (v + n.neighbor_offset).as_uvec3();
                let cell = self.honeycomb.get(n_xyz).unwrap_or(empty.clone());
                let point_type = cell.get(n.cell_xyz);
                w.update(false, point_type, n.self_xyz);
            }
            let c = w.collapse(&mut rng);
            if let Some(cell) = c {
                self.honeycomb.insert(*p, cell.clone());
                // Update neighbors after collapsing.
                for n in &neighbors {
                    let n_xyz = (v + n.neighbor_offset).as_uvec3().into_size3();
                    if let Some(w) = wfc_cells.get_mut(n_xyz) {
                        let point_type = cell.get(n.self_xyz);
                        w.update(true, point_type, n.cell_xyz);
                    }
                }
            }
        }
    }
}

/// A Cell2 Neighbor.
pub struct Neighbor {
    pub self_xyz: Size3,
    pub neighbor_offset: IVec3,
    pub cell_xyz: Size3,
}

impl Neighbor {
    const CELL_N: Size3 = (1, 2, 2);
    const CELL_NE: Size3 = (2, 2, 2);
    const CELL_E: Size3 = (2, 1, 2);
    const CELL_SE: Size3 = (2, 0, 2);
    const CELL_S: Size3 = (1, 0, 2);
    const CELL_SW: Size3 = (0, 0, 2);
    const CELL_W: Size3 = (0, 1, 2);
    const CELL_NW: Size3 = (0, 2, 2);
    /// Get the 2D neighbors of a Cell2.
    pub fn new(self_xyz: Size3, neighbor_offset: IVec3, cell_xyz: Size3) -> Neighbor {
        Neighbor {
            self_xyz,
            neighbor_offset,
            cell_xyz,
        }
    }

    /// Get the 2D neighbors of a Cell2.
    pub fn neighbors_2d() -> Vec<Neighbor> {
        vec![
            Neighbor::new(Self::CELL_N, IVec3::new(0, 1, 0), Self::CELL_S),
            Neighbor::new(Self::CELL_NE, IVec3::new(1, 1, 0), Self::CELL_SW),
            Neighbor::new(Self::CELL_E, IVec3::new(1, 0, 0), Self::CELL_W),
            Neighbor::new(Self::CELL_SE, IVec3::new(1, -1, 0), Self::CELL_NW),
            Neighbor::new(Self::CELL_S, IVec3::new(0, -1, 0), Self::CELL_N),
            Neighbor::new(Self::CELL_SW, IVec3::new(-1, -1, 0), Self::CELL_NE),
            Neighbor::new(Self::CELL_W, IVec3::new(-1, 0, 0), Self::CELL_E),
            Neighbor::new(Self::CELL_NW, IVec3::new(-1, 1, 0), Self::CELL_NW),
        ]
    }
}

/// A Cellset for WaveFunctionCollapse layer 0.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct CellsetLayer0(pub Cellset2);

impl CellsetLayer0 {
    /// Create a new empty CellsetLayer0.
    pub fn new() -> CellsetLayer0 {
        let mut cellset = Cellset2::new();
        cellset.push(Cell2::splat(PointType::Ground));
        cellset.push(Cell2::splat(PointType::BuildingBig));
        cellset.append(Self::with_buildings(Self::paths_2d()));
        // cellset.append(Self::paths_2d());
        CellsetLayer0(cellset)
    }
    
    /// Create all 2D path Cell2s.
    fn paths_2d() -> Vec<Cell2> {
        let mut paths = Vec::new();
        // Paths requiring 2 45 degree rotations.
        paths.append(&mut Self::path_2d_4way().rotated_z(2, 1));
        // Paths requiring 4 45 degree rotations.
        paths.append(&mut Self::path_2d_straight().rotated_z(4, 1));
        // Paths requiring 4 90 degree rotations.
        paths.append(&mut Self::path_2d_left_turn().rotated_z(4, 2));
        paths.append(&mut Self::path_2d_right_turn().rotated_z(4, 2));
        // Paths requiring 8 45 degree rotations.
        paths.append(&mut Self::path_2d_uturn().rotated_z(8, 1));
        paths.append(&mut Self::path_2d_end().rotated_z(8, 1));
        paths.append(&mut Self::path_2d_fork().rotated_z(8, 1));
        paths.append(&mut Self::path_2d_3way().rotated_z(8, 1));
        paths
    }

    fn with_buildings(cells: Vec<Cell2>) -> Vec<Cell2> {
        let mut buildings = Vec::new();
        for c in cells {
            buildings.append(&mut Self::with_building(c));
        }
        buildings
    }

    fn with_building(cell: Cell2) -> Vec<Cell2> {
        let mut ground = Vec::new();
        for (xyz, p) in cell.get_points().slice(s![.., .., 2]).indexed_iter() {
            if *p == PointType::Ground {
                ground.push(xyz);
            }
        }
        let mut buildings = Vec::new();
        let building_specs = Self::subsets(ground);
        for s in building_specs {
            let mut c = cell.clone();
            for p in s {
                *c.get_mut((p.0, p.1, 2)) = PointType::Building;
            }
            buildings.push(c);
        }
        buildings
    }

    /// Get all subsets of a set of points.
    fn subsets(points: Vec<Size2>) -> Vec<Vec<Size2>> {
        let mut result = vec![vec![]];

        for p in points {
            let len = result.len();
            for i in 0..len {
                let mut subset = result[i].clone();
                subset.push(p);
                result.push(subset);
            }
        }

        result
    }

    /// Create a new 2D path Cell2.
    fn path_2d_base() -> Cell2 {
        let mut base = Cell2::splat(PointType::Ground);
        *base.get_point_type_mut() = PointType::Path;
        base
    }

    /// Create a 2D straight east-west path Cell2.
    fn path_2d_straight() -> Cell2 {
        let mut base = Self::path_2d_base();
        let points = [
            (0, 1, 2),
            (1, 1, 2),
            (2, 1, 2),
        ];
        for p in points.iter() {
            *base.get_mut(*p) = PointType::Path;
        }
        base
    }

    /// Create a 2D left turn east-west path Cell2.
    fn path_2d_left_turn() -> Cell2 {
        let mut base = Self::path_2d_base();
        let points = [
            (0, 1, 2),
            (1, 1, 2),
            (2, 2, 2),
        ];
        for p in points.iter() {
            *base.get_mut(*p) = PointType::Path;
        }
        base
    }

    /// Create a 2D left turn east-west path Cell2.
    fn path_2d_right_turn() -> Cell2 {
        let mut base = Self::path_2d_base();
        let points = [
            (0, 1, 2),
            (1, 1, 2),
            (2, 0, 2),
        ];
        for p in points.iter() {
            *base.get_mut(*p) = PointType::Path;
        }
        base
    }

    /// Create a 2D dead-end path Cell2.
    fn path_2d_end() -> Cell2 {
        let mut base = Self::path_2d_base();
        let points = [
            (0, 1, 2),
            (1, 1, 2),
        ];
        for p in points.iter() {
            *base.get_mut(*p) = PointType::Path;
        }
        base
    }

    /// Create a 2D u-turn path Cell2.
    fn path_2d_uturn() -> Cell2 {
        let mut base = Self::path_2d_base();
        let points = [
            (0, 0, 2),
            (1, 1, 2),
            (2, 0, 2),
        ];
        for p in points.iter() {
            *base.get_mut(*p) = PointType::Path;
        }
        base
    }

    /// Create a 2D fork turn path Cell2.
    fn path_2d_fork() -> Cell2 {
        let mut base = Self::path_2d_left_turn();
        *base.get_mut((1, 0, 2)) = PointType::Path;
        base
    }

    /// Create a 2D 3-way path Cell2.
    fn path_2d_3way() -> Cell2 {
        let mut base = Self::path_2d_straight();
        *base.get_mut((1, 0, 2)) = PointType::Path;
        base
    }

    /// Create a 2D 4-way path Cell2.
    fn path_2d_4way() -> Cell2 {
        let mut base = Self::path_2d_straight();
        *base.get_mut((1, 0, 2)) = PointType::Path;
        *base.get_mut((1, 2, 2)) = PointType::Path;
        base
    }
}

/// A WaveFunctionCollapse cell.
#[derive(Clone, Debug)]
struct WfcCell2 {
    cellset: Cellset2,
    banned: Vec<bool>,
}
impl WfcCell2 {
    /// Create a new WfcCell with a given `cellset`.
    /// 
    /// All cells are initially banned.
    fn from(cellset: Cellset2) -> Self {
        let banned = vec![true; cellset.len()];
        Self {
            cellset,
            banned,
        }
    }

    /// Initialize the cell given a starting `point_type`.
    fn init(&mut self, point_type: PointType) {
        match point_type {
            PointType::Ground => self.init_unban(PointType::Ground),
            PointType::Path => self.init_unban(PointType::Path),
            _ => (),
        }
    }

    /// Require the given point type and point.
    fn init_unban(&mut self, point_type: PointType) {
        Zip::from(self.cellset.point_types()).and(&mut self.banned).for_each(|p, b| {
            if *p == point_type {
                *b = false;
            }
        });
    }

    /// Update the cell given a neighboring point type and point.
    fn update(&mut self, unban: bool, point_type: PointType, point: Size3) {
        match (point_type, point) {
            (PointType::Empty, _) => {
                for p in CellPoint2(point).adjacent() {
                    self.ban(PointType::Path, p.0);
                    // self.ban(PointType::Building, p);
                    // self.ban(PointType::BuildingBig, p);
                }
            },

            (PointType::Path, _) => {
                if unban {
                    self.unban(PointType::Path, point);
                }
                self.require(PointType::Path, point);
            },

            (PointType::Building, _) => {
                if unban {
                    self.unban(PointType::BuildingBig, point);
                }
            },

            (PointType::BuildingBig, _) => {
                if unban {
                    self.unban(PointType::Building, point);
                    self.unban(PointType::BuildingBig, point);
                }
            },

            _ => (),
        }
    }

    /// Ban the given point type and point.
    fn ban(&mut self, point_type: PointType, point: Size3) {
        Zip::from(self.cellset.points(point)).and(&mut self.banned).for_each(|p, b| {
            if *p == point_type {
                *b = true;
            }
        });
    }

    /// Require the given point type and point.
    fn unban(&mut self, point_type: PointType, point: Size3) {
        Zip::from(self.cellset.points(point)).and(&mut self.banned).for_each(|p, b| {
            if *p == point_type {
                *b = false;
            }
        });
    }

    /// Require the given point type and point.
    fn require(&mut self, point_type: PointType, point: Size3) {
        Zip::from(self.cellset.points(point)).and(&mut self.banned).for_each(|p, b| {
            if *p != point_type {
                *b = true;
            }
        });
    }

    /// Collapse the cell to a cell.
    fn collapse(&mut self, rng: &mut ChaCha8Rng) -> Option<Cell2> {
        let mut ids = Vec::new();
        for (id, b) in self.banned.iter().enumerate() {
            if !b {
                ids.push(id);
            }
        }
        let id = ids.iter().choose(rng);
        match id {
            Some(i) => {
                Some(self.cellset.get(CellId(*i)))
            },
            None => None,
        }
    }
}



/// wfc on a honeycomb.
pub struct WfcMap {
    pub honeycomb: Honeycomb,
}
impl WfcMap {
    pub fn from_honeycomb(honeycomb: Honeycomb) -> WfcMap {
        WfcMap {
            honeycomb,
        }
    }

    pub fn wfc(&mut self) {
        let mut wfc_cells = Self::initialize(self);
        // Wave Function Collapse.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let neighbors = CellPoint::neighbors();
        let empty = Cell::empty();
        for p in self.honeycomb.path.iter() {
            let w = wfc_cells.get_mut(*p).expect("Path out of bounds!");
            let v = p.into_ivec3();
            // Get updates from neighbors before collapsing.
            for (p, o) in &neighbors {
                let n = (v + *o).as_uvec3();
                let cell = self.honeycomb.cells.get(n.into_size3()).unwrap_or(&empty);
                let point_type = cell.get(p.opposite());
                w.update(false, point_type, *p);
            }
            let c = w.collapse(&mut rng);
            if let Some(cell) = c {
                self.honeycomb.cells[*p] = cell;
                // Update neighbors after collapsing.
                for (p, o) in &neighbors {
                    let p_n = (v + *o).as_uvec3().into_size3();
                    if let Some(w) = wfc_cells.get_mut(p_n) {
                        let point_type = cell.get(*p);
                        w.update(true, point_type, p.opposite());
                    }
                }
            }
        }
    }

    /// Initialize waves.
    fn initialize(&mut self) -> Array3<WfcCell> {
        let mut wfc_cells = Array3::from_elem(self.honeycomb.cells.dim(), WfcCell::new());
        for p in self.honeycomb.path.iter() {
            let w = wfc_cells.get_mut(*p).expect("Path out of bounds!");
            let g = Cell::paths_buildings();
            g.iter().for_each(|t| {
                w.insert(*t)
            });
            let current = self.honeycomb.cells.get(*p).expect("Path out of bounds!");
            if current == &Cell::ground() {
                w.ban_center(CellPointType::Path);
                w.ban_center(CellPointType::BuildingBig);
                w.ban_center(CellPointType::Edge);
            }
        }
        wfc_cells
    }
}


/// A WaveFunctionCollapse cell.
#[derive(Clone, Debug)]
struct WfcCell {
    cell_set: CellSet,
    banned: Vec<bool>,
}
impl WfcCell {
    /// Create a new WfcCell.
    fn new() -> Self {
        Self {
            cell_set: CellSet::new(),
            banned: Vec::new(),
        }
    }

    /// Insert an unbanned Cell.
    fn insert(&mut self, cell: Cell) {
        self.cell_set.insert(cell);
        self.banned.push(false);
    }

    /// Update the cell given a neighboring point type and point.
    fn update(&mut self, unban: bool, point_type: CellPointType, point: CellPoint) {
        match (point_type, point) {
            (CellPointType::Empty, _) => {
                for p in point.adjacent() {
                    self.ban(CellPointType::Path, p);
                    self.ban(CellPointType::Building, p);
                    self.ban(CellPointType::BuildingBig, p);
                }
            },

            (CellPointType::Path, _) => {
                if unban {
                    self.unban(CellPointType::Path, point);
                }
                self.require(CellPointType::Path, point);
            },

            (CellPointType::Building, _) => {
                if unban {
                    self.unban(CellPointType::BuildingBig, point);
                }
            },

            (CellPointType::BuildingBig, _) => {
                if unban {
                    self.unban(CellPointType::Building, point);
                    self.unban(CellPointType::BuildingBig, point);
                }
            },

            _ => (),
        }
    }

    /// Ban the given point type and point.
    fn ban(&mut self, point_type: CellPointType, point: CellPoint) {
        let vts = self.cell_set.point_types(point);
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt == point_type {
                *b = true;
            }
        }
    }

    /// Ban the given point type and point.
    fn ban_center(&mut self, point_type: CellPointType) {
        let vts = self.cell_set.point_types_center();
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt == point_type {
                *b = true;
            }
        }
    }

    /// Require the given point type and point.
    fn unban(&mut self, point_type: CellPointType, point: CellPoint) {
        let vts = self.cell_set.point_types(point);
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt == point_type {
                *b = false;
            }
        }
    }

    /// Require the given point type and point.
    fn require(&mut self, point_type: CellPointType, point: CellPoint) {
        let vts = self.cell_set.point_types(point);
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt != point_type {
                *b = true;
            }
        }
    }

    /// Collapse the cell to a cell.
    fn collapse(&mut self, rng: &mut ChaCha8Rng) -> Option<Cell> {
        let mut ids = Vec::new();
        for (id, b) in self.banned.iter().enumerate() {
            if !b {
                ids.push(id);
            }
        }
        let id = ids.iter().choose(rng);
        match id {
            Some(i) => {
                self.cell_set.get(*i)
            },
            None => None,
        }
    }
}

/// A set of Cells.
#[derive(Clone, Debug)]
pub struct CellSet {
    centers: Vec<CellPointType>,
    ns: Vec<CellPointType>,
    nes: Vec<CellPointType>,
    es: Vec<CellPointType>,
    ses: Vec<CellPointType>,
    ss: Vec<CellPointType>,
    sws: Vec<CellPointType>,
    ws: Vec<CellPointType>,
    nws: Vec<CellPointType>,
}
impl CellSet {
    /// Create a new CellSet.
    fn new() -> Self {
        Self {
            centers: Vec::new(),
            ns: Vec::new(),
            nes: Vec::new(),
            es: Vec::new(),
            ses: Vec::new(),
            ss: Vec::new(),
            sws: Vec::new(),
            ws: Vec::new(),
            nws: Vec::new(),
        }
    }

    /// Insert a Cell and return its ID.
    fn insert(&mut self, cell: Cell) -> usize {
        self.centers.push(cell.center);
        self.ns.push(cell.n);
        self.nes.push(cell.ne);
        self.es.push(cell.e);
        self.ses.push(cell.se);
        self.ss.push(cell.s);
        self.sws.push(cell.sw);
        self.ws.push(cell.w);
        self.nws.push(cell.nw);
        self.ns.len() - 1
    }

    /// Get a Cell by ID.
    fn get(&self, id: usize) -> Option<Cell> {
        match self.centers.get(id) {
            Some(c) => Some(Cell {
                    center: *c,
                    n: self.ns[id],
                    ne: self.nes[id],
                    e: self.es[id],
                    se: self.ses[id],
                    s: self.ss[id],
                    sw: self.sws[id],
                    w: self.ws[id],
                    nw: self.nws[id],
                }
            ),
            None => None,
        }
    }

    /// Get a CellPointType for each cell in a given CellPoint.
    fn point_types(&mut self, p: CellPoint) -> &Vec<CellPointType> {
        match p {
            CellPoint::Center => &self.centers,
            CellPoint::N => &self.ns,
            CellPoint::NE => &self.nes,
            CellPoint::E => &self.es,
            CellPoint::SE => &self.ses,
            CellPoint::S => &self.ss,
            CellPoint::SW => &self.sws,
            CellPoint::W => &self.ws,
            CellPoint::NW => &self.nws,
        }
    }

    /// Get a CellPointType for each cell center.
    fn point_types_center(&mut self) -> &Vec<CellPointType> {
        &self.centers
    }
}
