use glam::IVec3;
use ndarray::Array3;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rand::seq::IteratorRandom;

use crate::vector::{Pos3, Vector3};
use crate::wfc::honeycomb::{Cell, Point, PointType, Honeycomb};

/// A cubic honeycomb for Cells.
pub struct WfcMap {
    pub honeycomb: Honeycomb,
}
impl WfcMap {
    pub fn from_honeycomb(mut honeycomb: Honeycomb) -> WfcMap {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut wfc_cells = Array3::from_elem(honeycomb.cells.dim(), WfcCell::new());
        let neighbors = [
            (Point::N, IVec3::new(0, 1, 0)),
            (Point::NE, IVec3::new(1, 1, 0)),
            (Point::E, IVec3::new(1, 0, 0)),
            (Point::SE, IVec3::new(1, -1, 0)),
            (Point::S, IVec3::new(0, -1, 0)),
            (Point::SW, IVec3::new(-1, -1, 0)),
            (Point::W, IVec3::new(-1, 0, 0)),
            (Point::NW, IVec3::new(-1, 1, 0)),
        ];
        // Initialize waves.
        for p in honeycomb.path.iter() {
            let w = wfc_cells.get_mut(*p).expect("Path out of bounds!");
            let g = Cell::path_cells();
            g.iter().for_each(|t| {
                w.insert(*t)
            });
            let current = honeycomb.cells.get(*p).expect("Path out of bounds!");
            if current == &Cell::ground() {
                w.ban_center(PointType::Path);
            }
        }
        // Wave Function Collapse.
        let empty = Cell::empty();
        for p in honeycomb.path.iter() {
            let w = wfc_cells.get_mut(*p).expect("Path out of bounds!");
            let v = p.into_ivec3();
            for (p, o) in neighbors {
                let n = (v + o).as_uvec3();
                let cell = honeycomb.cells.get(n.into_pos()).unwrap_or(&empty);
                let point_type = cell.get(p.opposite());
                w.update(false, point_type, p);
            }
            let c = w.collapse(&mut rng);
            if let Some(cell) = c {
                honeycomb.cells[*p] = cell;
                for (p, o) in neighbors {
                    let p_n = (v + o).as_uvec3().into_pos();
                    if let Some(w) = wfc_cells.get_mut(p_n) {
                        let point_type = cell.get(p);
                        w.update(true, point_type, p.opposite());
                    }
                }
            }
        }
        WfcMap {
            honeycomb,
        }
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
    fn update(&mut self, add: bool, point_type: PointType, point: Point) {
        match (point_type, point) {
            (PointType::Empty, _) => {
                self.ban(PointType::Path, point);
            },

            (PointType::Path, _) => {
                self.require(add, PointType::Path, point);
            },

            _ => (),
        }
    }

    /// Ban the given point type and point.
    fn ban(&mut self, point_type: PointType, point: Point) {
        let vts = self.cell_set.point_types(point);
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt == point_type {
                *b = true;
            }
        }
    }

    /// Ban the given point type and point.
    fn ban_center(&mut self, point_type: PointType) {
        let vts = self.cell_set.point_types_center();
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt == point_type {
                *b = true;
            }
        }
    }

    /// Require the given point type and point.
    fn require(&mut self, add: bool, point_type: PointType, point: Point) {
        let vts = self.cell_set.point_types(point);
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt != point_type {
                *b = true;
            }
            if add && *vt == point_type {
                *b = false;
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
    centers: Vec<PointType>,
    ns: Vec<PointType>,
    nes: Vec<PointType>,
    es: Vec<PointType>,
    ses: Vec<PointType>,
    ss: Vec<PointType>,
    sws: Vec<PointType>,
    ws: Vec<PointType>,
    nws: Vec<PointType>,
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

    /// Get a PointType for each cell in a given Point.
    fn point_types(&mut self, p: Point) -> &Vec<PointType> {
        match p {
            Point::Center => &self.centers,
            Point::N => &self.ns,
            Point::NE => &self.nes,
            Point::E => &self.es,
            Point::SE => &self.ses,
            Point::S => &self.ss,
            Point::SW => &self.sws,
            Point::W => &self.ws,
            Point::NW => &self.nws,
        }
    }

    /// Get a PointType for each cell center.
    fn point_types_center(&mut self) -> &Vec<PointType> {
        &self.centers
    }
}
