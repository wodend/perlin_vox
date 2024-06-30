use glam::IVec3;
use ndarray::Array3;
use glam::UVec3;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
// use rand::rngs::ThreadRng;

use crate::noise::Perlin1;
use crate::render::normalize;
use crate::vector::{Vector3, Pos3};

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct Rgba([u8; 4]);

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum VertexType {
    Empty,
    Path,
    Ground,
    MaybePath,
}
impl VertexType {
    fn rgba(&self) -> Rgba {
        match *self {
            VertexType::Empty => Rgba([0; 4]),
            VertexType::Path => Rgba([51, 51, 51, 255]),
            VertexType::Ground => Rgba([120, 159, 138, 255]),
            VertexType::MaybePath => Rgba([90, 129, 108, 255]),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A 3D direction.
enum Direction {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}
impl Direction {
    /// Get the voxel space offset.
    fn values() -> Vec<Direction> {
        vec![
            Direction::N,
            Direction::NE,
            Direction::E,
            Direction::SE,
            Direction::S,
            Direction::SW,
            Direction::W,
            Direction::NW,
        ]
    }

    /// Get the voxel space offset.
    fn voxel_offset(&self) -> IVec3 {
        match *self {
            Direction::N => IVec3::new(0, 1, 0),
            Direction::NE => IVec3::new(1, 1, 0),
            Direction::E => IVec3::new(1, 0, 0),
            Direction::SE => IVec3::new(1, -1, 0),
            Direction::S => IVec3::new(0, -1, 0),
            Direction::SW => IVec3::new(-1, -1, 0),
            Direction::W => IVec3::new(-1, 0, 0),
            Direction::NW => IVec3::new(-1, 1, 0),
        }
    }
}

const TILE_SIZE: usize = 3;
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
/// A 3D tile.
struct Tile {
    center: VertexType,
    n: VertexType,
    ne: VertexType,
    e: VertexType,
    se: VertexType,
    s: VertexType,
    sw: VertexType,
    w: VertexType,
    nw: VertexType,
}
impl Tile {
    const DIRECTION_SPEC_SIZE: usize = 8;

    /// Create a new empty Tile.
    fn new() -> Tile {
        Tile {
            center: VertexType::Empty,
            n: VertexType::Empty,
            ne: VertexType::Empty,
            e: VertexType::Empty,
            se: VertexType::Empty,
            s: VertexType::Empty,
            sw: VertexType::Empty,
            w: VertexType::Empty,
            nw: VertexType::Empty,
        }
    }

    /// Create a new Tile from an direction spec.
    fn from_direction_spec(center: VertexType, array: [VertexType; Self::DIRECTION_SPEC_SIZE]) -> Tile {
        Tile {
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

    /// Create an direction spec from a tile.
    fn into_direction_spec(self) -> [VertexType; Self::DIRECTION_SPEC_SIZE] {
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

    /// Get the VertexType by Direction.
    fn vertex_type(&self, direction: Direction) -> VertexType {
        match direction {
            Direction::N => self.n,
            Direction::NE => self.ne,
            Direction::E => self.e,
            Direction::SE => self.se,
            Direction::S => self.s,
            Direction::SW => self.sw,
            Direction::W => self.w,
            Direction::NW => self.nw,
        }
    }

    /// Create a new debug Ground tile.
    fn debug_ground() -> Tile {
        Tile {
            center: VertexType::Ground,
            n: VertexType::Ground,
            ne: VertexType::Ground,
            e: VertexType::Ground,
            se: VertexType::Ground,
            s: VertexType::Ground,
            sw: VertexType::Ground,
            w: VertexType::Ground,
            nw: VertexType::Ground,
        }
    }

    /// Create a new debug Path tile.
    fn debug_path() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Path,
            ne: VertexType::Path,
            e: VertexType::Path,
            se: VertexType::Path,
            s: VertexType::Path,
            sw: VertexType::Path,
            w: VertexType::Path,
            nw: VertexType::Path,
        }
    }

    /// Create a new debug Path tile.
    fn debug_maybe_path() -> Tile {
        Tile {
            center: VertexType::MaybePath,
            n: VertexType::MaybePath,
            ne: VertexType::MaybePath,
            e: VertexType::MaybePath,
            se: VertexType::MaybePath,
            s: VertexType::MaybePath,
            sw: VertexType::MaybePath,
            w: VertexType::MaybePath,
            nw: VertexType::MaybePath,
        }
    }


    /// Create a debug tile.
    fn debug() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Path,
            ne: VertexType::Ground,
            e: VertexType::Ground,
            se: VertexType::Path,
            s: VertexType::Ground,
            sw: VertexType::Ground,
            w: VertexType::Ground,
            nw: VertexType::Ground,
        }
    }

    /// Create a new debug Path straight tile.
    fn debug_path_straight() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Ground,
            ne: VertexType::Ground,
            e: VertexType::Path,
            se: VertexType::Ground,
            s: VertexType::Ground,
            sw: VertexType::Ground,
            w: VertexType::Path,
            nw: VertexType::Ground,
        }
    }

    /// Create a new debug Path 3 way tile.
    fn debug_path_3way() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Ground,
            ne: VertexType::Ground,
            e: VertexType::Path,
            se: VertexType::Ground,
            s: VertexType::Path,
            sw: VertexType::Ground,
            w: VertexType::Path,
            nw: VertexType::Ground,
        }
    }

    /// Create a new debug Path left turn tile.
    fn debug_path_left() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Ground,
            ne: VertexType::Path,
            e: VertexType::Ground,
            se: VertexType::Ground,
            s: VertexType::Ground,
            sw: VertexType::Ground,
            w: VertexType::Path,
            nw: VertexType::Ground,
        }
    }

    /// Create a new debug Path right turn tile.
    fn debug_path_right() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Ground,
            ne: VertexType::Ground,
            e: VertexType::Ground,
            se: VertexType::Path,
            s: VertexType::Ground,
            sw: VertexType::Ground,
            w: VertexType::Path,
            nw: VertexType::Ground,
        }
    }

    /// Create a new debug Path right turn tile.
    fn debug_path_uturn() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Ground,
            ne: VertexType::Ground,
            e: VertexType::Ground,
            se: VertexType::Path,
            s: VertexType::Ground,
            sw: VertexType::Path,
            w: VertexType::Ground,
            nw: VertexType::Ground,
        }
    }

    /// Create a new debug Path end tile.
    fn debug_path_end() -> Tile {
        Tile {
            center: VertexType::Path,
            n: VertexType::Ground,
            ne: VertexType::Ground,
            e: VertexType::Path,
            se: VertexType::Ground,
            s: VertexType::Ground,
            sw: VertexType::Ground,
            w: VertexType::Ground,
            nw: VertexType::Ground,
        }
    }

    /// Generate all debug Path tiles.
    fn gen_all_debug_path_tiles() -> Vec<Tile> {
        let mut tiles = Vec::new();
        // Rotate straight Path tile.
        for i in 0..4 {
            let mut a = Tile::debug_path_straight().into_direction_spec();
            a.rotate_left(i);
            let t = Tile::from_direction_spec(VertexType::Path, a);
            tiles.push(t);
        }
        // Rotate left turn Path tile.
        for i in 0..4 {
            let mut a = Tile::debug_path_left().into_direction_spec();
            a.rotate_left(i * 2);
            let t = Tile::from_direction_spec(VertexType::Path, a);
            tiles.push(t);
        }
        // Rotate right turn Path tile.
        for i in 0..4 {
            let mut a = Tile::debug_path_right().into_direction_spec();
            a.rotate_left(i * 2);
            let t = Tile::from_direction_spec(VertexType::Path, a);
            tiles.push(t);
        }
        // Rotate u-turn Path tile.
        for i in 0..4 {
            let mut a = Tile::debug_path_uturn().into_direction_spec();
            a.rotate_left(i * 2);
            let t = Tile::from_direction_spec(VertexType::Path, a);
            tiles.push(t);
        }
        // Rotate end Path tile.
        for i in 0..4 {
            let mut a = Tile::debug_path_end().into_direction_spec();
            a.rotate_left(i * 2);
            let t = Tile::from_direction_spec(VertexType::Path, a);
            tiles.push(t);
        }
        // Rotate 3 way turn Path tile.
        for i in 0..4 {
            let mut a = Tile::debug_path_3way().into_direction_spec();
            a.rotate_left(i * 2);
            let t = Tile::from_direction_spec(VertexType::Path, a);
            tiles.push(t);
        }
        tiles
    }
}


/// A set of Tiles.
#[derive(Clone, Debug)]
pub struct TileSet {
    centers: Vec<VertexType>,
    ns: Vec<VertexType>,
    nes: Vec<VertexType>,
    es: Vec<VertexType>,
    ses: Vec<VertexType>,
    ss: Vec<VertexType>,
    sws: Vec<VertexType>,
    ws: Vec<VertexType>,
    nws: Vec<VertexType>,
}
impl TileSet {
    /// Create a new TileSet.
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

    /// Insert a Tile and return its ID.
    fn insert(&mut self, tile: Tile) -> usize {
        self.centers.push(tile.center);
        self.ns.push(tile.n);
        self.nes.push(tile.ne);
        self.es.push(tile.e);
        self.ses.push(tile.se);
        self.ss.push(tile.s);
        self.sws.push(tile.sw);
        self.ws.push(tile.w);
        self.nws.push(tile.nw);
        self.ns.len() - 1
    }

    /// Get a Tile by ID.
    fn get(&self, id: usize) -> Option<Tile> {
        match self.centers.get(id) {
            Some(c) => Some(Tile {
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

    /// Get a VertexType for each tile in a given Direction.
    fn vertex_types(&mut self, direction: Direction) -> &Vec<VertexType> {
        match direction {
            Direction::N => &self.ns,
            Direction::NE => &self.nes,
            Direction::E => &self.es,
            Direction::SE => &self.ses,
            Direction::S => &self.ss,
            Direction::SW => &self.sws,
            Direction::W => &self.ws,
            Direction::NW => &self.nws,
        }
    }
}

/// A WaveFunctionCollapse cell.
#[derive(Clone, Debug)]
struct WfcCell {
    tile_set: TileSet,
    banned: Vec<bool>,
}
impl WfcCell {
    /// Create a new WfcCell.
    fn new() -> Self {
        Self {
            tile_set: TileSet::new(),
            banned: Vec::new(),
        }
    }

    /// Insert an unbanned Tile.
    fn insert(&mut self, tile: Tile) {
        self.tile_set.insert(tile);
        self.banned.push(false);
    }

    /// Update the cell given a neighboring vertex type and direction.
    fn update(&mut self, vertex_type: VertexType, direction: Direction) {
        match (vertex_type, direction) {
            (VertexType::Empty, _) => {
                self.ban(VertexType::Path, direction);
            },

            (VertexType::Ground, _) => {
                self.ban(VertexType::Path, direction);
            },

            (VertexType::Path, _) => {
                self.require(VertexType::Path, direction);
            },

            // (VertexType::Path, _) => {
                // self.require(VertexType::Path, direction);
            // },
            _ => (),
        }
    }

    /// Ban the given vertex type and direction.
    fn ban(&mut self, vertex_type: VertexType, direction: Direction) {
        let vts = self.tile_set.vertex_types(direction);
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt == vertex_type {
                *b = true;
            }
        }
    }

    /// Require the given vertex type and direction.
    fn require(&mut self, vertex_type: VertexType, direction: Direction) {
        let vts = self.tile_set.vertex_types(direction);
        for (vt, b) in vts.iter().zip(self.banned.iter_mut()) {
            if *vt != vertex_type {
                *b = true;
            }
        }
    }

    /// Collapse the cell to a tile.
    fn collapse(&mut self, rng: &mut ChaCha8Rng) -> Option<Tile> {
        let mut ids = Vec::new();
        for (id, b) in self.banned.iter().enumerate() {
            if !b {
                ids.push(id);
            }
        }
        let id = ids.iter().choose(rng);
        match id {
            Some(i) => {
                self.tile_set.get(*i)
            },
            None => None,
        }
    }

    /// Show the unbanned tiles.
    fn unbanned_string(&self) -> String {
        let mut tiles = Vec::new();
        for (id, b) in self.banned.iter().enumerate() {
            if !b {
                tiles.push(self.tile_set.get(id));
            }
        }
        format!("{:?}", tiles)
    }
}

fn tile_voxels(tile: Tile) -> Array3<Rgba> {
    let base = tile.center.rgba();
    let mut voxels = Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), base);
    let center = IVec3::new(1, 1, 2);
    voxels[center.as_uvec3().into_pos()] = base;
    for d in Direction::values() {
        let p = (center + d.voxel_offset()).as_uvec3().into_pos();
        voxels[p] = tile.vertex_type(d).rgba();
    }
    voxels
}

fn gen_all_path_tiles() -> Vec<Tile> {
    let directions = vec![
        Direction::N,
        Direction::NE,
        Direction::E,
        Direction::SE,
        Direction::S,
        Direction::SW,
        Direction::W,
        Direction::NW,
    ];
    let subsets = subsets(directions);
    let mut tiles = Vec::new();
    for subset in subsets {
        tiles.push(path_tile(subset));
    }
    tiles
}

fn subsets<T>(nums: Vec<T>) -> Vec<Vec<T>>
where
    T: Copy + Clone,
{
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

fn path_tile(directions: Vec<Direction>) -> Tile {
    let mut tile = Tile::debug_ground();
    tile.center = VertexType::Path;
    for d in directions {
        match d {
            Direction::N => tile.n = VertexType::Path,
            Direction::NE => tile.ne = VertexType::Path,
            Direction::E => tile.e = VertexType::Path,
            Direction::SE => tile.se = VertexType::Path,
            Direction::S => tile.s = VertexType::Path,
            Direction::SW => tile.sw = VertexType::Path,
            Direction::W => tile.w = VertexType::Path,
            Direction::NW => tile.nw = VertexType::Path,
        }
    }
    tile
}

struct TileMap {
    tiles: Array3<Tile>,
    path: Vec<(usize, usize, usize)>,
}

fn tilemap(size: UVec3, seed: u32) -> TileMap {
    // Initialize RGBA voxel array.
    let voxels_size = size;
    let voxels_center = voxels_size / 2;
    let mut values = Array3::from_elem(voxels_size.into_pos(), Tile::new());
    let mut tilemap_path = Vec::new();
    let mut sprawl_path = Vec::new();

    // Generate main path.
    let path_len = voxels_size.x / 2;
    let path_scale = 0.08;
    let path_center = path_len / 2;
    let mut path = Perlin1::gen(path_len as usize, seed, path_scale);
    normalize(&mut path.values);

    // Generate path sprawl.
    let mut sprawl = Perlin1::gen(path_len as usize, seed + 1, path_scale);
    normalize(&mut sprawl.values);

    // Define voxel space scaling.
    let p_scale = 7.0;
    let s_scale = 3.0;
    for ((x, p), s) in path.values.indexed_iter().zip(sprawl.values) {
        // Center the path in the voxel array.
        let x_c = (x as i32 - path_center as i32) + voxels_center.x as i32;
        // Center and scale path.
        let y_c = voxels_center.y + ((p - 0.5) * p_scale) as u32;
        // Add sprawl noise values to path.
        let s_s = (1.0 + s) * s_scale;
        for i in 1..s_s as i32 {
            let p0 = (x_c as usize, (y_c as i32 - i) as usize, 0);
            let p1 = (x_c as usize, (y_c as i32 + i) as usize, 0);
            // Add sprawl to wfc path.
            sprawl_path.push(p0);
            sprawl_path.push(p1);
            // Set default sprawl tiles.
            // values[p0] = Tile::debug_ground();
            // values[p1] = Tile::debug_ground();
            values[p0] = Tile::debug_maybe_path();
            values[p1] = Tile::debug_maybe_path();
        }
        let p = (x_c as usize, y_c as usize, 0);
        // Add main Perlin road to wfc path.
        tilemap_path.push(p);
        // Set default sprawl tiles.
        values[p] = Tile::debug_path();
    }
    // tilemap_path.extend(sprawl_path[0..8].iter());
    // tilemap_path.extend(sprawl_path.iter());

    TileMap {
        tiles: values,
        path: tilemap_path,
    }
}

fn wfc(tilemap: &mut TileMap) {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut wave = Array3::from_elem(tilemap.tiles.dim(), WfcCell::new());
    let neighbors = [
        (Direction::N, IVec3::new(0, 1, 0)),
        (Direction::NE, IVec3::new(1, 1, 0)),
        (Direction::E, IVec3::new(1, 0, 0)),
        (Direction::SE, IVec3::new(1, -1, 0)),
        (Direction::S, IVec3::new(0, -1, 0)),
        (Direction::SW, IVec3::new(-1, -1, 0)),
        (Direction::W, IVec3::new(-1, 0, 0)),
        (Direction::NW, IVec3::new(-1, 1, 0)),
    ];
    for p in tilemap.path.iter() {
        let w = wave.get_mut(*p).expect("Path out of bounds!");
        // let g = gen_all_path_tiles();
        let g = Tile::gen_all_debug_path_tiles();
        g.iter().for_each(|t| {
            w.insert(*t)
        });
        let v = p.into_ivec3();
        for (d, o) in neighbors {
            let n = (v + o).as_uvec3();
            let tile = tilemap.tiles.get(n.into_pos()).expect("Neighbor out of bounds!");
            let vertex_type = match d {
                Direction::N => tile.s,
                Direction::NE => tile.sw,
                Direction::E => tile.w,
                Direction::SE => tile.nw,
                Direction::S => tile.n,
                Direction::SW => tile.ne,
                Direction::W => tile.e,
                Direction::NW => tile.se,
            };
            if *p == (8, 14, 0) && d == Direction::N {
                // println!("{:?}", vertex_type);
                println!("{:?}", tile);
            }
            w.update(vertex_type, d);
        }
        let t = w.collapse(&mut rng);
        if let Some(tile) = t {
            if tile == Tile::debug() {
                println!("{:?}", p);
                // println!("{:?}", w.unbanned_string());
            }
            tilemap.tiles[*p] = tile;
        }
    }
}

pub fn gen(seed: u32) -> Array3<[u8; 4]> {
    let tilemap_size = UVec3::new(32, 32, 4);
    let mut tilemap = tilemap(tilemap_size, seed);
    wfc(&mut tilemap);
    let voxels_size = tilemap_size * TILE_SIZE as u32;
    let mut voxels = Array3::from_elem(voxels_size.into_pos(), [0; 4]);
    for (xyz, tile) in tilemap.tiles.indexed_iter() {
        let tile_voxels = tile_voxels(*tile);
        let tile_vector = xyz.into_uvec3() * TILE_SIZE as u32;
        for (xyz, rgba) in tile_voxels.indexed_iter() {
            let vector = tile_vector + xyz.into_uvec3();
            voxels[vector.into_pos()] = rgba.0;
        }
    }
    voxels
}