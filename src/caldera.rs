use glam::IVec3;
use ndarray::Array3;
use glam::UVec3;
use glam::Vec3Swizzles;
use palette::Srgba;
use rand::seq::IteratorRandom;
use rand::rngs::ThreadRng;

use crate::noise::{Perlin1, Perlin2};
use crate::render::{normalize, binary_heatmap_gradient};
use crate::vector::{Vector3, Pos3};

#[non_exhaustive]
struct DebugPalette;

impl DebugPalette {
    pub const LIGHT_GREEN: [u8; 4] = [120, 159, 138, 255];
    pub const DARK_GREEN: [u8; 4] = [33, 69, 51, 255];
    pub const LIGHT_GREY: [u8; 4] = [92, 92, 92, 255];
    pub const DARK_GREY: [u8; 4] = [51, 51, 51, 255];
    pub const ERROR: [u8; 4] = [255, 0, 0, 255];
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum VertexType {
    Empty,
    Path,
    Ground,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
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
struct Tile {
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
    fn is_path(&self) -> bool {
        self.n == VertexType::Path
            || self.ne == VertexType::Path
            || self.e == VertexType::Path
            || self.se == VertexType::Path
            || self.s == VertexType::Path
            || self.sw == VertexType::Path
            || self.w == VertexType::Path
            || self.nw == VertexType::Path
    }
}

#[non_exhaustive]
#[derive(PartialEq, Eq)]
struct DebugTiles;

impl DebugTiles {
    pub const EMPTY: Tile = Tile {
        n: VertexType::Empty,
        ne: VertexType::Empty,
        e: VertexType::Empty,
        se: VertexType::Empty,
        s: VertexType::Empty,
        sw: VertexType::Empty,
        w: VertexType::Empty,
        nw: VertexType::Empty,
    };
    pub const GROUND: Tile = Tile {
        n: VertexType::Ground,
        ne: VertexType::Ground,
        e: VertexType::Ground,
        se: VertexType::Ground,
        s: VertexType::Ground,
        sw: VertexType::Ground,
        w: VertexType::Ground,
        nw: VertexType::Ground,
    };
    pub const MAYBE_PATH: Tile = Tile {
        n: VertexType::Path,
        ne: VertexType::Path,
        e: VertexType::Path,
        se: VertexType::Path,
        s: VertexType::Path,
        sw: VertexType::Path,
        w: VertexType::Path,
        nw: VertexType::Path,
    };
    pub const INTERSECTION: Tile = Tile {
        n: VertexType::Path,
        ne: VertexType::Ground,
        e: VertexType::Path,
        se: VertexType::Ground,
        s: VertexType::Path,
        sw: VertexType::Ground,
        w: VertexType::Path,
        nw: VertexType::Ground,
    };
    pub const PATH_EW: Tile = Tile {
        n: VertexType::Ground,
        ne: VertexType::Ground,
        e: VertexType::Path,
        se: VertexType::Ground,
        s: VertexType::Ground,
        sw: VertexType::Ground,
        w: VertexType::Path,
        nw: VertexType::Ground,
    };
}

/// A wave function collapse cell.
#[derive(Clone, Debug)]
pub struct WfcCell {
    ns: Vec<VertexType>,
    nes: Vec<VertexType>,
    es: Vec<VertexType>,
    ses: Vec<VertexType>,
    ss: Vec<VertexType>,
    sws: Vec<VertexType>,
    ws: Vec<VertexType>,
    nws: Vec<VertexType>,
    banned: Vec<bool>,
}
impl WfcCell {
    /// Create a new WfcCell.
    fn new() -> Self {
        Self {
            ns: Vec::new(),
            nes: Vec::new(),
            es: Vec::new(),
            ses: Vec::new(),
            ss: Vec::new(),
            sws: Vec::new(),
            ws: Vec::new(),
            nws: Vec::new(),
            banned: Vec::new(),
        }
    }

    /// Insert a new tile and return its ID (unbanned by default).
    fn insert(&mut self, tile: Tile) -> std::io::Result<usize> {
        self.ns.push(tile.n);
        self.nes.push(tile.ne);
        self.es.push(tile.e);
        self.ses.push(tile.se);
        self.ss.push(tile.s);
        self.sws.push(tile.sw);
        self.ws.push(tile.w);
        self.nws.push(tile.nw);
        self.banned.push(false);
        Ok(self.banned.len() - 1)
    }

    /// Update the cell given a neighboring vertex type and direction.
    fn update(&mut self, vertex_type: VertexType, direction: Direction) {
        match (vertex_type, direction) {
            // (VertexType::Path, Direction::E|Direction::W) => {
            //     self.insert(DebugTiles::PATH_EW);
            // }
            (VertexType::Empty, _) => {
                self.ban(VertexType::Path, direction);
            },

            (VertexType::Ground, _) => {
                self.ban(VertexType::Path, direction);
            },

            (VertexType::Path, _) => {
                self.constrain(VertexType::Path, direction);
            },
            _ => (),
        }
    }

    fn vertex_types(&self, direction: Direction) -> &Vec<VertexType> {
        match direction {
            Direction::N => {
                &self.ns
            },
            _ => &self.ns,
        }
    }
    /// Ban the given vertex type and direction.
    fn constrain(&mut self, vertex_type: VertexType, direction: Direction) {
        let vts = self.vertex_types(direction);
        // for (v, b) in vts.iter().zip(self.banned.iter_mut()) {
        //     if *v != vertex_type {
        //         *b = true;
        //     }
        // }

        match direction {
            Direction::N => {
                for (v, b) in self.ns.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::NE => {
                for (v, b) in self.nes.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::E => {
                for (v, b) in self.es.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::SE => {
                for (v, b) in self.ses.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::S => {
                for (v, b) in self.ss.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::SW => {
                for (v, b) in self.sws.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::W => {
                for (v, b) in self.ws.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::NW => {
                for (v, b) in self.nws.iter().zip(self.banned.iter_mut()) {
                    if *v != vertex_type {
                        *b = true;
                    }
                }
            },
        }
    }


    /// Ban the given vertex type and direction.
    fn ban(&mut self, vertex_type: VertexType, direction: Direction) {
        match direction {
            Direction::N => {
                for (v, b) in self.ns.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::NE => {
                for (v, b) in self.nes.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::E => {
                for (v, b) in self.es.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::SE => {
                for (v, b) in self.ses.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::S => {
                for (v, b) in self.ss.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::SW => {
                for (v, b) in self.sws.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::W => {
                for (v, b) in self.ws.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
            Direction::NW => {
                for (v, b) in self.nws.iter().zip(self.banned.iter_mut()) {
                    if *v == vertex_type {
                        *b = true;
                    }
                }
            },
        }
    }

    /// Collapse the cell to a tile.
    pub fn collapse(&mut self, rng: &mut ThreadRng) -> Option<Tile> {
        let mut ids = Vec::new();
        for (i, b) in self.banned.iter().enumerate() {
            if !b {
                ids.push(i);
            }
        }
        let id = ids.iter().choose(rng);
        match id {
            Some(i) => {
                Some(Tile {
                    n: self.ns[*i],
                    ne: self.nes[*i],
                    e: self.es[*i],
                    se: self.ses[*i],
                    s: self.ss[*i],
                    sw: self.sws[*i],
                    w: self.ws[*i],
                    nw: self.nws[*i],
                })
            },
            None => None,
        }
    }
}

fn tile_voxels(tile: Tile) -> Array3<[u8; 4]>{
    if tile.is_path() {
        return path_voxels(tile);
    } else {
        match tile {
            DebugTiles::EMPTY => {
                Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), [0; 4])
            },
            DebugTiles::GROUND => {
                Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), DebugPalette::LIGHT_GREEN)
            },
            DebugTiles::MAYBE_PATH => {
                let mut base = Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), DebugPalette::LIGHT_GREEN);
                base[[0, 1, 2]] = DebugPalette::DARK_GREY;
                base[[1, 0, 2]] = DebugPalette::DARK_GREY;
                base[[1, 2, 2]] = DebugPalette::DARK_GREY;
                base[[2, 1, 2]] = DebugPalette::DARK_GREY;
                base
            },
            DebugTiles::INTERSECTION => {
                let mut base = Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), DebugPalette::LIGHT_GREEN);
                base[[0, 1, 2]] = DebugPalette::DARK_GREY;
                base[[1, 0, 2]] = DebugPalette::DARK_GREY;
                base[[1, 2, 2]] = DebugPalette::DARK_GREY;
                base[[2, 1, 2]] = DebugPalette::DARK_GREY;
                base
            },
            DebugTiles::PATH_EW => {
                let mut base = Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), DebugPalette::DARK_GREEN);
                base[[0, 1, 2]] = DebugPalette::LIGHT_GREY;
                base[[1, 1, 2]] = DebugPalette::LIGHT_GREY;
                base[[2, 1, 2]] = DebugPalette::LIGHT_GREY;
                base
            },
            _ => {
                Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), DebugPalette::ERROR)
            }
        }
    }
}

fn path_voxels(tile: Tile) -> Array3<[u8; 4]> {
    // todo: use offsets bound to each direction: reuse below when finding neighbors
    let mut voxels = Array3::from_elem((TILE_SIZE, TILE_SIZE, TILE_SIZE), DebugPalette::DARK_GREEN);
    voxels[[1, 1, 2]] = DebugPalette::LIGHT_GREY;
    if tile.n == VertexType::Path {
        voxels[[1, 2, 2]] = DebugPalette::LIGHT_GREY;
    }
    if tile.ne == VertexType::Path {
        voxels[[2, 2, 2]] = DebugPalette::LIGHT_GREY;
    }
    if tile.e == VertexType::Path {
        voxels[[2, 1, 2]] = DebugPalette::LIGHT_GREY;
    }
    if tile.se == VertexType::Path {
        voxels[[2, 0, 2]] = DebugPalette::LIGHT_GREY;
    }
    if tile.s == VertexType::Path {
        voxels[[1, 0, 2]] = DebugPalette::LIGHT_GREY;
    }
    if tile.sw == VertexType::Path {
        voxels[[0, 0, 2]] = DebugPalette::LIGHT_GREY;
    }
    if tile.w == VertexType::Path {
        voxels[[0, 1, 2]] = DebugPalette::LIGHT_GREY;
    }
    if tile.nw == VertexType::Path {
        voxels[[0, 2, 2]] = DebugPalette::LIGHT_GREY;
    }
    voxels
}

fn path_tile(directions: Vec<Direction>) -> Tile {
    let mut tile = DebugTiles::GROUND;
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

pub fn subsets<T>(nums: Vec<T>) -> Vec<Vec<T>>
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
    // let subsets = vec![
    //     vec![Direction::E, Direction::W],
    //     vec![Direction::SE, Direction::W],
    //     // vec![Direction::E, Direction::W],
    // ];
    let mut tiles = Vec::new();
    for subset in subsets {
        tiles.push(path_tile(subset));
    }
    tiles
}

struct TileMap {
    tiles: Array3<Tile>,
    path: Vec<(usize, usize, usize)>,
}

fn tilemap(size: UVec3, seed: u32) -> TileMap {
    // Initialize RGBA voxel array.
    let voxels_size = size;
    let voxels_center = voxels_size / 2;
    // let mut values = Array3::from_elem(voxels_size.into_pos(), [0; 4]);
    let mut values = Array3::from_elem(voxels_size.into_pos(), DebugTiles::EMPTY);
    let mut tilemap_path = Vec::new();

    // Generate main path.
    let path_len = voxels_size.x / 2;
    let path_scale = 0.08;
    let path_center = path_len / 2;
    let mut path = Perlin1::gen(path_len as usize, seed, path_scale);
    normalize(&mut path.values);

    // Generate path sprawl.
    let mut sprawl = Perlin1::gen(path_len as usize, seed + 1, path_scale);
    normalize(&mut sprawl.values);

    // RGBA settings
    let gradient_size = 8;
    let gradient = binary_heatmap_gradient(gradient_size);

    // Define voxel space scaling.
    let p_scale = gradient_size as f32 - 1.0;
    let s_scale = 3.0;
    for ((x, p), s) in path.values.indexed_iter().zip(sprawl.values) {
        // Voxel RGBA.
        let index = (p * p_scale) as usize;
        let linsrgba = gradient[index];
        // let rgba = Srgba::from(linsrgba).into();

        // Center the path in the voxel array.
        let x_c = (x as i32 - path_center as i32) + voxels_center.x as i32;
        // Center and scale path.
        let y_c = voxels_center.y + ((p - 0.5) * p_scale) as u32;
        // Add sprawl noise values to path.
        let s_s = (1.0 + s) * s_scale;
        for i in 1..s_s as i32 {
            // values[[x_c as usize, (y_c as i32 - i) as usize, 0]] = DebugPalette::LIGHT_GREEN;
            // values[[x_c as usize, (y_c as i32 + i) as usize, 0]] = DebugPalette::LIGHT_GREEN;
            let p0 = (x_c as usize, (y_c as i32 - i) as usize, 0);
            let p1 = (x_c as usize, (y_c as i32 + i) as usize, 0);
            // Add sprawl to wfc path.
            // tilemap_path.push(p0);
            // tilemap_path.push(p1);
            values[p0] = DebugTiles::GROUND;
            values[p1] = DebugTiles::GROUND;
        }
        // values[[x_c as usize, (y_c as i32) as usize, 0]] = rgba;
        let p = (x_c as usize, y_c as usize, 0);
        tilemap_path.push(p);
        values[p] = DebugTiles::MAYBE_PATH;
    }

    TileMap {
        tiles: values,
        path: tilemap_path,
    }
}

fn wfc(tilemap: &mut TileMap) {
    let mut rng = rand::thread_rng();
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
    for (i, p) in tilemap.path.iter().enumerate() {
        let w = wave.get_mut(*p).expect("Path out of bounds!");
        let g = gen_all_path_tiles();
        // if tilemap.tiles[*p] == DebugTiles::MAYBE_PATH {
        g.iter().for_each(|t| {
            w.insert(*t).expect("Failed to insert tile!");
        });
        // }
        // if tilemap.tiles[*p] == DebugTiles::GROUND {
        //     w.insert(DebugTiles::INTERSECTION).expect("Failed to insert tile!");
        //     w.insert(DebugTiles::GROUND).expect("Failed to insert tile!");
        // }
        let v = p.into_ivec3();
            if i == 2 {
            println!("{:?} {:?}", p, tilemap.tiles[*p]);
            }
        for (d, o) in neighbors {
            let n = (v + o).as_uvec3();
                if i == 2 {
                println!("{:?} {:?}", d, tilemap.tiles[n.into_pos()]);
                }
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
            w.update(vertex_type, d);
                if i == 2 {
                println!("{:?}", w);
                }
            // if i == 2 {
            //     println!("{:?} {:?}", p, tilemap.tiles[*p]);
            //     tilemap.tiles[*p] = DebugTiles::EMPTY;
            // }
        }
        let t = w.collapse(&mut rng);
        if let Some(tile) = t {
            tilemap.tiles[*p] = tile;
        }
        // tilemap.tiles[*p] = DebugTiles::PATH_EW;
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
            voxels[vector.into_pos()] = *rgba;
        }
    }
    voxels
}