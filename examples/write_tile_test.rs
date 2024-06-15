use std::time::Instant;

use perlin_vox::tile_map::{tile_map, perlin_tile};
use perlin_vox::render::tile_array;
use perlin_vox::vox::Vox;
use perlin_vox::timing::print_elapsed;

fn main() {
    let now = Instant::now();
    let mut previous = now.elapsed();

    let n = tile_map();
    previous = print_elapsed(previous, now.elapsed(), "Generated tile_map");
    let v = tile_array(n);
    previous = print_elapsed(previous, now.elapsed(), "Rendered tile_map");
    Vox::from(&v)
        .unwrap()
        .write("output/tile_array.vox")
        .expect("Failed to write to file.");
    previous = print_elapsed(previous, now.elapsed(), "Wrote tile_array");

    let n = perlin_tile();
    previous = print_elapsed(previous, now.elapsed(), "Generated perlin_tile");
    let v = tile_array(n);
    previous = print_elapsed(previous, now.elapsed(), "Rendered perlin tile_array");
    Vox::from(&v)
        .unwrap()
        .write("output/perlin_tile.vox")
        .expect("Failed to write to file.");
    print_elapsed(previous, now.elapsed(), "Wrote perlin_tile");
}
