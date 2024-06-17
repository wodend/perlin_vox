use std::time::Instant;

use perlin_vox::render::tile_array;
use perlin_vox::tile_map::{test_1d_perlin, tile_map};
use perlin_vox::timing::print_elapsed;
use perlin_vox::vox::Vox;

fn main() {
    let now = Instant::now();
    let mut previous = now.elapsed();

    let n = test_1d_perlin();
    previous = print_elapsed(previous, now.elapsed(), "Generated test_1d_perlin");
    let v = tile_array(n);
    previous = print_elapsed(previous, now.elapsed(), "Rendered perlin tile_array");
    Vox::from(&v)
        .unwrap()
        .write("output/test_1d_perlin.vox")
        .expect("Failed to write to file.");
    print_elapsed(previous, now.elapsed(), "Wrote test_1d_perlin");
}
