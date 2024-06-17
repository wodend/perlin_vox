use std::time::Instant;

use perlin_vox::noise::Noise1;
use perlin_vox::render::tile_array;
use perlin_vox::render::Voxels;
use perlin_vox::tile_map::{test_1d_perlin, tile_map};
use perlin_vox::timing::print_elapsed;
use perlin_vox::vox::Vox;

fn main() {
    let now = Instant::now();
    let mut previous = now.elapsed();

    let mut noise1 = Noise1::new(80, 0.1);
    previous = print_elapsed(previous, now.elapsed(), "Generated 1D Perlin noise");
    let voxels = Voxels::noise1_heatmap(&mut noise1);
    previous = print_elapsed(previous, now.elapsed(), "Generated 1D Perlin heatmap");
    Vox::from(&voxels)
        .unwrap()
        .write("output/perlin_1d_heatmap.vox")
        .expect("Failed to write to file.");
    print_elapsed(
        previous,
        now.elapsed(),
        "Wrote output/perlin_1d_heatmap.vox",
    );
}
