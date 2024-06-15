use std::time::Instant;

use glam::Vec2;

use perlin_vox::noise::narray2;
use perlin_vox::render::varray2;
use perlin_vox::vox::Vox;
use perlin_vox::timing::print_elapsed;

fn main() {
    let now = Instant::now();
    let mut previous = now.elapsed();

    let mut n = narray2((8, 8), Vec2::new(0.1, 0.1));
    previous = print_elapsed(previous, now.elapsed(), "Generated narray2");
    let v = varray2(&mut n);
    previous = print_elapsed(previous, now.elapsed(), "Generated varray2");
    Vox::from(&v)
        .unwrap()
        .write("output/varray2.vox")
        .expect("Failed to write to file.");
    print_elapsed(previous, now.elapsed(), "Wrote output2");
}
