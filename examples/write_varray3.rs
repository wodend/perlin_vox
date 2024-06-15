use std::time::Instant;

use perlin_vox::noise::narray3;
use perlin_vox::render::varray3;
use perlin_vox::vox::Vox;
use perlin_vox::timing::print_elapsed;

fn main() {
    let now = Instant::now();
    let mut previous = now.elapsed();

    let mut n = narray3();
    previous = print_elapsed(previous, now.elapsed(), "Generated narray3");
    let v = varray3(&mut n);
    previous = print_elapsed(previous, now.elapsed(), "Generated varray3");
    Vox::from(&v)
        .unwrap()
        .write("output/varray3.vox")
        .expect("Failed to write to file.");
    print_elapsed(previous, now.elapsed(), "Wrote output3");
}
