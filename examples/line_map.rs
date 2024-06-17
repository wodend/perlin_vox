use perlin_vox::render::LineMap;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let mut timer = Timer::new();

    let line_map = LineMap::gen();
    timer.print_elapsed("Generated LineMap");

    Vox::from(&line_map.values)
        .unwrap()
        .write("output/line_map.vox")
        .expect("Failed to write to file.");
    timer.print_elapsed("Wrote output/line_map.vox");
}
