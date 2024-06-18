use perlin_vox::render::Render;
use perlin_vox::tile_map::TileMap;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let mut timer = Timer::new();

    let tile_map = TileMap::gen();
    timer.print_elapsed("Generated TileMap");

    let render = Render::gen(tile_map);
    timer.print_elapsed("Rendered TileMap");

    Vox::from(&render.values)
        .unwrap()
        .write("output/tile_rotations.vox")
        .expect("Failed to write to file.");
    timer.print_elapsed("Wrote output/tile_rotations.vox");
}
