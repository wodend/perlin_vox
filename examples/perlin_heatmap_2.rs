use perlin_vox::noise::Perlin2;
use perlin_vox::render::PerlinHeatmap2;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let mut timer = Timer::new();

    let mut perlin2 = Perlin2::gen((64, 64), 0, 0.1);
    timer.print_elapsed("Generated Perlin2");

    let heatmap2 = PerlinHeatmap2::gen(&mut perlin2);
    timer.print_elapsed("Generated PerlinHeatmap2");

    Vox::from(&heatmap2.values)
        .unwrap()
        .write("output/perlin_heatmap_2.vox")
        .expect("Failed to write to file.");
    timer.print_elapsed("Wrote output/perlin_heatmap_2.vox");
}
