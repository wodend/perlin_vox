use perlin_vox::noise::Perlin3;
use perlin_vox::render::PerlinHeatmap3;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let mut timer = Timer::new();

    let mut perlin3 = Perlin3::gen((64, 64, 64), 0, 0.1);
    let min = *perlin3
        .values
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max = *perlin3
        .values
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    timer.print_elapsed(&format!("Generated Perlin3 [min: {} max: {}]", min, max));

    let heatmap3 = PerlinHeatmap3::gen(&mut perlin3);
    timer.print_elapsed("Generated PerlinHeatmap3");

    Vox::from(&heatmap3.values)
        .unwrap()
        .write("output/perlin_heatmap_3.vox")
        .expect("Failed to write to file.");
    timer.print_elapsed("Wrote output/perlin_heatmap_3.vox");
}
