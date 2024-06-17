use perlin_vox::noise::Perlin1;
use perlin_vox::render::PerlinHeatmap1;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let mut timer = Timer::new();

    let mut perlin1 = Perlin1::gen(64, 0, 0.1);
    let min = *perlin1
        .values
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max = *perlin1
        .values
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    timer.print_elapsed(&format!("Generated Perlin1 [min: {} max: {}]", min, max));

    let heatmap1 = PerlinHeatmap1::gen(&mut perlin1);
    timer.print_elapsed("Generated PerlinHeatmap1");

    Vox::from(&heatmap1.values)
        .unwrap()
        .write("output/perlin_heatmap_1.vox")
        .expect("Failed to write to file.");
    timer.print_elapsed("Wrote output/perlin_heatmap_1.vox");
}
