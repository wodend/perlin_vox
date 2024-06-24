use perlin_vox::noise::Perlin1;
use perlin_vox::render::PerlinHeatmap1;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let seeds = [0, 4, 17];

    for seed in seeds {
        let name = format!("perlin1_{}", seed);
        let file_name = format!("output/{}.vox", &name);
        let mut timer = Timer::new();

        let mut perlin = Perlin1::gen(64, seed, 0.1);
        timer.print_elapsed(&format!("Generated {}", &name));

        let heatmap = PerlinHeatmap1::gen(&mut perlin);
        timer.print_elapsed(&format!("Generated {} heatmap", &name));

        Vox::from(&heatmap.values)
            .unwrap()
            .write(&file_name)
            .expect("Failed to write to file.");
        timer.print_elapsed(&format!("wrote {}", &file_name));
    }
}