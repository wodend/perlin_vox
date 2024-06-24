use perlin_vox::caldera::gen;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let seeds = [0, 4, 17];

    for seed in seeds {
        let name = format!("caldera_{}", seed);
        let file_name = format!("output/{}.vox", &name);
        let mut timer = Timer::new();
        let values = gen(seed);
        timer.print_elapsed(&format!("Generated {}", &name));
        Vox::from(&values)
            .unwrap()
            .write(&file_name)
            .expect("Failed to write to file.");
        timer.print_elapsed(&format!("Wrote {}", &file_name));
    }
}
