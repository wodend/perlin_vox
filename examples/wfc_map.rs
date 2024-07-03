use perlin_vox::wfc::honeycomb::Honeycomb;
use perlin_vox::wfc::wfc_map::WfcMap;
use perlin_vox::timing::Timer;
use perlin_vox::vox::Vox;

fn main() {
    let seeds = [0, 4, 731];

    for seed in seeds {
        let name = format!("wfc_map_{}", seed);
        let file_name = format!("output/{}.vox", &name);
        let mut timer = Timer::new();
        let honeycomb = Honeycomb::gen_perlin(seed);
        timer.print_elapsed(&format!("Generated honeycomb {}", &name));
        let wfc_map = WfcMap::from_honeycomb(honeycomb);
        timer.print_elapsed(&format!("Generated wfc_map {}", &name));
        let render = wfc_map.honeycomb.debug_render();
        timer.print_elapsed(&format!("Rendered {}", &name));
        Vox::from(&render)
            .unwrap()
            .write(&file_name)
            .expect("Failed to write to file.");
        timer.print_elapsed(&format!("wrote {}", &file_name));
    }
}
