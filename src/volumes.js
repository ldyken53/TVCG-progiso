export var datasets = {
    plane_x: {
        compressionRate: 2,
        name: "plane_x_32x32x32_float32.gen.crate2",
        range: [0, 1],
        scale: [1, 1, 1],
    },
    sphere: {
        compressionRate: 2,
        name: "sphere_32x32x32_float32.gen.crate2",
        range: [0, 1],
        scale: [1, 1, 1],
    },
    quarter_sphere: {
        compressionRate: 2,
        name: "quarter_sphere_32x32x32_float32.gen.crate2",
        range: [0, 1],
        scale: [1, 1, 1],
    },
    wavelet: {
        compressionRate: 2,
        name: "wavelet_32x32x32_float32.gen.crate2",
        range: [-3, 3],
        scale: [1, 1, 1],
    },
    fuel: {
        compressionRate: 4,
        name: "fuel_64x64x64_uint8.raw.crate4",
        range: [10, 255],
        scale: [1, 1, 1],
    },
    aneurism: {
        compressionRate: 4,
        name: "vertebra_512x512x512_uint16.raw.crate4",
        range: [550, 2100],
        scale: [1, 1, 1],
    },
    duct: {
        compressionRate: 2,
        name: "duct_193x194x1000_float32.raw.crate2",
        range: [0, 4],
        scale: [1, 1, 1000 / 193],
        step: 4 / 100
    },
    stagbeetle: {
        compressionRate: 2,
        name: "stag_beetle_832x832x494_uint16.raw.crate2",
        range: [100, 4096],
        scale: [1, 1, 1],
        step: 1.0 / 4096,
    },
    foot: {
        compressionRate: 2,
        name: "foot_256x256x256_uint8.raw.crate2",
        range: [10, 255],
        scale: [1, 1, 1],
    },
    backpack: {
        compressionRate: 4,
        name: "backpack_512x512x373_uint16.raw.crate4",
        range: [200, 4000],
        scale: [1, 1, 1],
        step: 1.0 / 3800.0,
    },
    // For benchmarks:
    tacc_turbulence: {
        compressionRate: 2,
        name: "tacc_turbulence_256x256x256_float32.raw.crate2",
        range: [1, 10],
        scale: [1, 1, 1],
        step: 10 / 100,
    },
    skull: {
        compressionRate: 2,
        name: "skull_256x256x256_uint8.raw.crate2",
        // range: [10, 255],
        range: [40, 140],
        scale: [1, 1, 1],
    },
    magnetic: {
        compressionRate: 4,
        name: "magnetic_reconnection_512x512x512_float32.raw.crate4",
        range: [0.1, 3.5],
        scale: [1, 1, 1],
        step: 1.0 / 8192,
    },
    kingsnake: {
        compressionRate: 2,
        name: "kingsnake_1024x1024x795_uint8.raw.crate2",
        range: [100, 150],
        scale: [1, 1, 1],
    },
    chameleon: {
        compressionRate: 2,
        name: "chameleon_1024x1024x1080_uint16.raw.crate2",
        range: [11000, 33000],
        scale: [1, 1, 1],
        step: 1.0 / 8192,
    },
    beechnut: {
        compressionRate: 1,
        name: "beechnut_1024x1024x1546_uint16.raw.crate1",
        range: [13200, 17000],
        scale: [1, 1, 1],
        step: (17000 - 13200) / 100.0,
    },
    miranda: {
        compressionRate: 4,
        name: "miranda_1024x1024x1024_float32.raw.crate4",
        range: [1.05, 2.9],
        scale: [1, 1, 1],
        step: 1.0 / 8192,
    },
    jicf_q: {
        compressionRate: 2,
        name: "jicf_q_1408x1080x1100_float32.raw.crate2",
        range: [-1, 1],
        scale: [1, 1, 1],
        step: 6 / 100
    },
    truss: {
        compressionRate: 2,
        name: "synthetic_truss_with_five_defects_1200x1200x1200_float32.raw.crate2",
        range: [0, 0.01],
        scale: [1, 1, 1],
    },
    dns_large: {
        compressionRate: 2,
        name: "dns_1920x1440x288_float64.raw.crate2",
        range: [0.75, 1.15],
        scale: [1, 1440 / 1920, 288 / 1920],
        step: 1.0 / 8192,
    },
    richtmyer_meshkov: {
        compressionRate: 1,
        name: "richtmyer_meshkov_2048x2048x1920_uint8.raw.crate1",
        range: [40, 190],
        scale: [1, 1, 1920.0 / 2048.0],
        step: 1.0,
    },
};


export var getVolumeDimensions = function(filename) {
    var fileRegex = /(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;
    var m = filename.match(fileRegex);
    return [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];
};

