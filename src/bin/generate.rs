use std::{
    fs::File,
    io::{Read, Write},
};

use bytemuck::cast_slice;
use itertools::Itertools;

fn main() {
    let mut histo = vec![0_u16; 256 * 256 * 256];
    let data = File::open(std::env::args().nth(1).unwrap()).unwrap();
    let data = std::io::BufReader::new(data);
    for (a, b, c) in data.bytes().map(|x| x.unwrap() as usize).tuple_windows() {
        // let (a, b, c) = ar.collect_tuple().unwrap();
        let ix = a + (b << 8) + (c << 16);
        histo[ix] = histo[ix].saturating_add(1);
    }

    File::create(std::env::args().nth(2).unwrap())
        .unwrap()
        .write_all(cast_slice(&histo))
        .unwrap();
}
