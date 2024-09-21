use std::path::Path;

use image::ImageReader;

pub fn get_image_size<P: AsRef<Path>>(file_path: P) -> Result<(u32, u32), image::ImageError> {
    let reader = ImageReader::open(file_path)?;
    let dimensions = reader.into_dimensions()?;
    Ok(dimensions)
}

pub fn uv_to_pixel_coords(uv_coords: &[(f64, f64)], width: u32, height: u32) -> Vec<(u32, u32)> {
    uv_coords
        .iter()
        .map(|(u, v)| {
            (
                (u.clamp(0.0, 1.0) * width as f64).min(width as f64 - 1.0) as u32,
                ((1.0 - v.clamp(0.0, 1.0)) * height as f64).min(height as f64 - 1.0) as u32,
            )
        })
        .collect()
}

pub fn calc_bbox(pixel_coords: &[(u32, u32)]) -> (u32, u32, u32, u32) {
    pixel_coords.iter().fold(
        (u32::MAX, u32::MAX, 0, 0),
        |(min_x, min_y, max_x, max_y), (x, y)| {
            (min_x.min(*x), min_y.min(*y), max_x.max(*x), max_y.max(*y))
        },
    )
}
