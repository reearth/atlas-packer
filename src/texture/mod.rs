use std::{
    path::{Path, PathBuf},
    sync::mpsc,
};

use image::{DynamicImage, GenericImageView, ImageBuffer};
use rayon::prelude::*;
use utils::{calc_bbox, uv_to_pixel_coords};

pub mod cache;
mod utils;

#[derive(Debug, Clone)]
pub struct DownsampleFactor(f32);

impl DownsampleFactor {
    pub fn new(factor: &f32) -> Self {
        if (0.0..=1.0).contains(factor) {
            DownsampleFactor(*factor)
        } else {
            panic!("The argument must be entered between 0~1.") //FIXME: panic! is not recommended
        }
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

// A structure that retains an image cut out from the original image.
#[derive(Debug, Clone)]
pub struct CroppedTexture {
    pub image_path: PathBuf,
    // The origin of the cropped image in the original image (top-left corner).
    pub origin: (u32, u32),
    pub width: u32,
    pub height: u32,
    pub downsample_factor: DownsampleFactor,
    // UV coordinates for the cropped texture (bottom-left origin).
    pub cropped_uv_coords: Vec<(f64, f64)>,
}

impl CroppedTexture {
    pub fn new(
        image_path: &Path,
        size: (u32, u32),
        uv_coords: &[(f64, f64)],
        downsample_factor: DownsampleFactor,
    ) -> Self {
        let pixel_coords = uv_to_pixel_coords(uv_coords, size.0, size.1);
        let (min_x, min_y, max_x, max_y) = calc_bbox(&pixel_coords);

        let cropped_width = max_x - min_x;
        let cropped_height = max_y - min_y;

        let dest_uv_coords = pixel_coords
            .iter()
            .map(|(x, y)| {
                (
                    (*x - min_x) as f64 / cropped_width as f64,
                    1.0 - (*y - min_y) as f64 / cropped_height as f64,
                )
            })
            .collect::<Vec<(f64, f64)>>();

        CroppedTexture {
            image_path: image_path.to_path_buf(),
            origin: (min_x, min_y),
            width: cropped_width,
            height: cropped_height,
            downsample_factor,
            cropped_uv_coords: dest_uv_coords,
        }
    }

    /// Check if the two textures partially or completely overlap.
    pub(super) fn overlaps(&self, other: &Self) -> bool {
        if self.image_path != other.image_path {
            return false;
        }

        let (x1, y1, w1, h1) = (self.origin.0, self.origin.1, self.width, self.height);
        let (x2, y2, w2, h2) = (other.origin.0, other.origin.1, other.width, other.height);

        !(x1 + w1 < x2 || x2 + w2 < x1 || y1 + h1 < y2 || y2 + h2 < y1)
    }

    /// Check if the texture completely covers the other texture.
    /// If the texture completely covers the other texture, return the offset.
    pub(super) fn covers(&self, other: &Self) -> Option<(u32, u32)> {
        if self.image_path != other.image_path {
            return None;
        }

        let (x1, y1, w1, h1) = (self.origin.0, self.origin.1, self.width, self.height);
        let (x2, y2, w2, h2) = (other.origin.0, other.origin.1, other.width, other.height);

        if x1 <= x2 && y1 <= y2 && x1 + w1 >= x2 + w2 && y1 + h1 >= y2 + h2 {
            Some((x2 - x1, y2 - y1))
        } else {
            None
        }
    }

    pub fn crop(&self, image: &DynamicImage) -> DynamicImage {
        let (x, y) = self.origin;
        let cropped_image = image.view(x, y, self.width, self.height).to_image();

        // Collect pixels into a Vec and then process in parallel
        let pixels: Vec<_> = cropped_image.enumerate_pixels().collect();

        let samples = 1;
        let num_threads = rayon::current_num_threads();
        let chunk_size = (pixels.len() / num_threads).clamp(1, pixels.len() + 1);

        let (sender, receiver) = mpsc::channel();

        // If the center coordinates of the pixel are contained within a polygon composed of UV coordinates, the pixel is written
        pixels
            .par_chunks(chunk_size)
            .for_each_with(sender, |s, chunk| {
                let mut local_results = Vec::new();

                for &(px, py, pixel) in chunk {
                    let mut is_inside = false;

                    'subpixels: for sx in 0..samples {
                        for sy in 0..samples {
                            let x = (px as f64 + (sx as f64 + 0.5) / samples as f64)
                                / self.width as f64;
                            let y = 1.0
                                - (py as f64 + (sy as f64 + 0.5) / samples as f64)
                                    / self.height as f64;
                            // Adjust x and y to the center of the pixel
                            let center_x = x + 0.5 / self.width as f64;
                            let center_y = y - 0.5 / self.height as f64;

                            if is_point_inside_polygon(
                                (center_x, center_y),
                                &self.cropped_uv_coords,
                            ) {
                                is_inside = true;
                                break 'subpixels;
                            }
                        }
                    }

                    if is_inside {
                        local_results.push((px, py, *pixel));
                    } else {
                        // FIXME: Do not crop temporarily because pixel boundary jaggies will occur.
                        local_results.push((px, py, *pixel));
                    }
                }

                s.send(local_results).unwrap();
            });

        // Collect results in the main thread
        let mut clipped = ImageBuffer::new(self.width, self.height);
        for received in receiver {
            for (px, py, pixel) in received {
                clipped.put_pixel(px, py, pixel);
            }
        }

        // Downsample
        let scaled_width = (clipped.width() as f32 * self.downsample_factor.value()) as u32;
        let scaled_height = (clipped.height() as f32 * self.downsample_factor.value()) as u32;

        DynamicImage::ImageRgba8(image::imageops::resize(
            &clipped,
            scaled_width,
            scaled_height,
            image::imageops::FilterType::Triangle,
        ))
    }
}

fn is_point_inside_polygon(test_point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let mut is_inside = false;
    let mut previous_vertex_index = polygon.len() - 1;

    for current_vertex_index in 0..polygon.len() {
        let (current_x, current_y) = polygon[current_vertex_index];
        let (previous_x, previous_y) = polygon[previous_vertex_index];

        let is_y_between_vertices = (current_y > test_point.1) != (previous_y > test_point.1);
        let does_ray_intersect = test_point.0
            < (previous_x - current_x) * (test_point.1 - current_y) / (previous_y - current_y)
                + current_x;

        if is_y_between_vertices && does_ray_intersect {
            is_inside = !is_inside;
        }

        previous_vertex_index = current_vertex_index;
    }

    is_inside
}
