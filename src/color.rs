/// RGB color in the working color space.
///
/// All channel values are normalized and thus are free from color depth limits.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Color {
    /// Red channel value between `0.0` and `1.0`.
    pub r: f64,
    /// Green channel value between `0.0` and `1.0`.
    pub g: f64,
    /// Blue channel value between `0.0` and `1.0`.
    pub b: f64,
}

unsafe impl Send for Color {}
unsafe impl Sync for Color {}

impl Default for Color {
    /// Returns the default color, which is black.
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }
}

impl Eq for Color {}

impl Color {
    /// Sets the RGB components of this color.
    pub fn set(&mut self, r: f64, g: f64, b: f64) {
        self.r = r;
        self.g = g;
        self.b = b;
    }
}
