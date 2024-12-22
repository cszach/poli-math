use std::ops;

/// 3D vector for quantities such as 3D points, 3D directions, etc.
///
/// You can convert a tuple or an array of three floats to a 3D vector using
/// `.into()`.
///
/// ## Supported operators
///
/// All binary operations support vector and scalar values. Vector binary
/// operations are element-wise. For dot and cross product, see [`Self::dot`]
/// and [`Self::cross`], respectively.
///
/// - [`ops::Add`]
/// - [`ops::AddAssign`]
/// - [`ops::Sub`]
/// - [`ops::SubAssign`]
/// - [`ops::Mul`]
/// - [`ops::MulAssign`]
/// - [`ops::Div`]
/// - [`ops::DivAssign`]
/// - [`ops::Neg`]
///
/// You can use operators such as `+`, `-`, `*`, `/` for element-wise addition,
/// subtraction, multiplication, division, and negation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vector3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

unsafe impl Send for Vector3 {}
unsafe impl Sync for Vector3 {}

impl Eq for Vector3 {}

impl From<(f32, f32, f32)> for Vector3 {
    fn from(tuple: (f32, f32, f32)) -> Self {
        Vector3 {
            x: tuple.0,
            y: tuple.1,
            z: tuple.2,
        }
    }
}

impl From<[f32; 3]> for Vector3 {
    fn from(array: [f32; 3]) -> Self {
        Vector3 {
            x: array[0],
            y: array[1],
            z: array[2],
        }
    }
}

impl_op_ex!(+ |a: &Vector3, b: &Vector3| -> Vector3 {
    Vector3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
});

impl_op_ex!(+= |a: &mut Vector3, b: &Vector3| {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
});

impl_op_ex!(+|v: &Vector3, s: &f32| -> Vector3 {
    Vector3 {
        x: v.x + s,
        y: v.y + s,
        z: v.z + s,
    }
});

impl_op_ex!(+= |v: &mut Vector3, s: &f32| {
    v.x += s;
    v.y += s;
    v.z += s;
});

impl_op_ex!(-|a: &Vector3, b: &Vector3| -> Vector3 {
    Vector3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
});

impl_op_ex!(-= |a: &mut Vector3, b: &Vector3| {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
});

impl_op_ex!(-|v: &Vector3, s: &f32| -> Vector3 {
    Vector3 {
        x: v.x - s,
        y: v.y - s,
        z: v.z - s,
    }
});

impl_op_ex!(-= |v: &mut Vector3, s: &f32| {
    v.x -= s;
    v.y -= s;
    v.z -= s;
});

impl_op_ex!(*|a: &Vector3, b: &Vector3| -> Vector3 {
    Vector3 {
        x: a.x * b.x,
        y: a.y * b.y,
        z: a.z * b.z,
    }
});

impl_op_ex!(*= |a: &mut Vector3, b: &Vector3| {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
});

impl_op_ex!(*|v: &Vector3, s: &f32| -> Vector3 {
    Vector3 {
        x: v.x * s,
        y: v.y * s,
        z: v.z * s,
    }
});

impl_op_ex!(*= |v: &mut Vector3, s: &f32| {
    v.x *= s;
    v.y *= s;
    v.z *= s;
});

impl_op_ex!(/ |a: &Vector3, b: &Vector3| -> Vector3 {
    Vector3 {
        x: a.x / b.x,
        y: a.y / b.y,
        z: a.z / b.z,
    }
});

impl_op_ex!(/= |a: &mut Vector3, b: &Vector3| {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
});

impl_op_ex!(/|v: &Vector3, s: &f32| -> Vector3 {
    Vector3 {
        x: v.x / s,
        y: v.y / s,
        z: v.z / s,
    }
});

impl_op_ex!(/= |v: &mut Vector3, s: &f32| {
    v.x /= s;
    v.y /= s;
    v.z /= s;
});

impl_op_ex!(-|v: &Vector3| -> Vector3 {
    Vector3 {
        x: -v.x,
        y: -v.y,
        z: -v.z,
    }
});

impl Vector3 {
    /// Sets the elements of this vector.
    pub fn set(&mut self, x: f32, y: f32, z: f32) -> &Self {
        self.x = x;
        self.y = y;
        self.z = z;

        self
    }

    /// Returns the length of this vector.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalizes this vector.
    pub fn normalize(&mut self) -> &Self {
        let length = self.length();

        self.x /= length;
        self.y /= length;
        self.z /= length;

        self
    }

    /// Returns the normalized version of this vector.
    ///
    /// Note that this does not modify this vector.
    pub fn normalized(&self) -> Self {
        let length = self.length();

        Self {
            x: self.x / length,
            y: self.y / length,
            z: self.z / length,
        }
    }

    pub fn dot(&self, rhs: &Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Returns the cross product of this vector with another vector.
    ///
    /// Note that this does not modify this vector.
    pub fn cross(&self, rhs: &Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}
