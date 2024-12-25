use std::ops;

use super::{Euler, EulerOrder, Vector3};

/// Quaternion, which can be used to represent rotations around arbitrary axes.
///
/// ## Quaternion primer
///
/// Quaternions are a better way to represent 3D rotations compared to
/// [Euler angles](super::Euler).
///
/// A quaternion represents rotation around an arbitrary axis, while Euler
/// angles represent three rotations around the local X, Y, and Z axis,
/// respectively, in a specified order.
///
/// Compared to Euler angles, quaternions are faster to operate on, use less
/// memory, easier to interpolate, and are less prone to the gimbal lock issue.
///
/// A quaternion has four components: x, y, z, and w. A rotation by `α` radians
/// around the axis `β` is represented by:
/// - `x = (β.x * (α / 2.0)).sin()`
/// - `y = (β.y * (α / 2.0)).sin()`
/// - `z = (β.z * (α / 2.0)).sin()`
/// - `w = (α / 2.0).cos()`
///
/// ## Creating a quaternion
///
/// - Use [`Quaternion::from`] if you already have the Euler angles.
/// - Use [`Quaternion::from_axis_angle`] for a rotation around an arbitrary
///   axis.
/// - Or manually instantiate a new struct if you already have the components.
///
/// ## Quaternion operations
///
/// - To combine rotations, multiply two quaternions. `a * b` is the rotation
///   obtained by first applying `b` and then `a`, where `a` and `b` are both
///   quaternions.
/// - The reverse of a rotation represented by quaternion `q` is the conjugate
///   of `q`. For rotation quaternions, the conjugate is the same as the
///   inverse. These correspond to [`Quaternion::conjugate`] and
///   [`Quaternion::invert`], respectively.
/// - Rotation quaternions must be unit (normalized) quaternions. To normalize
///   a quaternion, use [`Quaternion::normalize`].
///
/// ## Conversion between quaternions and Euler angles
///
/// ```rust
/// use poli_math::{Quaternion, Euler, EulerOrder};
/// let euler = Euler::default();
///
/// // From Euler angles to quaternion
/// let q: Quaternion = (&euler).into();
/// let q = Quaternion::from(&euler);
///
/// // From quaternion to Euler angles
/// let euler = Euler::from_quaternion(&q, EulerOrder::default());
/// ```
///
/// ## Supported operators
///
/// - [`ops::Mul`]
/// - [`ops::MulAssign`]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

unsafe impl Send for Quaternion {}
unsafe impl Sync for Quaternion {}

impl Default for Quaternion {
    /// Returns the identity quaternion (i.e. no rotation).
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl Eq for Quaternion {}

impl From<&Euler> for Quaternion {
    /// Converts the given Euler angles to a rotation quaternion.
    fn from(euler: &Euler) -> Self {
        let c1 = (euler.x / 2.0).cos();
        let c2 = (euler.y / 2.0).cos();
        let c3 = (euler.z / 2.0).cos();

        let s1 = (euler.x / 2.0).sin();
        let s2 = (euler.y / 2.0).sin();
        let s3 = (euler.z / 2.0).sin();

        match euler.order {
            EulerOrder::Xyz => Self {
                x: s1 * c2 * c3 + c1 * s2 * s3,
                y: c1 * s2 * c3 - s1 * c2 * s3,
                z: c1 * c2 * s3 + s1 * s2 * c3,
                w: c1 * c2 * c3 - s1 * s2 * s3,
            },

            EulerOrder::Yxz => Self {
                x: s1 * c2 * c3 + c1 * s2 * s3,
                y: c1 * s2 * c3 - s1 * c2 * s3,
                z: c1 * c2 * s3 - s1 * s2 * c3,
                w: c1 * c2 * c3 + s1 * s2 * s3,
            },

            EulerOrder::Zxy => Self {
                x: s1 * c2 * c3 - c1 * s2 * s3,
                y: c1 * s2 * c3 + s1 * c2 * s3,
                z: c1 * c2 * s3 + s1 * s2 * c3,
                w: c1 * c2 * c3 - s1 * s2 * s3,
            },

            EulerOrder::Zyx => Self {
                x: s1 * c2 * c3 - c1 * s2 * s3,
                y: c1 * s2 * c3 + s1 * c2 * s3,
                z: c1 * c2 * s3 - s1 * s2 * c3,
                w: c1 * c2 * c3 + s1 * s2 * s3,
            },

            EulerOrder::Yzx => Self {
                x: s1 * c2 * c3 + c1 * s2 * s3,
                y: c1 * s2 * c3 + s1 * c2 * s3,
                z: c1 * c2 * s3 - s1 * s2 * c3,
                w: c1 * c2 * c3 - s1 * s2 * s3,
            },

            EulerOrder::Xzy => Self {
                x: s1 * c2 * c3 - c1 * s2 * s3,
                y: c1 * s2 * c3 - s1 * c2 * s3,
                z: c1 * c2 * s3 + s1 * s2 * c3,
                w: c1 * c2 * c3 + s1 * s2 * s3,
            },
        }
    }
}

impl_op_ex!(*|a: &Quaternion, b: &Quaternion| -> Quaternion {
    Quaternion {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    }
});

impl_op_ex!(*= |a: &mut Quaternion, b: &Quaternion| {
    *a = *a * b;
});

impl Quaternion {
    /// Creates a new quaternion for the rotation by the given angle around the
    /// given axis. The axis must be normalized and the angle must be in
    /// radians.
    pub fn from_axis_angle(axis: &Vector3, angle: f32) -> Self {
        let s = (angle / 2.0).sin();

        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: (angle / 2.0).cos(),
        }
    }

    /// Sets the x, y, z, and w properties.
    pub fn set(&mut self, x: f32, y: f32, z: f32, w: f32) -> &Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;

        self
    }

    /// Returns the norm. The norm of a quaternion has no inherent geometric
    /// meaning, but all rotation quaternions must have a norm of `1.0`.
    pub fn norm(&self) -> f32 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalizes this quaternion.
    pub fn normalize(&mut self) -> &Self {
        let norm = self.norm();

        self.w /= norm;
        self.x /= norm;
        self.y /= norm;
        self.z /= norm;

        self
    }

    /// Returns the conjugate. The conjugate represents the same rotation in the
    /// opposite direction.
    pub fn conjugate(&self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Inverts this quaternion using the conjugate. If this is a rotation
    /// quaternion, this effectively reverses the rotation.
    pub fn invert(&mut self) -> &Self {
        *self = self.conjugate();

        self
    }
}

#[cfg(test)]
mod tests {
    use core::f32;

    use super::*;
    use assert_float_eq::assert_float_absolute_eq;

    #[test]
    fn test_from_axis_angle() {
        let zero = Quaternion::default();

        let a = Quaternion::from_axis_angle(&(1.0, 0.0, 0.0).into(), 0.0);
        assert_eq!(a, zero);
        let a = Quaternion::from_axis_angle(&(0.0, 1.0, 0.0).into(), 0.0);
        assert_eq!(a, zero);
        let a = Quaternion::from_axis_angle(&(0.0, 0.0, 1.0).into(), 0.0);
        assert_eq!(a, zero);

        let b1 = Quaternion::from_axis_angle(&(1.0, 0.0, 0.0).into(), f32::consts::PI);
        assert_ne!(a, b1);
        let b2 = Quaternion::from_axis_angle(&(1.0, 0.0, 0.0).into(), -f32::consts::PI);
        assert_ne!(a, b2);

        assert_eq!(a, b1 * b2);
    }

    #[test]
    fn test_set() {
        let mut a = Quaternion::default();

        assert_eq!(a.x, 0.0);
        assert_eq!(a.y, 0.0);
        assert_eq!(a.z, 0.0);
        assert_eq!(a.w, 1.0);

        a.set(1.0, 2.0, 3.0, 4.0);

        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(a.z, 3.0);
        assert_eq!(a.w, 4.0);
    }

    #[test]
    fn test_norm_and_normalize() {
        let mut a = Quaternion {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 4.0,
        };

        assert_ne!(a.norm(), 1.0);
        a.normalize();
        assert_float_absolute_eq!(a.norm(), 1.0);
    }

    #[test]
    fn test_conjugate() {
        let a = Quaternion {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 4.0,
        };

        let b = a.conjugate();

        assert_eq!(a.x, -b.x);
        assert_eq!(a.y, -b.y);
        assert_eq!(a.z, -b.z);
        assert_eq!(a.w, b.w);
    }

    #[test]
    fn test_invert() {
        let a = Quaternion {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 4.0,
        };

        let mut b = a.clone();
        b.invert();

        assert_eq!(a.x, -b.x);
        assert_eq!(a.y, -b.y);
        assert_eq!(a.z, -b.z);
        assert_eq!(a.w, b.w);
    }
}
