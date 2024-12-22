use super::{Matrix4, Quaternion};

/// Order of Euler rotations.
///
/// For example, the XYZ order ([`Self::Xyz`]) means the rotation around the
/// local X axis is applied first, then Y, then Z.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EulerOrder {
    Xyz,
    Xzy,
    Yxz,
    Yzx,
    Zxy,
    Zyx,
}

unsafe impl Send for EulerOrder {}
unsafe impl Sync for EulerOrder {}

impl Default for EulerOrder {
    /// Returns the default order of Euler angles, which is XYZ.
    fn default() -> Self {
        Self::Xyz
    }
}

/// Euler angles, which describes rotations as chained rotations around the
/// local XYZ axes.
///
/// Euler angles describe a rotation in 3D space by the amount of rotation
/// by each local axis and an axis order. They are intuitive to use but suffer
/// from the [gimbal lock problem](https://en.wikipedia.org/wiki/Gimbal_lock).
///
/// For a better representation of rotations, use [`Quaternion`], which
/// represents a rotation around an arbitrary axis.
#[derive(Default)]
pub struct Euler {
    /// Angle of the X axis in radians.
    pub x: f32,
    /// Angle of the Y axis in radians.
    pub y: f32,
    /// Angle of the Z axis in radians.
    pub z: f32,
    /// Order that the rotations are applied.
    pub order: EulerOrder,
}

impl Euler {
    /// Creates Euler angles from the given rotation matrix and axis order.
    pub fn from_rotation_matrix(m: &Matrix4, order: EulerOrder) -> Self {
        // Extract the top-left 3x3 matrix.

        let m11 = m.elements[0];
        let m12 = m.elements[4];
        let m13 = m.elements[8];
        let m21 = m.elements[1];
        let m22 = m.elements[5];
        let m23 = m.elements[9];
        let m31 = m.elements[2];
        let m32 = m.elements[6];
        let m33 = m.elements[10];

        match order {
            EulerOrder::Xyz => {
                let y = m13.clamp(-1.0, 1.0).asin();
                let (x, z) = if m13.abs() < 0.9999999 {
                    ((-m23).atan2(m33), (-m12).atan2(m11))
                } else {
                    (m32.atan2(m22), 0.0)
                };

                Self { x, y, z, order }
            }
            EulerOrder::Xzy => {
                let z = (-(m12.clamp(-1.0, 1.0))).asin();

                let (x, y) = if m12.abs() < 0.9999999 {
                    (m32.atan2(m22), m13.atan2(m11))
                } else {
                    ((-m23).atan2(m33), 0.0)
                };

                Self { x, y, z, order }
            }
            EulerOrder::Yxz => {
                let x = (-(m23.clamp(-1.0, 1.0))).asin();

                let (y, z) = if m23.abs() < 0.9999999 {
                    (m13.atan2(m33), m21.atan2(m22))
                } else {
                    ((-m31).atan2(m11), 0.0)
                };

                Self { x, y, z, order }
            }
            EulerOrder::Yzx => {
                let z = m21.clamp(-1.0, 1.0).asin();

                let (x, y) = if m21.abs() < 0.9999999 {
                    ((-m23).atan2(m22), (-m31).atan2(m11))
                } else {
                    (0.0, m13.atan2(m33))
                };

                Self { x, y, z, order }
            }
            EulerOrder::Zxy => {
                let x = m32.clamp(-1.0, 1.0).asin();

                let (y, z) = if m32.abs() < 0.9999999 {
                    ((-m31).atan2(m33), (-m12).atan2(m22))
                } else {
                    (0.0, m21.atan2(m11))
                };

                Self { x, y, z, order }
            }
            EulerOrder::Zyx => {
                let y = (-(m31.clamp(-1.0, 1.0))).asin();

                let (x, z) = if m31.abs() < 0.9999999 {
                    (m32.atan2(m33), m21.atan2(m11))
                } else {
                    (0.0, (-m12).atan2(m22))
                };

                Self { x, y, z, order }
            }
        }
    }

    /// Creates Euler angles from the given rotation quaternion and axis order.
    pub fn from_quaternion(q: &Quaternion, order: EulerOrder) -> Self {
        Self::from_rotation_matrix(&Matrix4::from_quaternion(q), order)
    }

    /// Sets the X, Y, and Z angles, and optionally the order.
    pub fn set(&mut self, x: f32, y: f32, z: f32, order: Option<EulerOrder>) {
        self.x = x;
        self.y = y;
        self.z = z;

        if let Some(order) = order {
            self.order = order;
        };
    }
}
