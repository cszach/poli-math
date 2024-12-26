use std::ops;

use impl_ops::impl_op_ex;

use super::{Euler, Quaternion, Vector3};

/// 4x4 matrix, commonly used to encode transformations i.e. translation,
/// rotation, and scale.
///
/// ## Supported operators
///
/// - [`ops::Mul`], [`ops::MulAssign`]
///   - Matrix multiplication
///   - Element-wise multiplication by a scalar (commutative)
/// - [`ops::Div`], [`ops::DivAssign`]
///   - Element-wise division by a scalar (commutative)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Matrix4 {
    /// The elements of this matrix in column-major order.
    pub elements: [f32; 16],
}

unsafe impl Send for Matrix4 {}
unsafe impl Sync for Matrix4 {}

impl Default for Matrix4 {
    /// Returns the default 4x4 matrix, which is the 4x4 identity matrix.
    fn default() -> Self {
        Self::identity()
    }
}

impl AsRef<Matrix4> for Matrix4 {
    fn as_ref(&self) -> &Matrix4 {
        self
    }
}

impl_op_ex!(*|a: &Matrix4, b: &Matrix4| -> Matrix4 {
    let a11 = a.elements[0];
    let a21 = a.elements[1];
    let a31 = a.elements[2];
    let a41 = a.elements[3];
    let a12 = a.elements[4];
    let a22 = a.elements[5];
    let a32 = a.elements[6];
    let a42 = a.elements[7];
    let a13 = a.elements[8];
    let a23 = a.elements[9];
    let a33 = a.elements[10];
    let a43 = a.elements[11];
    let a14 = a.elements[12];
    let a24 = a.elements[13];
    let a34 = a.elements[14];
    let a44 = a.elements[15];

    let b11 = b.elements[0];
    let b21 = b.elements[1];
    let b31 = b.elements[2];
    let b41 = b.elements[3];
    let b12 = b.elements[4];
    let b22 = b.elements[5];
    let b32 = b.elements[6];
    let b42 = b.elements[7];
    let b13 = b.elements[8];
    let b23 = b.elements[9];
    let b33 = b.elements[10];
    let b43 = b.elements[11];
    let b14 = b.elements[12];
    let b24 = b.elements[13];
    let b34 = b.elements[14];
    let b44 = b.elements[15];

    Matrix4::new(
        a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41,
        a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42,
        a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43,
        a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44,
        a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41,
        a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42,
        a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43,
        a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44,
        a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41,
        a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42,
        a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43,
        a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44,
        a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41,
        a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42,
        a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43,
        a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44,
    )
});

impl_op_ex!(*= |a: &mut Matrix4, b: &Matrix4| {
    *a = *a * b;
});

impl_op_ex_commutative!(*|a: &Matrix4, b: &f32| -> Matrix4 {
    Matrix4 {
        elements: a.elements.map(|x| x * b),
    }
});

impl_op_ex!(*= |a: &mut Matrix4, b: &f32| {
    a.elements.iter_mut().for_each(|x| {
        *x *= b;
    });
});

impl_op_ex!(/|a: &Matrix4, b: &f32| -> Matrix4 {
    Matrix4 {
        elements: a.elements.map(|x| x / b),
    }
});

impl_op_ex!(/= |a: &mut Matrix4, b: &f32| {
    a.elements.iter_mut().for_each(|x| {
        *x /= b;
    });
});

impl Matrix4 {
    /// Creates a new 4x4 matrix with the given row-major elements. The elements
    /// will be stored internally in column-major order.
    #[rustfmt::skip]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n11: f32, n12: f32, n13: f32, n14: f32,
        n21: f32, n22: f32, n23: f32, n24: f32,
        n31: f32, n32: f32, n33: f32, n34: f32,
        n41: f32, n42: f32, n43: f32, n44: f32,
    ) -> Self {
        Self {
            elements: [
                n11, n21, n31, n41,
                n12, n22, n32, n42,
                n13, n23, n33, n43,
                n14, n24, n34, n44,
            ],
        }
    }

    /// Returns the 4x4 identity matrix.
    pub fn identity() -> Self {
        Self {
            #[rustfmt::skip]
            elements: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Returns the 4x4 zero matrix.
    pub fn zero() -> Self {
        Self {
            elements: [0.0; 16],
        }
    }

    /// Returns the translation matrix of the given displacement vector.
    #[rustfmt::skip]
    pub fn from_translation(v: &Vector3) -> Self {
        Self::new(
            1.0, 0.0, 0.0, v.x,
            0.0, 1.0, 0.0, v.y,
            0.0, 0.0, 1.0, v.z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Returns the rotation matrix around the X axis by the given angle in
    /// radians.
    #[rustfmt::skip]
    pub fn from_rotation_x(theta: f32) -> Self {
        let cos = theta.cos();
        let sin = theta.sin();

        Self::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos, -sin, 0.0,
            0.0, sin, cos, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Returns the rotation matrix around the Y axis by the given angle in
    /// radians.
    #[rustfmt::skip]
    pub fn from_rotation_y(theta: f32) -> Self {
        let cos = theta.cos();
        let sin = theta.sin();

        Self::new(
            cos, 0.0, sin, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin, 0.0, cos, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Returns the rotation matrix around the Z axis by the given angle in
    /// radians.
    #[rustfmt::skip]
    pub fn from_rotation_z(theta: f32) -> Self {
        let cos = theta.cos();
        let sin = theta.sin();

        Self::new(
            cos, -sin, 0.0, 0.0,
            sin, cos, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Returns the rotation matrix from the given Euler angles.
    ///
    /// The implementation is based on formulae on [this page][rotmat].
    ///
    /// [rotmat]: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    pub fn from_euler(euler: &Euler) -> Self {
        let mut m = Matrix4::identity();

        let a = euler.x.cos();
        let b = euler.x.sin();
        let c = euler.y.cos();
        let d = euler.y.sin();
        let e = euler.z.cos();
        let f = euler.z.sin();

        match euler.order {
            super::EulerOrder::Xyz => {
                let ae = a * e;
                let af = a * f;
                let be = b * e;
                let bf = b * f;

                m.elements[0] = c * e;
                m.elements[4] = -c * f;
                m.elements[8] = d;

                m.elements[1] = af + be * d;
                m.elements[5] = ae - bf * d;
                m.elements[9] = -b * c;

                m.elements[2] = bf - ae * d;
                m.elements[6] = be + af * d;
                m.elements[10] = a * c;
            }
            super::EulerOrder::Xzy => {
                let ac = a * c;
                let ad = a * d;
                let bc = b * c;
                let bd = b * d;

                m.elements[0] = c * e;
                m.elements[4] = -f;
                m.elements[8] = d * e;

                m.elements[1] = ac * f + bd;
                m.elements[5] = a * e;
                m.elements[9] = ad * f - bc;

                m.elements[2] = bc * f - ad;
                m.elements[6] = b * e;
                m.elements[10] = bd * f + ac;
            }
            super::EulerOrder::Yxz => {
                let ce = c * e;
                let cf = c * f;
                let de = d * e;
                let df = d * f;

                m.elements[0] = ce + df * b;
                m.elements[4] = de * b - cf;
                m.elements[8] = a * d;

                m.elements[1] = a * f;
                m.elements[5] = a * e;
                m.elements[9] = -b;

                m.elements[2] = cf * b - de;
                m.elements[6] = df + ce * b;
                m.elements[10] = a * c;
            }
            super::EulerOrder::Yzx => {
                let ac = a * c;
                let ad = a * d;
                let bc = b * c;
                let bd = b * d;

                m.elements[0] = c * e;
                m.elements[4] = bd - ac * f;
                m.elements[8] = bc * f + ad;

                m.elements[1] = f;
                m.elements[5] = a * e;
                m.elements[9] = -b * e;

                m.elements[2] = -d * e;
                m.elements[6] = ad * f + bc;
                m.elements[10] = ac - bd * f;
            }
            super::EulerOrder::Zxy => {
                let ce = c * e;
                let cf = c * f;
                let de = d * e;
                let df = d * f;

                m.elements[0] = ce - df * b;
                m.elements[4] = -a * f;
                m.elements[8] = de + cf * b;

                m.elements[1] = cf + de * b;
                m.elements[5] = a * e;
                m.elements[9] = df - ce * b;

                m.elements[2] = -a * d;
                m.elements[6] = b;
                m.elements[10] = a * c;
            }
            super::EulerOrder::Zyx => {
                let ae = a * e;
                let af = a * f;
                let be = b * e;
                let bf = b * f;

                m.elements[0] = c * e;
                m.elements[4] = be * d - af;
                m.elements[8] = ae * d + bf;

                m.elements[1] = c * f;
                m.elements[5] = bf * d + ae;
                m.elements[9] = af * d - be;

                m.elements[2] = -d;
                m.elements[6] = b * c;
                m.elements[10] = a * c;
            }
        }

        m
    }

    /// Returns the rotation matrix from the given rotation quaternion.
    ///
    /// The implementation is based on the formulae on [this page][rotmatquat].
    ///
    /// [rotmatquat]: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    pub fn from_quaternion(q: &Quaternion) -> Self {
        Self::compose(&Vector3::default(), q, &(1.0, 1.0, 1.0).into())
    }

    /// Returns the transformation matrix for the given scale transform.
    #[rustfmt::skip]
    pub fn from_scale(v: &Vector3) -> Self {
        Self::new(
            v.x, 0.0, 0.0, 0.0,
            0.0, v.y, 0.0, 0.0,
            0.0, 0.0, v.z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Creates a matrix for the transformation composed of the given
    /// translation, rotation, and scale. This uses TRS ordering: scale first,
    /// then rotation, then translation.
    ///
    /// The implementation is based on the formulae for
    /// [quaternion to 4x4 matrix conversion][rotmatquat], with the addition of
    /// scale and translation.
    ///
    /// [rotmatquat]: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    pub fn compose(translation: &Vector3, rotation: &Quaternion, scale: &Vector3) -> Self {
        let mut elements = [0.0f32; 16];

        let x = rotation.x;
        let y = rotation.y;
        let z = rotation.z;
        let w = rotation.w;

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;

        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        let sx = scale.x;
        let sy = scale.y;
        let sz = scale.z;

        elements[0] = (1.0 - (yy + zz)) * sx;
        elements[1] = (xy + wz) * sx;
        elements[2] = (xz - wy) * sx;
        elements[3] = 0.0;

        elements[4] = (xy - wz) * sy;
        elements[5] = (1.0 - (xx + zz)) * sy;
        elements[6] = (yz + wx) * sy;
        elements[7] = 0.0;

        elements[8] = (xz + wy) * sz;
        elements[9] = (yz - wx) * sz;
        elements[10] = (1.0 - (xx + yy)) * sz;
        elements[11] = 0.0;

        elements[12] = translation.x;
        elements[13] = translation.y;
        elements[14] = translation.z;
        elements[15] = 1.0;

        Self { elements }
    }

    /// Returns a rotation matrix looking from `eye` towards `target` oriented
    /// by the `up` vector.
    pub fn look_at(eye: &Vector3, target: &Vector3, up: &Vector3) -> Self {
        let z = (eye - target).normalized();
        let x = up.cross(&z).normalized();
        let y = z.cross(&x).normalized();

        Self {
            elements: [
                x.x,
                x.y,
                x.z,
                0.0,
                y.x,
                y.y,
                y.z,
                0.0,
                z.x,
                z.y,
                z.z,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
            ],
        }
    }

    /// Sets the elements of this matrix with the given row-major elements.
    #[rustfmt::skip]
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &mut self,
        n11: f32, n12: f32, n13: f32, n14: f32,
        n21: f32, n22: f32, n23: f32, n24: f32,
        n31: f32, n32: f32, n33: f32, n34: f32,
        n41: f32, n42: f32, n43: f32, n44: f32,
    ) {
        self.elements[0] = n11;
        self.elements[1] = n21;
        self.elements[2] = n31;
        self.elements[3] = n41;
        self.elements[4] = n12;
        self.elements[5] = n22;
        self.elements[6] = n32;
        self.elements[7] = n42;
        self.elements[8] = n13;
        self.elements[9] = n23;
        self.elements[10] = n33;
        self.elements[11] = n43;
        self.elements[12] = n14;
        self.elements[13] = n24;
        self.elements[14] = n34;
        self.elements[15] = n44;
    }

    /// Returns the translation component of this matrix.
    pub fn translation(&self) -> Vector3 {
        Vector3 {
            x: self.elements[12],
            y: self.elements[13],
            z: self.elements[14],
        }
    }

    /// Translates by the given vector.
    pub fn translate(&mut self, v: &Vector3) {
        *self *= Self::from_translation(v);
    }

    /// Applies the given rotation quaternion.
    ///
    /// If you have Euler angles, you can use [`Quaternion::from`].
    pub fn rotate(&mut self, q: &Quaternion) {
        *self *= &Self::from_quaternion(q);
    }

    /// Scales by the given scale factor.
    pub fn scale(&mut self, v: &Vector3) {
        *self *= &Self::from_scale(v);
    }

    /// Returns the determinant of this matrix.
    ///
    /// The algorithm can be found
    /// [here](http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm).
    pub fn determinant(&self) -> f32 {
        let n11 = self.elements[0];
        let n21 = self.elements[1];
        let n31 = self.elements[2];
        let n41 = self.elements[3];
        let n12 = self.elements[4];
        let n22 = self.elements[5];
        let n32 = self.elements[6];
        let n42 = self.elements[7];
        let n13 = self.elements[8];
        let n23 = self.elements[9];
        let n33 = self.elements[10];
        let n43 = self.elements[11];
        let n14 = self.elements[12];
        let n24 = self.elements[13];
        let n34 = self.elements[14];
        let n44 = self.elements[15];

        n14 * n23 * n32 * n41 - n13 * n24 * n32 * n41 - n14 * n22 * n33 * n41
            + n12 * n24 * n33 * n41
            + n13 * n22 * n34 * n41
            - n12 * n23 * n34 * n41
            - n14 * n23 * n31 * n42
            + n13 * n24 * n31 * n42
            + n14 * n21 * n33 * n42
            - n11 * n24 * n33 * n42
            - n13 * n21 * n34 * n42
            + n11 * n23 * n34 * n42
            + n14 * n22 * n31 * n43
            - n12 * n24 * n31 * n43
            - n14 * n21 * n32 * n43
            + n11 * n24 * n32 * n43
            + n12 * n21 * n34 * n43
            - n11 * n22 * n34 * n43
            - n13 * n22 * n31 * n44
            + n12 * n23 * n31 * n44
            + n13 * n21 * n32 * n44
            - n11 * n23 * n32 * n44
            - n12 * n21 * n33 * n44
            + n11 * n22 * n33 * n44
    }

    /// Returns the adjugate of this matrix.
    pub fn adjugate(&self) -> Self {
        let n11 = self.elements[0];
        let n21 = self.elements[1];
        let n31 = self.elements[2];
        let n41 = self.elements[3];
        let n12 = self.elements[4];
        let n22 = self.elements[5];
        let n32 = self.elements[6];
        let n42 = self.elements[7];
        let n13 = self.elements[8];
        let n23 = self.elements[9];
        let n33 = self.elements[10];
        let n43 = self.elements[11];
        let n14 = self.elements[12];
        let n24 = self.elements[13];
        let n34 = self.elements[14];
        let n44 = self.elements[15];

        Self::new(
            n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44
                + n22 * n33 * n44,
            n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44
                - n12 * n33 * n44,
            n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44
                + n12 * n23 * n44,
            n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34
                - n12 * n23 * n34,
            n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44
                - n21 * n33 * n44,
            n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44
                + n11 * n33 * n44,
            n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44
                - n11 * n23 * n44,
            n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34
                + n11 * n23 * n34,
            n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44
                + n21 * n32 * n44,
            n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44
                - n11 * n32 * n44,
            n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44
                + n11 * n22 * n44,
            n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34
                - n11 * n22 * n34,
            n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43
                - n21 * n32 * n43,
            n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43
                + n11 * n32 * n43,
            n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43
                - n11 * n22 * n43,
            n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33
                + n11 * n22 * n33,
        )
    }

    /// Returns the inverse of this matrix. If this matrix has no inverse i.e.
    /// the determinant is zero, then return the 4x4 zero matrix.
    ///
    /// The inverse is calculated in terms of its [adjugate](Self::adjugate).
    pub fn inverse(&self) -> Self {
        let det = self.determinant();

        if det == 0.0 {
            Self::zero()
        } else {
            self.adjugate() / det
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use assert_float_eq::assert_float_absolute_eq;

    use crate::EulerOrder;

    use super::*;

    /// Converts the given column-major index to its row-major equivalent.
    ///
    /// That is, returns the index that would return the same element if the
    /// elements of the matrix was stored in row-major instead of column-major.
    fn cm_to_rm(i: usize) -> usize {
        i % 4 * 4 + i / 4
    }

    /// Compares if two matrices element-by-element with a tolerance for
    /// floating-point precision error.
    fn matrix4_equals(a: Matrix4, b: Matrix4) {
        for i in 0..16 {
            assert_float_absolute_eq!(a.elements[i], b.elements[i]);
        }
    }

    #[test]
    fn test_new() {
        #[rustfmt::skip]
        let m = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        for i in 0..16 {
            assert_eq!(m.elements[i], (cm_to_rm(i) + 1) as f32);
        }
    }

    #[test]
    fn test_identity() {
        let m = Matrix4::identity();

        for i in 0..16 {
            assert_eq!(m.elements[i], if i % 5 == 0 { 1.0 } else { 0.0 });
        }
    }

    #[test]
    fn test_zero() {
        let m = Matrix4::zero();

        for i in 0..16 {
            assert_eq!(m.elements[i], 0.0);
        }
    }

    #[test]
    fn test_from_translation() {
        let m = Matrix4::from_translation(&Vector3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        });

        #[rustfmt::skip]
        let expected = Matrix4::new(
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0
        );

        matrix4_equals(m, expected);
    }

    #[test]
    fn test_from_rotation_x() {
        let m = Matrix4::from_rotation_x(PI / 6.0);

        let cos = 3.0f32.sqrt() / 2.0;

        #[rustfmt::skip]
        let expected = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos, -0.5, 0.0,
            0.0, 0.5, cos, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        matrix4_equals(m, expected);
    }

    #[test]
    fn test_from_rotation_y() {
        let m = Matrix4::from_rotation_y(PI / 6.0);

        let cos = 3.0f32.sqrt() / 2.0;

        #[rustfmt::skip]
        let expected = Matrix4::new(
            cos, 0.0, 0.5, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -0.5, 0.0, cos, 0.0,
            0.0, 0.0, 0.0, 1.0 
        );

        matrix4_equals(m, expected);
    }

    #[test]
    fn test_from_rotation_z() {
        let m = Matrix4::from_rotation_z(PI / 6.0);

        let cos = 3.0f32.sqrt() / 2.0;

        #[rustfmt::skip]
        let expected = Matrix4::new(
            cos, -0.5, 0.0, 0.0,
            0.5, cos, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        matrix4_equals(m, expected);
    }

    #[test]
    fn test_from_euler() {
        let test_values = [
            Euler {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                order: EulerOrder::Xyz,
            },
            Euler {
                x: 1.0,
                y: 0.0,
                z: 0.0,
                order: EulerOrder::Xyz,
            },
            Euler {
                x: 0.0,
                y: 1.0,
                z: 0.0,
                order: EulerOrder::Zyx,
            },
            Euler {
                x: 0.0,
                y: 0.0,
                z: 0.5,
                order: EulerOrder::Yzx,
            },
            Euler {
                x: 0.0,
                y: 0.0,
                z: -0.5,
                order: EulerOrder::Yzx,
            },
        ];

        for euler in test_values {
            let m = Matrix4::from_euler(&euler);
            let m2 = Matrix4::from_euler(&Euler::from_rotation_matrix(&m, euler.order));

            matrix4_equals(m, m2);
        }
    }

    #[test]
    fn test_from_scale() {
        let m = Matrix4::from_scale(&Vector3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        });

        #[rustfmt::skip]
        let expected = Matrix4::new(
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        matrix4_equals(m, expected);
    }

    #[test]
    fn test_look_at() {
        let m = Matrix4::look_at(
            &Vector3::default(),
            &(0.0, 1.0, -1.0).into(),
            &(0.0, 1.0, 0.0).into(),
        );

        let rotation_xyz = Euler::from_rotation_matrix(&m, EulerOrder::Xyz);

        assert_float_absolute_eq!(rotation_xyz.x * (180.0 / PI), 45.0);
    }

    #[test]
    fn test_set() {
        #[rustfmt::skip]
        let mut m = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        #[rustfmt::skip]
        m.set(
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0,
            29.0, 30.0, 31.0, 32.0,
        );

        for i in 0..16 {
            assert_eq!(m.elements[i], (cm_to_rm(i) + 17) as f32);
        }
    }

    #[test]
    fn test_translation() {
        #[rustfmt::skip]
        let m = Matrix4::new(
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 2.0,
            0.0, 0.0, 1.0, 3.0,
            0.0, 0.0, 0.0, 1.0,
        );

        assert_eq!(m.translation(), (1.0, 2.0, 3.0).into());
    }

    #[test]
    fn test_translate() {
        let mut m = Matrix4::identity();

        m.translate(&(1.0, 2.0, 3.0).into());

        assert_eq!(m.translation(), (1.0, 2.0, 3.0).into());
    }

    #[test]
    fn test_matrix_multiplication() {
        #[rustfmt::skip]
        let mut a = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        #[rustfmt::skip]
        let b = Matrix4::new(
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0,
            29.0, 30.0, 31.0, 32.0,
        );

        #[rustfmt::skip]
        let expected = Matrix4::new(
            250.0, 260.0, 270.0, 280.0,
            618.0, 644.0, 670.0, 696.0,
            986.0, 1028.0, 1070.0, 1112.0,
            1354.0, 1412.0, 1470.0, 1528.0
        );

        let a_mul_b = a * b;
        matrix4_equals(a_mul_b, expected);

        a *= b;
        matrix4_equals(a, expected);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        #[rustfmt::skip]
        let mut m = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        let m2 = m * 2.0;

        for i in 0..16 {
            assert_float_absolute_eq!(m2.elements[i], ((cm_to_rm(i) + 1) * 2) as f32);
        }

        m *= 2.0;

        for i in 0..16 {
            assert_float_absolute_eq!(m.elements[i], ((cm_to_rm(i) + 1) * 2) as f32);
        }
    }

    #[test]
    fn test_matrix_scalar_division() {
        #[rustfmt::skip]
        let mut m = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        let m2 = m / 2.0;

        for i in 0..16 {
            assert_float_absolute_eq!(m2.elements[i], (cm_to_rm(i) + 1) as f32 / 2.0);
        }

        m /= 2.0;

        for i in 0..16 {
            assert_float_absolute_eq!(m.elements[i], (cm_to_rm(i) + 1) as f32 / 2.0);
        }
    }

    #[test]
    fn test_determinant() {
        #[rustfmt::skip]
        let m = Matrix4::new(
            2.0, -3.0, 1.0, 5.0,
            4.0, 0.0, -2.0, 1.0,
            -1.0, 2.0, 3.0, 4.0,
            3.0, 1.0, 2.0, -2.0,
        );

        #[rustfmt::skip]
        let degenerate = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        assert_float_absolute_eq!(m.determinant(), -420.0);
        assert_eq!(degenerate.determinant(), 0.0);
    }

    #[test]
    fn test_inverse() {
        #[rustfmt::skip]
        let m = Matrix4::new(
            0.0, 0.0, -1.0, 2.0,
            0.0, 1.0, 0.0, 0.0,
            9.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        let i = m.inverse();

        assert_eq!(i.elements[0], 0.0);
        assert_eq!(i.elements[1], 0.0);
        assert_eq!(i.elements[2], -1.0);
        assert_eq!(i.elements[3], 0.0);
        assert_eq!(i.elements[4], 0.0);
        assert_eq!(i.elements[5], 1.0);
        assert_eq!(i.elements[6], 0.0);
        assert_eq!(i.elements[7], 0.0);
        assert_eq!(i.elements[8], 1.0 / 9.0);
        assert_eq!(i.elements[9], 0.0);
        assert_eq!(i.elements[10], 0.0);
        assert_eq!(i.elements[11], 0.0);
        assert_eq!(i.elements[12], 0.0);
        assert_eq!(i.elements[13], 0.0);
        assert_eq!(i.elements[14], 2.0);
        assert_eq!(i.elements[15], 1.0);

        #[rustfmt::skip]
        let degenerate = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        matrix4_equals(degenerate.inverse(), Matrix4::zero());
    }
}
