use std::ops;

use crate::Vector3;

use super::Matrix4;

/// 3x3 matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Matrix3 {
    /// The elements in column-major order.
    pub elements: [f32; 9],
}

unsafe impl Send for Matrix3 {}
unsafe impl Sync for Matrix3 {}

impl Default for Matrix3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Eq for Matrix3 {}

impl_op_ex!(*|a: &Matrix3, b: &Vector3| -> Vector3 {
    let a11 = a.elements[0];
    let a21 = a.elements[1];
    let a31 = a.elements[2];
    let a12 = a.elements[3];
    let a22 = a.elements[4];
    let a32 = a.elements[5];
    let a13 = a.elements[6];
    let a23 = a.elements[7];
    let a33 = a.elements[8];

    Vector3 {
        x: a11 * b.x + a12 * b.y + a13 * b.z,
        y: a21 * b.x + a22 * b.y + a23 * b.z,
        z: a31 * b.x + a32 * b.y + a33 * b.z,
    }
});

impl_op_ex_commutative!(/|a: &Matrix3, b: &f32| -> Matrix3 {
    Matrix3 {
        elements: a.elements.map(|x| x / b),
    }
});

impl Matrix3 {
    /// Creates a new 3x3 matrix with the given row-major elements. The elements
    /// will be stored internally in column-major order.
    #[rustfmt::skip]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n11: f32, n12: f32, n13: f32,
        n21: f32, n22: f32, n23: f32,
        n31: f32, n32: f32, n33: f32,
    ) -> Self {
        Self {
            elements: [
                n11, n21, n31,
                n12, n22, n32,
                n13, n23, n33
            ],
        }
    }

    /// Returns the 3x3 identity matrix.
    pub fn identity() -> Self {
        Matrix3 {
            #[rustfmt::skip]
            elements: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ],
        }
    }

    /// Returns the 3x3 zero matrix.
    pub fn zero() -> Self {
        Matrix3 { elements: [0.0; 9] }
    }

    /// Returns the top-left 3x3 matrix of the given 4x4 matrix.
    pub fn from_matrix4(m4: &Matrix4) -> Self {
        Self {
            elements: [
                m4.elements[0],
                m4.elements[1],
                m4.elements[2],
                m4.elements[4],
                m4.elements[5],
                m4.elements[6],
                m4.elements[8],
                m4.elements[9],
                m4.elements[10],
            ],
        }
    }

    /// Sets the elements of this matrix with the given row-major elements.
    #[rustfmt::skip]
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &mut self,
        n11: f32, n12: f32, n13: f32,
        n21: f32, n22: f32, n23: f32,
        n31: f32, n32: f32, n33: f32,
    ) {
        self.elements[0] = n11;
        self.elements[1] = n21;
        self.elements[2] = n31;
        self.elements[3] = n12;
        self.elements[4] = n22;
        self.elements[5] = n32;
        self.elements[6] = n13;
        self.elements[7] = n23;
        self.elements[8] = n33;
    }

    /// Returns the normal matrix for the given transformation matrix, which is
    /// multiplied with normal vectors to correct for deforms such as scaling
    /// and skewing.
    ///
    /// The normal matrix is calculated as the adjoint of the transformation
    /// matrix, not the inverse transpose. See
    /// <https://github.com/graphitemaster/normals_revisited>.
    pub fn normal_matrix(m4: &Matrix4) -> Self {
        Matrix3::from_matrix4(m4).adjugate()
    }

    /// Returns the determinant of this matrix.
    pub fn determinant(&self) -> f32 {
        let n11 = self.elements[0];
        let n21 = self.elements[1];
        let n31 = self.elements[2];
        let n12 = self.elements[3];
        let n22 = self.elements[4];
        let n32 = self.elements[5];
        let n13 = self.elements[6];
        let n23 = self.elements[7];
        let n33 = self.elements[8];

        n11 * n22 * n33 + n12 * n23 * n31 + n13 * n21 * n32
            - n11 * n23 * n32
            - n12 * n21 * n33
            - n13 * n22 * n31
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Self {
        Self {
            elements: [
                self.elements[0],
                self.elements[3],
                self.elements[6],
                self.elements[1],
                self.elements[4],
                self.elements[7],
                self.elements[2],
                self.elements[5],
                self.elements[8],
            ],
        }
    }

    /// Returns the adjugate of this matrix, also known as the classical adjoint
    /// or the adjunct.
    pub fn adjugate(&self) -> Self {
        let n11 = self.elements[0];
        let n21 = self.elements[1];
        let n31 = self.elements[2];
        let n12 = self.elements[3];
        let n22 = self.elements[4];
        let n32 = self.elements[5];
        let n13 = self.elements[6];
        let n23 = self.elements[7];
        let n33 = self.elements[8];

        Self {
            elements: [
                n22 * n33 - n23 * n32,
                n23 * n31 - n21 * n33,
                n21 * n32 - n22 * n31,
                n13 * n32 - n12 * n33,
                n11 * n33 - n13 * n31,
                n12 * n31 - n11 * n32,
                n12 * n23 - n13 * n22,
                n13 * n21 - n11 * n23,
                n11 * n22 - n12 * n21,
            ],
        }
    }

    /// Returns the inverse of this matrix. If this matrix has no inverse i.e.
    /// the determinant is zero, then return the 3x3 zero matrix.
    ///
    /// The inverse is calculated in terms of its [adjugate](Self::adjugate).
    pub fn inverse(&self) -> Self {
        let det = self.determinant();

        if det != 0.0 {
            self.adjugate() / det
        } else {
            Self::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_float_eq::assert_float_absolute_eq;

    use super::*;

    /// Converts the given column-major index to its row-major equivalent.
    ///
    /// That is, returns the index that would return the same element if the
    /// elements of the matrix was stored in row-major instead of column-major.
    fn cm_to_rm(i: usize) -> usize {
        i % 3 * 3 + i / 3
    }

    #[test]
    fn test_default() {
        assert_eq!(Matrix3::default(), Matrix3::identity());
    }

    #[test]
    fn test_new() {
        #[rustfmt::skip]
        let m = Matrix3::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        );

        for i in 0..9 {
            assert_eq!(m.elements[i], (cm_to_rm(i) + 1) as f32)
        }
    }

    #[test]
    fn test_identity() {
        let m = Matrix3::identity();

        for i in 0..9 {
            assert_eq!(m.elements[i], if i % 4 == 0 { 1.0 } else { 0.0 });
        }
    }

    #[test]
    fn test_zero() {
        let m = Matrix3::zero();

        for i in 0..9 {
            assert_eq!(m.elements[i], 0.0);
        }
    }

    #[test]
    fn test_from_matrix4() {
        #[rustfmt::skip]
        let m4 = Matrix4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        let m3 = Matrix3::from_matrix4(&m4);

        assert_eq!(
            m3,
            Matrix3 {
                elements: [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
            }
        );
    }

    #[test]
    fn test_set() {
        #[rustfmt::skip]
        let mut m = Matrix3::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        );

        #[rustfmt::skip]
        m.set(
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0,
        );

        for i in 0..9 {
            assert_eq!(m.elements[i], (cm_to_rm(i) + 10) as f32);
        }
    }

    #[test]
    fn test_normal() {
        #[rustfmt::skip]
        let m = Matrix4::new(
            1.0, 2.0, 3.0, 3.0,
            0.0, 1.0, 4.0, 4.0,
            5.0, 6.0, 0.0, 5.0,
            6.0, 7.0, 8.0, 9.0
        );

        let actual = Matrix3::normal_matrix(&m);

        #[rustfmt::skip]
        let expected = Matrix3::new(
            -24.0, 18.0, 5.0,
            20.0, -15.0, -4.0,
            -5.0, 4.0, 1.0
        );

        for i in 0..9 {
            assert_float_absolute_eq!(actual.elements[i], expected.elements[i]);
        }
    }

    #[test]
    fn test_determinant() {
        let mut m = Matrix3::identity();
        assert_eq!(m.determinant(), 1.0);

        m.elements[0] = 2.0;
        assert_eq!(m.determinant(), 2.0);

        m.elements[0] = 0.0;
        assert_eq!(m.determinant(), 0.0);

        #[rustfmt::skip]
        m.set(
            2.0, 3.0, 4.0,
            5.0, 13.0, 7.0,
            8.0, 9.0, 11.0
        );
        assert_eq!(m.determinant(), -73.0);
    }

    #[test]
    fn test_transpose() {
        #[rustfmt::skip]
        let m = Matrix3::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        );

        assert_eq!(
            m.transpose(),
            Matrix3 {
                elements: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }
        );
    }

    #[test]
    fn test_adjugate() {
        #[rustfmt::skip]
        let m = Matrix3::new(
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0
        );

        let actual = m.adjugate();

        #[rustfmt::skip]
        let expected = Matrix3::new(
            -24.0, 18.0, 5.0,
            20.0, -15.0, -4.0,
            -5.0, 4.0, 1.0
        );

        assert_float_absolute_eq!(actual.elements[0], expected.elements[0]);
        assert_float_absolute_eq!(actual.elements[1], expected.elements[1]);
        assert_float_absolute_eq!(actual.elements[2], expected.elements[2]);
        assert_float_absolute_eq!(actual.elements[3], expected.elements[3]);
        assert_float_absolute_eq!(actual.elements[4], expected.elements[4]);
        assert_float_absolute_eq!(actual.elements[5], expected.elements[5]);
        assert_float_absolute_eq!(actual.elements[6], expected.elements[6]);
        assert_float_absolute_eq!(actual.elements[7], expected.elements[7]);
        assert_float_absolute_eq!(actual.elements[8], expected.elements[8]);
    }

    #[test]
    fn test_inverse() {
        #[rustfmt::skip]
        let m = Matrix3::new(
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0
        );

        #[rustfmt::skip]
        let expected = Matrix3::new(
            -24.0, 18.0, 5.0,
            20.0, -15.0, -4.0,
            -5.0, 4.0, 1.0,
        );

        assert_eq!(m.inverse(), expected);

        #[rustfmt::skip]
        let degenerate = Matrix3::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        );

        assert_eq!(degenerate.inverse(), Matrix3::zero());
    }
}
