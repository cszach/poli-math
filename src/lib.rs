//! WebGPU-friendly 3D graphics math library.
//!
//! This library supports math operations for matrix transformations, rotations,
//! and colors. It supports all compilation targets, including WebAssembly.
//!
//! This library uses C-style layout for types. This allows quantites such as
//! matrices and vectors to be written to GPU buffers and passed to shaders
//! correctly.

#[macro_use]
extern crate impl_ops;

mod color;
mod euler;
mod matrix3;
mod matrix4;
mod quaternion;
mod vector3;

pub use color::*;
pub use euler::*;
pub use matrix3::*;
pub use matrix4::*;
pub use quaternion::*;
pub use vector3::*;
