//! Fixed-point vector operations

use crate::error::{FixedPointError, Result};
use crate::fixed::{Fixed, DEFAULT_SCALE};

/// A vector of fixed-point values with common scale
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedVector {
    /// Raw i32 values
    pub data: Vec<i32>,
    /// Common scale factor for all elements
    pub scale: u8,
}

impl FixedVector {
    /// Create a new vector from raw data with the given scale
    pub fn from_raw(data: Vec<i32>, scale: u8) -> Self {
        Self { data, scale }
    }

    /// Create a vector from floating-point values
    pub fn from_f64_slice(values: &[f64], scale: u8) -> Result<Self> {
        let mut data = Vec::with_capacity(values.len());
        for &v in values {
            let fixed = Fixed::from_f64(v, scale)?;
            data.push(fixed.raw);
        }
        Ok(Self { data, scale })
    }

    /// Create a vector from floating-point values with default scale
    pub fn from_f64_slice_default(values: &[f64]) -> Result<Self> {
        Self::from_f64_slice(values, DEFAULT_SCALE)
    }

    /// Convert to floating-point vector
    pub fn to_f64_vec(&self) -> Vec<f64> {
        let scale_factor = (1u64 << self.scale) as f64;
        self.data.iter().map(|&x| x as f64 / scale_factor).collect()
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<Fixed> {
        self.data
            .get(index)
            .map(|&raw| Fixed { raw, scale: self.scale })
    }

    /// Create a zero vector of given length
    pub fn zeros(len: usize, scale: u8) -> Self {
        Self {
            data: vec![0; len],
            scale,
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.scale != other.scale {
            return Err(FixedPointError::ScaleMismatch {
                expected: self.scale,
                got: other.scale,
            });
        }
        if self.len() != other.len() {
            return Err(FixedPointError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        let data: Vec<i32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a.wrapping_add(b))
            .collect();

        Ok(Self {
            data,
            scale: self.scale,
        })
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Result<Self> {
        if self.scale != other.scale {
            return Err(FixedPointError::ScaleMismatch {
                expected: self.scale,
                got: other.scale,
            });
        }
        if self.len() != other.len() {
            return Err(FixedPointError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        let data: Vec<i32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a.wrapping_sub(b))
            .collect();

        Ok(Self {
            data,
            scale: self.scale,
        })
    }

    /// Dot product with another vector
    pub fn dot(&self, other: &Self) -> Result<Fixed> {
        if self.scale != other.scale {
            return Err(FixedPointError::ScaleMismatch {
                expected: self.scale,
                got: other.scale,
            });
        }
        if self.len() != other.len() {
            return Err(FixedPointError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        // Accumulate in i64 to avoid overflow
        let sum: i64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| (a as i64) * (b as i64))
            .sum();

        // Rescale
        let rescaled = sum >> self.scale;

        Ok(Fixed {
            raw: rescaled as i32,
            scale: self.scale,
        })
    }

    /// Multiply each element by a scalar
    pub fn scale_by(&self, scalar: Fixed) -> Result<Self> {
        if self.scale != scalar.scale {
            return Err(FixedPointError::ScaleMismatch {
                expected: self.scale,
                got: scalar.scale,
            });
        }

        let data: Vec<i32> = self
            .data
            .iter()
            .map(|&x| {
                let product = (x as i64) * (scalar.raw as i64);
                (product >> self.scale) as i32
            })
            .collect();

        Ok(Self {
            data,
            scale: self.scale,
        })
    }

    /// Negate all elements
    pub fn neg(&self) -> Self {
        Self {
            data: self.data.iter().map(|&x| x.wrapping_neg()).collect(),
            scale: self.scale,
        }
    }

    /// Encode to bytes (little-endian i32 values)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Decode from bytes (little-endian i32 values)
    pub fn from_bytes(bytes: &[u8], scale: u8) -> Result<Self> {
        if bytes.len() % 4 != 0 {
            return Err(FixedPointError::DimensionMismatch {
                expected: (bytes.len() / 4) * 4,
                got: bytes.len(),
            });
        }

        let data: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Self { data, scale })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_roundtrip() {
        let values = vec![1.0, 2.0, 3.0, -1.0, 0.5];
        let vec = FixedVector::from_f64_slice_default(&values).unwrap();
        let back = vec.to_f64_vec();

        for (expected, got) in values.iter().zip(&back) {
            assert!((expected - got).abs() < 0.0001);
        }
    }

    #[test]
    fn test_vector_add() {
        let a = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let b = FixedVector::from_f64_slice_default(&[4.0, 5.0, 6.0]).unwrap();
        let sum = a.add(&b).unwrap();
        let result = sum.to_f64_vec();

        assert!((result[0] - 5.0).abs() < 0.0001);
        assert!((result[1] - 7.0).abs() < 0.0001);
        assert!((result[2] - 9.0).abs() < 0.0001);
    }

    #[test]
    fn test_vector_dot() {
        let a = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let b = FixedVector::from_f64_slice_default(&[4.0, 5.0, 6.0]).unwrap();
        let dot = a.dot(&b).unwrap();
        let result = dot.to_f64();

        // 1*4 + 2*5 + 3*6 = 32
        assert!((result - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_bytes_roundtrip() {
        let vec = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let bytes = vec.to_bytes();
        let decoded = FixedVector::from_bytes(&bytes, DEFAULT_SCALE).unwrap();
        assert_eq!(vec, decoded);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = FixedVector::from_f64_slice_default(&[1.0, 2.0]).unwrap();
        let b = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        assert!(matches!(
            a.add(&b),
            Err(FixedPointError::DimensionMismatch { .. })
        ));
    }
}
