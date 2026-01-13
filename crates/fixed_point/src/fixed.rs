//! Fixed-point scalar type

use crate::error::{FixedPointError, Result};

/// Default scale factor exponent (S=16 means 2^16 = 65536)
pub const DEFAULT_SCALE: u8 = 16;

/// Maximum scale (to prevent overflow in intermediate computations)
pub const MAX_SCALE: u8 = 30;

/// A fixed-point number represented as i32 with implicit scaling factor 2^S
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fixed {
    /// The raw integer value
    pub raw: i32,
    /// Scale factor exponent (value represents raw / 2^scale)
    pub scale: u8,
}

impl Fixed {
    /// Create a new fixed-point value from raw integer and scale
    pub fn from_raw(raw: i32, scale: u8) -> Result<Self> {
        if scale > MAX_SCALE {
            return Err(FixedPointError::InvalidScale(scale));
        }
        Ok(Self { raw, scale })
    }

    /// Create a fixed-point value from a floating-point number
    pub fn from_f64(value: f64, scale: u8) -> Result<Self> {
        if scale > MAX_SCALE {
            return Err(FixedPointError::InvalidScale(scale));
        }

        let scale_factor = (1u64 << scale) as f64;
        let scaled = value * scale_factor;

        if scaled > i32::MAX as f64 {
            return Err(FixedPointError::Overflow { value });
        }
        if scaled < i32::MIN as f64 {
            return Err(FixedPointError::Underflow { value });
        }

        Ok(Self {
            raw: scaled.round() as i32,
            scale,
        })
    }

    /// Create a fixed-point value from a floating-point number using default scale
    pub fn from_f64_default(value: f64) -> Result<Self> {
        Self::from_f64(value, DEFAULT_SCALE)
    }

    /// Convert back to floating-point
    pub fn to_f64(self) -> f64 {
        let scale_factor = (1u64 << self.scale) as f64;
        self.raw as f64 / scale_factor
    }

    /// Create a zero value with the given scale
    pub fn zero(scale: u8) -> Result<Self> {
        Self::from_raw(0, scale)
    }

    /// Create a one value with the given scale
    pub fn one(scale: u8) -> Result<Self> {
        if scale > MAX_SCALE {
            return Err(FixedPointError::InvalidScale(scale));
        }
        // 1.0 = 2^scale in raw form
        let raw = 1i32 << scale;
        Ok(Self { raw, scale })
    }

    /// Add two fixed-point values (must have same scale)
    pub fn add(self, other: Self) -> Result<Self> {
        if self.scale != other.scale {
            return Err(FixedPointError::ScaleMismatch {
                expected: self.scale,
                got: other.scale,
            });
        }
        // Note: wrapping add for modular arithmetic in secret sharing
        Ok(Self {
            raw: self.raw.wrapping_add(other.raw),
            scale: self.scale,
        })
    }

    /// Subtract two fixed-point values (must have same scale)
    pub fn sub(self, other: Self) -> Result<Self> {
        if self.scale != other.scale {
            return Err(FixedPointError::ScaleMismatch {
                expected: self.scale,
                got: other.scale,
            });
        }
        // Note: wrapping sub for modular arithmetic in secret sharing
        Ok(Self {
            raw: self.raw.wrapping_sub(other.raw),
            scale: self.scale,
        })
    }

    /// Multiply two fixed-point values
    /// Result needs to be rescaled (shifted right by scale bits)
    pub fn mul(self, other: Self) -> Result<Self> {
        if self.scale != other.scale {
            return Err(FixedPointError::ScaleMismatch {
                expected: self.scale,
                got: other.scale,
            });
        }
        // Use i64 intermediate to avoid overflow
        let product = (self.raw as i64) * (other.raw as i64);
        // Rescale back: divide by 2^scale
        let rescaled = product >> self.scale;

        // Check for overflow when converting back to i32
        if rescaled > i32::MAX as i64 || rescaled < i32::MIN as i64 {
            return Err(FixedPointError::Overflow {
                value: rescaled as f64,
            });
        }

        Ok(Self {
            raw: rescaled as i32,
            scale: self.scale,
        })
    }

    /// Negate the value
    pub fn neg(self) -> Self {
        Self {
            raw: self.raw.wrapping_neg(),
            scale: self.scale,
        }
    }

    /// Get the absolute value
    pub fn abs(self) -> Self {
        Self {
            raw: self.raw.abs(),
            scale: self.scale,
        }
    }

    /// Encode to little-endian bytes
    pub fn to_le_bytes(self) -> [u8; 4] {
        self.raw.to_le_bytes()
    }

    /// Decode from little-endian bytes
    pub fn from_le_bytes(bytes: [u8; 4], scale: u8) -> Result<Self> {
        Self::from_raw(i32::from_le_bytes(bytes), scale)
    }
}

impl Default for Fixed {
    fn default() -> Self {
        Self {
            raw: 0,
            scale: DEFAULT_SCALE,
        }
    }
}

impl std::fmt::Display for Fixed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.6}", self.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_f64() {
        let values = [0.0, 1.0, -1.0, 0.5, -0.5, 0.123456, -0.123456, 100.0, -100.0];

        for &v in &values {
            let fixed = Fixed::from_f64_default(v).unwrap();
            let back = fixed.to_f64();
            let error = (v - back).abs();
            // Error should be less than 1/2^16 ≈ 0.000015
            assert!(
                error < 0.0001,
                "Roundtrip error too large for {}: got {}, error {}",
                v,
                back,
                error
            );
        }
    }

    #[test]
    fn test_add() {
        let a = Fixed::from_f64_default(1.5).unwrap();
        let b = Fixed::from_f64_default(2.5).unwrap();
        let sum = a.add(b).unwrap();
        let result = sum.to_f64();
        assert!((result - 4.0).abs() < 0.0001);
    }

    #[test]
    fn test_sub() {
        let a = Fixed::from_f64_default(5.0).unwrap();
        let b = Fixed::from_f64_default(2.0).unwrap();
        let diff = a.sub(b).unwrap();
        let result = diff.to_f64();
        assert!((result - 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_mul() {
        let a = Fixed::from_f64_default(2.0).unwrap();
        let b = Fixed::from_f64_default(3.0).unwrap();
        let prod = a.mul(b).unwrap();
        let result = prod.to_f64();
        assert!((result - 6.0).abs() < 0.0001);
    }

    #[test]
    fn test_scale_mismatch() {
        let a = Fixed::from_f64(1.0, 16).unwrap();
        let b = Fixed::from_f64(1.0, 12).unwrap();
        assert!(matches!(
            a.add(b),
            Err(FixedPointError::ScaleMismatch { .. })
        ));
    }

    #[test]
    fn test_overflow() {
        // Try to encode a value too large for i32
        let result = Fixed::from_f64(100000.0, DEFAULT_SCALE);
        assert!(matches!(result, Err(FixedPointError::Overflow { .. })));
    }

    #[test]
    fn test_bytes_roundtrip() {
        let fixed = Fixed::from_f64_default(3.14159).unwrap();
        let bytes = fixed.to_le_bytes();
        let decoded = Fixed::from_le_bytes(bytes, DEFAULT_SCALE).unwrap();
        assert_eq!(fixed, decoded);
    }

    #[test]
    fn test_one() {
        let one = Fixed::one(DEFAULT_SCALE).unwrap();
        assert!((one.to_f64() - 1.0).abs() < 0.0001);
    }
}
