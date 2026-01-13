//! OT Extension traits and implementations
//!
//! This module defines the interface for OT extensions that can be plugged in.
//! The actual cryptographic implementation is abstracted behind these traits.

use crate::error::Result;

/// Trait for OT extension implementations
pub trait OtExtension: Send + Sync {
    /// Generate base OT setup message (sender side)
    fn generate_base_ot_sender(&mut self) -> Result<Vec<u8>>;

    /// Process base OT setup message and generate response (receiver side)
    fn process_base_ot_receiver(&mut self, msg: &[u8]) -> Result<Vec<u8>>;

    /// Process base OT response (sender side)
    fn process_base_ot_sender(&mut self, msg: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Check if base OT is complete
    fn is_base_ot_complete(&self) -> bool;

    /// Generate OT query for selecting items (receiver/client side)
    /// `indices` are the selected indices (0..V-1)
    /// Returns opaque query blob
    fn generate_query(&mut self, indices: &[u32], session_id: &[u8], ctr: u64) -> Result<Vec<u8>>;

    /// Process OT query and generate response (sender/server side)
    /// `query` is the opaque query blob
    /// `database` is the embedding table (V rows × row_bytes each)
    /// Returns opaque response blob
    fn process_query(
        &mut self,
        query: &[u8],
        database: &[u8],
        row_bytes: usize,
        session_id: &[u8],
        ctr: u64,
    ) -> Result<Vec<u8>>;

    /// Decode OT response to get selected items (receiver/client side)
    /// Returns the selected rows
    fn decode_response(
        &mut self,
        response: &[u8],
        num_items: usize,
        row_bytes: usize,
    ) -> Result<Vec<u8>>;
}

/// A simple OT extension implementation for testing/development.
/// WARNING: This is NOT cryptographically secure - it's a placeholder
/// that demonstrates the interface. A real implementation would use
/// IKNP or KOS OT extension.
pub struct SimpleOtExtension {
    /// Whether base OT is complete
    base_ot_complete: bool,
    /// Stored indices for decoding (in real OT, this would be hidden)
    stored_indices: Vec<u32>,
}

impl SimpleOtExtension {
    pub fn new() -> Self {
        Self {
            base_ot_complete: false,
            stored_indices: vec![],
        }
    }
}

impl Default for SimpleOtExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl OtExtension for SimpleOtExtension {
    fn generate_base_ot_sender(&mut self) -> Result<Vec<u8>> {
        // In real OT: generate Diffie-Hellman or similar parameters
        Ok(vec![0x01, 0x02, 0x03, 0x04]) // Placeholder
    }

    fn process_base_ot_receiver(&mut self, _msg: &[u8]) -> Result<Vec<u8>> {
        // In real OT: process sender's parameters and generate response
        Ok(vec![0x05, 0x06, 0x07, 0x08]) // Placeholder
    }

    fn process_base_ot_sender(&mut self, _msg: &[u8]) -> Result<Option<Vec<u8>>> {
        // In real OT: complete the base OT setup
        self.base_ot_complete = true;
        Ok(None) // No more messages needed
    }

    fn is_base_ot_complete(&self) -> bool {
        self.base_ot_complete
    }

    fn generate_query(&mut self, indices: &[u32], _session_id: &[u8], _ctr: u64) -> Result<Vec<u8>> {
        // Store indices for later decoding
        self.stored_indices = indices.to_vec();

        // In real OT: generate masked selection bits using OT extension
        // For now, just encode the indices (NOT private - this is a placeholder!)
        let mut query = Vec::with_capacity(indices.len() * 4);
        for &idx in indices {
            query.extend_from_slice(&idx.to_le_bytes());
        }
        Ok(query)
    }

    fn process_query(
        &mut self,
        query: &[u8],
        database: &[u8],
        row_bytes: usize,
        _session_id: &[u8],
        _ctr: u64,
    ) -> Result<Vec<u8>> {
        // Decode indices from query (in real OT, server wouldn't see these)
        let indices: Vec<u32> = query
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // In real OT: use OT extension to generate encrypted responses
        // For now, just extract the requested rows
        let mut response = Vec::with_capacity(indices.len() * row_bytes);
        for idx in indices {
            let start = idx as usize * row_bytes;
            let end = start + row_bytes;
            if end <= database.len() {
                response.extend_from_slice(&database[start..end]);
            } else {
                // Pad with zeros if index is out of range
                response.extend(std::iter::repeat(0u8).take(row_bytes));
            }
        }
        Ok(response)
    }

    fn decode_response(
        &mut self,
        response: &[u8],
        num_items: usize,
        row_bytes: usize,
    ) -> Result<Vec<u8>> {
        // In real OT: decrypt the response using receiver's keys
        // For now, response is already plaintext
        let expected_len = num_items * row_bytes;
        if response.len() < expected_len {
            return Err(crate::error::OtError::InvalidMessageFormat);
        }
        Ok(response[..expected_len].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_ot_extension() {
        let mut sender = SimpleOtExtension::new();
        let mut receiver = SimpleOtExtension::new();

        // Base OT setup
        let msg1 = sender.generate_base_ot_sender().unwrap();
        let msg2 = receiver.process_base_ot_receiver(&msg1).unwrap();
        sender.process_base_ot_sender(&msg2).unwrap();

        assert!(sender.is_base_ot_complete());

        // Create a small database (4 rows of 8 bytes each)
        let database: Vec<u8> = (0..32).collect();
        let row_bytes = 8;

        // Client wants rows 1 and 3
        let indices = vec![1, 3];
        let session_id = [0u8; 16];
        let ctr = 1;

        let query = receiver
            .generate_query(&indices, &session_id, ctr)
            .unwrap();
        let response = sender
            .process_query(&query, &database, row_bytes, &session_id, ctr)
            .unwrap();
        let result = receiver
            .decode_response(&response, indices.len(), row_bytes)
            .unwrap();

        // Verify we got the right rows
        assert_eq!(result.len(), 16); // 2 rows * 8 bytes
        assert_eq!(&result[0..8], &database[8..16]); // Row 1
        assert_eq!(&result[8..16], &database[24..32]); // Row 3
    }
}
