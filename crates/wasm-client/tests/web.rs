//! WASM tests for the ShardLM client

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_create_iknp_client() {
    use shardlm_ot::IknpOtExtension;

    // This should work - just creating the extension
    let _ext = IknpOtExtension::new_client();
}

#[wasm_bindgen_test]
fn test_generate_base_ot_init() {
    use shardlm_ot::{IknpOtExtension, OtExtension};

    let mut ext = IknpOtExtension::new_client();

    // This is where the panic likely happens
    let result = ext.generate_base_ot_sender();
    assert!(result.is_ok(), "generate_base_ot_sender failed: {:?}", result.err());
}
