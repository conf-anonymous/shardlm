//! CUDA availability check example
//!
//! Run with: cargo run -p shardlm-v2-core --features cuda --example cuda_check

fn main() {
    println!("ShardLM v2 CUDA Check");
    println!("=====================\n");

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaDevice;

        match CudaDevice::new(0) {
            Ok(device) => {
                println!("CUDA initialized successfully!");
                println!("Device 0: Available");

                // Try to get device count
                match cudarc::driver::result::device::get_count() {
                    Ok(count) => println!("Total CUDA devices: {}", count),
                    Err(e) => println!("Could not get device count: {:?}", e),
                }

                // Allocate a small test buffer
                match device.alloc_zeros::<f32>(1024) {
                    Ok(_) => println!("GPU memory allocation: OK"),
                    Err(e) => println!("GPU memory allocation failed: {:?}", e),
                }

                println!("\nCUDA is ready for ShardLM v2!");
            }
            Err(e) => {
                eprintln!("Failed to initialize CUDA device 0: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled.");
        println!("Rebuild with: cargo run -p shardlm-v2-core --features cuda --example cuda_check");
        std::process::exit(1);
    }
}
