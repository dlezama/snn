import torch
import snntorch as snn

def test_hardware():
    print("--- Hardware Verification ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"Active GPU: {props.name}")
        print(f"VRAM Capacity: {props.total_memory / 1024**3:.2f} GB")
    
    # 1. Create a tensor on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        x = torch.ones(10).to(device)
        print(f"\nTensor successfully moved to: {x.device}")

        # 2. Simple snnTorch test
        lif = snn.Leaky(beta=0.9).to(device)
        mem = lif.init_leaky().to(device)
        # Ensure mem is on the right device
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)
        
        spk, mem = lif(x, mem)
        print(f"snnTorch neuron initialized and processed a spike on {device}.")
        print("--- Readiness Check Passed ---")
    except Exception as e:
        print(f"\n[Error during GPU operations]: {e}")
        print("Note: ROCm on Windows can sometimes require specific environment variables or driver versions.")

if __name__ == "__main__":
    test_hardware()
