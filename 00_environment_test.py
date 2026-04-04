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
    
    # 2. Simple snnTorch test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        x = torch.ones(10).to(device)
        print(f"\nTensor successfully moved to: {x.device}")

        lif = snn.Leaky(beta=0.9).to(device)
        mem = lif.init_leaky()
        # If mem is a tensor, ensure it's on the right device
        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)

        spk, mem = lif(x, mem)
        print(f"snnTorch neuron initialized and processed a spike on {device}.")
    except Exception as e:
        print(f"\n[Error during GPU operations]: {e}")

    # 3. Akida Hardware Verification
    print("\n--- Akida Hardware Verification ---")
    try:
        import akida
        print(f"Akida version: {akida.__version__}")

        devices = akida.devices()
        if len(devices) > 0:
            print(f"Found {len(devices)} Akida device(s).")
            for i, dev in enumerate(devices):
                print(f"  [{i}]: {dev.version}")
        else:
            print("No Akida hardware (AKD1000) found. Check PCIe/M.2 connections and drivers.")

        import cnn2snn
        print(f"cnn2snn version: {cnn2snn.__version__}")
        print("--- Akida Stack Ready ---")
    except ImportError as e:
        print(f"[ImportError]: {e}")
        print("Ensure 'akida' and 'cnn2snn' are installed in the current environment.")
    except Exception as e:
        print(f"[Akida Error]: {e}")

    print("\n--- Readiness Check Passed ---")

if __name__ == "__main__":
    test_hardware()
