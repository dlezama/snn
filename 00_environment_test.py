import torch
import snntorch as snn
import sys
import os
import platform
import importlib


def _report_version(module_name, display_name=None):
    """Import a module and print its __version__. Returns True on success."""
    label = display_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {label}: {version}")
        return True
    except Exception as e:
        print(f"  {label}: [FAIL] {type(e).__name__}: {e}")
        return False


def test_hardware():
    # 0. Host identification — which machine's stack are we verifying?
    print("--- Host ---")
    print(f"Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform: {platform.system()} {platform.release()} / {platform.machine()}")

    print("\n--- Hardware Verification ---")
    print(f"PyTorch version: {torch.__version__}")

    is_avail = torch.cuda.is_available()
    print(f"CUDA/ROCm Available: {is_avail}")

    if is_avail:
        props = torch.cuda.get_device_properties(0)
        print(f"Active GPU: {props.name}")
        print(f"VRAM Capacity: {props.total_memory / 1024**3:.2f} GB")

    # 2. Simple snnTorch test
    device = torch.device("cuda" if is_avail else "cpu")

    try:
        x = torch.ones(10).to(device)
        print(f"\nTensor successfully moved to: {x.device}")

        lif = snn.Leaky(beta=0.9).to(device)
        mem = lif.init_leaky()

        if isinstance(mem, torch.Tensor):
            mem = mem.to(device)

        spk, mem = lif(x, mem)
        print(f"snnTorch neuron initialized and processed a spike on {device}.")

        # Explicitly delete tensors to prevent ROCm context deadlocks
        del x, spk, mem, lif

    except Exception as e:
        print(f"\n[Error during GPU operations]: {e}")

    # 3. Core library stack — every pinned package that later lessons will touch.
    # A version print that fails loudly here beats a cryptic ImportError mid-lesson.
    print("\n--- Library Stack ---")
    print("PyTorch ecosystem:")
    _report_version("snntorch")
    _report_version("torchvision")
    _report_version("torchaudio")
    print("Scientific stack:")
    _report_version("numpy")
    _report_version("scipy")
    _report_version("matplotlib")
    _report_version("h5py")
    print("ONNX / export (Lesson 17):")
    _report_version("onnx")
    _report_version("onnxruntime")
    _report_version("onnxscript")

    # 4. Tonic — installed with --no-deps to bypass its numpy<2 metadata pin.
    # Importing a dataset class (without instantiation) confirms the eagerly-loaded
    # module tree (datasets/mvsec → importRosbag, datasets/ntidigits18 → tqdm) is intact.
    print("\n--- Tonic (Lesson 11) ---")
    try:
        import tonic
        from tonic.datasets import DVSGesture
        print(f"  tonic: {tonic.__version__}")
        print(f"  tonic.datasets.DVSGesture resolved to {DVSGesture.__module__}.{DVSGesture.__name__} "
              "(tqdm + importRosbag reachable).")
    except Exception as e:
        print(f"  tonic: [FAIL] {type(e).__name__}: {e}")

    # 5. Akida Hardware Verification
    print("\n--- Akida Hardware Verification ---")
    try:
        # Suppress TensorFlow INFO and WARNING logs, but KEEP ERRORS
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        import akida
        print(f"Akida version: {akida.__version__}")

        devices = akida.devices()
        has_hardware = len(devices) > 0
        if has_hardware:
            print(f"Found {len(devices)} Akida device(s).")
            for i, dev in enumerate(devices):
                print(f"  [{i}]: {dev.version}")
        else:
            print("No Akida hardware (AKD1000) found. Check PCIe/M.2 connections and drivers.")

        import cnn2snn
        print(f"cnn2snn version: {cnn2snn.__version__}")
        
        if has_hardware:
            print("--- Akida Stack Ready (Hardware Accelerated) ---")
        else:
            print("--- Akida Stack Ready (Simulator Only) ---")
    except ImportError as e:
        print(f"[ImportError]: {e}")
        print("Ensure 'akida' and 'cnn2snn' are installed in the current environment.")
    except Exception as e:
        print(f"[Akida Error]: {e}")

    # Clean up ROCm/CUDA context to prevent hangs on Windows
    if is_avail:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    sys.stdout.flush()
    # os._exit(0)

if __name__ == "__main__":
    test_hardware()
