# Learning Spiking Neural Networks with snnTorch: A Comprehensive Course

This repository is dedicated to mastering Spiking Neural Networks (SNNs) using the [snnTorch](https://github.com/jeshraghian/snntorch) library. The goal is to build deep intuition and practical skills in neuromorphic computing, spike-based communication, and training SNNs using surrogate gradients and advanced training algorithms.

---

## 🧠 The Neuromorphic Mindset: Time vs. Space
Standard Deep Learning is largely **spatial**—information is processed in parallel snapshots (images, word embeddings). Neuromorphic computing is **temporal**. 
*   **Information is in the Timing:** A single spike has no value; the *time* it arrives relative to others is everything.
*   **Memory is Internal:** Every neuron has a "membrane potential" that acts as a short-term memory of previous inputs.
*   **Sparsity is Efficiency:** In SNNs, "silence is golden." If nothing is happening, no energy is consumed.

---

## 📚 Core Theory & Reading List
Before diving into code, it is highly recommended to skim these foundational resources:
1.  **The Bible of SNNs:** [Neuronal Dynamics (Gerstner et al.)](https://neuronaldynamics.epfl.ch/) - Specifically Chapter 1 (LIF models).
2.  **The snnTorch Paper:** [Eshraghian et al. (2021) - "snnTorch: Deep Learning with Spiking Neural Networks"](https://arxiv.org/abs/2109.12894) - The definitive guide to the library's design and surrogate gradients.
3.  **Surrogate Gradients:** [Neftci et al. (2019) - "Surrogate Gradient Learning in Spiking Neural Networks"](https://arxiv.org/abs/1901.09948).

---

## Environment Setup (AMD GPU / ROCm)

This project uses `uv` and Python 3.12 for managing dependencies, specifically configured for AMD GPUs using ROCm on Windows.

### 1. Create Virtual Environment
```powershell
uv venv --python 3.12
.venv\Scripts\activate
```

### 2. Install ROCm SDK Components
```powershell
uv pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
```

### 3. Install PyTorch with ROCm Support
```powershell
uv pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
```

### 4. Install snnTorch & Utilities
```powershell
uv pip install snntorch matplotlib tonic
```

---

## The Complete snnTorch Course Curriculum

This curriculum is designed as a series of hands-on lessons. You will write the code for each lesson as a standalone Python executable. Each lesson builds upon the previous one and concludes with a **short quiz (3-5 questions)** to test your knowledge.

### Part 1: Spike Encoding, Simulation, and Visualization
The foundational stage. Learn how single neurons work and how to translate real-world continuous data into binary spikes.

*   [x] **`01_hello_snn.py`**: **The Leaky Integrate-and-Fire (LIF) Neuron**
    *   **Concept:** The biological mechanism of membrane potential, leak, integration, threshold, and refractory reset.
    *   **Objective:** Simulate a single `snntorch.Leaky` neuron. Pass a hardcoded spike train and observe the membrane potential decay.
    *   **Deep Dive:** [Tutorial 2: LIF Neuron](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html) and [Neuronal Dynamics: The LIF Model](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html).
    *   **Key APIs:** `snn.Leaky`, `lif.init_leaky()`, `lif(input, mem)`.

*   [ ] **`02_spike_encoding.py`**: **Data to Spikes (Spikegen)**
    *   **Concept:** SNNs require temporal binary data. We can encode a static image into a spike train using Rate or Latency.
    *   **Objective:** Load MNIST. Use `snntorch.spikegen` to convert a static image into a temporal spike train using Rate and Latency coding.
    *   **Deep Dive:** [Tutorial 1: Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html).
    *   **Key APIs:** `spikegen.rate`, `spikegen.latency`.

*   [ ] **`03_population_coding.py`**: **Robustness via Numbers**
    *   **Concept:** "Population coding" uses a group of neurons to represent a single value.
    *   **Objective:** Use `snntorch.spikegen.delta` to encode a signal.
    *   **Akida Cross-Check:** Use the `akida` SDK (Simulator) to visualize how the AKD1000's digital "Events" differ from floating-point "Spikes."
    *   **Deep Dive:** [Theoretical Neuroscience (Dayan & Abbott) - Population Codes](http://www.rctn.org/bruno/public/dayan_abbott_ch3.pdf).
    *   **Key APIs:** `spikegen.delta`.

*   [ ] **`04_spike_visualization.py`**: **Plotting SNN Activity**
    *   **Concept:** Debugging SNNs requires temporal visualization.
    *   **Objective:** Take the spike trains generated in previous lessons and visualize them using `snntorch.spikeplot`. Create a raster plot and a membrane potential trace.
    *   **Deep Dive:** [Tutorial 1 & 2 Visualization](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html).
    *   **Key APIs:** `spikeplot.raster`, `spikeplot.traces`, `spikeplot.animator`.

*   [ ] **`05_synaptic_currents.py`**: **2nd-Order Neuron Models**
    *   **Concept:** Real neurons have a slow synaptic conductance wave.
    *   **Objective:** Upgrade from `Leaky` to `Synaptic` neurons. Observe the interplay between `alpha` (synaptic decay) and `beta` (membrane decay).
    *   **Deep Dive:** [Tutorial 4: 2nd Order Models](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html).
    *   **Key APIs:** `snn.Synaptic`, `lif.init_synaptic()`.

### Part 2: Building & Training Feedforward Networks
Transitioning from isolated neurons to deep architectures and addressing the challenge of gradients.

*   [ ] **`06_feedforward_snn.py`**: **Connecting Neurons with PyTorch**
    *   **Concept:** snnTorch neurons act as activation functions that integrate over time.
    *   **Objective:** Build a 2-layer Fully Connected SNN using `nn.Sequential`.
    *   **Akida Cross-Check:** Map this network to the Akida NPU mesh. See how neurons are distributed across the hardware's 20 NPUs using the simulator.
    *   **Deep Dive:** [Tutorial 3: Feedforward SNNs](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html).
    *   **Key APIs:** `nn.Linear`, `snn.Leaky`, `snntorch.utils.reset`.

*   [ ] **`07_surrogate_gradients.py`**: **The Dead Neuron Problem**
    *   **Concept:** The spike function derivative is zero. Surrogate gradients provide a smooth approximation for backprop.
    *   **Objective:** Implement `atan` and `fast_sigmoid` surrogates.
    *   **Deep Dive:** [Tutorial 5: Surrogate Gradients](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) and [Neftci et al. (2019)](https://arxiv.org/abs/1901.09948).
    *   **Key APIs:** `snntorch.surrogate.atan`, `snntorch.surrogate.fast_sigmoid`.

*   [ ] **`08_training_bptt.py`**: **Backpropagation Through Time (BPTT)**
    *   **Concept:** SNNs are trained by unrolling them over time steps.
    *   **Objective:** Train the network on MNIST. **Crucial:** Normalize inputs to $[0, 1]$ to avoid spike saturation.
    *   **Deep Dive:** [snnTorch Paper - Section III: Training Algorithms](https://arxiv.org/pdf/2109.12894.pdf).
    *   **Key APIs:** `snntorch.functional.ce_rate_loss`, `snntorch.functional.acc`.

### Part 3: Advanced Architectures & High-Resolution Datasets
Scaling SNNs to complex spatial patterns and native event-based sensors using high-performance hardware.

*   [ ] **`09_convolutional_snn.py`**: **Spiking ResNets (CIFAR-10)**
    *   **Concept:** MNIST is too small for high-end GPUs. Real-world SNNs use Deep Residual Networks.
    *   **Objective:** Build a **Spiking ResNet-18** architecture. Train it on the **CIFAR-10** dataset. Compare the "Spike Sparsity" of different layers in a deep network.
    *   **Deep Dive:** [Tutorial 6: Convolutional SNNs](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html) and [Spiking ResNets (Fang et al., 2021)](https://arxiv.org/abs/2102.04159).
    *   **Key APIs:** `nn.Conv2d`, `snn.Leaky`.

*   [ ] **`10_neuromorphic_data_tonic.py`**: **High-Res Event Cameras (DVS128)**
    *   **Concept:** DVS cameras naturally output sparse events. High-resolution DVS data requires massive temporal unrolling.
    *   **Objective:** Use `Tonic` to load the **DVS128 Gesture** dataset ($128 \times 128$ resolution). Train a 3D-CSNN to recognize human gestures in real-time.
    *   **Deep Dive:** [Tutorial 7: Neuromorphic Datasets](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html).
    *   **Key APIs:** `tonic.datasets.DVSGesture`.

*   [ ] **`11_recurrent_snn.py`**: **Long-Term Temporal Memory**
    *   **Concept:** Explicit recurrent weights for complex temporal dynamics.
    *   **Objective:** Implement `snn.RLeaky`. Train it on the **Sequential MNIST** task (pixels fed 1-by-1) to push the limits of temporal integration.
    *   **Deep Dive:** [snnTorch Documentation: Recurrent Layers](https://snntorch.readthedocs.io/en/latest/snntorch.html#recurrent-layers).

### Part 4: Optimization, Scaling & Local Learning

*   [ ] **`12_loss_and_regularization.py`**: **Sparsity & Power Efficiency**
    *   **Concept:** Penalize excessive firing to improve energy efficiency.
    *   **Objective:** Add L1/L2 spike regularization to your Spiking ResNet. Visualize the "Energy-Accuracy" frontier.
    *   **Deep Dive:** [snnTorch Documentation: Regularization](https://snntorch.readthedocs.io/en/latest/snntorch.functional.html#regularization).

*   [ ] **`13_stdp_online_learning.py`**: **STDP vs. Incremental Learning**
    *   **Concept:** Biological STDP vs. Industrial "One-Shot" learning.
    *   **Objective:** Implement unsupervised STDP in `snnTorch`.
    *   **Akida Parallel:** Use the Akida card's native **Incremental Learning** to "learn" a new data class (e.g., a new gesture) in one-shot without a training loop.
    *   **Deep Dive:** [Akida Learning Whitepaper](https://brainchip.com/akida-learning/).

*   [ ] **`14_ann_to_snn.py`**: **Deep Model Conversion**
    *   **Concept:** Fast-track SNN development by converting pre-trained deep ANNs.
    *   **Objective:** Convert a pre-trained **VGG-16** or **Inception** model to an SNN and compare performance on ImageNet-1K (subset).
    *   **Deep Dive:** [snnTorch Documentation: ANN-to-SNN](https://snntorch.readthedocs.io/en/latest/snntorch.utils.html#ann-to-snn-conversion).

*   [ ] **`15_energy_and_export.py`**: **SynOps & NIR Export**
    *   **Concept:** Measure energy (Synaptic Operations) and export to hardware (NIR).
    *   **Objective:** Benchmark your Spiking ResNet-18 and export to `.nir` format.

*   [ ] **`16_spiking_transformers.py`**: **The Cutting Edge (SSA)**
    *   **Concept:** Standard Self-Attention is quadratic. Spiking Self-Attention (SSA) can be linear and ultra-sparse.
    *   **Objective:** Implement a basic **Spiking Transformer (Spikformer)**. Train it on CIFAR-100 to utilize your 7900 XTX's memory.
    *   **Deep Dive:** [Spikformer (Zhou et al., 2023)](https://arxiv.org/abs/2211.12681).

*   [ ] **`17_akida_deployment.py`**: **Edge AI with BrainChip Akida**
    *   **Concept:** Moving from high-power training (GPU) to ultra-low-power inference (AKD1000).
    *   **Objective:** Take a trained CSNN from Lesson 09. Quantize the weights to 4-bit using `quantizeml`. Convert to Akida format (`.fbz`) and run real-time inference on the AKD1000 M.2 card.
    *   **Hardware Required:** BrainChip Akida AKD1000 M.2 Card.
    *   **Deep Dive:** [BrainChip MetaTF Documentation](https://doc.brainchipinc.com/).

---

## 🛠️ Neuromorphic Hardware Stack
This course supports a heterogeneous hardware environment:
*   **Training (The Muscle):** **AMD Radeon RX 7900 XTX** (ROCm). Used for deep BPTT and large-scale architectural exploration.
*   **Deployment (The Brain):** **BrainChip Akida AKD1000**. Used for edge inference, event-based processing, and incremental learning.
*   **Sensing (The Eyes):** **`v2e` (Video-to-Events) Simulation**. Uses your GPU to generate high-fidelity event streams from standard video files. (Hardware DVS optional).

---

## Capstone Projects

*   **Project A: Neuromorphic Pong (RL):** Policy Gradients with spikes to play Pong.
*   **Project B: Keyword Spotting (KWS):** Audio processing using the `SHHD` dataset.
*   **Project C: Anomaly Detection:** RSNN for industrial sensor streams.
