# Learning Spiking Neural Networks with snnTorch: A Comprehensive Course

This repository is dedicated to mastering Spiking Neural Networks (SNNs) using the [snnTorch](https://github.com/jeshraghian/snntorch) library. The goal is to build deep intuition and practical skills in neuromorphic computing, spike-based communication, and training SNNs using surrogate gradients and advanced training algorithms.

---

## 🧠 The Neuromorphic Mindset: Time vs. Space
Standard Deep Learning is largely **spatial**—information is processed in parallel snapshots (images, word embeddings). Neuromorphic computing is **temporal**. 
*   **Information is in the Timing:** A single spike has no value; the *time* it arrives relative to others is everything.
*   **Memory is Internal:** Every neuron has a "membrane potential" that acts as a short-term memory of previous inputs.
*   **Sparsity is Efficiency:** In SNNs, "silence is golden." If nothing is happening, no energy is consumed.
*   **Hybrid Computation:** Bridging continuous numerical calculus (for training) with discrete logic events (for hardware).

---

## 📚 Core Theory & Reading List
Before diving into code, it is highly recommended to skim these foundational resources:
1.  **The Bible of SNNs:** [Neuronal Dynamics (Gerstner et al.)](https://neuronaldynamics.epfl.ch/) - Specifically Chapter 1 (LIF models).
2.  **The snnTorch Paper:** [Eshraghian et al. (2021) - "snnTorch: Deep Learning with Spiking Neural Networks"](https://arxiv.org/abs/2109.12894).
3.  **Surrogate Gradients:** [Neftci et al. (2019) - "Surrogate Gradient Learning in Spiking Neural Networks"](https://arxiv.org/abs/1901.09948).
4.  **Community SNN Textbook:** [Spiking Neural Networks (snnbook.net)](https://snnbook.net/) - (Accessed: April 4, 2026). This is an active community resource bridging theory and hardware.

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

### 5. Install BrainChip Akida (MetaTF)
To simulate and deploy to the AKD1000 M.2 card, install the MetaTF toolchain:
```powershell
uv pip install akida cnn2snn
```

---

## The Complete snnTorch Course Curriculum

### Part 1: Foundations & Real-World Sensing

*   **`00_environment_test.py`**: **Environment & Hardware Verification**
    *   **[Docs]**: [Tutorial 7: Backpropagation Through Time](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html) (Focus on the introduction to BPTT and memory overhead)
    *   **Concept:** Ensuring the high-end Heterogeneous Stack (7900 XTX + Akida) is visible to the software. The script is already written and provided; **no coding is required by the student for this lesson**.
    *   **Objective:** Run the script to verify ROCm 7.2.1, PyTorch 2.9.1, and `snntorch` visibility. Confirm ~24GB VRAM availability.
    *   **Quiz Topics:** GPU vs. CPU acceleration in SNNs, VRAM management for temporal unrolling, ROCm/CUDA compatibility layers.

*   **`01_hello_snn.py`**: **The Leaky Integrate-and-Fire (LIF) Neuron**
    *   **[Docs]**: [Tutorial 1: Spiking Neurons with snnTorch](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
    *   **Concept:** Membrane potential, leak, integration, threshold, and refractory reset.
    *   **Quiz Topics:** Passive vs. Active membrane properties, `beta`, discrete-time recurrence, reset mechanisms.

*   **`02_real_world_sensing_v2e.py`**: **Video-to-Events (v2e)**
    *   **Concept:** Simulation of high-resolution event cameras (DVS). Understanding polarity and microsecond timestamps.
    *   **Deep Dive:** [v2e documentation](https://github.com/SensorsINI/v2e).

*   **`03_spike_encoding.py`**: **Data to Spikes (Spikegen)**
    *   **[Docs]**: [Tutorial 2: Spike Encoding with snnTorch](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html)
    *   **Concept:** Rate vs. Latency coding. The "Silicon Retina" concept.
    *   **Objective:** Load MNIST. Use `snntorch.spikegen` to convert images to temporal spike trains.

*   **`04_population_coding.py`**: **Robustness via Numbers & Akida Cross-Check**
    *   **Concept:** Representing values across a group of neurons. 
    *   **Akida Cross-Check:** Visualize how the AKD1000's digital "Events" differ from floating-point "Spikes."

*   **`05_spike_visualization.py`**: **Plotting SNN Activity**
    *   **[Docs]**: [Tutorial 3: Spiking Neural Networks with snnTorch](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html)
    *   **Objective:** Visualize spike trains using `snntorch.spikeplot`. Raster plots, membrane potential traces, and identifying "Bursting" patterns.

### Part 2: Mathematical Rigor & Training Dynamics

*   **`06_synaptic_currents.py`**: **2nd-Order Neuron Models**
    *   **[Docs]**: [Tutorial 4: 2nd Order Spiking Neurons](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html)
    *   **Concept:** Biological realism with synaptic conductance waves. Interplay of `alpha` and `beta` decay.

*   **`07_surrogate_gradients_math.py`**: **The Dead Neuron Problem (Deep Dive)**
    *   **[Docs]**: [Tutorial 5: Training SNNs with Surrogate Gradients](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)
    *   **Concept:** Mathematical derivation of Surrogate Gradients from `snnbook.net`. 
    *   **Objective:** Implement custom surrogates (ATan, Fast Sigmoid). Visualize forward (discrete) vs. backward (smooth) gradients.

*   **`08_feedforward_snn.py`**: **Connecting Neurons with PyTorch**
    *   **[Docs]**: [Tutorial 6: Building SNNs with PyTorch](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
    *   **Objective:** Build a 2-layer FC SNN. 
    *   **Akida Cross-Check:** Map the network to the Akida NPU mesh (20 NPUs).

*   **`09_training_bptt.py`**: **Backpropagation Through Time (BPTT)**
    *   **[Docs]**: [Tutorial 7: Backpropagation Through Time](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)
    *   **Objective:** Train on MNIST. Focus on Rate vs. Latency loss and GPU VRAM utilization on the 7900 XTX.

### Part 3: Scaling, High-Res Data & Hardware Deep Dives

*   **`10_convolutional_snn.py`**: **Spiking ResNets (CIFAR-10)**
    *   **[Docs]**: [Tutorial 6: Building SNNs with PyTorch (CSNN Section)](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
    *   **Objective:** Build Spiking ResNet-18 architecture utilizing 24GB VRAM.

*   **`11_neuromorphic_data_tonic.py`**: **High-Res Event Data (DVS128)**
    *   **[Docs]**: [Tonic Documentation](https://tonic.readthedocs.io/en/latest/)
    *   **Objective:** Use `Tonic` to load DVS128 Gesture dataset. Train a 3D-CSNN.

*   **`12_hardware_deep_dive.py`**: **7900 XTX vs. BrainChip Akida**
    *   **[Docs]**: [Akida Documentation](https://doc.brainchipinc.com/)
    *   **Concept:** Parallel GPU simulation vs. Edge NPU event-driven inference.
    *   **Objective:** Profile energy, latency, and memory constraints for both paradigms.

### Part 4: Advanced Architectures & Deployment

*   **`13_recurrent_snn.py`**: **Long-Term Temporal Memory**
    *   **[Docs]**: [Tutorial 8: Recurrent Spiking Neural Networks](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_8.html)
    *   **Objective:** Implement `snn.RLeaky` on Sequential MNIST to test temporal integration limits.

*   **`14_loss_and_regularization.py`**: **Sparsity & Power Efficiency**
    *   **[Docs]**: [snntorch.functional (Loss Functions)](https://snntorch.readthedocs.io/en/latest/snntorch.functional.html)
    *   **Objective:** Add L1/L2 spike regularization. Analyze the Energy-Accuracy frontier.

*   **`15_stdp_online_learning.py`**: **STDP vs. Incremental Learning**
    *   **[Docs]**: [Tutorial: Unsupervised Learning with STDP](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_stdp.html)
    *   **Objective:** Unsupervised STDP in `snntorch` vs. Akida's native "One-Shot" incremental learning.

*   **`16_ann_to_snn.py`**: **Deep Model Conversion**
    *   **[Docs]**: [cnn2snn (Akida) Documentation](https://doc.brainchipinc.com/api_reference/cnn2snn/index.html)
    *   **Objective:** Convert pre-trained VGG-16 to SNN. Threshold balancing and weight rescaling.

*   **`17_energy_and_export.py`**: **SynOps & NIR Export**
    *   **[Docs]**: [NIR: Neuromorphic Intermediate Representation](https://github.com/neuromorphs/NIR)
    *   **Concept:** Measuring Synaptic Operations (SynOps) and exporting to `.nir` format.

*   **`18_spiking_transformers.py`**: **The Cutting Edge (SSA)**
    *   **[Docs]**: [Spiking Transformers Research](https://arxiv.org/abs/2205.00319)
    *   **Objective:** Implement Spiking Self-Attention (SSA) to utilize high-bandwidth memory.

*   **`19_akida_deployment.py`**: **Edge AI Inference**
    *   **[Docs]**: [Akida Model Zoo & Deployment](https://doc.brainchipinc.com/api_reference/akida/index.html)
    *   **Objective:** Quantize to 4-bit and deploy as `.fbz` to AKD1000 M.2 card.

---

## 🛠️ Neuromorphic Hardware Stack
*   **Training (The Muscle):** **AMD Radeon RX 7900 XTX** (ROCm).
*   **Deployment (The Brain):** **BrainChip Akida AKD1000**.
*   **Sensing (The Eyes):** **`v2e` Simulation**.
