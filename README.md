# Learning Spiking Neural Networks with snnTorch: A Comprehensive Course

This repository is dedicated to mastering Spiking Neural Networks (SNNs) using the [snnTorch](https://github.com/jeshraghian/snntorch) library. The goal is to build deep intuition and practical skills in neuromorphic computing, spike-based communication, and training SNNs using surrogate gradients and advanced training algorithms.

**Official Documentation Reference:** [snnTorch ReadTheDocs](https://snntorch.readthedocs.io/)

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

This curriculum is designed as a series of hands-on lessons. You will write the code for each lesson as a standalone Python executable. Each lesson builds upon the previous one.

### Part 1: Spike Encoding, Simulation, and Visualization
The foundational stage. Learn how single neurons work and how to translate real-world continuous data into binary spikes.

*   [x] **`01_hello_snn.py`**: **The Leaky Integrate-and-Fire (LIF) Neuron**
    *   **Concept:** The biological mechanism of membrane potential, leak, integration, threshold, and refractory reset.
    *   **Objective:** Simulate a single `snntorch.Leaky` neuron. Pass a hardcoded spike train and observe the membrane potential decay.
    *   **Key APIs:** `snn.Leaky`, `lif.init_leaky()`, `lif(input, mem)`.
    *   **Reference:** [Tutorial 2: LIF Neuron](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html)

*   [ ] **`02_spike_encoding.py`**: **Data to Spikes (Spikegen)**
    *   **Concept:** Artificial neural networks take continuous values (like pixel intensity). SNNs require temporal binary data.
    *   **Objective:** Load the MNIST dataset (using `torchvision`). Use `snntorch.spikegen` to convert a static image into a temporal spike train using Rate Coding, Latency Coding, and Delta Modulation.
    *   **Key APIs:** `spikegen.rate`, `spikegen.latency`, `spikegen.delta`.
    *   **Reference:** [Tutorial 1: Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)

*   [ ] **`03_spike_visualization.py`**: **Plotting SNN Activity**
    *   **Concept:** Debugging SNNs requires temporal visualization.
    *   **Objective:** Take the spike trains generated in Lesson 02 and visualize them using `snntorch.spikeplot`. Create a raster plot and a membrane potential trace.
    *   **Key APIs:** `spikeplot.raster`, `spikeplot.traces`, `spikeplot.animator`.
    *   **Reference:** [Tutorial 1 & 2 Visualization](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)

*   [ ] **`04_synaptic_currents.py`**: **2nd-Order Neuron Models**
    *   **Concept:** Real neurons don't instantly jump in voltage when hit by a spike. Neurotransmitters create a slow synaptic conductance wave.
    *   **Objective:** Upgrade from the 1st-order `Leaky` neuron to the 2nd-order `Synaptic` neuron. Observe the interplay between the synaptic decay (`alpha`) and membrane decay (`beta`).
    *   **Key APIs:** `snn.Synaptic`, `lif.init_synaptic()`.
    *   **Reference:** [Tutorial 4: 2nd Order Models](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html)

### Part 2: Building & Training Feedforward Networks
Transitioning from isolated neurons to deep architectures and addressing the fundamental challenge of training SNNs.

*   [ ] **`05_feedforward_snn.py`**: **Connecting Neurons with PyTorch**
    *   **Concept:** snnTorch neurons act as activation functions. They can be seamlessly integrated with PyTorch's `nn.Linear` layers.
    *   **Objective:** Build a 2-layer Fully Connected SNN using `nn.Sequential`. Pass an encoded MNIST image through the network.
    *   **Key APIs:** `nn.Linear`, `snn.Leaky`, `snntorch.utils.reset`.
    *   **Reference:** [Tutorial 3: Feedforward SNNs](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html)

*   [ ] **`06_surrogate_gradients.py`**: **The Dead Neuron Problem**
    *   **Concept:** The Heaviside step function used to generate a spike has a derivative of zero almost everywhere. Backpropagation fails because gradients vanish. Surrogate gradients replace the step function's derivative with a smooth approximation during the backward pass.
    *   **Objective:** Implement different surrogate gradients and observe how they wrap the spiking mechanism.
    *   **Key APIs:** `snntorch.surrogate.atan`, `snntorch.surrogate.sigmoid`, `snntorch.surrogate.fast_sigmoid`.
    *   **Reference:** [Tutorial 5: Surrogate Gradients](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)

*   [ ] **`07_training_bptt.py`**: **Backpropagation Through Time (BPTT)**
    *   **Concept:** Training SNNs requires unrolling the network over time steps.
    *   **Objective:** Write a complete PyTorch training loop for the network built in Lesson 05. Use CrossEntropyLoss applied to the total spike count (rate coding) of the output layer to classify MNIST digits.
    *   **Key APIs:** `snntorch.functional.ce_rate_loss`, `snntorch.functional.acc`.
    *   **Reference:** [Tutorial 5: Training SNNs](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)

### Part 3: Advanced Architectures & Neuromorphic Datasets
Applying SNNs to spatial data and native event-based sensors.

*   [ ] **`08_convolutional_snn.py`**: **Spatial Features (CSNNs)**
    *   **Concept:** SNNs can utilize weight-sharing just like standard CNNs.
    *   **Objective:** Replace `nn.Linear` with `nn.Conv2d` and `nn.MaxPool2d` to build a Convolutional SNN. Train it on MNIST and compare parameter counts/accuracy against the fully connected version.
    *   **Key APIs:** `nn.Conv2d`, integrating SNN layers with CNN layers.
    *   **Reference:** [Tutorial 6: CSNNs](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)

*   [ ] **`09_neuromorphic_data_tonic.py`**: **Event-Based Cameras (DVS)**
    *   **Concept:** Spikegen is slow and artificial. Neuromorphic cameras (like DVS) naturally output sparse, asynchronous spikes.
    *   **Objective:** Use the `Tonic` library to download and load the Neuromorphic-MNIST (N-MNIST) dataset. Feed these native events directly into your CSNN.
    *   **Key APIs:** `tonic.datasets.NMNIST`, `tonic.transforms.ToFrame`.
    *   **Reference:** [Tutorial 7: Neuromorphic Datasets](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)

*   [ ] **`10_recurrent_snn.py`**: **Temporal Memory (RSNNs)**
    *   **Concept:** SNNs inherently have memory via membrane potential, but adding explicit recurrent weights allows them to learn complex temporal dynamics.
    *   **Objective:** Implement `snntorch.RLeaky` or `snntorch.RSynaptic`. Train the network to recognize a sequence of inputs over time.
    *   **Key APIs:** `snn.RLeaky`, `snn.RSynaptic`.

### Part 4: Optimization, Regularization, and Local Learning
Refining the network for deployment on neuromorphic hardware and exploring biological learning rules.

*   [ ] **`11_loss_and_regularization.py`**: **Sparsity & Power Efficiency**
    *   **Concept:** Spikes cost energy on neuromorphic hardware. We want to penalize excessive firing while maintaining accuracy.
    *   **Objective:** Update your training loop to include L1/L2 regularization on the hidden layer spike counts. Explore Latency Loss (forcing the network to decide quickly).
    *   **Key APIs:** `snntorch.functional.reg.l1_rate_sparsity`, `snntorch.functional.mse_membrane_loss`.

*   [ ] **`12_stdp_online_learning.py`**: **Spike-Timing-Dependent Plasticity**
    *   **Concept:** Biological brains don't use BPTT. They use local learning rules based on the timing of spikes (STDP). "Neurons that fire together wire together."
    *   **Objective:** Use the `snntorch.stdp` module to implement unsupervised learning. Update weights step-by-step during the forward pass based on pre- and post-synaptic spike timings.
    *   **Key APIs:** `snntorch.stdp.STDP`.

*   [ ] **`13_export_nir.py`**: **Neuromorphic Intermediate Representation**
    *   **Concept:** To run a model on actual neuromorphic hardware (like Loihi or SpiNNaker), it needs to be exported from PyTorch.
    *   **Objective:** Export your trained CSNN to the NIR format, demonstrating readiness for hardware deployment.
    *   **Key APIs:** `snntorch.export_nir`.
