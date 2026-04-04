# Learning Spiking Neural Networks: A Comprehensive Course

This repository is dedicated to mastering Spiking Neural Networks (SNNs) through deep theoretical understanding and hands-on practice. The primary tool is [snnTorch](https://github.com/jeshraghian/snntorch), but the course draws on the best resources from computational neuroscience, neuromorphic engineering, and cutting-edge research to ensure the student builds real intuition — not just API familiarity.

---

## The Neuromorphic Mindset: Time vs. Space

Standard Deep Learning is largely **spatial** — information is processed in parallel snapshots (images, word embeddings). Neuromorphic computing is **temporal**.
*   **Information is in the Timing:** A single spike has no value; the *time* it arrives relative to others is everything.
*   **Memory is Internal:** Every neuron has a "membrane potential" that acts as a short-term memory of previous inputs.
*   **Sparsity is Efficiency:** In SNNs, "silence is golden" (achieving "Inbox-Zero" for neurons). If nothing is happening, no energy is consumed — crucial for Edge AI.
*   **Hybrid Computation:** Bridging continuous numerical calculus (for training) with discrete logic events (for hardware).

---

## Prerequisites

This course assumes the following background. If you are weak in any area, shore it up before starting.

| Area | What You Need | Why |
|------|---------------|-----|
| **Python** | Comfortable writing classes, loops, and using libraries | All exercises are in Python |
| **PyTorch** | Tensors, autograd, `nn.Module`, training loops, `DataLoader` | snnTorch is built on PyTorch — you must be fluent |
| **Calculus** | Derivatives, chain rule, gradient descent | Surrogate gradients and BPTT require this |
| **Linear Algebra** | Matrix multiplication, vector operations | Network layers are matrix ops |
| **Probability** (helpful) | Bernoulli trials, distributions | Rate coding and stochastic spiking |
| **Neuroscience** (not required) | Will be taught from scratch | The course covers all necessary biology |

---

## Core Theory & Reading List

### Foundational Textbooks & Surveys

These are the pillars of SNN knowledge. You will be assigned specific chapters before each lesson.

1.  **Neuronal Dynamics** — Gerstner, Kistler, Naud & Paninski (FREE online textbook + video lectures)
    *   [Online textbook](https://neuronaldynamics.epfl.ch/online/) | [Video lectures](https://neuronaldynamics.epfl.ch/lectures.html)
    *   *The* definitive resource for computational neuroscience. Covers LIF models, synaptic dynamics, STDP, and network-level behavior with mathematical rigor and biological grounding. **Start here.**

2.  **Spiking Neural Networks (snnbook.net)** — Open Neuromorphic Community (FREE, actively written)
    *   [snnbook.net](https://snnbook.net/)
    *   Community-driven textbook bridging theory and practice. Interactive examples, hardware-aware content. Written by ~10 active researchers. Good complement to Gerstner.

3.  **Training Spiking Neural Networks Using Lessons From Deep Learning** — Eshraghian et al. (Proceedings of the IEEE, 2023)
    *   [arxiv.org/abs/2109.12894](https://arxiv.org/abs/2109.12894)
    *   A 40-page tutorial-survey that bridges deep learning and SNNs. Covers encoding, surrogate gradients, BPTT for spikes, and connections to STDP. **The single best entry point for deep learning practitioners.**

4.  **Theoretical Neuroscience** — Dayan & Abbott (MIT Press)
    *   The standard graduate textbook for computational neuroscience. Rigorous mathematical treatment of neural encoding/decoding, network models, and plasticity. Recommended for students wanting deeper theoretical grounding.

### Key Research Papers

Organized by topic. You will be pointed to specific papers before relevant lessons.

**Surrogate Gradient Learning:**
*   Neftci, Mostafa & Zenke (2019) — [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948). The foundational paper.
*   Zenke & Vogels (2021) — [The Remarkable Robustness of Surrogate Gradient Learning](https://direct.mit.edu/neco/article/33/4/899/97482). Demonstrates that the exact shape of the surrogate barely matters — a surprising and liberating result.

**Deep Spiking Architectures:**
*   Fang et al. (2021) — [Deep Residual Learning in Spiking Neural Networks (SEW-ResNet)](https://arxiv.org/abs/2102.04159). Solves vanishing/exploding gradients in deep SNNs via spike-element-wise residual connections.

**Local Learning Rules:**
*   Bellec et al. (2020) — [A Solution to the Learning Dilemma for Recurrent Spiking Networks of Winner-Take-All Circuits (e-prop)](https://www.nature.com/articles/s41467-020-17236-y). Biologically plausible gradient approximation without BPTT. Nature Communications.
*   Frenkel (2022) — [e-prop PyTorch Implementation](https://github.com/ChFrenkel/eprop-PyTorch). Clean reference implementation for experimentation.

**Spiking Attention & State Space Models:**
*   Zhou et al. (2023) — [Spikformer: When SNN Meets Transformer](https://arxiv.org/abs/2209.15425). First pure-spike self-attention.
*   Yao et al. (2023) — Spike-Driven Transformer. Addition-only attention with 87x lower energy than vanilla attention.
*   SpikeMamba (2024) — [arxiv.org/abs/2404.01198](https://arxiv.org/abs/2404.01198). Spiking meets state space models.

**ANN-to-SNN Conversion:**
*   Jiang et al. (2023) — Unified Optimization Framework for ANN-SNN Conversion (ICML). Treats spike-rate mapping as a differentiable optimization problem.

**Neuromorphic Hardware Benchmarks:**
*   NeuroBench (2025) — [A Framework for Benchmarking Neuromorphic Computing](https://www.nature.com/articles/s41467-025-56739-4). Nature Communications. Collaborative benchmark from 100+ authors.

**Frontier (2025-2026):**
*   SpikeGPT — Zhu, Zhao & Eshraghian (2023) — [arxiv.org/abs/2302.13939](https://arxiv.org/abs/2302.13939). The largest backprop-trained SNN for language generation (216M params). 20x fewer operations on neuromorphic hardware.
*   P-SpikeSSM (2025) — Probabilistic Spiking State Space Models (ICLR 2025). Integrates SNNs with Mamba-style dynamics.

### University Lectures, Workshops & Community

*   **EPFL Neuronal Dynamics Video Lectures** — [neuronaldynamics.epfl.ch/lectures.html](https://neuronaldynamics.epfl.ch/lectures.html). Free, high-quality video lectures accompanying the textbook. Watch these alongside reading assignments.
*   **SNUFA Workshop** — [snufa.net](http://snufa.net/). Annual online workshop with recorded talks from top SNN researchers. All past videos on YouTube. Essential for staying current.
*   **Open Neuromorphic** — [open-neuromorphic.org](https://open-neuromorphic.org/). 2,000+ member community. Workshops, tutorials, Discord, and framework benchmarks. The central hub for the neuromorphic open-source ecosystem.
*   **Tonic Documentation** — [tonic.readthedocs.io](https://tonic.readthedocs.io/). "PyTorch Vision for neuromorphic data." Essential for lessons 04 and 11.
*   **NIR (Neuromorphic Intermediate Representation)** — [github.com/neuromorphs/NIR](https://github.com/neuromorphs/NIR). Universal exchange format for 7 simulators and 4 hardware platforms. Published in Nature Communications (2024).

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

## The Course Curriculum

### Part 1: Neuroscience Foundations & Spike Representations

The goal of Part 1 is to build biological intuition. You will understand *what* a spiking neuron does, *how* information is encoded as spikes, and *why* event-driven sensing matters — all before writing a single network.

*   **`00_environment_test.py`**: **Environment & Hardware Verification**
    *   **Concept:** Ensuring the heterogeneous stack (7900 XTX + Akida) is visible to the software. **No coding required** — the script is provided.
    *   **Objective:** Run the script to verify ROCm 7.2.1, PyTorch 2.9.1, snnTorch, and Akida visibility. Confirm ~24GB VRAM availability.
    *   **Quiz Topics:** GPU vs. CPU acceleration in SNNs, why temporal unrolling is VRAM-hungry, heterogeneous computing (GPU for training, NPU for inference).

*   **`01_lif_neuron.py`**: **The Leaky Integrate-and-Fire (LIF) Neuron**
    *   **Theory:** [Neuronal Dynamics Ch. 1](https://neuronaldynamics.epfl.ch/online/Ch1.html) (LIF derivation from biophysics) | Eshraghian et al. (2023) Sec. II (LIF in the deep learning context)
    *   **Practice:** [snnTorch Tutorial 2](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html)
    *   **Concept:** Membrane potential as a leaky integrator. The RC-circuit analogy. Threshold, spike generation, and refractory reset. Why the LIF model sacrifices biophysical detail for computational tractability.
    *   **Objective:** Simulate a single LIF neuron (`snn.Leaky`). Explore `beta` (decay rate), `threshold`, and reset mechanisms (subtract, zero, none). Plot membrane potential trajectories under constant and varying input currents.
    *   **Quiz Topics:** Passive vs. active membrane properties, the biological meaning of `beta`, discrete-time recurrence relation, reset mechanisms and their tradeoffs, when LIF breaks down (bursting, adaptation).

*   **`02_spike_encoding.py`**: **Data to Spikes**
    *   **Theory:** [Neuronal Dynamics Ch. 7](https://neuronaldynamics.epfl.ch/online/Ch7.html) (neural coding) | Eshraghian et al. (2023) Sec. II-B (encoding strategies)
    *   **Practice:** [snnTorch Tutorial 1](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
    *   **Concept:** The fundamental question: how does the brain (and an SNN) represent information? Rate coding (frequency), latency coding (time-to-first-spike), and delta modulation (change detection). The "Silicon Retina" concept. Why encoding choice shapes everything downstream.
    *   **Objective:** Load MNIST. Use `snntorch.spikegen` to convert images to temporal spike trains using rate, latency, and delta encoding. Compare the resulting spike trains visually and in terms of spike count / information density.
    *   **Quiz Topics:** Rate vs. temporal codes in biology, tradeoffs (accuracy vs. latency vs. sparsity), how `num_steps` affects information capacity, why rate coding is robust but slow, why latency coding is fast but fragile.

*   **`03_spike_visualization.py`**: **Reading the Language of Spikes**
    *   **Theory:** [snnbook.net](https://snnbook.net/) — raster plots and population analysis
    *   **Practice:** [snnTorch Tutorial 3](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html) (visualization sections)
    *   **Concept:** You cannot debug what you cannot see. Raster plots, membrane potential traces, spike frequency analysis, and identifying pathological patterns (silence, saturation, synchronous bursting).
    *   **Objective:** Visualize spike trains using `snntorch.spikeplot`. Create raster plots for populations of neurons. Overlay membrane potential traces. Identify and explain "healthy" vs. "pathological" firing patterns.
    *   **Quiz Topics:** How to read a raster plot, what synchronous bursting indicates, membrane potential trace interpretation, the relationship between input strength and firing rate, visual signatures of different encoding schemes.

*   **`04_event_cameras_v2e.py`**: **Event-Driven Sensing**
    *   **Theory:** Gallego et al. (2020) — "Event-based Vision: A Survey" (skim Sec. I-III for principles) | [v2e documentation](https://github.com/SensorsINI/v2e)
    *   **Concept:** Now that you understand spikes and encoding, meet the hardware that *natively* produces them. Event cameras (DVS) output asynchronous per-pixel brightness changes with microsecond resolution and 120+ dB dynamic range. How v2e simulates this from conventional video. Polarity, timestamps, and the address-event representation (AER).
    *   **Objective:** Use v2e to convert a short video clip into an event stream. Visualize the events as a spatiotemporal point cloud. Compare information density with frame-based representations.
    *   **Quiz Topics:** How a DVS pixel works vs. a conventional pixel, polarity and what it encodes, advantages of event cameras (latency, dynamic range, power), the AER protocol, when event cameras fail (static scenes).

*   **`05_population_coding.py`**: **Robustness via Numbers**
    *   **Theory:** Dayan & Abbott, Ch. 3 (Neural Encoding and Decoding) — population coding theory | [Neuronal Dynamics Ch. 11](https://neuronaldynamics.epfl.ch/online/Ch11.html) — population models
    *   **Practice:** [snnTorch Population Coding Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html)
    *   **Concept:** A single neuron's output is noisy. Populations of neurons encoding the same variable achieve robustness through redundancy. Tuning curves, population vectors, and why the brain uses millions of neurons for simple tasks.
    *   **Objective:** Implement population coding for a continuous variable. Decode a stimulus from population activity. Explore how population size affects accuracy and robustness.
    *   **Quiz Topics:** What a tuning curve is, how population size trades off with precision, the population vector method, biological examples of population coding (motor cortex, place cells), advantages for hardware with noisy/quantized neurons.

### Part 2: Mathematical Rigor & Training Dynamics

Part 2 is where the math gets serious. You will understand *why* training SNNs is hard (the dead neuron problem), learn the elegant surrogate gradient workaround, build your first trainable network, and train it end-to-end.

*   **`06_synaptic_currents.py`**: **2nd-Order Neuron Models**
    *   **Theory:** [Neuronal Dynamics Ch. 3](https://neuronaldynamics.epfl.ch/online/Ch3.html) (synaptic ion channels and conductance)
    *   **Practice:** [snnTorch Tutorial 4](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html)
    *   **Concept:** The LIF model assumes instantaneous synaptic input. Real synapses produce conductance waves that take time to rise and decay. The `Synaptic` and `Alpha` neuron models add this second time constant, capturing the interplay of synaptic current decay (`alpha`) and membrane decay (`beta`). When and why this matters.
    *   **Objective:** Simulate `snn.Synaptic` and `snn.Alpha` neurons. Compare their impulse responses to `snn.Leaky`. Explore how the `alpha`/`beta` ratio affects temporal filtering and spike timing.
    *   **Quiz Topics:** Why a 2nd-order model is needed, the biological basis of synaptic conductance, how `alpha` and `beta` interact, when to use `Synaptic` vs. `Alpha` vs. `Leaky`, computational cost implications.

*   **`07_surrogate_gradients.py`**: **The Dead Neuron Problem**
    *   **Theory:** [Neftci et al. (2019)](https://arxiv.org/abs/1901.09948) — the foundational surrogate gradient paper | [Zenke & Vogels (2021)](https://direct.mit.edu/neco/article/33/4/899/97482) — robustness of surrogate gradient choice | [snnbook.net](https://snnbook.net/) — surrogate gradient chapter
    *   **Practice:** [snnTorch Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) (surrogate gradient sections)
    *   **Concept:** The Heaviside step function (spike/no-spike) has zero gradient almost everywhere. This kills backpropagation. Surrogate gradients replace the step function's backward pass with a smooth approximation (ATan, Fast Sigmoid, etc.) while keeping the forward pass discrete. This is *the* key insight that makes modern SNN training possible.
    *   **Objective:** Implement and visualize multiple surrogate gradient functions. Plot the forward (discrete) vs. backward (smooth) pass for each. Compare their gradients at different membrane potentials. Use `snntorch.surrogate.custom_surrogate` to define your own.
    *   **Quiz Topics:** Why the Heaviside derivative is zero almost everywhere, how surrogate gradients solve this, the forward/backward asymmetry, why Zenke & Vogels showed the shape barely matters, the connection to the straight-through estimator (STE), biological plausibility of surrogate gradients.

*   **`08_feedforward_snn.py`**: **Building Networks with PyTorch**
    *   **Theory:** Eshraghian et al. (2023) Sec. III (SNN architectures)
    *   **Practice:** [snnTorch Tutorial 3](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html) (architecture and forward pass sections)
    *   **Concept:** Connecting neurons into layers using standard PyTorch `nn.Linear` for weights and snnTorch neurons for activation. The temporal forward pass: iterating over timesteps, passing spikes between layers, and accumulating output spikes. This is where Tutorial 3's architecture content comes together with the surrogate gradients from lesson 07.
    *   **Objective:** Build a 2-layer fully-connected SNN (`nn.Linear` -> `snn.Leaky` -> `nn.Linear` -> `snn.Leaky`). Run a forward pass over multiple timesteps. Verify correct tensor shapes: `[num_steps, batch, features]`. Understand the role of `snntorch.utils.reset(net)`.
    *   **Quiz Topics:** Why `reset()` is critical between samples, the temporal forward loop structure, where weights live vs. where dynamics live, batch dimension vs. time dimension (a common source of bugs), how to read output spike counts for classification.

*   **`09_training_bptt.py`**: **Backpropagation Through Time**
    *   **Theory:** Eshraghian et al. (2023) Sec. IV (training SNNs) | [snnTorch Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) (BPTT and loss functions)
    *   **Practice:** [snnTorch Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)
    *   **Concept:** BPTT unrolls the temporal forward pass and backpropagates through every timestep. The surrogate gradients from lesson 07 are what make this possible. Loss functions: rate-based (cross-entropy on spike counts), latency-based (time-to-first-spike), and membrane-based (max membrane potential). VRAM cost scales linearly with `num_steps`.
    *   **Objective:** Train the feedforward SNN from lesson 08 on MNIST. Experiment with `ce_rate_loss`, `ce_count_loss`, and `mse_count_loss`. Monitor VRAM utilization on the 7900 XTX. Achieve >95% test accuracy.
    *   **Quiz Topics:** How BPTT unrolling works for SNNs, why VRAM scales with `num_steps`, rate-based vs. latency-based loss tradeoffs, learning rate sensitivity in SNNs, the role of the surrogate gradient during backward pass, comparison with training RNNs.

### Milestone Project A: Consolidation

**You are now ready to work independently.** Before advancing to convolutional networks and hardware, prove your mastery by building an SNN from scratch without scaffolding.

*   **`milestone_a_fashion_mnist.py`**: **Design, Train & Analyze Your Own SNN**
    *   **Task:** Independently design, train, and analyze an SNN on **Fashion-MNIST**:
        1.  Choose a neuron model (`Leaky` or `Synaptic`) with written justification.
        2.  Choose a spike encoding strategy with written justification.
        3.  Design the network architecture (depth, width, neuron parameters).
        4.  Train using BPTT with a surrogate gradient and loss function of your choice.
        5.  Achieve **>85% test accuracy**.
        6.  Produce: spike raster plots for each layer, training curves, confusion matrix.
        7.  Write a brief analysis (~1 page in journal.md): how does spike sparsity change across layers? What is the accuracy/sparsity tradeoff?
    *   **No scaffolding is provided.** This is your first unguided exercise.

### Part 3: Scaling, Real-World Data & Hardware

Part 3 scales up. You will build deep convolutional SNNs, work with real neuromorphic sensor data, and understand the fundamental difference between GPU simulation and neuromorphic hardware execution.

*   **`10_convolutional_snn.py`**: **Spiking ConvNets & ResNets**
    *   **Theory:** [Fang et al. (2021) — SEW-ResNet](https://arxiv.org/abs/2102.04159) (spike-element-wise residual connections)
    *   **Practice:** [snnTorch Tutorial 6](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)
    *   **Concept:** Spiking convolutional layers for spatial feature extraction over time. The challenge of vanishing/exploding gradients in deep SNNs and how residual connections (SEW-ResNet) solve it. Leveraging 24GB VRAM for deep temporal unrolling.
    *   **Objective:** Build a spiking CNN for CIFAR-10. Implement spike-element-wise (SEW) residual connections. Train and compare with/without residuals. Profile VRAM usage with different `num_steps` values.
    *   **Quiz Topics:** How convolutions interact with temporal dynamics, why vanilla deep SNNs suffer gradient issues, the SEW residual mechanism, VRAM budgeting for `[num_steps, batch, channels, H, W]` tensors, batch size vs. num_steps tradeoff.

*   **`11_neuromorphic_data_tonic.py`**: **Real Event Data**
    *   **Theory:** Gallego et al. (2020) — Event-based Vision Survey (revisit Sec. IV on event representations) | [Tonic documentation](https://tonic.readthedocs.io/)
    *   **Practice:** [snnTorch Tutorial 7](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)
    *   **Concept:** Moving beyond frame-converted MNIST to natively asynchronous sensor data. The DVS128 Gesture dataset captures real hand gestures via a dynamic vision sensor. Key challenges: variable-length event streams, spatial and temporal transforms, binning strategies (time windows, event count windows).
    *   **Objective:** Use `Tonic` to load DVS128 Gesture. Apply event transforms (denoise, time quantization). Bin events into frames for the convolutional SNN from lesson 10. Train and evaluate. Optionally explore SHD (Spiking Heidelberg Digits) for audio spike data.
    *   **Quiz Topics:** How DVS data differs from frame-based data, event binning strategies and their tradeoffs, why Tonic is needed (raw events are irregular), the SHD and SSC audio datasets, challenges of variable-length sequences.

*   **`12_hardware_deep_dive.py`**: **GPU Simulation vs. Neuromorphic Hardware**
    *   **Theory:** [NeuroBench (2025)](https://www.nature.com/articles/s41467-025-56739-4) — benchmarking framework | [Akida documentation](https://doc.brainchipinc.com/) | [Open Neuromorphic framework benchmarks](https://open-neuromorphic.org/)
    *   **Concept:** The 7900 XTX runs SNNs as a parallel matrix math engine — fast for training but energy-hungry. The Akida AKD1000 processes events asynchronously with near-zero idle power — perfect for inference. Understanding when each paradigm wins. Profiling energy, latency, memory, and throughput.
    *   **Objective:** Profile a trained SNN on both GPU and Akida (or Akida simulator). Measure: wall-clock latency, memory footprint, and (estimated) energy per inference. Understand the NPU mesh architecture (20 NPUs in AKD1000). Map your lesson 10 network onto the NPU mesh.
    *   **Akida-Absent Fallback:** If Akida hardware is not detected, use the Akida simulator for all experiments. Note the simulator-only status in `journal.md` and skip hardware energy measurements (use published specs instead).
    *   **Quiz Topics:** Clock-driven (GPU) vs. event-driven (NPU) execution, why SNNs are inefficient on GPUs but efficient on neuromorphic chips, the NPU mesh and how layers map to it, NeuroBench metrics, when to use GPU vs. NPU.

### Part 4: Advanced Architectures & Deployment

Part 4 covers the full breadth of modern SNN research: recurrence, local learning, model conversion, hardware export, attention mechanisms, and the bleeding edge.

*   **`13_recurrent_snn.py`**: **Long-Term Temporal Memory**
    *   **Theory:** [Bellec et al. (2020)](https://www.nature.com/articles/s41467-020-17236-y) Sec. on ALIF (Adaptive LIF) neurons | Eshraghian et al. (2023) Sec. V (recurrent SNNs)
    *   **Practice:** [snnTorch Regression Tutorial Part II](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html) (recurrent feedback)
    *   **Concept:** Feedforward SNNs have limited temporal memory (one decay constant). Recurrent connections (`snn.RLeaky`, `snn.RSynaptic`) and Spiking LSTMs (`snn.SLSTM`) extend memory horizons. Adaptive LIF (ALIF) neurons dynamically adjust their threshold based on firing history — a key mechanism in e-prop.
    *   **Objective:** Implement `snn.RLeaky` and `snn.SLSTM` on Sequential MNIST (pixels fed one-at-a-time). Compare temporal memory capacity. Explore how recurrence depth affects accuracy on long sequences. Optionally implement a simple ALIF mechanism.
    *   **Quiz Topics:** Why feedforward SNNs forget quickly, how recurrent connections extend memory, RLeaky vs. SLSTM tradeoffs, the ALIF threshold adaptation mechanism, vanishing gradients in recurrent SNNs, comparison with classical LSTMs.

*   **`14_loss_and_regularization.py`**: **Sparsity & Power Efficiency**
    *   **Theory:** [snnbook.net](https://snnbook.net/) — loss functions and regularization | [snntorch.functional docs](https://snntorch.readthedocs.io/en/latest/snntorch.functional.html)
    *   **Concept:** In neuromorphic hardware, every spike costs energy. Regularization techniques (L1/L2 on spike counts, activity budgets) trade accuracy for power efficiency. The Energy-Accuracy Pareto frontier. Understanding SynOps (Synaptic Operations) as the hardware-relevant cost metric.
    *   **Objective:** Add spike regularization to a trained SNN. Sweep the regularization strength and plot the accuracy vs. spike count Pareto frontier. Calculate SynOps for each operating point. Identify the "sweet spot" for edge deployment.
    *   **Quiz Topics:** Why spike count correlates with energy on neuromorphic hardware, L1 vs. L2 regularization effects on spike patterns, the SynOps metric, activity budgets, how regularization interacts with surrogate gradient training, dead neurons from over-regularization.

*   **`15_stdp_online_learning.py`**: **STDP, e-prop & Local Learning Rules**
    *   **Theory:** [Neuronal Dynamics Ch. 19](https://neuronaldynamics.epfl.ch/online/Ch19.html) (Spike-Timing Dependent Plasticity) | [Bellec et al. (2020) — e-prop](https://www.nature.com/articles/s41467-020-17236-y) | [Frenkel — e-prop PyTorch](https://github.com/ChFrenkel/eprop-PyTorch)
    *   **Concept:** BPTT requires storing all timesteps in memory and computing gradients globally. Biological brains learn locally: STDP strengthens synapses where a presynaptic spike *precedes* a postsynaptic spike (Hebb's rule with timing). E-prop bridges local and global by maintaining eligibility traces that approximate BPTT gradients using only local information. These are essential for on-device learning where BPTT is infeasible.
    *   **Note:** snnTorch does not natively support STDP. This lesson implements STDP from scratch in PyTorch to build understanding, then studies e-prop as the practical bridge to on-device learning.
    *   **Objective:** Implement a minimal STDP rule in raw PyTorch (exponential time window, additive weight updates). Apply it to an unsupervised feature extraction task. Study Frenkel's e-prop implementation. Compare STDP, e-prop, and BPTT in terms of memory, compute, and biological plausibility.
    *   **Quiz Topics:** The STDP time window and why timing direction matters, Hebbian vs. anti-Hebbian learning, why BPTT is biologically implausible, how eligibility traces work in e-prop, the memory advantage of local learning, where STDP and e-prop fall short vs. BPTT.

*   **`16_ann_to_snn.py`**: **Deep Model Conversion**
    *   **Theory:** Jiang et al. (2023) — Unified ANN-SNN Conversion Framework (ICML) | [Akida & cnn2snn documentation](https://doc.brainchipinc.com/)
    *   **Concept:** Instead of training SNNs from scratch, convert a pre-trained ANN. The key insight: ReLU activation rates approximate spike firing rates over time. Challenges: threshold balancing (setting neuron thresholds to match ANN activation ranges), weight rescaling, and the accuracy-latency tradeoff (more timesteps = higher accuracy but slower inference).
    *   **Objective:** Convert a pre-trained VGG-16 (or smaller model) to SNN using Akida's `cnn2snn` pipeline. Apply threshold balancing. Measure accuracy as a function of `num_steps`. Compare with a directly-trained SNN from earlier lessons.
    *   **Quiz Topics:** Why ReLU-to-spike conversion works, threshold balancing and why it matters, the accuracy-latency tradeoff curve, when to convert vs. train from scratch, limitations of conversion (temporal dynamics are lost), quantization effects.

*   **`17_energy_and_export.py`**: **SynOps & Cross-Platform Export**
    *   **Theory:** [NIR: Neuromorphic Intermediate Representation](https://github.com/neuromorphs/NIR) | NIR paper (Nature Communications, 2024)
    *   **Concept:** Measuring the true computational cost of SNNs via Synaptic Operations (SynOps) rather than FLOPs. Exporting trained models to NIR format for cross-platform deployment. NIR connects 7 simulators (snnTorch, Norse, SpikingJelly, Lava, Nengo, Rockpool, Sinabs) and 4 hardware platforms (Loihi 2, Speck, SpiNNaker2, Xylo).
    *   **Objective:** Calculate SynOps for your trained models. Export a model using `snntorch.export_nir`. Verify the NIR graph structure. Discuss which hardware targets the model could run on and what adaptations would be needed.
    *   **Quiz Topics:** SynOps vs. FLOPs as cost metrics, why NIR exists (fragmentation problem), what the NIR graph represents, which neuron models NIR supports, limitations of cross-platform export, the role of `torch.fx` tracing.

*   **`18_spiking_transformers.py`**: **Spiking Self-Attention**
    *   **Theory:** [Spikformer — Zhou et al. (2023)](https://arxiv.org/abs/2209.15425) | Yao et al. (2023) — Spike-Driven Transformer | SpikingResformer (CVPR 2024) — Dual Spike Self-Attention
    *   **Stability Warning:** Spiking Transformers are an active research area without stable, pip-installable implementations. This lesson requires building components from paper descriptions. Expect to reference source code from paper repositories.
    *   **Concept:** Standard self-attention uses expensive matrix multiplications. Spiking Self-Attention (SSA) replaces Q, K, V with binary spike tensors, making attention addition-only (87x lower energy). The tradeoff: reduced representational capacity. How Spike-Driven Transformer V2 and SpikingResformer address this.
    *   **Objective:** Implement a minimal SSA module from the Spikformer paper. Integrate it into a spiking vision transformer. Train on CIFAR-10. Profile memory and compute vs. a standard attention baseline.
    *   **Quiz Topics:** How SSA differs from standard attention, why binary Q/K/V enables addition-only compute, the accuracy gap and what causes it, Spike-Driven Transformer improvements, when spiking attention makes sense vs. standard attention, implications for neuromorphic hardware.

*   **`19_akida_deployment.py`**: **Edge AI Inference**
    *   **Theory:** [Akida documentation & Model Zoo](https://doc.brainchipinc.com/) | SpikeFit (2025) — quantization + pruning for neuromorphic deployment
    *   **Concept:** Moving from GPU simulation to real neuromorphic hardware. Quantization-Aware Training (QAT) reduces weights to 1, 2, or 4-bit precision while maintaining accuracy. The `.fbz` format for Akida deployment. Profiling on-chip power, latency, and throughput.
    *   **Objective:** Perform QAT on a model using the Akida/MetaTF pipeline. Quantize to 4-bit and 2-bit. Measure accuracy degradation at each bit width. Deploy as `.fbz` to the AKD1000 M.2 card (or simulator). Profile real inference metrics.
    *   **Akida-Absent Fallback:** If hardware is not available, deploy to the Akida simulator and use published power specifications for energy estimates.
    *   **Quiz Topics:** Why low-bit quantization is essential for neuromorphic chips, QAT vs. post-training quantization, the `.fbz` format, how the AKD1000 NPU mesh processes quantized events, accuracy vs. bit-width tradeoff, comparison with GPU INT8 inference.

*   **`20_spiking_mamba_ssm.py`**: **Spiking State Space Models (Frontier Research)**
    *   **Theory:** [SpikeMamba (2024)](https://arxiv.org/abs/2404.01198) | P-SpikeSSM (ICLR 2025) — probabilistic spiking SSMs | [SpikeGPT (2023)](https://arxiv.org/abs/2302.13939) — spiking language models
    *   **Stability Warning:** This is bleeding-edge 2025-2026 research. Implementations may be incomplete, unstable, or require manual reproduction from papers. Expect to work directly from arxiv code repositories.
    *   **Concept:** State Space Models (Mamba, S4) achieve near-Transformer performance with O(1) inference complexity via recurrent state evolution. Combining SSM dynamics with spiking neurons yields models that are both long-range capable and hardware-friendly. This represents the convergence of two major research threads: efficient sequence modeling and neuromorphic computing.
    *   **Objective:** Study the SpikeMamba architecture. Implement a simplified spiking SSM layer. Evaluate on a sequential task (Sequential MNIST or SHD). Discuss how P-SpikeSSM's probabilistic approach differs from deterministic spiking SSMs. Reflect on where the field is heading.
    *   **Quiz Topics:** What SSMs are and why Mamba matters, how spiking neurons fit into the SSM framework, O(1) inference complexity and why it matters for edge, deterministic vs. probabilistic spiking SSMs, SpikeGPT and the path to spiking language models, open research problems.

### Capstone Project: End-to-End Neuromorphic Pipeline

**You have now covered the full landscape of modern SNN research.** The capstone project integrates everything into a single pipeline from data to deployed model.

*   **`capstone_project.py`**: **From Sensor to Silicon**
    *   **Task:** Build a complete neuromorphic pipeline:
        1.  **Data:** Choose a neuromorphic dataset (DVS128 Gesture, SHD, or CIFAR10-DVS).
        2.  **Model:** Design and train a deep SNN (convolutional, recurrent, or hybrid). Justify your architecture choices.
        3.  **Regularize:** Apply spike regularization to optimize the energy-accuracy tradeoff.
        4.  **Export:** Export via NIR format.
        5.  **Deploy:** Quantize (QAT) and deploy to Akida (or simulator).
        6.  **Report:** Measure and report: test accuracy, total spike count, SynOps, estimated energy per inference, and latency. Compare GPU vs. Akida numbers.
        7.  **Write-up:** A comprehensive analysis in `journal.md` covering design decisions, results, and lessons learned.

---

## Neuromorphic Hardware Stack

*   **Training (The Muscle):** **AMD Radeon RX 7900 XTX** (ROCm 7.2.1). 24GB VRAM for deep temporal unrolling.
*   **Deployment (The Brain):** **BrainChip Akida AKD1000** M.2 card. 20 NPUs, event-driven inference, microwatt idle power.
*   **Sensing (The Eyes):** **v2e Simulation** of Dynamic Vision Sensors. Future: Prophesee GENX320 or similar.

---

## Companion Software Ecosystem

While snnTorch is the primary tool, the broader ecosystem matters for context:

| Tool | Role | Used In |
|------|------|---------|
| [snnTorch](https://github.com/jeshraghian/snntorch) | Primary SNN framework (PyTorch-based) | All lessons |
| [Tonic](https://tonic.readthedocs.io/) | Neuromorphic dataset loading & transforms | Lessons 04, 11 |
| [NIR](https://github.com/neuromorphs/NIR) | Cross-platform model export | Lesson 17 |
| [Akida/MetaTF](https://doc.brainchipinc.com/) | Quantization & Akida deployment | Lessons 12, 16, 19 |
| [v2e](https://github.com/SensorsINI/v2e) | Event camera simulation | Lesson 04 |
| [Norse](https://github.com/norse/norse) | Alternative SNN framework (comparison) | Reference only |
| [SpikingJelly](https://github.com/fangwei123456/spikingjelly) | Alternative with ANN2SNN & STDP | Reference only |
| [Lava](https://github.com/lava-nc/lava) | Intel's Loihi framework | Reference only |
