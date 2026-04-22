# SNN Mastery Journal

This journal documents progress through the SNN curriculum defined in `README.md`. It captures reading completed, quiz results, technical milestones, and discoveries.

---

## Hardware & Environment Setup

**Training host (Windows)**
*   **OS:** Windows 11 (win32)
*   **GPU:** AMD Radeon RX 7900 XTX (24 GB VRAM)
*   **Environment:** Python 3.12 (via `uv`) with ROCm 7.2.1

**Deployment host (Raspberry Pi 5)**
*   **OS:** Raspberry Pi OS Trixie, kernel `6.12.75+rpt-rpi-2712` (arm64)
*   **CPU-only PyTorch:** `torch 2.9.1+cpu` + snnTorch 0.9.4 in Python 3.12 venv
*   **Akida hardware:** BrainChip AKD1000 M.2, detected as `BC.00.000.002` (hardware, not simulator) on 2026-04-21
*   **Akida stack:** `akida` / `cnn2snn` 2.19.1, TensorFlow 2.19.1
*   **Kernel driver:** `akida_pcie` (built from [Brainchip-Inc/akida_dw_edma](https://github.com/Brainchip-Inc/akida_dw_edma)) — **requires a local Makefile patch** to build on kernel 6.12; see Lesson 00 discoveries below

**Sensing Strategy:** v2e (Video-to-Events) Simulation

---

## Lesson Log

Use this template for each lesson:

```
### Lesson NN: [Title] — [YYYY-MM-DD]
*   **Reading Completed:** [what was read before the lesson]
*   **Quiz Score:** X/Y — [pass/marginal/retry]
*   **Weak Areas:** [topics to revisit, if any]
*   **Code:** `NN_filename.py` — [complete/partial/skipped]
*   **Key Insight:** [one sentence: the most important thing learned]
*   **Discoveries:** [surprises, bugs, connections to other lessons]
```

---

### Lesson 00: Environment & Hardware Verification — 2026-04-21 (pi setup; quiz pending)
*   **Reading Completed:** N/A (setup-only lesson)
*   **Quiz Score:** [pending — to be administered before lesson is marked complete]
*   **Code:** `00_environment_test.py` — ran on Raspberry Pi 5, all checks passed (PyTorch 2.9.1+cpu, snnTorch 0.9.4, Akida 2.19.1, cnn2snn 2.19.1, `Found 1 Akida device(s): BC.00.000.002`)
*   **Key Insight:** AKD1000 detection is gated on an out-of-tree PCIe kernel driver, not just the userspace wheel — without `akida_pcie` loaded, `lspci` sees the card but its memory BARs stay `[disabled]` and `akida.devices()` returns empty.
*   **Discoveries:**
    *   BrainChip's [`akida_dw_edma`](https://github.com/Brainchip-Inc/akida_dw_edma) repo does **not** use DKMS — the repo name refers to the embedded DesignWare eDMA engine, but the install path is a plain `./install.sh` wrapping `make` + `modprobe akida-pcie`. The loaded module is named `akida_pcie`, not `akida_dw_edma`. Our README originally documented a nonexistent DKMS path; corrected in this commit.
    *   The driver Makefile's `AKIDA_KERNEL_VERSION_RANK` table tops out at kernel 6.9. Raspberry Pi OS Trixie ships 6.12, which trips `$(error Kernel 6.12 not supported)` before any C compiles. Fixed with a local patch (see `patches/akida_dw_edma-kernel-6.12.patch`) that extends the table to `[6.9, 6.13)` and maps it to the 5.16 header set. Built cleanly against kernel `6.12.75+rpt-rpi-2712`; two harmless `-Wmissing-prototypes` warnings, no errors.
    *   The Makefile has explicit `CONFIG_ARCH_BCM2835` handling that forces 32-bit PCIe accesses on Pi hardware (BrainChip documents this as a Compute Module 4 workaround). It fires automatically on the Pi 5 kernel and is the reason the `#pragma message "PCIe 64bit accesses forced to 32bit accesses"` appears in the build log. Keep an eye on this for Lesson 12 / Lesson 19 — it may affect peak PCIe DMA throughput numbers on the Pi vs. published AKD1000 benchmarks (which assume 64-bit hosts).
    *   `99-akida-pcie.rules` chmods `/dev/akida0` to 0666. Fine for this single-user Pi; edit the rules file before install if you want stricter access.

---

## Milestone & Capstone Progress

*   **Milestone Project A (Fashion-MNIST):** [not started]
*   **Capstone Project (End-to-End Pipeline):** [not started]

---

## Side Quest Notes

*   **Akida Hardware Cross-Checks:** Planned for Lessons 05 (population coding), 08 (feedforward SNN), 12 (hardware deep dive), 16 (ANN-to-SNN), and 19 (Akida deployment).
*   **High-Resolution Scaling:** Utilizing 24GB VRAM on the 7900 XTX for deep BPTT (lesson 09), Spiking ResNets (lesson 10), and Spiking Transformers (lesson 18).
*   **Audio Spike Data:** Optional exploration of SHD (Spiking Heidelberg Digits) in lesson 11 and SNN audio benchmarks.
