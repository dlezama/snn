# 📓 SNN Mastery Journal: A Learning Journey

This journal documents my progress through the [snnTorch](https://github.com/jeshraghian/snntorch) curriculum, capturing key insights, technical milestones, and exploratory side quests.

---

## 🛠️ Hardware & Environment Setup
*   **Operating System:** Windows (win32)
*   **Training GPU:** AMD Radeon RX 7900 XTX (24GB VRAM)
*   **Environment:** Python 3.12 (via `uv`) with ROCm 7.2.1 support.
*   **Deployment Hardware:** BrainChip Akida AKD1000 M.2 Card.
*   **Sensing Strategy:** `v2e` (Video-to-Events) Simulation.

---

## 📖 Log Entries

### [2026-04-02] - The Neuromorphic Reset
*   **Status:** Initialized Curriculum & Teacher Mandates.
*   **Key Concepts:**
    *   **Temporal vs. Spatial:** Learned that SNNs process information in time, not just space.
    *   **The Silicon Retina:** Discovered the link between DVS cameras and biological eye processing (asynchronous events vs. synchronous frames).
    *   **The Leaky Bucket:** Introduced to the Leaky Integrate-and-Fire (LIF) model.
*   **Side Quests:**
    *   Explored the capabilities of the BrainChip Akida AKD1000 for ultra-low-power deployment.
    *   Discussed the integration of `v2e` for event-based simulation on the 7900 XTX.
*   **Milestones:**
    *   [x] Repo initialized and curriculum refined for high-end AMD hardware.
    *   [ ] Lesson 01: The "Hello World" of SNNs (In Progress).

---

## 🚀 Side Quest Notes
*   **Akida Integration:** Plan to use the Akida card earlier in the course for "Hardware Cross-Checks" (Lessons 03, 06, 13).
*   **High-Resolution Scaling:** Modified the course to include Spiking ResNets and Transformers to leverage the 24GB of VRAM on the 7900 XTX.
