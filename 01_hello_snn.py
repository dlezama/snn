import torch
import snntorch as snn

def hello_snn():
    # 1. Define the LIF neuron parameters
    # beta is the membrane potential decay rate
    # reset_mechanism="zero" resets the membrane potential to 0 after a spike
    beta = 0.9  
    lif = snn.Leaky(beta=beta, reset_mechanism="zero")

    # 2. Create an interleaved spike train (10 steps of 1-0-1-0...) followed by 15 steps of silence
    num_steps = 25
    # Create [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    interleaved = torch.tensor([1.0, 0.0] * 5)
    # Combine with 15 zeros
    input_spikes = torch.cat((interleaved, torch.zeros(15))).reshape(num_steps, 1, 1)

    # 3. Initialize membrane potential
    mem = lif.init_leaky()

    # 4. Simulation loop
    print(f"Simulating {num_steps} time steps...")
    spk_count = 0
    for step in range(num_steps):
        spk, mem = lif(input_spikes[step], mem)
        if spk > 0:
            spk_count += 1
            print(f"Step {step:2d}: Spike Out! (Mem: {mem.item():.3f})")
        else:
            print(f"Step {step:2d}: No spike.  (Mem: {mem.item():.3f})")

    print("-" * 30)
    print(f"Total output spikes: {spk_count}")
    print("Hello SNN World! snntorch is working correctly.")

if __name__ == "__main__":
    hello_snn()
