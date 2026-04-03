import torch
import snntorch as snn
import matplotlib.pyplot as plt

def run_lesson_01():
    # 1. Define the Neuron
    # beta is the decay rate. 0.95 means 5% of the membrane potential leaks per step.
    beta = 0.95
    lif = snn.Leaky(beta=beta)

    # 2. Create Input: A single spike at t=10
    num_steps = 200
    input_current = torch.zeros(num_steps)
    input_current[10] = 1.0 

    # 3. Simulation Loop
    mem = lif.init_leaky() # Initialize membrane potential
    mem_history = []
    spike_history = []

    # --- TODO: [Learning Part] ---
    # Loop through each time step in input_current.
    # Pass the current input and previous 'mem' into the 'lif' neuron.
    # Store the resulting spike and new 'mem' in the history lists.
    
    # for x in input_current:
    #     ... your code here ...
    
    # -----------------------------

    # 4. Visualization
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    
    # Plot Input Current
    ax[0].stem(range(num_steps), input_current.numpy(), use_line_collection=True)
    ax[0].set_ylabel("Input Current")
    ax[0].set_title("LIF Neuron Response to a Single Spike")

    # Plot Membrane Potential
    if mem_history:
        ax[1].plot(range(num_steps), [m.item() for m in mem_history])
        ax[1].axhline(y=1.0, color='r', linestyle='--', label="Threshold")
        ax[1].set_ylabel("Membrane Potential (U)")
        ax[1].set_xlabel("Time Step")
        ax[1].legend()

    plt.tight_layout()
    plt.show()
    print("Simulation complete. Check the plot!")

if __name__ == "__main__":
    run_lesson_01()
