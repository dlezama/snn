import os
import torch
from torchvision import datasets, transforms
from snntorch import utils, spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

# Training Parameters
BATCH_SIZE = 128
DATA_PATH = 'temp'
NUM_CLASSES = 10  # MNIST has 10 output classes
DTYPE = torch.float
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DATA_PATH, exist_ok=True)

def load_dataset():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])

    mnist_train = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    
    subset = 1
    mnist_train = utils.data_subset(mnist_train, subset)
    print(f"The size of mnist_train is {len(mnist_train)}")
    
    train_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} samples")
    return train_loader

def main():
    train_loader = load_dataset()
    data = iter(train_loader)
    data_it, targets_it = next(data)
    data_it = data_it.to(DEVICE, non_blocking=True)

    num_steps = 100

    # Spiking Data
    spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=1)
    print("Data size:", spike_data.size())

    spike_data_sample = spike_data[:, 0, 0]
    print("Sample size:", spike_data_sample.size())

    fig_orig, ax_orig = plt.subplots()
    ax_orig.imshow(data_it[0].squeeze().cpu(), cmap='gray')
    ax_orig.set_title(f"Original Digit: {targets_it[0]}")
    ax_orig.axis('off')
    plt.savefig('temp/original_digit.png', bbox_inches='tight', dpi=100)
    plt.close()

    plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)
    anim.save('temp/animation.mp4')
    plt.close()

    plt.figure(facecolor="w")
    plt.subplot(1,2,1)
    plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 1')

    spike_data2 = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)
    spike_data_sample2 = spike_data2[:, 0, 0]

    plt.subplot(1,2,2)
    plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 0.25')

    plt.savefig('temp/output_digit.png', bbox_inches='tight', dpi=100)
    plt.close()

    # Reshape
    spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))

    # raster plot
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data_sample2, ax, s=1.5, c="black")

    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.savefig('temp/raster_plot.png', bbox_inches='tight', dpi=100)
    plt.close()

    idx = 210  # index into 210th neuron
    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    splt.raster(spike_data_sample.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
    plt.title("Input Neuron " + str(idx))
    plt.xlabel("Time step")
    plt.yticks([])
    plt.savefig('temp/single_neuron.png', bbox_inches='tight', dpi=100)
    plt.close()

    print(f"The corresponding target is: {targets_it[0]}")

if __name__ == "__main__":
    main()
