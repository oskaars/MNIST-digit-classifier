# MNIST Handwritten Digit Classifier (PyTorch)

This project is a complete, from-scratch implementation of an Artificial Neural Network designed to recognize handwritten digits using the **PyTorch** library. It serves as a foundational project in Computer Vision, classifying 28x28 pixel images from the classic MNIST dataset into one of 10 categories (digits 0-9).

## Key Features

* **Custom Multilayer Perceptron (MLP):** A fully connected feed-forward neural network architecture (`784 -> 128 -> 128 -> 10`) utilizing `ReLU` activation functions.
* **Automated Data Pipeline:** Utilizes `torchvision.datasets` and `DataLoader` to automatically download, normalize (-1.0 to 1.0), and batch the dataset for optimized training.
* **Supervised Learning Loop:** Implements a robust training loop using the `Adam` optimizer and `CrossEntropyLoss` for precise classification.
* **Persistence (Save/Load):** The script automatically saves the trained `state_dict` (`mnist_brain.pth`) after training. On subsequent runs, it loads the pre-trained weights, skipping the training phase.
* **Inference & Visualization:** Uses `matplotlib` to randomly sample a test image, run it through the network with gradient tracking disabled (`torch.no_grad()`), and visually display the AI's prediction versus the ground truth.

## 🛠️ Technologies Used

* **Python 3.x**
* **PyTorch** (`torch`, `torch.nn`, `torch.optim`)
* **Torchvision** (Data transformation and MNIST dataset access)
* **Matplotlib** (Visualizing the inference results)

## How to Run

### 1. Install Dependencies
Ensure you have an active virtual environment, then install the required libraries:
```bash
pip install torch torchvision matplotlib
```
2. Run the Script
Simply execute the main file:

```bash
python main.py
```
First Run: The script will download the MNIST dataset, train the network for 5 epochs, save the model to mnist_brain.pth, and display a test result.

Subsequent Runs: The script will detect the saved model, load the weights instantly, and directly output a visual inference test.

Code Architecture
Formatter(): Configures the image transformations and prepares the DataLoader batching system.

NumAnalyzer (nn.Module): Defines the spatial dimensions and forward pass logic of the neural network.

trainingLoop(): Handles the "Big Five" of PyTorch training (zero gradients, forward pass, calculate loss, backward pass, optimizer step).

Inference(): Sets the model to .eval() mode and runs a prediction on a sample batch to verify accuracy.
