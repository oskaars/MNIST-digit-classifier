import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os



def Formatter():
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformer)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    return loader
print("Dane przygotowane!")


class NumAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()


        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # x to paczka obrazków o wymiarach [64, 1, 28, 28]

        # spłaszcza pixele do formatu [64, 784]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output
model = NumAnalyzer()
print("model ready")


def save_model(model, filename="mnist_brain.pth"):
    torch.save(model.state_dict(), filename)
    print(f"saved to file: {filename}")

def load_model(model, filename="mnist_brain.pth"):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename, weights_only=True))
        print(f"loaded from file: {filename}")
        return True
    else:
        print("model is empty.")
        return False

def trainingLoop(epochs=3):
    print("training loop start")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = Formatter()
    for epoch in range(epochs):
        error_sum = 0.0
        for images, labels in loader:

            #optimizer reset
            optimizer.zero_grad()

            #forward pass (passing data into the net)
            outputs = model(images)

            #loss count
            loss = criterion(outputs, labels)

           #back propagation
            loss.backward()

            #updates weights and biases
            optimizer.step()


            error_sum += loss.item()

        avg_error = error_sum / len(loader)
        print(f"Epoch [{epoch + 1}/{epochs}] | avg loss: {avg_error:.4f}")
def tests():

    loader = Formatter()
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Sprawdźmy wymiary paczki (powinno być [64, 1, 28, 28])
    print(f"Wymiary paczki obrazków: {images.shape}")

    test_output = model(images)
    print(f"Kształt wyniku sieci: {test_output.shape}")

    plt.imshow(images[0].numpy().squeeze(), cmap='gray')
    plt.title(f"Prawdziwa etykieta: {labels[0]}")
    plt.show()
#tests()

def Inference(loader):
    print("\nInference start")

    model.eval()

    data_iter = iter(loader)
    images, labels = next(data_iter)

    #turn off gradient following
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)


    pierwszy_obrazek = images[0].numpy().squeeze()
    prawdziwa_etykieta = labels[0].item()
    werdykt_ai = predictions[0].item()

    plt.imshow(pierwszy_obrazek, cmap='gray')

    kolor = 'green' if prawdziwa_etykieta == werdykt_ai else 'red'

    plt.title(f"AI widzi: {werdykt_ai} | Prawda: {prawdziwa_etykieta}", color=kolor, fontsize=14)
    plt.show()

loader = Formatter()

if not load_model(model):
    trainingLoop(5)
    save_model(model)

Inference(loader)

