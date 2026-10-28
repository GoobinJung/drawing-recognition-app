import sys
from importlib.metadata import version

import numpy as np
import torch
from PIL import Image
from flask import Flask

print("Python:", sys.version.split()[0])

for package in ["torch", "numpy", "pillow", "flask"]:
    print(f"{package}: {version(package)}")

numbers = np.array([1, 2, 3], dtype=np.float32)
tensor = torch.from_numpy(numbers)
print("Tensor calculation:", tensor * 2)

image = Image.new("L", (28, 28))
print("Image size:", image.size)

app = Flask(__name__)
print("Flask app created:", app.name)

print("Environment check passed.")