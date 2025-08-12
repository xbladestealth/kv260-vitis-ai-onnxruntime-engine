from PIL import Image
import numpy as np


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img = np.array(img, dtype=np.float32) / 255.0
    return img.reshape(1, 1, 28, 28)  # [batch, channels, height, width]


img = preprocess_image("digit.png")
np.save("input.npy", img)
