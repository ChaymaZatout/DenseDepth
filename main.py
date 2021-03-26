import numpy as np
from matplotlib import pyplot as plt
import torch
from model import PTModel
from PIL import Image
import time


def my_DepthNorm(x, maxDepth):
    return maxDepth / x


def my_predict(model, image, minDepth=10, maxDepth=4000):
    with torch.no_grad():
        pytorch_input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0)
        # Compute predictions
        predictions = model(pytorch_input)
        # Put in expected range
    return (np.clip(my_DepthNorm(predictions.numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth)[0][0]


# load model:
pytorch_model = PTModel().float()
checkpoint = torch.load("nyu.pth.tar", map_location=torch.device('cpu'))
pytorch_model.load_state_dict(checkpoint['state_dict'])
pytorch_model.eval()

# Input images:
file = "_in/sample.jpg"
input = np.clip(np.asarray(Image.open(file), dtype='float32') / 255, 0, 1)

# Compute results
start = time.time()
output = my_predict(pytorch_model, input)
print(f"Predicted in {time.time() - start} s.")

print(output.shape)
print(f"min: {output.min()}")
print(f"max: {output.max()}")

# # save image:
# im = Image.fromarray(np.uint8(output[0, 0, :, :] * 255))
# im.save("_out/sample_depth.png")

# # display:
# plt.imshow(output[0, 0, :, :])
# plt.show()
