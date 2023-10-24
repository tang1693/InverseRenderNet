import os
import pickle as pk
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image


np.set_printoptions(threshold=np.inf)
file_list = glob.glob("Data/037/037_0001.pk")

if file_list:
    with open(file_list[0], 'rb') as f:
        batch_data = pk.load(f)
    
    input_data = np.float32(batch_data['input'])
    input_data = input_data / 255.0  # Normalize the input data from [0, 255] to [0, 1] range
    dm = batch_data['dm']
    nm = np.float32(batch_data['nm'])
    nm = (nm + 255) / 510.0  # Normalize the range from [-255, 255] needs to change to [0, 1]
    mask = np.float32(batch_data['mask'])

    # Create Raw directory if it doesn't exist
    if not os.path.exists("Data/037/Raw"):
        os.makedirs("Data/037/Raw")

   # Save each data to a separate text file
    with open("Data/037/Raw/input_data.txt", 'w') as txt_file:
        txt_file.write(str(input_data))
    
    with open("Data/037/Raw/dm_data.txt", 'w') as txt_file:
        txt_file.write(str(dm))

    with open("Data/037/Raw/nm_data.txt", 'w') as txt_file:
        txt_file.write(str(nm))

    with open("Data/037/Raw/mask_data.txt", 'w') as txt_file:
        txt_file.write(str(mask))

    with open("Data/037/Raw/misc_data.txt", 'w') as txt_file:
        for key, value in batch_data.items():
            if key not in ['input', 'dm', 'nm', 'mask']:
                txt_file.write(f"{key}:\n{value}\n\n")




    # Loop through and visualize each Input image
    for i, img in enumerate(input_data):
        plt.imshow(img)
        plt.title(f'Input RGB Image {i + 1}')
        plt.axis('off')
        plt.savefig(f"Data/037/Raw/input_image_{i + 1}.png")
        plt.show()

    # Loop through and visualize each Depth Map
    for i, depth in enumerate(dm):
        plt.imshow(depth, cmap='gray')
        plt.title(f'Depth Map {i + 1}')
        plt.axis('off')
        plt.savefig(f"Data/037/Raw/depth_map_{i + 1}.png")
        plt.show()

    # Loop through and visualize each Normal Map
    for i, normal in enumerate(nm):
        plt.imshow(normal)
        plt.title(f'Normal Map {i + 1}')
        plt.axis('off')
        plt.savefig(f"Data/037/Raw/normal_map_{i + 1}.png")
        plt.show()

    # Loop through and visualize each Mask
    for i, m in enumerate(mask):
        plt.imshow(m, cmap='gray')
        plt.title(f'Mask Image {i + 1}')
        plt.axis('off')
        plt.savefig(f"Data/037/Raw/mask_image_{i + 1}.png")
        plt.show()

else:
    print("No .pk files found in Data/037/ directory")
