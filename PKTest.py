import pickle as pk
import numpy as np
import glob

# Get a list of all .pk files in the Data/037/ directory
file_list = glob.glob("Data/037/*.pk")

# Check if the list is not empty
if file_list:
    # Open the first .pk file in the list
    with open(file_list[0], 'rb') as f:
        batch_data = pk.load(f)
    
    input_data = np.float32(batch_data['input'])
    dm = batch_data['dm']
    nm = np.float32(batch_data['nm'])
    cam = np.float32(batch_data['cam'])
    scaleX = batch_data['scaleX']
    scaleY = batch_data['scaleY']
    mask = np.float32(batch_data['mask'])

    # Print data structure for verification
    print("Input Data:", input_data)
    print("Depth Map:", dm)
    print("Normal Map:", nm)
    print("Camera Data:", cam)
    print("Scale X:", scaleX)
    print("Scale Y:", scaleY)
    print("Mask:", mask)
else:
    print("No .pk files found in Data/037/ directory")
