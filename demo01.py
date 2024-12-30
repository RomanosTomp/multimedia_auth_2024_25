import numpy as np
import soundfile as sf
import os
from encoder import RPE_frame_st_coder
from decoder import RPE_frame_st_decoder


#Read file
input_file = 'ena_dio_tria.wav'
output_file = 'ena_dio_tria_reconstructed.wav'

#Data
s, fs = sf.read(input_file)

#Frame edit
frame_size = 160
frames = [s[i:i+frame_size] for i in range(0, len(s), frame_size)]
 
decoded_frames = []

for frame in frames:
    if len(frame) < frame_size:
        frame = np.pad(frame, (0, frame_size - len(frame)))

    #Encode
    LARc, residual = RPE_frame_st_coder(frame)
    #print(LARc)
    #print(residual)

    #decode
    decoded_frame = RPE_frame_st_decoder(residual, LARc)

    decoded_frames.append(decoded_frame)

decoded_signal = np.hstack(decoded_frames)

sf.write(output_file, decoded_signal, fs)

print("Done file is saved as", output_file)
# Get the current directory where the script is located
current_directory = os.getcwd()

# List all files in the directory
files_in_directory = os.listdir(current_directory)

# Iterate over the files and get their sizes
print(f"Files in '{current_directory}':\n")
for file_name in files_in_directory:
    full_path = os.path.join(current_directory, file_name)
    if os.path.isfile(full_path):  # Check if it's a file
        file_size = os.path.getsize(full_path)
        print(f"{file_name}: {file_size / 1024:.2f} KB")  # Convert bytes to KB