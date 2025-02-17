import numpy as np
import soundfile as sf
import os
from encoder import RPE_frame_st_coder
from decoder import RPE_frame_st_decoder
import matplotlib.pyplot as plt


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

    #encode
    LARc, residual = RPE_frame_st_coder(frame)

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

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 5))

# Plot the original signal
axs[0].plot(s, label="Original Signal", linewidth=2)
axs[0].set_title("Original Signal")
axs[0].set_xlabel("Samples")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# Plot the reconstructed signal
axs[1].plot(decoded_signal, label="Reconstructed Signal", linewidth=2)
axs[1].set_title("Reconstructed Signal")
axs[1].set_xlabel("Samples")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)

# Display the legend and title
for ax in axs:
    ax.legend()

# Add a title for the entire figure
plt.suptitle("Original vs Reconstructed Short Term Analysis", fontsize=16)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()