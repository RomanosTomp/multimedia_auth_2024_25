import numpy as np
import soundfile as sf
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
