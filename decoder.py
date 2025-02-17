import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff

def RPE_frame_st_decoder(curr_frame_st_resd: np.ndarray, LARc: np.ndarray):

    # dequantization of LAR
    A = np.array([20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824])
    B = np.array([0.0, 0.0, 4.0, -5.0, 0.184, -3.5, -0.666, -2.235])
    LAR = (LARc - B) / A

    # calculating refl coeffs
    abs_LAR = np.abs(LAR)
    # conditions indexes
    cond1 = abs_LAR < 0.675
    cond2 = (abs_LAR >= 0.675) & (abs_LAR < 1.225)
    cond3 = (abs_LAR >= 1.225) & (abs_LAR <= 1.625)
    
    refl_coeffs = np.empty_like(LAR)
    refl_coeffs[cond1] = LAR[cond1]  # Condition 1
    refl_coeffs[cond2] = np.sign(LAR[cond2]) * (0.5 * abs_LAR[cond2] + 0.3375)  # Condition 2
    refl_coeffs[cond3] = np.sign(LAR[cond3]) * (0.125 * abs_LAR[cond3] + 0.796875)  # Condition 3

    #calculating polynomial coeffs
    poly_coeffs, _ = reflection_coeff_to_polynomial_coeff(refl_coeffs)

    #reconstructing the signal from residual
    b = np.array([1.0])
    a = np.concatenate((np.array([1.0]), - poly_coeffs[1:]))
    s = lfilter(b, a, curr_frame_st_resd)

    # post-processing
    beta = 28180 * pow(2,-15)
    a1 = [1, -beta]
    b1 = [1]
    s0 = lfilter(b1, a1, s)

    return s0

def RPE_frame_slt_decoder(LARc, Nc, bc, curr_frame_ex_full, prev_frame_st_resd=None):

    # decode bc values to gain b
    b_values = [0.0, 0.3, 0.7, 1.0]  # Approximate mapped values for bc
    bc_decoded = np.array([b_values[b] for b in bc])

    if prev_frame_st_resd is None:
        prev_frame_st_resd = np.zeros(160)  # first frame case

    # previous + current frame residuals
    both_frame_resd = np.concatenate((prev_frame_st_resd, np.zeros(160)))
    curr_frame_st_resd = np.zeros(160)

    # Loop through subframes
    for j in range(4):
        i = j * 40  # subframe index for d'(n) in current frame
        p = 160 + j * 40  # subframe index in previous + current frame residuals

        # reconstruct long-term residual
        curr_frame_st_resd[i:i+40] = curr_frame_ex_full[i:i+40] + bc_decoded[j] * both_frame_resd[p-Nc[j]: p-Nc[j]+40]
        both_frame_resd[p:p+40] = curr_frame_st_resd[i:i+40]

    # short term decoding
    s0 = RPE_frame_st_decoder(curr_frame_st_resd, LARc)

    return s0, curr_frame_st_resd