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
    
    refl_coeffs = np.empty(8)
    refl_coeffs[cond1] = LAR[cond1]  # Condition 1
    refl_coeffs[cond2] = np.sign(LAR[cond2]) * (0.5 * abs_LAR[cond2] + 0.3375)  # Condition 2
    refl_coeffs[cond3] = np.sign(LAR[cond3]) * (0.125 * abs_LAR[cond3] + 0.796875)  # Condition 3

    #calculating polynomial coeffs
    poly_coeffs = reflection_coeff_to_polynomial_coeff(refl_coeffs)
    
    #reconstructing the signal from residual
    b = 1
    a = np.concatenate([1], - poly_coeffs)
    s = lfilter(b, a, curr_frame_st_resd)

    # post-processing
    beta = 28180 * pow(2,-15)
    a1 = [1 -beta]
    b1 = 1
    s0 = lfilter(b1, a1, s)

    return s0