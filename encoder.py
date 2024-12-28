import numpy as np
from scipy.signal import lfilter


def RPE_frame_st_coder(s0: np.ndarray):
    
    #Preprocessing
    alpha = 32735 * pow(2,-15)
    beta = 28180 * pow(2,-15)

    # Offset Compensation: y[k] = x[k] - x[k-1] + alpha * y[k-1]
    b1 = [1, -1]  # Numerator coefficients
    a1 = [1, -alpha]  # Denominator coefficients
    offset_compensated = lfilter(b1, a1, s0)
    # Pre-emphasis: y[k] = x[k] - beta * x[k-1]
    b2 = [1, -beta]  # Numerator coefficients
    a2 = [1]  # Denominator coefficients
    pre_emphasized = lfilter(b2, a2, offset_compensated)

    #Short-term analysis
    ak = compute_predictor_coefficients(s_preprocessed) 
    LAR = compute_LAR(ak)
    LARc = quantize_LAR(LAR)

    residual = compute_residual_signal(s_preprocessed, ak)

    return LARc, residual


def offset_compensation(s0: np.ndarray):
    # Offset compensation
    s = s0 - np.mean(s0)
    return s

