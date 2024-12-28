import numpy as np


def RPE_frame_st_coder(s0: np.ndarray):
    #Preprocessing
    s_preprocessed = pre_emphasis(offset_compensation(s0))

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

