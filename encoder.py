import numpy as np
from scipy.signal import lfilter
from hw_utils import polynomial_coeff_to_reflection_coeff


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
    s = lfilter(b2, a2, offset_compensated)

    #Short-term analysis
    # finding ACFs
    acf = np.zeros(9)
    for k in range(9):
        for i in range(k,160):
            acf[k] += s[i]*s[i-k]
    # constructing R and r to find w
    n = len(acf) - 1
    index_matrix = np.abs(np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1))
    R = acf[index_matrix]
    r = acf[1:9]
    w = np.linalg.solve(R, r) # maybe not linalg (?)

    # calculating refl coeffs and LAR
    refl_coeffs = polynomial_coeff_to_reflection_coeff(w)
    abs_r = np.abs(refl_coeffs)
    # conditions indexes
    cond1 = abs_r < 0.675
    cond2 = (abs_r >= 0.675) & (abs_r < 0.950)
    cond3 = (abs_r >= 0.950) & (abs_r <= 1.000)
    LAR = np.empty(8)
    print(refl_coeffs[cond1])
    LAR[cond1] = refl_coeffs[cond1]  # Condition 1
    LAR[cond2] = np.sign(r[cond2]) * (2 * abs_r[cond2] - 0.675)  # Condition 2
    LAR[cond3] = np.sign(r[cond3]) * (8 * abs_r[cond3] - 6.375)  # Condition 3

    # quantization of LAR
    A = np.array([20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824])
    B = np.array([0.0, 0.0, 4.0, -5.0, 0.184, -3.5, -0.666, -2.235])
    minLARc = np.array([-32, -32, -16, -16, -8, -8, -4, -4])
    maxLARc = np.array([31, 31, 15, 15, 7, 7, 3, 3])
    LARcc = A * LAR + B
    LARc = round(LARcc + np.sign(LARcc) * 0.5)
    LARc[LARc < minLARc] = minLARc[LARc < minLARc]
    LARc[LARc > maxLARc] = maxLARc[LARc > maxLARc]

    # calculating the recidual
    # for now we are not gonna do the linear interpolation with the previous frame
    coeffs = np.concatenate((np.array([1]), -LARc))
    curr_frame_st_resd = lfilter(coeffs, 1, s) # FIR

    return LARc, curr_frame_st_resd
