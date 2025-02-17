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
    index_matrix = np.abs(np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1)) # indexing for R
    R = acf[index_matrix]
    r = acf[1:9]
    w = np.linalg.solve(R, r)
    w = np.concatenate(([1], w))
    # calculating refl coeffs and LAR
    refl_coeffs = polynomial_coeff_to_reflection_coeff(w)
    
    abs_r = np.abs(refl_coeffs)
    # conditions indexes
    cond1 = abs_r < 0.675
    cond2 = (abs_r >= 0.675) & (abs_r < 0.950)
    cond3 = (abs_r >= 0.950) & (abs_r <= 1.000)
    LAR = np.empty(8)
    LAR[cond1] = refl_coeffs[cond1]  # Condition 1
    LAR[cond2] = np.sign(r[cond2]) * (2 * abs_r[cond2] - 0.675)  # Condition 2
    LAR[cond3] = np.sign(r[cond3]) * (8 * abs_r[cond3] - 6.375)  # Condition 3

    # quantization of LAR
    A = np.array([20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824])
    B = np.array([0.0, 0.0, 4.0, -5.0, 0.184, -3.5, -0.666, -2.235])
    minLARc = np.array([-32, -32, -16, -16, -8, -8, -4, -4])
    maxLARc = np.array([31, 31, 15, 15, 7, 7, 3, 3])
    LARcc = A * LAR + B
    LARc = np.round(LARcc + np.sign(LARcc) * 0.5)
    LARc[LARc < minLARc] = minLARc[LARc < minLARc]
    LARc[LARc > maxLARc] = maxLARc[LARc > maxLARc]

    # calculating the recidual
    # for now we are not gonna do the linear interpolation with the previous frame
    coeffs = np.concatenate((np.array([1]), -LARc)) ## what are the coeffs here??
    curr_frame_st_resd = lfilter(coeffs, 1, s) # FIR

    return LARc, curr_frame_st_resd

def RPE_frame_slt_coder(s0: np.ndarray, prev_frame_st_resd: np.ndarray = None):
    # curr_frame_st_resd_d is d(n) of current frame
    # prev_frame_st_resd is d'(n) of previous frame
    # curr_frame_st_resd will be d'(n) of current frame
    Nc = []
    bc = []
    curr_frame_ex_full = np.zeros(160)
    LARc, curr_frame_st_resd_d = RPE_frame_st_coder(s0)
    if prev_frame_st_resd is None: # if first frame of voice sample
        prev_frame_st_resd = np.zeros(160) 

    both_frame_resd = np.concatenate((prev_frame_st_resd, np.zeros(160)))
    for j in range(4):
        i = j * 40 # starting subframe index for d(n)
        l = j * 40 + 40 # starting subframe index for d'(n) (previous+current)
        p = j * 40 + 160 # starting subframe of current frame  for d'(n)
        temp_d = curr_frame_st_resd_d[i:i+40] # d(n)
        temp_prev_d = both_frame_resd[l:l+120] # d'(n)
        N, b = RPE_subframe_slt_lte(temp_d, temp_prev_d)
        Nc.append(N) # quantization and coding of N,b
        if b <= 0.2:
            bc.append(0)
        elif (b <= 0.5) & (b > 0.2):
            bc.append(1)
        elif (b <= 0.8) & (b > 0.5):
            bc.append(2)
        else:
            bc.append(3)
        
        curr_frame_ex_full[i:i+40] = temp_d - bc[-1] * both_frame_resd[p-Nc[-1]: p-Nc[-1]+40]
        both_frame_resd[p:p+40] = curr_frame_ex_full[i:i+40] + b * both_frame_resd[p-N:p-N+40]

    curr_frame_st_resd = both_frame_resd[160:320]
    return LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd


def RPE_subframe_slt_lte(
    d: np.ndarray,
    prev_d: np.ndarray
):
    # Define possible lag values (40 ≤ λ ≤ 120)
    lag_range = np.arange(40, 121)

    # Compute cross-correlation R(λ) for each λ
    R_values = np.array([
        np.sum(d * prev_d[120-lag:160-lag]) for lag in lag_range
    ])

    # Find the lag N that maximizes cross-correlation
    N = lag_range[np.argmax(R_values)]

    # Compute gain factor b using the formula
    numerator = np.sum(d * prev_d[120-N:160-N])
    denominator = np.sum(prev_d[120-N:160-N] ** 2)

    b = numerator / denominator if denominator != 0 else 0  # Avoid division by zero
    return N, b