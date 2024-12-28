import numpy as np
from scipy.signal import lfilter

    #Preprocessing
    alpha = 32735 * pow(2,-15)
    beta = 28180 * pow(2,-15)

    # Reverse Pre-emphasis: x[k] = y[k] + beta * x[k-1]
    b1 = [1]  # Numerator coefficients
    a1 = [1, -beta]  # Denominator coefficients
    reversed_pre_emphasis = lfilter(b1, a1, pre_emphasized)

    # Reverse Offset Compensation: x[k] = y[k] + x[k-1] - alpha * y[k-1]
    b2 = [1, -1]  # Numerator coefficients
    a2 = [1, -alpha]  # Denominator coefficients
    decoded_signal = lfilter(b2, a2, reversed_pre_emphasis)
