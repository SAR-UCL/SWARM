import pywt
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', None) #or 10 or None


#https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html

path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/in-flight data/LP/test/'
load_data = path + 'test-single-pass2022-01-24.csv'
df = pd.read_csv(load_data)
data = df["Ne"].values

WAVELET_FAMILY = "haar"

#print(df)

def wavelet_forward(data):
    #2D discrete wavelet transform
    return pywt.dwt2(data.reshape(-1, 1), WAVELET_FAMILY)

def wavelet_inverse(coefficients):
    return pywt.idwt2(coefficients, WAVELET_FAMILY)[:,0].reshape(-1)

def thresholding(data):
    # original
    #plt.plot(data, color="k", label="original")

    # sanity check
    assert_allclose(data, wavelet_inverse(wavelet_forward(data)), rtol=1e-15)

    approximation, (horizontal, zero_vertical, zero_diagonal) = wavelet_forward(data)
    thresholded = np.where(approximation < 1e6, 0, approximation)
    #New coefficients
    new_coefficients = approximation, (horizontal, zero_vertical, zero_diagonal)
    modified_signal = wavelet_inverse(new_coefficients)
    #print(modified_signal)
    plt.plot(modified_signal, label="thresholded")
    # assert_array_equal(data, modified_signal)
    plt.legend()
    #plt.show()
 

if __name__ == "__main__":
    thresholding(data)