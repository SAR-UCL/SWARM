from scipy import signal
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

'''
def butter_highpass(low_cut, high_cut, fs, order=5):
    """
    Design band pass filter.

    Args:
        - low_cut  (float) : the low cutoff frequency of the filter.
        - high_cut (float) : the high cutoff frequency of the filter.
        - fs       (float) : the sampling rate.
        - order      (int) : order of the filter, by default defined to 5.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs

    # design filter
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    # returns the filter coefficients: numerator and denominator
    return b, a 


df = pd.DataFrame({'col1': [1, 3, 10, 23, 5, 10], 'col2': [3, 4, 34, 10, 3, 4]})
df = df.loc[:,'col1']
df_data = np.array(df.values) 
#print(df_data)

#butter = butter_highpass(0.5, 2.5, 100)
#print(len(butter[0]))
#print(butter)

length = 10

b, a = signal.butter(4, length, 'low', analog=True)
w, h = signal.freqs(b, a)

#print(b, a)
#print(w, h)


#plt.plot(w, 20 * np.log10(abs(h)))
plt.plot(w, h)
#plt.plot()
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(length, color='green') # cutoff frequency
#plt.show()

'''
#
#
#
#

from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        sos = butter_bandpass(lowcut, highcut, fs, order)
        y = sosfilt(sos, data)
        return y

#data = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass', output='sos')
#data = np.delete(data, (0), axis=0)
#print(data)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    #fs must be x2 larger than highcut
    fs = 3000
    lowcut = 500.0
    highcut = 1250.0
    order = 3

    # Plot the frequency response for a few different orders.
    #plt.figure(1)
    #plt.clf()
    #sos = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    #print(sos)
    #w, h = sosfreqz(data, worN=1500)
    #print(h)

    
    #plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    
    '''
    for order in [3, 6, 9]:
        #sos = butter_bandpass(lowcut, highcut, fs, order=order)
        sos = butter_bandpass_filter(data, lowcut, highcut, fs, order=order)
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)'''

    #plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
    #         '--', label='sqrt(0.5)')
    
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.legend(loc='best')

    #plt.show()


#####
#####
#####
from scipy.signal import savgol_filter
x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])

window_len = 5
poly = 2

sav = savgol_filter(x, window_len, poly)
print(sav)

plt.figure(1)
plt.plot(x, sav, label="x and sav")
plt.legend()
plt.show()
