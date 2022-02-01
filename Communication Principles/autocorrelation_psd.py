# %%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import fft
from scipy.fftpack import fftshift


# %%
def awgn(sig, snr):
    '''
    Additive White Gaussian Noise
    
    Parameters
    ----------
    sig: original signal
    snr: SNR in dB
    ----------
    Returns
    ----------
    sig+noise: signal with awgn
    ----------
    '''
    #SNR in dB
    snr_w = 10**(snr/10)
    sig_pow = np.sum(sig**2)/len(sig)
    noise_pow = sig_pow/snr_w 
    noise = np.random.randn(len(sig))*np.sqrt(noise_pow)
    return sig+noise
    
def main():
    fs = 1e5 #sampling frequency
    fm = 1e3 #signal frequency

    t = np.arange(0,10/fm, 1/fs)
    sig = np.cos(2*np.pi*fm*t)
    sig = awgn(sig, 10)


    N = 4096        
    f = np.arange(0,N)/N*fs #analog angle frequency
    f = f-fs/2      #shift to 0

    r = np.correlate(sig, sig, mode = 'same') #correlation function R(t)
    psd = fft(r,N)                             #power spectrum density 
    psd_shift = fftshift(psd)

    plt.subplot(1,2,1)
    plt.plot(f/fm, np.abs(psd_shift))
    plt.title('Power Spectrum Density')

    plt.subplot(1,2,2)
    plt.plot(t, r)
    plt.title('Autocorrelation Function')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()

