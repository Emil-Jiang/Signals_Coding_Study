# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft,fft
from scipy.fftpack import fftshift
from scipy.signal import firwin, filtfilt


# %%
def LPF(n, fcut):
    '''
    Use Hamming window to design a n-th order LPF
    
    Parameters
    ----------
    n: Filter order 
    fcut: cutoff frequency
    '''
    lpf = firwin(n, fcut, window='hamming')
    return lpf

# %%
def recover_signal(sig, fcut):
    '''
    Recover the signal by using 30th order LPF
    
    Paramaters
    ----------
    sig: Received signal
    fcut: Cutoff frequency
    ----------
    Returns 
    ----------
    recover: Recovered signal
    ----------
    '''
    lpf = LPF(30, fcut)
    recover = filtfilt(lpf, 1, sig)
    return recover

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
    
    

# %%
def AM(sig, fm, fc, fs,  N):
    '''
    AM modulation
    
    Parameters
    ----------
    sig: original signal
    fm: original signal frequency
    fc: carrier frequency
    fs: sampling frequency
    N: the number of points on the frequency domain
    ----------
    '''
    t = np.arange(0,10/fm, 1/fs)

    a0 = max(sig)
    am = a0+sig*np.cos(2*np.pi*fc*t)
    psd = fftshift(fft(am, n = N))
    
    f = np.arange(0,N)/N*fs 
    f = f-fs/2
    
    rec = am*np.cos(2*np.pi*fc*t)
    rec_psd = fftshift(fft(rec, n = N))
    
    fcut = 1.5*2*np.pi*fm/fs
    recover = recover_signal(rec, fcut)
    recover_fft = fftshift(fft(recover, N))
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(f/fm, np.abs(psd))
    plt.xlabel('Frequency')
    plt.title('AM Modulation')
    
    plt.subplot(2,2,2)
    plt.plot(f/fm, np.abs(rec_psd))
    plt.xlabel('Frequency')
    plt.title('AM Demodulation')
    
    plt.subplot(2,2,3)
    plt.plot(f/fm, np.abs(recover_fft))
    plt.xlabel('Frequency')
    plt.title('AM Recovered Signal')
    
    plt.subplot(2,2,4)
    plt.plot(range(len(recover)), np.abs(recover))
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()
    

# %%
def DSB(sig, fm, fc, fs,  N):
    '''
    DSB modulation
    
    Parameters
    ----------
    sig: original signal
    fm: original signal frequency
    fc: carrier frequency
    fs: sampling frequency
    N: the number of points on the frequency domain
    ----------
    '''
    t = np.arange(0,10/fm, 1/fs)

    dsb = sig*np.cos(2*np.pi*fc*t)
    psd = fftshift(fft(dsb, n = N))
    
    f = np.arange(0,N)/N*fs 
    f = f-fs/2
    
    rec = dsb*np.cos(2*np.pi*fc*t)
    rec_psd = fftshift(fft(rec, n = N))
    
    fcut = 1.5*2*np.pi*fm/fs
    recover = recover_signal(rec, fcut)
    recover_fft = fftshift(fft(recover, N))
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(f/fm, np.abs(psd))
    plt.xlabel('Frequency')
    plt.title('DSB Modulation')
    
    plt.subplot(2,2,2)
    plt.plot(f/fm, np.abs(rec_psd))
    plt.xlabel('Frequency')
    plt.title('DSB Demodulation')
    
    plt.subplot(2,2,3)
    plt.plot(f/fm, np.abs(recover_fft))
    plt.xlabel('Frequency')
    plt.title('DSB Recovered Signal')
    
    plt.subplot(2,2,4)
    plt.plot(range(len(recover)), np.abs(recover))
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()
    

# %%
def SSB(sig, fm, fc, fs,  N):
    '''
    SSB modulation
    
    Parameters
    ----------
    sig: original signal
    fm: original signal frequency
    fc: carrier frequency
    fs: sampling frequency
    N: the number of points on the frequency domain
    ----------
    '''
    t = np.arange(0,10/fm, 1/fs)

    ssb = sig*np.cos(2*np.pi*fc*t)
    psd = fftshift(fft(ssb, n = N))
    
    f = np.arange(0,N)/N*fs 
    f = f-fs/2
    
    rec = ssb*np.cos(2*np.pi*fc*t)
    rec_psd = fftshift(fft(rec, n = N))
    
    fcut = 1.5*2*np.pi*fm/fs
    recover = recover_signal(rec, fcut)
    recover_fft = fftshift(fft(recover, N))
    recover_fft[:int(len(recover_fft)/2)] = 0
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(f/fm, np.abs(psd))
    plt.xlabel('Frequency')
    plt.title('SSB Modulation')
    
    plt.subplot(2,2,2)
    plt.plot(f/fm, np.abs(rec_psd))
    plt.xlabel('Frequency')
    plt.title('SSB Demodulation')
    
    plt.subplot(2,2,3)
    plt.plot(f/fm, np.abs(recover_fft))
    plt.xlabel('Frequency')
    plt.title('SSB Recovered Signal')
    
    plt.subplot(2,2,4)
    plt.plot(range(len(recover)), np.abs(recover))
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

# %%
def main():
    fc = 1e4
    fs = 1e5
    fm = 1e3
    
    t = np.arange(0,10/fm,1/fs)
    A = 5
    m = A*np.sin(2*np.pi*fm*t)
    m = awgn(m,100)
    N = 4096

    
    AM(m, fm, fc, fs, N)
    DSB(m, fm, fc, fs, N)
    SSB(m, fm, fc, fs, N)

# %%
if __name__ == '__main__':
    main()


