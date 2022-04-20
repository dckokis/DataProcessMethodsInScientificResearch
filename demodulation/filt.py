import numpy as np

def gaus(x, sigma=1, mu=0):
    return np.array([np.exp(-(i - mu) ** 2 / (sigma ** 2)) for i in x])


def lff(x, x0=3):
    return np.array(
        [complex(1., abs(i) * 2 * np.pi / x0) if -(x0 + 0.01) < i < (x0 + 0.01) else complex(0., 0.) for i in x])


def low_freq_filter(signal, freq, t):
    df = 1. / t
    N = len(signal)
    xf = np.fft.fftfreq(N) * N * df
    ideal_lff_F = lff(xf, freq)
    ideal_lff_F_c = ideal_lff_F
    g = gaus(np.concatenate((xf[int(N / 2):], xf[:int(N / 2)])), freq)
    ideal_lff = np.fft.ifft(ideal_lff_F_c)
    ideal_lff_real = ideal_lff
    ideal_lff_real = np.concatenate((ideal_lff_real[int(N / 2):], ideal_lff_real[:int(N / 2)]))
    ideal_lff_real_w = ideal_lff_real[int(N / 2 - 25):int(N / 2) + 25]
    graph_ideal_lff_real_w = np.concatenate(
        (np.zeros((int(N / 2) - 25)), ideal_lff_real_w, np.zeros((int(N / 2) - 25))))
    ideal_lff_real_wg = graph_ideal_lff_real_w * g
    return np.convolve(signal, ideal_lff_real_wg[int(N / 2) - 25:int(N / 2) + 25])[25:N + 25].real
