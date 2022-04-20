import numpy as np
import matplotlib.pyplot as plt
from filt import low_freq_filter

plt.rcParams['figure.dpi'] = 200


def modulate(S_t, fs, Um, fm, Uc, fc, L, size):
    t = np.linspace(0, L, size)
    cos = Um * np.cos(2 * np.pi * fm * t)
    sin = - Uc * np.sin(2 * np.pi * fc * t)
    Re = np.real(S_t)
    Im = np.imag(S_t)
    Re = Re * cos
    Im = Im * sin
    return Re + Im


def demod(S_t, fs, Um, fm, Uc, fc, L, size):
    t = np.linspace(0, L, size)
    cos = Um * np.cos(2 * np.pi * fm * t)
    sin = Uc * np.sin(2 * np.pi * fc * t)
    Re = S_t * sin
    Im = S_t * cos
    Re = low_freq_filter(Re, fc, L)
    Im = low_freq_filter(Im, fc, L)
    return np.array([complex(Re[i], Im[i]) for i in range(len(Re))])


fm = 15
fc = 15
Uc = 1
Um = 1
L = 5
size = 1000
fs = 25
t = np.linspace(0, L, size)
noise = np.array([(np.random.rand() - 0.5) for i in range(size)])
S_t = np.sin(2 * np.pi * fs * t) + 2 * np.sin(2 * np.pi * (fs - 20) * t) + 0.5 * np.sin(2 * np.pi * (fs + 25) * t) + noise
res = demod(S_t, fc, Um, fm, Uc, fc, L, size)
plt.subplot(2, 1, 1)
plt.plot(t, S_t)
plt.subplot(2, 1, 2)
plt.plot(np.abs(np.fft.fft(S_t)))
plt.show()
plt.subplot(2, 1, 1)
plt.plot(t, res)
plt.subplot(2, 1, 2)
plt.plot(np.abs(np.fft.fft(res)))
plt.show()
# cos = Um * np.cos(2 * np.pi * fm * t)
# sin = Uc * np.sin(2 * np.pi * fc * t)
# plt.plot(t, cos)
# plt.plot(t, sin)
# plt.show()
