import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
from math import sin, cos, pi, gcd

plt.rcParams['figure.dpi'] = 200


def filtration_lpf(size, K):
    left_border = size / (2 * K)
    right_border = size * (1 - 1 / (2 * K))
    A = np.array([0 if left_border <= i <= right_border else 1 for i in range(size)])
    # (
    #         np.cos(i * np.pi / left_border) + 1j * np.sin((i - size) * np.pi / left_border)) for i in range(size)])
    h = np.fft.ifft(A)
    # отзеркалить sinc
    f1, f2 = np.array_split(h, 2)
    sinc = np.concatenate([f2, f1])
    # обоконивание
    cut = 70
    cp_sinc = sinc.copy()
    gauss = scp.windows.gaussian(size, 100)
    cp_sinc = cp_sinc * np.array(
        [1 if cp_sinc.size / 2 - cut <= i <= cp_sinc.size / 2 + cut else 0 for i in range(cp_sinc.size)])
    cp_sinc = cp_sinc * gauss
    f1, f2 = np.array_split(cp_sinc, 2)
    cp_sinc = np.concatenate([f2, f1])
    h = np.fft.fft(cp_sinc)
    return h, A, cp_sinc


def lpf(x: np.ndarray, K) -> np.ndarray:
    size = x.size
    h = filtration_lpf(size, K)[0]
    f = np.fft.fft(x)
    a = np.concatenate((f, np.zeros(h.size - x.size, dtype=complex)))
    y = np.real(np.fft.ifft(h * a))
    return y


def upsample(signal, K):
    if K == 1:
        return signal
    res = np.zeros(int(K * signal.size - K + 1))
    for i in range(int(K * signal.size - K + 1)):
        if i % K == 0:
            res[i] = signal[int(i / K)]

    res = lpf(res, K)
    return res * K


def downsample(signal, M):
    signal = lpf(signal, M)
    new_size = int(signal.size - (M - 1) * (signal.size - 1) / M)
    res = np.zeros(new_size)
    for i in range(signal.size):
        if i % M == 0:
            res[int(i / M)] = signal[i]
    return res


def resample(signal, new_len):
    K = new_len // gcd(len(signal), new_len)
    M = len(signal) // gcd(len(signal), new_len)
    a = upsample(signal, K)
    return downsample(a, M)


def sinus(N, freq):
    t = np.linspace(-np.pi, np.pi, N)
    noise = np.array([np.random.rand() for i in range(N)])
    return np.sin(freq * t) + np.sin(40 * freq * t) + noise


def draw():
    size = 1000
    new_size = 1700
    signal = sinus(size, 10)

    plt.subplot(211)
    plt.plot(signal)
    plt.title('Signal')
    plt.subplot(212)
    res = resample(signal, new_len=new_size)
    plt.plot(res)
    plt.title('Resampled')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


# draw()

# if N % 2 == 0:
#     for i in range(M):
#         if 1 <= i <= (N + 1) / 2:
#             fft_res[i] = fft_signal[i]
#         elif (N + 1) / 2 + 1 <= i <= (N + 1) / 2 + M - N:
#             fft_res[i] = 0
#         elif (N + 1) / 2 + M - N + 1 <= i <= M:
#             fft_res[i] = signal[i - M + N]
# else:
#     for i in range(M):
#         if 1 <= i <= N / 2:
#             fft_res[i] = fft_signal[i]
#         elif N / 2 + 2 <= i <= N / 2 + M - N:
#             fft_res[i] = 0
#         elif N / 2 + M - N + 2 <= i <= M:
#             fft_res[i] = signal[i - M + N]
#         elif i == N / 2 + 1 or i == N / 2 + M - N + 1:
#             fft_res[i] = fft_signal[N / 2 + 1] / 2
# return np.real(np.fft.ifft(fft_res))


def draw_fft():
    size = 1000
    new_size = 3000
    signal = sinus(size, 1)
    fft_signal = np.fft.fft(signal)
    plt.subplot(4, 1, 1)
    plt.plot(np.abs(fft_signal))
    N = len(signal)
    fft_res = None
    if new_size > N:
        left, right = np.array_split(fft_signal, 2)
        zeros = np.zeros(new_size - N)
        fft_res = np.concatenate([left, zeros, right])
        fft_res *= new_size / N
    else:
        left, right = np.array_split(fft_signal, 2)
        new_left = left[:int(N / 2) - int(new_size / 2)]
        new_right = right[int(N / 2) - int(new_size / 2):]
        fft_res = np.concatenate([new_left, new_right])
        fft_res *= new_size / N
    plt.subplot(4, 1, 2)
    plt.plot(np.abs(fft_res))
    res = np.real(np.fft.ifft(fft_res))
    plt.subplot(4, 1, 3)
    plt.plot(signal)
    plt.subplot(4, 1, 4)
    plt.plot(res)
    plt.show()


draw_fft()
