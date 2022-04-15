import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin


def convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    N = a.size
    M = b.size
    S = np.zeros(N + M - 1, dtype=complex)
    for n in range(N + M - 1):
        for m in range(N):
            if M > n - m >= 0:
                S[n] += a[m] * b[n - m]
    return S


def fft_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.size + b.size - 1
    return np.fft.ifft(np.fft.fft(a, n) * np.fft.fft(b, n))


def correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = a.size + b.size - 1
    a = np.concatenate((np.zeros(size - a.size), a))
    b = np.concatenate((b, np.zeros(size - b.size)))
    C = np.zeros(size, dtype=complex)
    for n in range(size):
        for i in range(a.size):
            if min(a.size, b.size) > i - n >= 0:
                C[n] += a[i] * b[i - n]
    return C


def fft_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.size + b.size - 1
    return np.fft.ifft(np.fft.fft(a, n) * np.fft.fft(b, n).conj())


def draw(f, g):
    c = convolution(f, g)
    c1 = np.convolve(f, g)
    corr = correlation(f, g)
    corr1 = np.correlate(f, g, 'full')
    # corr1 = fft_correlation(f, g)
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(np.real(f))
    plt.plot(np.real(g))
    plt.subplot(3, 1, 2)
    plt.plot(np.real(c), '-')
    plt.plot(np.real(c1), '--')
    plt.subplot(3, 1, 3)
    plt.plot(np.real(corr), '-')
    plt.plot(np.real(corr1), '--')
    plt.show()


def square():
    f = np.array([1 if 30 < i < 280 else 0 for i in range(511)], dtype=complex)
    g = np.array([1 if 50 < i < 300 else 0 for i in range(501)], dtype=complex)
    draw(f, g)


def sinus():
    f = np.array([sin((pi * i + 100) / 100) for i in range(501)], dtype=complex)
    g = np.array([sin(pi * i / 100) for i in range(501)], dtype=complex)
    draw(f, g)


def saw():
    f = np.array([200 - 2 * i if i > 100 else -(200 - (-2 * i)) for i in range(201)] * 15, dtype=complex)
    g = np.array([300 - 3 * i + 150 if i > 140 else -(300 - (-3 * i)) + 150 for i in range(201)] * 15, dtype=complex)
    draw(f, g)


def noise():
    f = np.array([np.random.rand() for _ in range(111)])
    g = np.array([np.random.rand() for _ in range(247)])
    draw(f, g)


# square()
sinus()
# saw()
# noise()
