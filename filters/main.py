import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin
import scipy.signal as scp

plt.rcParams['figure.dpi'] = 200


def filtration_hpf(size):
    left_border = size * 0.25
    right_border = size * 0.75
    A = np.array([(np.cos(i * np.pi / left_border) + 1j * np.sin(
        (i - size) * np.pi / left_border)) if left_border <= i <= right_border else 0 for i in range(size)])
    h = np.fft.ifft(A)
    # отзеркалить
    f1, f2 = np.array_split(h, 2)
    sinc = np.concatenate([f2, f1])
    # обоконивание
    cut = 70
    cp_sinc = sinc.copy()
    gauss = scp.windows.gaussian(size, 40)
    cp_sinc = cp_sinc* np.array(
        [1 if cp_sinc.size / 2 - cut <= i <= cp_sinc.size / 2 + cut else 0 for i in range(cp_sinc.size)])
    cp_sinc = cp_sinc * gauss
    f1, f2 = np.split(cp_sinc, 2)
    cp_sinc = np.concatenate([f2, f1])
    h = np.fft.fft(cp_sinc)
    return h, A, cp_sinc


def filtration_lpf(size):
    left_border = size * 0.125
    right_border = size * 0.875
    A = np.array([0 if left_border <= i <= right_border else (
                np.cos(i * np.pi / left_border) + 1j * np.sin((i - size) * np.pi / left_border)) for i in range(size)])
    h = np.fft.ifft(A)
    # отзеркалить sinc
    f1, f2 = np.array_split(h, 2)
    sinc = np.concatenate([f2, f1])
    # обоконивание
    cut = 70
    cp_sinc = sinc.copy()
    gauss = scp.windows.gaussian(size, 40)
    cp_sinc = cp_sinc * np.array(
        [1 if cp_sinc.size / 2 - cut <= i <= cp_sinc.size / 2 + cut else 0 for i in range(cp_sinc.size)])
    cp_sinc = cp_sinc * gauss
    f1, f2 = np.split(cp_sinc, 2)
    cp_sinc = np.concatenate([f2, f1])
    h = np.fft.fft(cp_sinc)
    return h, A, cp_sinc


def Strip(x: np.ndarray) -> np.ndarray:
    size = x.size
    h1 = filtration_lpf(size)[0]
    h2 = filtration_hpf(size)[0]
    h = h1*h2
    f = np.fft.fft(x)
    a = np.concatenate((f, np.zeros(h.size - x.size, dtype=complex)))
    y = np.real(np.fft.ifft(h * a))
    return y


def Reject(x: np.ndarray) -> np.ndarray:
    size = x.size
    h1 = filtration_lpf(size)[0]
    h2 = filtration_hpf(size)[0]
    h = h1 + h2
    f = np.fft.fft(x)
    a = np.concatenate((f, np.zeros(h.size - x.size, dtype=complex)))
    y = np.real(np.fft.ifft(h * a))
    return y


def draw(f):
    plt.figure()
    # # Strip
    # filteredStrip = Strip(f)
    # # сигнал
    # plt.subplot(5, 1, 1)
    # plt.plot(np.real(f), '-')
    # plt.ylabel('signal')
    # h1 = filtration_lpf(f.size)[0]
    # h2 = filtration_hpf(f.size)[0]
    # plt.subplot(5, 1, 2)
    # plt.plot(np.abs(h1 * h2))
    # plt.subplot(5, 1, 3)
    # plt.plot(np.abs(np.fft.fft(f)))
    # # сигнал после фильтра
    # plt.subplot(5, 1, 4)
    # plt.plot(np.real(filteredStrip))
    # plt.ylabel('Strip')
    # plt.subplot(5, 1, 5)
    # plt.plot(np.abs(np.fft.fft(filteredStrip)))

    # Reject
    filteredRej = Reject(f)
    # сигнал
    plt.subplot(6, 1, 1)
    plt.plot(np.real(f), '-')
    plt.ylabel('signal')
    h1 = filtration_lpf(f.size)[0]
    h2 = filtration_hpf(f.size)[0]
    plt.subplot(6, 1, 2)
    plt.plot(np.abs(h1+h2))
    plt.subplot(6, 1, 3)
    plt.plot(np.abs(np.fft.fft(f)))
    # сигнал после фильтра
    plt.subplot(6, 1, 4)
    plt.plot(np.real(filteredRej), '-')
    plt.ylabel('Reject')
    plt.subplot(6, 1, 5)
    plt.plot(np.abs(np.fft.fft(filteredRej)))
    plt.subplot(6, 1, 6)
    plt.plot(np.fft.ifft(h1 + h2))
    plt.subplots_adjust(hspace=1.1)
    plt.show()


def sinus():
    f = np.array([sin((pi * i) / 2) for i in range(1000)])
    draw(f)


def sum_sinus():
    t = np.linspace(-np.pi, np.pi, 1000)
    freq1 = 1
    freq3 = 10
    x_sin1 = np.sin(1 * freq1 * t)
    x_sin3 = np.sin(np.pi * freq3 * t)
    f = x_sin1 + x_sin3
    draw(f)


def noise_sin():
    f = np.array([sin((pi * i) / 4) + 2 * sin((pi * i) / 160) for i in range(1000)])
    draw(f)


# sinus()
sum_sinus()
# noise_sin()
