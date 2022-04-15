import cmath
import numpy as np
from math import pi, sin
import matplotlib.pyplot as plt


def direct_fourier(f: np.ndarray):
    N = f.size
    F = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            F[k] += f[n] * cmath.exp(complex(0, -2 * pi / N * k * n))
    return F


def inverse_fourier(F: np.ndarray):
    N = F.size
    f = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            f[n] += F[k] * cmath.exp(complex(0, 2 * pi / N * k * n))
    return f


def generate_signal(sig_type: int, add_noise: bool):
    if sig_type == 1:
        if add_noise:
            f = np.array([sin(2 * pi * i / 100) + 2 * (np.random.rand() - 0.5) for i in range(1001)], dtype=complex)
        else:
            f = np.array([sin(2 * pi * i / 100) for i in range(1001)], dtype=complex)
        return f
    elif sig_type == 2:
        if add_noise:
            f = np.array(
                [sin(2 * pi * i / 100) + 5 * sin(9 * pi * i / 100) + 10 * (np.random.rand() - 0.5) for i in
                 range(1001)],
                dtype=complex)
        else:
            f = np.array([sin(2 * pi * i / 100) + 3 * sin(10 * pi * i / 100) for i in range(1001)], dtype=complex)
        return f
    elif sig_type == 3:
        if add_noise:
            f = np.array([(1 if i < 25 else -1) + 10 * (np.random.rand() - 0.5) for i in range(51)], dtype=complex)
        else:
            f = np.array([1 if i < 50 else -1 for i in range(101)] * 10, dtype=complex)
        return f
    else:
        print("Wrong signal type")


def main():
    f = generate_signal(1, True)

    f1 = direct_fourier(f)
    f2 = np.fft.fft(f)

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(np.real(f))

    plt.subplot(3, 1, 2)
    plt.plot(np.abs(f1), '-')
    plt.plot(np.abs(f2), '--')

    plt.subplot(3, 1, 3)
    plt.plot(np.abs(f2 - f1))

    plt.show()


main()
