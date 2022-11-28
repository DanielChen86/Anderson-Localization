import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


class Localization:
    def __init__(self, N=800, x0=400, mu=0, t=10, W=0, dt=1e-3) -> None:
        self.N = N
        self.x0 = x0
        self.mu = mu
        self.t = t
        self.W = W
        self.dt = dt
        self.gen_init_state()
        self.gen_hamiltonian()
        self.ipr = []

    def gen_init_state(self):
        self.psi = np.zeros(self.N)
        self.psi[self.x0] = self.x0

    def gen_hamiltonian(self):
        self.hamiltonian = np.zeros((self.N, self.N))
        for n in range(self.N):
            self.hamiltonian[n, n] += self.mu * \
                np.cos(2 * np.pi * n * (2 / (1 + np.sqrt(5))))

        self.hamiltonian[0, 1] = self.t
        self.hamiltonian[0, self.N-1] = self.t
        for n in range(1, self.N-1):
            self.hamiltonian[n, n-1] = self.t
            self.hamiltonian[n, n+1] = self.t
        self.hamiltonian[self.N-1, self.N-2] = self.t
        self.hamiltonian[self.N-1, 0] = self.t
        for n in range(self.N):
            self.hamiltonian[n, n] += (np.random.rand() * 2 - 1) * self.W

    def norm(self, vec: np.array):
        return np.sqrt(np.sum(np.conjugate(vec) * vec))

    def update(self):
        d_psi = 1j * self.dt * (self.hamiltonian @ self.psi)
        self.psi = self.psi + d_psi
        self.psi = self.psi / self.norm(self.psi)
        self.ipr.append(self.get_ipr())

    def prob(self):
        return np.real(np.conjugate(self.psi) * self.psi)

    def get_ipr(self):
        return np.sum(np.abs(self.psi) ** 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', '-N', default=100, type=int)
    parser.add_argument('--x0', '-x', default=None, type=int)
    parser.add_argument('--mu', '-mu', default=0, type=float)
    parser.add_argument('--t', '-t', default=10, type=float)
    parser.add_argument('--W', '-W', default=0, type=float)
    parser.add_argument('--dt', '-dt', default=1e-3, type=float)
    parser.add_argument('--frames', '-f', default=1000, type=int)
    args = parser.parse_args()
    if not args.x0:
        args.x0 = int(args.N / 2)
    for k, v in (args.__dict__.items()):
        print(f'{k}: {v}')

    localization = Localization(
        args.N, args.x0, args.mu, args.t, args.W, args.dt)

    fig, ax = plt.subplots(2)

    def animate(i):
        x1 = np.arange(localization.N)
        y1 = localization.prob()

        ax[0].clear()
        ax[0].plot(x1, y1)
        ax[0].set_xlim([0, localization.N])
        ax[0].set_ylim([0, 1.2 * np.max(y1)])

        x2 = np.arange(len(localization.ipr))
        y2 = localization.ipr

        ax[1].clear()
        ax[1].plot(x2, y2)
        ax[1].set_xlim([0, args.frames])
        ax[1].set_ylim([0, 0.5])

        localization.update()

    ani = FuncAnimation(fig, animate, frames=args.frames,
                        interval=1, repeat=False)

    plt.show()
