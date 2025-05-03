import cv2
import numpy as np
from ex4_utils import *
import sympy as sp

class Particle_filter():

    def __init__(self, alpha=0.05, q=10, sigma=0.1, N=100, dynamic_model="NCA"):
        self.target_width = None
        self.target_height = None
        self.region_width = None
        self.region_height = None
        self.x = None
        self.y = None
        self.h_tar = None
        self.scaling_parameter = 1
        self.epanechnik_kernel = None

        self.Q_i = None
        self.A = None

        self.particles = None
        self.weights = None
        self.dynamic_model = dynamic_model

        self.alpha = alpha
        self.q = q
        self.sigma = sigma
        self.N = N

    def name(self):
        return "pf"

    def initialize(self, image, region):


        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        position = (region[0] + region[2] / 2, region[1] + region[3] / 2) # X x Y
        self.x = position[0]
        self.y = position[1]

        self.target_width = region[2] if region[2] %2 == 1 else region[2] + 1
        self.target_height = region[3] if region[3] %2 == 1 else region[3] + 1

        self.region_width = (math.floor((region[2] * self.scaling_parameter) / 2) * 2) + 1
        self.region_height = (math.floor((region[3] * self.scaling_parameter) / 2) * 2) + 1

        patch, mask = get_patch(image, position, (self.region_width, self.region_height))
        if self.dynamic_model == "NCV":
            T, q, AS, Q_iS = NCV()
            part_matr = [self.x, self.y, 0, 0]
        elif self.dynamic_model == "NCA":
            T, q, AS, Q_iS = NCA()
            part_matr = [self.x, self.y, 0, 0, 0, 0]
        elif self.dynamic_model == "RW":
            T, q, AS, Q_iS = RW()
            part_matr = [self.x, self.y]

        self.epanechnik_kernel = create_epanechnik_kernel(self.region_width, self.region_height, 1)
        patch, mask = get_patch(image, (self.x, self.y), (self.region_width, self.region_height))
        self.h_tar = extract_histogram(patch, 16, self.epanechnik_kernel)

        T_val = 1  # time step
        q_val = min(self.region_height, self.region_width)*0.1
        Q_i = np.array(Q_iS.subs({T: T_val, q: q_val}).evalf(), dtype=np.float32)
        self.Q_i = Q_i

        A = np.array(AS.subs({T: T_val}).evalf(), dtype=np.float32)
        self.A = A

        self.particles = sample_gauss(part_matr, self.Q_i, self.N)
        self.weights = np.ones(self.particles.shape[0])/self.particles.shape[0]
 

    def track(self, image):
        particles_new = resample_particles(self.particles, self.weights, self.N)
        particles_new = (self.A @ particles_new.T).T
        particles_new += np.random.multivariate_normal(np.zeros(self.particles.shape[1]), self.Q_i, particles_new.shape[0])

        self.particles = particles_new

        for i, particle in enumerate(self.particles):
            xc, yc = particle[0], particle[1]

            if (xc - self.region_width / 2 < 0 or xc + self.region_width / 2 > image.shape[1] or
            yc - self.region_height / 2 < 0 or yc + self.region_height / 2 > image.shape[0]):
            # This particle is too close to the edge; ignore it
                self.weights[i] = 0
                continue
            particle_patch, mask = get_patch(image, (xc, yc), (self.region_width, self.region_height))
            h_new = extract_histogram(particle_patch, 16, self.epanechnik_kernel)

            dist = hellinger_distance(h_new, self.h_tar)
            self.weights[i] = np.exp(-dist**2 / (2 * self.sigma**2))
        
        self.weights = self.weights / np.sum(self.weights)

        position = np.average(self.particles[:, :2], axis=0, weights=self.weights)

        self.x = position[0]
        self.y = position[1]

        target_patch, mask = get_patch(image, (self.x, self.y), (self.region_width, self.region_height))
        new_hist = extract_histogram(target_patch, 16, self.epanechnik_kernel)
        self.h_tar = (1 - self.alpha) * self.h_tar + self.alpha * new_hist

        return(round(self.x - (self.region_width/2)), round(self.y - (self.region_height/2)), self.region_width, self.region_height)


def NCV():
    T, q = sp.symbols("T, q")
    F = sp.Matrix([[0,0,1,0],
                [0,0,0,1],
                [0,0,0,0],
                [0,0,0,0]])

    AS = sp.exp(F*T)
    LS = sp.Matrix([[0,0],
                [0,0],
                [1,0],
                [0,1]])
    Q_iS = sp.integrate((AS*LS)*q*(AS*LS).T, (T, 0, T))
    
    return T, q, AS, Q_iS

def resample_particles(particles, weights, N):
    weights_norm = weights/np.sum(weights)
    weights_cumsumed = np.cumsum(weights_norm)
    rand_samples = np.random.rand(N, 1)
    sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
    particles_new = particles[sampled_idxs.flatten(), :]
    return particles_new

def hellinger_distance(h1, h2):
    # Ensure L1-normalized histograms
    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)
    
    return np.sqrt(1 - np.sum(np.sqrt(h1 * h2)))

def NCA():
    T, q = sp.symbols("T, q")
    F = sp.Matrix([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    AS = sp.exp(F*T)

    LS = sp.Matrix([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    Q_iS = sp.integrate((AS*LS)*q*(AS*LS).T, (T, 0, T))

    CS = sp.Matrix([
        [1, 0, 0, 0, 0, 0],  # observe x
        [0, 1, 0, 0, 0, 0]   # observe y
    ])

    r = sp.symbols("r")
    R_iS = sp.Matrix([[r, 0],
                    [0, r]])
    
    return T, q, AS, Q_iS

def RW():
    T, q = sp.symbols("T, q")
    F = sp.Matrix([[0, 0],
                   [0, 0]])

    AS = sp.exp(F*T)

    LS = sp.Matrix([[1, 0],
                    [0, 1]])

    Q_iS = sp.integrate((AS*LS)*q*(AS*LS).T, (T, 0, T))


    r = sp.symbols("r")
    R_iS = sp.Matrix([[r, 0],
                    [0, r]])
    
    return T, q, AS, Q_iS