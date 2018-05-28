import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from numpy.linalg import cholesky

High_D = 500
High_mu = torch.rand(1, High_D) * 100
for i in range(High_D):
    High_mu[0, i] = i * 0.1 + 1
# High_cov = np.random.rand(High_D, High_D)
# High_cov = (High_cov + High_cov.T) / 2.0
# High_cov[np.arange(High_D), np.arange(High_D)] = 1.0

High_cov = np.zeros(shape=(High_D, High_D))
for i in range(High_D):
    High_cov[i, i] = (i+1) * 1.0 / High_D
High_Sigma = torch.from_numpy(High_cov).float()
# High_Sigma = torch.eye(High_D, High_D)

def kinetic_energy(velocity):
    """
    calculate kinetic energy
    :param velocity: torch.tensor [num_samples, D_velocity]
    :return: a vector with length of num_samples
    """
    # temp = torch.matmul(velocity.data, High_Sigma)
    # temp = torch.matmul(temp, velocity.data.t())

    # return 0.5 * torch.sum(temp)

    return 0.5 * torch.sum((velocity.data)**2)

# compute energy: U + K
def hamiltonian(pos, vel, energy_fn):
    """
    Hamiltionian energy = potential + kinetic
    :param pos: torch.tensor [num_samples, D_position]
    :param vel: torch.tensor [num_samples, D_velocity]
    :param energy_fn: a function
    :return: a vector with length of num_samples
    """
    U = energy_fn(pos)
    U = U.data[0][0]
    K = kinetic_energy(vel)
    return U + K


def metropolis_hastings_accept(energy_prev, energy_next):
    """
    Performs a Metropolis-Hastings accept-reject move.
    :param energy_prev: torch.tensor [num_samples],
        the energy in time-step t
    :param energy_next: torch.tensor [num_samples],
        the energy in time-step t+1
    :return: boolean torch.tensor [num_samples]
        true->accept
    """
    energy_diff = energy_prev - energy_next
    # print("energy:", energy_prev, "energy diff:", energy_diff)
    random_sample = torch.rand(1)[0]
    return (np.exp(energy_diff) - random_sample) >= 0


# obtain a single sample after n_steps leapfrog
def simulate_dynamic(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    """
    return final (position, velocity) after n_steps leapfrog
    :param initial_pos:
    :param initial_vel:
    :param stepsize:
    :param n_steps:
    :param energy_fn:
    :return: final_position, final_velocity
    """

    def leapfrog(pos, vel, step):
        """

        :param pos: torch.tensor [D_position],
            position at time t
        :param vel: torch.tensor [D_velocity],
            velocity at time (t-stepsize/2)
        :param step: scalar
        :return: new_pos, new_vel
            position at time (t+stepsize)
            velocity at time (t+stepsize/2)
        """
        # grad of potential nergy
        # energy_fn.backward(torch.ones(initial_pos.size()))
        if not pos.requires_grad:
            pos.requires_grad = True
            pos.volatile = False
        potential_of_give_pos = energy_fn(pos)
        potential_of_give_pos.backward()
        dE_dpos = pos.grad

        new_vel = vel - step * dE_dpos  # at time (t + stepsize/2)
        new_pos = pos + step * new_vel  # at time (t + stepsize)
        return new_pos, new_vel

    # velocity at time t+stepsize/2
    # initial_pos = Variable(initial_pos, requires_grad=True)
    # initial_vel = Variable(initial_vel, requires_grad=False)

    initial_potential = energy_fn(initial_pos)
    initial_potential.backward()
    dE_dpos = initial_pos.grad
    vel_half_step = initial_vel - 0.5 * stepsize * dE_dpos

    # position at time t+stepsize
    pos_full_step = initial_pos + stepsize * vel_half_step

    # perform leapfrog
    temp_pos = pos_full_step
    temp_vel = vel_half_step
    for lf_step in range(n_steps):
        temp_pos, temp_vel = leapfrog(temp_pos, temp_vel, stepsize)

    final_pos = temp_pos
    final_vel = temp_vel

    # final_pos = Variable(final_pos, requires_grad=True)
    final_pos.requires_grad = True
    final_pos.volatile = False
    potential = energy_fn(final_pos)
    potential.backward()
    final_vel = final_vel - 0.5 * stepsize * final_pos.grad

    return final_pos, final_vel


def hmc_move(positions, energy_fn, stepsize, n_steps):
    """
    Perform one iteration of sampling.
    1. Start by sampling a random velocity from a Gaussian.
    2. Perform n_steps leapfrog
    3. decide whether to accept or reject
    :param positions: start sampling from positions
    :param energy_fn: potential energy function
    :param stepsize: leapfrog stepsize
    :param n_steps: leapfrog steps
    :return: accept(bool), final_pos(torch.tensor)
    """
    # initial_vel = torch.randn(positions.size()) # with zero mean and unit variance
    np_inital_vel = np.random.multivariate_normal(np.zeros(High_D), High_cov, 1)
    initial_vel = torch.from_numpy(np_inital_vel).float()
    positions = Variable(positions, requires_grad=True)
    initial_vel = Variable(initial_vel, requires_grad=True)

    final_pos, final_vel = simulate_dynamic(initial_pos=positions,
                                            initial_vel=initial_vel,
                                            stepsize=stepsize,
                                            n_steps=n_steps,
                                            energy_fn=energy_fn)

    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(positions, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn)
    )

    return accept, final_pos.data

def hmc_sampling(init_pos, energy_fn, n_samples, stepsize=0.01, n_steps=20, gap=20):
    result_samples = torch.zeros(n_samples+gap, 1, init_pos.size(1))
    # last_pos = init_pos
    # result_samples.append(init_pos)
    result_samples[0, :, :] = init_pos
    accept_num = 0

    for i in range(1, n_samples+gap):
        last_pos = result_samples[i-1, :, :]

        accept, new_pos = hmc_move(last_pos, energy_fn, stepsize, n_steps)

        if accept:
            # result_samples.append([new_pos[0][0], new_pos[0][1]])
            result_samples[i, :, :] = new_pos
            accept_num += 1
        else:
            # result_samples.append([last_pos[0][0], last_pos[0][1]])
            result_samples[i, :, :] = last_pos

    accept_rate = accept_num * 1.0 / (n_samples+gap)
    return result_samples[gap:, :, :], accept_rate

Gaussian_u = torch.rand(1, 2) * 2
conv = np.random.rand(2, 2)
conv = (conv + conv.T) / 2
conv[np.arange(2), np.arange(2)] = 1.0
Gaussian_Sigma = torch.from_numpy(conv).float()


def NormalEnergy(x):
    # x = Variable(x, requires_grad=True)
    # u = torch.zeros(1, 2)
    # u[0, :] = torch.FloatTensor([2, 2])
    # Sigma = torch.FloatTensor([[1.0, 0.8], [0.8, 1.0]])
    u = Gaussian_u
    Sigma = Gaussian_Sigma

    Sigma = torch.inverse(Sigma)
    # Sigma = Sigma.t()

    if isinstance(x, Variable):
        u = Variable(u, requires_grad=True)
        Sigma = Variable(Sigma, requires_grad=True)

    diff = x - u

    temp = 0.5 * torch.matmul(torch.matmul(diff, Sigma), diff.t())

    return temp

def grad_test():
    x = torch.ones(1, 2)
    x = Variable(x, requires_grad=True)
    y = NormalEnergy(x)
    y.backward()
    print(x.grad)

def vis_test():
    n_samples = 1000
    stepsize = 0.1
    n_steps = 20
    dim = 2

    initial_pos = torch.randn(1, dim)
    samples, _ = hmc_sampling(initial_pos, NormalEnergy, n_samples, stepsize, n_steps)
    samples = samples.view(samples.size(0), -1)
    # print(torch.mean(samples, 0))
    # samples = np.array(samples)
    samples = samples.numpy()

    print("mean")
    print(np.mean(samples, axis=0))
    print("covariance")
    print(np.cov(samples.T))
    # print(torch.std(samples, 0))

    fig = plt.figure(0)
    plt.title('Dynamics Sampling')
    plt.xlabel('x')
    plt.ylabel('y')

    x = samples[:, 0]
    y = samples[:, 1]
    plt.scatter(x, y, c='red', marker='+')


    # mu = np.array([2, 2])
    # Sigma = np.array([[1, 0.8], [0.8, 1]])
    mu = Gaussian_u[0, :].numpy()
    Sigma = Gaussian_Sigma.numpy()

    x, y = np.random.multivariate_normal(mu, Sigma, 1000).T
    print("true mean:")
    print(mu)
    print("true covariance:")
    s = [x, y]
    print(np.cov(s))
    plt.scatter(x, y, c='green', marker='*')
    plt.show()

def high_dimension_gaussain_energy(x):
    u = High_mu
    Sigma = High_Sigma
    Sigma = torch.inverse(Sigma)

    if isinstance(x, Variable):
        u = Variable(u, requires_grad=True)
        Sigma = Variable(Sigma, requires_grad=True)

    diff = x - u
    temp = 0.5 * torch.matmul(torch.matmul(diff, Sigma), diff.t())

    return temp


def high_dimension_test():
    n_samples = 2
    stepsize = 0.01
    n_steps = 150
    dim = High_D
    n_gap = 100
    epoch = 1
    total_n = epoch * n_samples
    samples = torch.zeros(total_n, 1, High_D)

    for e in range(epoch):
        initial_pos = torch.randn(1, dim)
        e_samples, accept_rate = hmc_sampling(initial_pos, high_dimension_gaussain_energy, n_samples, stepsize, n_steps, n_gap)
        samples[e*n_samples:(e+1)*n_samples, :, :] = e_samples

    samples = samples.view(samples.size(0), -1)
    samples = samples.numpy()

    print("samples mu:")
    samples_mu = np.mean(samples, axis=0)
    print(samples_mu)
    # print("samples covariance")
    samples_sigma = np.cov(samples.T)
    # print(samples_sigma)

    print("\ntrue mu:")
    print(High_mu)
    # print("true covariance:")
    # print(High_cov)

    L2_mu = ((samples_mu - High_mu.numpy())**2).mean()
    print("L2 mu:", L2_mu)
    L2_Sigma = ((High_Sigma.numpy() - samples_sigma)**2).mean()
    print("L2 covariance:", L2_Sigma)

    print("accept rate: ", accept_rate)


if __name__ == '__main__':
    # grad_test()
    # vis_test()
    high_dimension_test()








