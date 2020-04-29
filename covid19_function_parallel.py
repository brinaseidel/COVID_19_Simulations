import numpy as np
import pdb
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count
from functools import partial
from multiprocessing.pool import Pool

# Matplotlib doesn't play nicely with multiprocessing, so
# we have to create a separate graphing function & import matplotlib inside of that.
def graphing_function(args, days, norms):
    import matplotlib.pyplot as plt
    susceptible_pop_norm, infected_pop_norm, recovered_pop_norm = norms
    # plot results and save the plot
    fig = plt.figure()
    ax = plt.axes()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.plot(np.arange(days), susceptible_pop_norm, label='Susceptible', color='#4aa5f0', linewidth=2)
    ax.plot(np.arange(days), infected_pop_norm, label='Infected', color='#f03737', linewidth=2)
    ax.plot(np.arange(days), recovered_pop_norm, label='Recovered', color='#82e88a', linewidth=2)
    ax.legend(frameon=False)
    ax.set_xlabel("Days")
    ax.set_ylabel("Share of Population")
    if 'fig_name' in args:
        ax.figure.savefig('figures/' + args.fig_name + '.png')
    else:
        ax.figure.savefig('figures/sir_plot.png')
    plt.close()

# Convenience function: access Python dict keys as dict.key instead of dict['key']
class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

# Run n simulations in parallel: for each run, parameters are randomly sampled from an interval
# NOTE: Are these intervals reasonable?
def multiprocess(args):

    chunks = []
    num_procs = args.randomize
    for _ in range(num_procs):
        d = dict()
        d['origin_matrix_path'] = args.origin_matrix_path
        d['thresh1'] = np.random.randint(0, 350)
        d['thresh2'] = np.random.randint(d['thresh1'], 700)
        d['infection_magnitude'] = np.random.randint(5, 15)
        d['beta'] = 1.6
        d['gamma'] = 0.04
        d['public_trans'] = np.random.randint(1, 7) * 0.1
        d['days'] = 200
        d['fig_name'] = 'public_trans_{:0.1f}_({}, {})_inf_{}'.format(d['public_trans'], d['thresh1'], d['thresh2'], d['infection_magnitude'])
        d = objdict(d)
        chunks.append(d)

    with Pool(num_procs) as p:
        p.map(covid_sim, chunks)

    p.join()



def covid_sim(args):
    # Read in origin-destination flow matrix
    OD = np.genfromtxt(args.origin_matrix_path, delimiter=',')

    # initialize the population vector from the origin-destination flow matrix
    N_k = np.abs(np.diagonal(OD) + OD.sum(axis=0) - OD.sum(axis=1))
    locs_len = len(N_k)                 # number of locations
    SIR = np.zeros(shape=(locs_len, 3)) # make a numpy array with 3 columns for keeping track of the S, I, R groups
    SIR[:,0] = N_k                      # initialize the S group with the respective populations
    thresh1 = args.thresh1
    thresh2 = args.thresh2

    first_infections = np.where((thresh1 <= SIR[:, 0]) & (SIR[:, 0] <= thresh2), np.random.randint(1, args.infection_magnitude), 0)   # for demo purposes, randomly introduce infections
    # NOTE: this is arbitrary but not actually random.... 
    SIR[:, 0] = SIR[:, 0] - first_infections
    SIR[:, 1] = SIR[:, 1] + first_infections                           # move infections to the I group

    # row normalize the SIR matrix for keeping track of group proportions
    row_sums = SIR.sum(axis=1)
    SIR_n = SIR / row_sums[:, np.newaxis]

    # initialize parameters
    beta = args.beta
    gamma = args.gamma
    public_trans = args.public_trans                                 # alpha
    R0 = beta/gamma
    beta_vec = np.random.gamma(beta, 2, locs_len)
    gamma_vec = np.full(locs_len, gamma)
    public_trans_vec = np.full(locs_len, public_trans)

    # make copy of the SIR matrices 
    SIR_sim = SIR.copy()
    SIR_nsim = SIR_n.copy()

    # run model
    #print(SIR_sim.sum(axis=0).sum() == N_k.sum())
    infected_pop_norm = []
    susceptible_pop_norm = []
    recovered_pop_norm = []
    days = 100
    for time_step in tqdm(range(days)):
        infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
        OD_infected = np.round(OD*infected_mat)
        inflow_infected = OD_infected.sum(axis=0)
        inflow_infected = np.round(inflow_infected*public_trans_vec)
        #print('total infected inflow: ', inflow_infected.sum())
        new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
        new_recovered = gamma_vec*SIR_sim[:, 1]
        new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
        SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
        SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered
        SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
        SIR_sim = np.where(SIR_sim<0,0,SIR_sim)
        # recompute the normalized SIR matrix
        row_sums = SIR_sim.sum(axis=1)
        SIR_nsim = SIR_sim / row_sums[:, np.newaxis]
        S = SIR_sim[:,0].sum()/N_k.sum()
        I = SIR_sim[:,1].sum()/N_k.sum()
        R = SIR_sim[:,2].sum()/N_k.sum()
        #print(S, I, R, (S+I+R)*N_k.sum(), N_k.sum())
        #print('\n')
        infected_pop_norm.append(I)
        susceptible_pop_norm.append(S)
        recovered_pop_norm.append(R)

    graphing_function(args, days, (susceptible_pop_norm, infected_pop_norm, recovered_pop_norm))
    '''
    print('fig_name' in args)
    # plot results and save the plot
    fig = plt.figure()
    ax = plt.axes()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.plot(np.arange(days), susceptible_pop_norm, label='Susceptible', color='#4aa5f0', linewidth=2)
    ax.plot(np.arange(days), infected_pop_norm, label='Infected', color='#f03737', linewidth=2)
    ax.plot(np.arange(days), recovered_pop_norm, label='Recovered', color='#82e88a', linewidth=2)
    ax.legend(frameon=False)
    ax.set_xlabel("Days")
    ax.set_ylabel("Share of Population")
    if 'fig_name' in args:
        ax.figure.savefit('figures/' + args.fig_name + '.png')
    else:
        ax.figure.savefig('figures/sir_plot.png')
    '''



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID Simulator')
    parser.add_argument('--randomize', type=int, default=0)
    parser.add_argument('--origin-matrix-path', default='data/Yerevan_OD_coronavirus.csv')
    parser.add_argument('--thresh1', type=int, default=0)
    parser.add_argument('--thresh2', type=int, default=200)
    parser.add_argument('--infection-magnitude', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.75)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--public-trans', type=float, default=0.56)
    parser.add_argument('--public-trans-vec', type=float, default=0.56)
    parser.add_argument('--days', type=int, default=100)
    parser.add_argument('--img-folder', default='figures/')

    args = parser.parse_args()

    if args.randomize == 0:
        covid_sim(args)
    else:
        multiprocess(args)
