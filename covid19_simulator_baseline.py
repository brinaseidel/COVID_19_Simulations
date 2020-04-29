import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Read in origin-destination flow matrix
#OD = np.genfromtxt('data/Yerevan_OD_coronavirus.csv', delimiter=',')
OD = np.genfromtxt('data/nyc_od.csv', delimiter=',')


# initialize the population vector from the origin-destination flow matrix
N_k = np.abs(np.diagonal(OD) + OD.sum(axis=0) - OD.sum(axis=1))
locs_len = len(N_k)                 # number of locations
SIR = np.zeros(shape=(locs_len, 3)) # make a numpy array with 3 columns for keeping track of the S, I, R groups
SIR[:,0] = N_k                      # initialize the S group with the respective populations
thresh=200
first_infections = np.where(SIR[:, 0]<=thresh, SIR[:, 0]//20, 0)   # for demo purposes, randomly introduce infections
# NOTE: this is arbitrary but not actually random.... 
SIR[:, 0] = SIR[:, 0] - first_infections
SIR[:, 1] = SIR[:, 1] + first_infections                           # move infections to the I group

# row normalize the SIR matrix for keeping track of group proportions
row_sums = SIR.sum(axis=1)
SIR_n = SIR / row_sums[:, np.newaxis]

# initialize parameters
beta = 1.6
gamma = 0.04
public_trans = 0.5                                 # alpha
R0 = beta/gamma
beta_vec = np.random.gamma(1.6, 2, locs_len)
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
ax.figure.savefig('figures/sir_plot.png')



def simulation(beta=0.75, gamma=0.2, days=100, public_trans=0.56, public_trans_vec=None, plot=False):
    # Read in origin-destination flow matrix
    #OD = np.genfromtxt('data/Yerevan_OD_coronavirus.csv', delimiter=',')
    OD = np.genfromtxt('data/nyc_od.csv', delimiter=',')


    # initialize the population vector from the origin-destination flow matrix
    N_k = np.abs(np.diagonal(OD) + OD.sum(axis=0) - OD.sum(axis=1))
    locs_len = len(N_k)                 # number of locations
    SIR = np.zeros(shape=(locs_len, 3)) # make a numpy array with 3 columns for keeping track of the S, I, R groups
    SIR[:,0] = N_k                      # initialize the S group with the respective populations
    thresh=200
    first_infections = np.where(SIR[:, 0]<=thresh, SIR[:, 0]//20, 0)   # for demo purposes, randomly introduce infections
    # NOTE: this is arbitrary but not actually random....
    SIR[:, 0] = SIR[:, 0] - first_infections
    SIR[:, 1] = SIR[:, 1] + first_infections                           # move infections to the I group

    # row normalize the SIR matrix for keeping track of group proportions
    row_sums = SIR.sum(axis=1)
    SIR_n = SIR / row_sums[:, np.newaxis]

    # initialize parameters
    beta = beta         # Number of ppl an infectious person infects per day
    gamma = gamma        # 1/how many days you would infect ppl for (0.2 assumes 5 days, and that people quarantine once symptoms develop)
    public_trans = public_trans                                # alpha (how many people are moving around)
    R0 = beta/gamma
    beta_vec = np.random.gamma(beta, 2, locs_len)   # gamma dist multiplies beta * 2 to give a mean of 1.5
    gamma_vec = np.full(locs_len, gamma)
    if public_trans_vec is None:
        public_trans_vec = np.full(locs_len, public_trans)

    # make copy of the SIR matrices
    SIR_sim = SIR.copy()
    SIR_nsim = SIR_n.copy()

    # run model
    #print(SIR_sim.sum(axis=0).sum() == N_k.sum())
    infected_pop_norm = []
    susceptible_pop_norm = []
    recovered_pop_norm = []

    for time_step in tqdm(range(days)):
        infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
        OD_infected = np.round(OD*infected_mat)
        inflow_infected = OD_infected.sum(axis=0)
        inflow_infected = np.round(inflow_infected*public_trans_vec[time_step])
#         print('total infected inflow: ', inflow_infected.sum())
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
#         print(S, I, R, (S+I+R)*N_k.sum(), N_k.sum())
#         print('\n')
        infected_pop_norm.append(I)
        susceptible_pop_norm.append(S)
        recovered_pop_norm.append(R)

    # plot results
    if plot==True:
        fig = plt.figure()
        ax = plt.axes()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.plot(np.arange(days), susceptible_pop_norm, label='Susceptible', color='#4aa5f0', linewidth=2)
        ax.plot(np.arange(days), infected_pop_norm, label='Infected', color='#f03737', linewidth=2)
        ax.plot(np.arange(days), recovered_pop_norm, label='Recovered', color='#82e88a', linewidth=2)
        ax.plot(np.arange(days), public_trans_vec, '--', label='public_trans_vec', color='k', linewidth=2)
        ax.legend(frameon=False)
        ax.set_xlabel("Days")
        ax.set_ylabel("Share of Population")
        #ax.figure.savefig('figures/sir_plot.png')
        plt.show()