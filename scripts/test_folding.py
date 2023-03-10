import sys
sys.path.append('../')
from sampler import MCMC

Sampler = MCMC.SequenceOptimizer(native_seq='PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK',
                                 n_iter=5000, n_traj=5, T=100, mut_p=(0.2,0.6,0.1))
Sampler.run()