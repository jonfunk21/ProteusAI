import sys
sys.path.append('../')
from sampler import MCMC

Sampler = MCMC.Hallucination(native_seq='PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK',
                                 n_iter=5000, n_traj=5, T=100, mut_p=(0.5,0.3,0.1))
Sampler.run()