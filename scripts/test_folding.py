import sys
sys.path.append('../')
from sampler import MCMC

Sampler = MCMC.SequenceOptimizer(native_seq='CNLARCQLSCKSLGLKGGCQGSFCTCG',
                                 n_iter=1000, n_traj=5, mut_p=(0.6,0.3,0.1))
Sampler.run()