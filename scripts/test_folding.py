import sys
sys.path.append('../')
from sampler import MCMC

Sampler = MCMC.SequenceOptimizer(native_seq='LNLKDSIGLRIKTERERQQMSREVLCLDGAE', n_iter=20, n_traj=3)
Sampler.run()