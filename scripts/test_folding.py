import sys
sys.path.append('../')
from sampler import MCMC

Sampler = MCMC.SequenceOptimizer(native_seq='LNLKDSIGLRIKTERERQQMSREVLCLDGAE', n_iter=1000, n_traj=5)
Sampler.run()