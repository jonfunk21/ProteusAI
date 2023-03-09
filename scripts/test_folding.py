import sys
sys.path.append('../')
from sampler import MCMC

Sampler = MCMC.SequenceOptimizer(native_seq='LNLKDSIGLRIKTERERQQMSREVLCLDGAE', n_iter=1, n_traj=2)
Sampler.run()