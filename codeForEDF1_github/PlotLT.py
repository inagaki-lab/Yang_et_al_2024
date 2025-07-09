import numpy as np
from scipy.io import savemat

# other network_type:
# 'externally_driven_1input'
# 'two_region_integrator_1input'
# 'two_integrator_1input'
# 'one_integrator_follower_opposite_1input'
# 'one_integrator_follower_opposite_leaky_1input'
# 'data'

data = np.load(r'C:\Users\yangzd\Inagaki Lab Dropbox\Zidan Yang\ED1code\data\LickTimes.npy', allow_pickle=True)

savemat(r'C:\Users\yangzd\Inagaki Lab Dropbox\Zidan Yang\ED1code\data\LickTimes.mat', {'LT': data})