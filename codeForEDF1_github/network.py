import numpy as np
from scipy.linalg import eig


def configure_network(network_type):
    if network_type == 'data':
        cc = 0.4

        W = np.array([
            [0.4, 0.0, 0.0, 0.0],
            [0.00, 0.3, cc, 0.0],
            [0.03, cc, 0.7, -0.3],
            [0.0, cc, -0.3, 0.7]
        ])          

        inputVector = np.array([50, 0, 0, 0])  # inputVector = np.array([6, 0, 0, 0])

        cueVector = np.array([0.0, 0.0, 0.0, 0.0])

        s1, s2 = -1, -0.3
           

    elif network_type == 'two_region_integrator_1input':
        cc = 0.4
        dd = 0.6
        
        W = np.array([
            [cc,       -0.3,          0.7 - cc,           dd ],
            [-0.3,     cc,            0.0,   0.7 - cc + dd   ],
            [0.7 - cc + dd,  0.0 ,    cc,            -0.3    ],
            [dd, 0.7 - cc,            -0.3,          cc      ]
        ])
        
        inputVector = np.array([3, 0, 0, 0])
        cueVector = np.array([0.0, 0.0, 0.0, 0.0])

        s1, s2 = -1, -0.2
        
    
    elif network_type == 'two_integrator_1input':    
        cc = 0.4  # 0.2

        W = np.array([
            [0.7, -0.3, cc, 0.0],
            [-0.3, 0.7, cc, 0.0],
            [cc, 0.0, 0.7, -0.3],
            [cc, 0.0, -0.3, 0.7]
        ])
        
       
        inputVector = np.array([3, 0, 0, 0])
        cueVector = np.array([0.0, 0.0, 0.0, 0.0])

        s1, s2 = -1, -0.2

    elif network_type == 'one_integrator_follower_1input':
        cc = 0.3
        
        W = np.array([
            [0.6, -0.3, 0.0, cc],
            [-0.3, 0.6, 0.0, cc],
            [cc, 0.0, 0.7, -0.3],
            [cc, 0.0, -0.3, 0.7]
        ])
        
        inputVector = np.array([2, 0, 0, 0])
        cueVector = np.array([0.0, 0.0, 0.0, 0.0])

        s1, s2 = -1, -0.2
        

    elif network_type == 'one_integrator_follower_opposite_1input':
        cc = 0.5 #0.3
        
        W = np.array([
            [0.7, -0.3, cc, 0.0],
            [-0.3, 0.7, cc, 0.0],
            [cc, 0.0, 0.6, -0.3],
            [cc, 0.0, -0.3, 0.6]
        ])
        inputVector = np.array([2, 0, 0, 0])
        cueVector = np.array([0.0, 0.0, 0.0, 0.0])

        s1, s2 = -0.2, -1        

    elif network_type == 'one_integrator_follower_opposite_leaky_1input':  
        cc = 0.3
        
        W = np.array([
            [0.69, -0.3, cc, 0.4],
            [-0.3, 0.69, 0.0, 0.4],
            [0.0, 0.0, 0.4, 0.0],
            [0.4, 0.0, 0.0, 0.3]
        ])
        

        inputVector = np.array([0, 0, 10, 0]) # input to STR only
        cueVector = np.array([0.0, 0.0, 0.0, 0.0])

        s1, s2 = -0.2, -1     
        
    elif network_type == 'externally_driven_1input':
        cc = 0.6

        W = np.array([
            [0.4, -0.3, cc, 0.0],
            [-0.3, 0.4, cc, 0.0],
            [cc, 0.0, 0.4, -0.3],
            [cc, 0.0, -0.3, 0.4]
        ])
        inputVector = np.array([0.5, 0, 0, 0])
        cueVector = np.array([0.0, 0.0, 0.0, 0.0])

        s1, s2 = -2, -2        

    else:
        raise ValueError("Invalid network type provided")

    return W, inputVector, cueVector, s1, s2
