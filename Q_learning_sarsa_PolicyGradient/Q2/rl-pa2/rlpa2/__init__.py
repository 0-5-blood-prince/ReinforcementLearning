#import logging
from gym.envs.registration import register

#logger = logging.getLogger(__name__)
# print("in registering")
register(
    id = 'chakra-v0',
    entry_point='rlpa2.envs:chakra',
    #timestep_limit=40
)
register(
    id = 'vishamC-v0',
    entry_point='rlpa2.envs:vishamC',
    #timestep_limit=40
)
