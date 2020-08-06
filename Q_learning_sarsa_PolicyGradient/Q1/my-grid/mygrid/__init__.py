from gym.envs.registration import register

#logger = logging.getLogger(__name__)
# print("in registering")
register(
    id = 'gridenv-v0',
    entry_point='mygrid.envs:gridenv',
    #timestep_limit=40
)
register(
    id = 'gridenv-v1',
    entry_point='mygrid.envs:gridenv1',
    #timestep_limit=40
)
register(
    id = 'gridenv-v2',
    entry_point='mygrid.envs:gridenv2',
    #timestep_limit=40
)
