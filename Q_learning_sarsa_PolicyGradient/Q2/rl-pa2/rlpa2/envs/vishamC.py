from gym import Env
# from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np


class vishamC(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low=-0.025, high=0.025, shape=(2,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
        self._seed()
        self.viewer = None
        self.state = None
        
        _=self._reset()
        
    def isT(self,state):
        return np.linalg.norm(state)<0.01
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reward(self,state):
        return -( (0.5*pow(state[0],2)) + ( 5*pow(state[1],2) ) )
    def offspace(self,state):
        return state[0]>=1 or state[0]<=-1 or state[1]<=-1 or state[1]>=1
    def _step(self, action):
        #Fill your code here
        st = self.state
        resulting_state = self.state + action
        if self.offspace(resulting_state):
            _=self._reset()
            return self.state,-self.reward(st)+self.reward(resulting_state),self.isT(self.state),None
        else:
            self.state = resulting_state
            return self.state,-self.reward(st)+self.reward(resulting_state),self.isT(self.state),None# Return the next state and the reward, along with 2 additional quantities : False, {}

    def _reset(self):
        while True:
            self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
            # Sample states that are far away
            if np.linalg.norm(self.state) > 0.9:
                break
        return np.array(self.state)

    # method for rendering

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            origin = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            trans = rendering.Transform(translation=(0, 0))
            agent.add_attr(trans)
            self.trans = trans
            agent.set_color(1, 0, 0)
            origin.set_color(0, 0, 0)
            origin.add_attr(rendering.Transform(
                translation=(screen_width // 2, screen_height // 2)))
            self.viewer.add_geom(agent)
            self.viewer.add_geom(origin)

        # self.trans.set_translation(0, 0)
        self.trans.set_translation(
            (self.state[0] + 1) / 2 * screen_width,
            (self.state[1] + 1) / 2 * screen_height,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# register(
#     'vishamC-v0',
#     entry_point='rlpa2.vishamC:vishamC',
#     timestep_limit=40,
# )
