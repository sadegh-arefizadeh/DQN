import collections
import numpy as np
import torch
import gym
from gym import spaces
from numpy.random import randn

class my_env(gym.Env):

    def __init__(self,tmp_fix,initial_tmp,humid_fix,initial_Humid,tmp_out,humid_out,number_of_timesteps=1000,change_setting=False, limit=False,seed=None):

        self.step_period = 0
        self.tmp_fix = tmp_fix
        self.initial_Humid = initial_Humid
        self.initial_tmp = initial_tmp
        self.tmp_old = initial_tmp
        self.humid_fix = humid_fix
        self.humid_old = initial_Humid
        self.tmp_out = tmp_out
        self.humid_out = humid_out
        self.change_setting = change_setting
        self.seed = seed

        self.A_1, self.A_2, self.A_3, self.A_4 = 0.006, 0.004, 0.05, 0.01
        self.B_1, self.B_2 = 0.05, 0.8
        self.C_1, self.C_2, self.C_3 = 0.02, 0.06, 1
        self.D_1, self.D_2 = 0.05, 0.01
        self.E_1, self.E_2 = 0.05, 0.1


        self.limit = limit
        self.number_of_timesteps = number_of_timesteps

        A1_A2 = [0.7,1.3]
        C1_C2 = [0.7,1.3]


        if seed is None:
            pass
        elif seed == 0:
            self.A_1, self.A_2 = A1_A2[0]*self.A_1, A1_A2[0]*self.A_2
            self.B_2 = A1_A2[1]*self.B_2

            self.C_1, self.C_2 = C1_C2[0]*self.C_1, C1_C2[0]*self.C_2

        elif seed == 1:
            self.A_1, self.A_2 = A1_A2[1]*self.A_1, A1_A2[1]*self.A_2
            self.B_2 = A1_A2[0] * self.B_2

            self.C_1, self.C_2 = C1_C2[0]*self.C_1, C1_C2[0]*self.C_2

        elif seed == 2:
            self.A_1, self.A_2 = A1_A2[0]*self.A_1, A1_A2[0]*self.A_2
            self.B_2 = A1_A2[1] * self.B_2

            self.C_1, self.C_2 = C1_C2[1]*self.C_1, C1_C2[1]*self.C_2

        elif seed == 3:
            self.A_1, self.A_2 = A1_A2[1]*self.A_1, A1_A2[1]*self.A_2
            self.B_2 = A1_A2[0] * self.B_2

            self.C_1, self.C_2 = C1_C2[1]*self.C_1, C1_C2[1]*self.C_2


        self.tmp_element_new = initial_tmp
        self.tmp_element_old = initial_tmp
        self.tmp_water_new = initial_tmp
        self.tmp_water_old = initial_tmp
        self.tmp_humidfier_new =initial_tmp
        self.tmp_humidfier_old = initial_tmp

        self.h = 10
        self.i = 2 #1
        self.j = 0.01 #0.03
        self.temprature = collections.deque(np.ones(15)*initial_tmp, maxlen=15)
        self.humid = collections.deque(np.ones(15) * initial_Humid, maxlen=15)


        self.temprature_observed = collections.deque(np.ones(15)*initial_tmp, maxlen=15)
        self.humid_observed = collections.deque(np.ones(15) * initial_Humid, maxlen=15)



        self.tmp_new = np.array(initial_tmp)
        self.humid_new = np.array(initial_Humid)
        self.element = collections.deque(np.zeros(15), maxlen=15)
        self.humider = collections.deque(np.zeros(15), maxlen=15)
        self.fans = collections.deque(np.zeros(15), maxlen=15)


        self.error_new_tmp = 0
        self.all_error_tmp = 0
        self.error_old_tmp = 0

        self.error_new_humid = 0
        self.all_error_humid = 0
        self.error_old_humid = 0


        self.action_space = spaces.Discrete(8)




        self.observation_space = spaces.Dict(
            {"temp_new": spaces.Box(low=20, high=60, shape=(15,)),
             "temp_fix": spaces.Box(low=20, high=60, shape=(1,)),
             "temp_out": spaces.Box(low=0, high=60, shape=(1,)),
             "humid_new": spaces.Box(low=0, high=100, shape=(15,)),
             "humid_fix": spaces.Box(low=0, high=100, shape=(1,)),
             "humid_out": spaces.Box(low=0, high=100, shape=(1,)),
             "elem": spaces.Box(low=-0, high=1, shape=(15,)),
             "humider": spaces.Box(low=-0, high=1, shape=(15,)),
             "fan": spaces.Box(low=-0, high=1, shape=(15,))})


    def reset(self):
        print("env reset")
        self.step_period = 0


        self.tmp_out = 16 + np.random.uniform(low=0,high=16)
        self.humid_out = 20 + np.random.uniform(low=0,high=40)

        self.tmp_new = self.initial_tmp + np.random.uniform(low=0,high=4)
        self.humid_new = self.initial_Humid + np.random.uniform(low=0,high=30)
        self.tmp_old = np.array(self.tmp_new)
        self.humid_old = np.array(self.humid_new)

        self.tmp_element_new = 40
        self.tmp_element_old = 40
        self.tmp_water_new = 40
        self.tmp_water_old = 40
        self.tmp_humidfier_new = 40
        self.tmp_humidfier_old = 40

        self.error_new_tmp = 0
        self.all_error_tmp = 0
        self.error_old_tmp = 0

        self.error_new_humid = 0
        self.all_error_humid = 0
        self.error_old_humid = 0

        self.temprature = collections.deque(np.ones(15)*self.tmp_new, maxlen=15)
        self.humid = collections.deque(np.ones(15) * self.humid_new, maxlen=15)

        self.temprature_observed = collections.deque(np.ones(15)*self.tmp_new, maxlen=15)
        self.humid_observed = collections.deque(np.ones(15) * self.humid_new, maxlen=15)

        self.element = collections.deque(np.zeros(15), maxlen=15)
        self.humider = collections.deque(np.zeros(15), maxlen=15)
        self.fans = collections.deque(np.zeros(15), maxlen=15)





        return self.current_state(), {}

    def current_state(self):


        state = {"temp_new":np.array(self.temprature, dtype=np.float32),
                "temp_fix": np.array(np.array([self.tmp_fix]),dtype=np.float32),
                 "temp_out": np.array(np.array([self.tmp_out]),dtype=np.float32),
                 "humid_new": np.array(self.humid_observed, dtype=np.float32),
                 "humid_fix":np.array(np.array([self.humid_fix]), dtype=np.float32),
                 "humid_out":np.array(np.array([self.humid_out]), dtype=np.float32)}




        # for i in range(0,5):
        state["elem"] = np.array(self.element, dtype=np.float32)
        state["humider"] = np.array(self.humider, dtype=np.float32)
        state["fan"] = np.array(self.fans, dtype=np.float32)



        return state

    def PID_(self, P_tmp, I_tmp, D_tmp, P_humid, I_humid, D_humid):
          action = [0, 0, 0]
          self.error_new_tmp = self.temprature_observed[-1] - self.tmp_fix
          controller_tmp = P_tmp * self.error_new_tmp



          self.error_new_humid = self.humid_observed[-1] - self.humid_fix
          self.all_error_humid = self.all_error_humid + self.error_new_humid
          controller_humid = P_humid * self.error_new_humid



          if controller_tmp < -2:
              action[0] = 1

          elif controller_tmp < -1:
              action[0] = 0.25

          elif controller_tmp < -0.5:
              action[0] = 0.125

          elif controller_tmp < 0:
              action[0] = 0.0625

          elif controller_tmp <3:
              action[0] = 0
          else:
              action[0] = 0
              action[2] = 1

          if controller_humid < 0:
              action[1] = 1
          elif controller_humid < 5:
              action[1] = 0
          else:
              action[1] = 0
              action[2] = 1
          return action

    def step(self,actionn, mode="learning_mode",pid=False):
        if actionn == 0:
            action = [0, 0, 0]
        elif actionn == 1:
            action = [1, 0, 0]
        elif actionn == 2:
            action = [0, 1, 0]
        elif actionn == 3:
            action = [1, 1, 0]
        elif actionn == 4:
            action = [0, 0, 1]
        elif actionn == 5:
            action = [1, 0, 1]
        elif actionn == 6:
            action = [0, 1, 1]
        elif actionn == 7:
            action = [0, 1, 1]
        elif pid == True:
            action = actionn
        else:
            print("error!!!")

        elem, humiderr, fan = action




        self.fans.append(np.array(fan).item())
        self.element.append(np.array(elem).item())
        self.humider.append(np.array(humiderr).item())

        noises = randn(2)*1


        self.tmp_old = self.tmp_new
        self.tmp_element_old = self.tmp_element_new
        self.humid_old = self.humid_new

        self.tmp_water_old = self.tmp_water_new
        self.tmp_humidfier_old = self.tmp_humidfier_new





        self.tmp_new = self.tmp_old + (self.A_1 + self.A_2*fan)*(self.tmp_out-self.tmp_old) + self.A_3*(self.tmp_element_old - self.tmp_old) +0.3* self.A_4*(self.tmp_water_old - self.tmp_old)
        self.tmp_element_new = self.tmp_element_old + self.B_1*(self.tmp_old - self.tmp_element_old) + self.B_2*elem
        self.humid_new = self.humid_old + (self.C_1 + self.C_2*fan) * (self.humid_out - self.humid_old) + self.C_3 * (self.tmp_water_old-self.tmp_fix)*(self.tmp_water_old-self.tmp_fix>0)
        self.tmp_water_new = self.tmp_water_old + self.D_1*(self.tmp_humidfier_old - self.tmp_water_old) + self.D_2 * (self.tmp_old - self.tmp_water_old)
        self.tmp_humidfier_new = self.tmp_humidfier_old + self.E_1 * (self.tmp_water_old - self.tmp_humidfier_old) + self.E_2 * humiderr




        self.temprature_observed.append(np.array(self.tmp_new +noises[0]*0.1).item())
        self.humid_observed.append(np.array(self.humid_new +noises[1]*0.4).item())

        self.temprature.append(np.array(self.tmp_new).item())
        self.humid.append(np.array(self.humid_new).item())

        if self.step_period >= self.number_of_timesteps-1:
            # print("whole steps : {}".format(self.step_period_whole))

            self.step_period = 0
            rewardd = self.reward()
            return self.current_state(), rewardd, False, True, {}
        else:
            self.step_period = self.step_period + 1
            return self.current_state(), self.reward(), False, False, {}


    def reward(self):

        R = -self.h * 1.0* (np.abs(self.tmp_fix - self.tmp_new)) ** 1 + -self.i * (
            np.abs(self.humid_fix - self.humid_new)) ** 1 - self.j * (self.element[-1] + self.humider[-1] - 4*self.fans[-1])

        return R
    def random_sample(self):
        return [np.random.choice([0,1,2,3,4,5,6,7])]

    def close(self):
        pass

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print(self.current_state())

