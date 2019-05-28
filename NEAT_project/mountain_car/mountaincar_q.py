import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt

EPISODES = 1000
ENVIRONMENT = 'MountainCar-v0'
TRAIN = False

class DQNAgent:
    "Discretized action DQN agent"
    def __init__(self, env = ENVIRONMENT, generation = EPISODES, batch_size = 100, train = True):
        self.env = gym.make(env)
        self.d_state = self.env.observation_space.shape[0]
        self.nb_action = self.env.action_space.n
        self.memory =  deque(maxlen=1000)
        self.gamma = 0.95  
        self.epsilon = 0.8
        self.decay = 0.8
        self.epsilon_min = 0.1    
        self.l_r = 0.01
        self.model = self.Network()
        self.generation = generation
        self.batch_size = batch_size
        self.train = train
        self.ep_reward = []
        

    def Network(self):
        model = Sequential()
        model.add(Dense(8, input_dim=self.d_state, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.nb_action, activation='linear'))
        print(model.summary())
        model.compile( optimizer=Adam(lr=self.l_r), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nb_action)
        action = self.model.predict(state)
        return np.argmax(action[0])  

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay

    def load(self, file_name):
        self.model.load_weights(file_name)

    def save(self, file_name):
        self.model.save_weights(file_name, overwrite = True)

    def Train(self):
        
        for e in range(self.generation):
            state = self.env.reset()
            state = np.reshape(state, [1, self.d_state])
            done = False
            q = 0
            while not done:
                #self.env.render()
                action = self.epsilon_greedy(state)
                observation, reward, done, _ = self.env.step(action)

                observation = np.reshape(observation, [1, self.d_state])
                self.remember(state, action, reward, observation, done)
                state = observation
                q += reward

            self.ep_reward.append(q)
            print("total reward: {}, episode: {}".format(q, e))
                # warm up play
            if len(self.memory) > self.batch_size:
                self.replay()

        self.save('DQN_{}_weights'.format(ENVIRONMENT))

    def Test(self):
        self.load('DQN_{}_weights'.format(ENVIRONMENT))
        for i in range(5):
            state = self.env.reset()
            state = np.reshape(state, [1, self.d_state])
            done = False
            q = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state)[0])
                observation, reward, done, _ = self.env.step(action)
                q += reward
                observation = np.reshape(observation, [1, self.d_state])
                state = observation
   
            
            print(print("total reward: {}".format(q)))
            self.env.close()

    def main(self):
        if self.train == True:
            self.Train()
        else:
            self.Test()


if __name__ ==  "__main__":
    mc_agent = DQNAgent(train = TRAIN)
    mc_agent.main()

    #ep_time = np.linspace(0, len(np.array(mc_agent.ep_reward)), len(np.array(mc_agent.ep_reward)))

    #plt.plot(np.array(ep_time), np.array(mc_agent.ep_reward), 'b')
    #plt.xlabel('time episode')
    #plt.ylabel('total episode reward')
    #plt.show()
    