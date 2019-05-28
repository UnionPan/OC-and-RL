import os
import neat
import gym
import visualize
import pickle
import multiprocessing as mp


os.chdir("./checkpoints")

NUM_GENERATION = 100
CONFIG_FILE = "../config"

ENVIRONMENT = "MountainCar-v0"
TRAIN = False

CHECKPOINT = 99


class NEAT:
    def __init__(self, num_generation, parallel=2, train = True):
        self.num_generation = num_generation
        self.parallel = parallel
        self.train = train


    @staticmethod
    def _policy_evaluation(genome, config, obj):
        "static method: used for evaluate a single genome's fitness"
        env = gym.make(ENVIRONMENT)

        state = env.reset()

        done = False # stopping criterion
        q = 0 # cumulative reward

        net = neat.nn.FeedForwardNetwork.create(genome, config)  # create the network using config file

        while not done:
            value = net.activate(state.flatten()) # activate the network to have the value output
            action = value.index(max(value)) # choose the action based on index if max value

            observation, reward, done, _ = env.step(action) # step forward and see what heppens
            
            q += reward 
            state = observation # state transition

        fitness = q 
        obj.put(fitness) 

        env.close()
        


    def eval_genomes(self, genomes, config):
        "for all the genomes run p of them at once in every iteration"
        idx, genomes = zip(*genomes)

        for i in range(0, len(genomes), self.parallel):
            output = mp.Queue()
            processes = [mp.Process(target=self._policy_evaluation, args=(genome, config, output)) 
                             for genome in genomes[i:i + self.parallel]]  # Define all the processes

            # Run the processes
            [p.start() for p in processes]
            [p.join() for p in processes]

            results = [output.get() for _ in processes]
            for n, r in enumerate(results):
                genomes[i + n].fitness = r



    def startbreed(self, config_file, generations):
        
        "training process, evaluate the genomes using parallel staticmehod in every generation"
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        #pop = neat.Population(config)
        pop = neat.Checkpointer.restore_checkpoint("neat-checkpoint-"+str(CHECKPOINT))

        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(20))

        winner = pop.run(self.eval_genomes, generations)

        pickle.dump(winner, open('winner.pkl', 'wb'))

        node_names = {-1: 'position', -2: 'velocity',
                        0: 'action1', 1: 'action2', 2: 'action3'}
        visualize.draw_net(config, winner, True, node_names = node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)


    def test_genomes(self, config_file = CONFIG_FILE):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)

        genome = pickle.load(open("winner.pkl", "rb"))
        fitness = 0
        env = gym.make(ENVIRONMENT)

        for i in range(5):
            state = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            q = 0

            while not done:
                output = net.activate(state.flatten())
                action = output.index(max(output))
                observation, reward, done, _= env.step(action)
                state = observation
                q += reward
                env.render()  


            fitness = q
            print("Fitness {}".format(fitness))

            env.close()


    def main(self, config_file=CONFIG_FILE):
        local_dir = os.path.dirname(__file__)   
        config_path = os.path.join(local_dir, config_file)
        if self.train == True:
            self.startbreed(config_path, self.num_generation)
        else:
            self.test_genomes()



if __name__ == "__main__":
    evolution = NEAT(NUM_GENERATION, train = TRAIN)
    evolution.main()



