import torch 
import random 
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import NN,QTrainer
from Helper import plot
import pygame
MAX_MEMORY = 100_000 #max memory size for deque
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        #display n_game in the top corner
        
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = NN(11,256,3) #11 states, 256 hidden layer, 3 action outputs (e.g. [1,0,0])
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         
        # TODO: model,trainer

    # state (11 Values)
    #[ danger straight, danger right, danger left,
    #   
    # direction left, direction right,
    # direction up, direction down
    # 
    # food left,food right,
    # food up, food down]
    def get_state(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y) #create 4 points around the head
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT #check the current direction: one of these will be true and the rest will be false
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [ #list of the 11 states:
            # Danger Straight-- all the options where the danger is straight
            (dir_u and game.is_collision(point_u))or #for example, if the current directoin is up, and the point above the head is a collision, then the snake is in danger straight
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger Left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, # food is in left
            game.food.x > game.head.x, # food is in right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state,dtype=int) #convert the states to boolean values

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE): #if the memoery is greater than the batch size, than take a random sample from the memory
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones) #The long memory takes in lists of states, actions, rewards, next_states, and dones, 
                                                                            #as opposed to a single state, action, reward, next_state, and done.
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # exploration / exploitation tradeoff using epsilon value for randomness
        self.epsilon = 80 - self.n_game #the further into the game we are, the smaller epsilon gets.
        final_move = [0,0,0] #intialized to 0
        if(random.randint(0,200)<self.epsilon): #if the random number is less than the epsilon value, then the snake will move randomly
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0) # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while agent.n_game < 1000:
        # Get Old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done) #store in the deque

        #if done or n_game is divisible by 100, then train long memory
        if done or agent.n_game % 100 == 0:
            # Train long memory,plot result
            game.reset_game() #resets the game state
            agent.n_game += 1
            agent.train_long_memory() 
            if(score > reward): # new High score #TODO I think this is wrong
                reward = score
                agent.model.save()
                if score > record:
                  record = score
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            #save the plot        


if(__name__=="__main__"):
    train()