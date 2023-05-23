import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
pygame.init()
font = pygame.font.Font('arial.ttf',25)

# Reset 
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Point = namedtuple('Point','x , y')

BLOCK_SIZE=20
SPEED = 80
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

class SnakeGameAI:
    def __init__(self,w=640,h=480):
        self.w=w
        self.h=h
        #init display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('SnakeAi by Shua Friedman and Dovid Sparzadeh')
        #add a count in the corner keeping track of which number round we are in        
        self.clock = pygame.time.Clock()
        #init game state
        self.reset_game()

    def reset_game(self): #Add a reset function for the training to automatically reset the game.n #TODO
        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,  Point(self.head.x-BLOCK_SIZE,self.head.y),  Point(self.head.x-(2*BLOCK_SIZE),self.head.y)] #snake initialization 

        self.direction = Direction.RIGHT
        self.food = None
        self._place__food()
        self.score = 0
        self.frame_count = 0 #Add a frame iteration to keep track of the number of frames the snake has been alive.
      
    def _place__food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if(self.food in self.snake):
            self._place__food()

    #For the agent, we need to change this whole function to take in the action from the model #TODO 
    def play_step(self,action): #action comes from "final_move" in the agent file. 
        self.frame_count+=1
        # 1. Collect the user input
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            
        # 2. Move
        self._move(action)
        self.snake.insert(0,self.head) #adds in a node to the snake

        # 3. Check if game Over
        #HEre, we are adding in the reward values for the model #TODO
        reward = 0  # eat food: +10 , game over: -10 , else: 0
        game_over = False 
        if(self.is_collision() or self.frame_count > 100*len(self.snake) ): #if the snake hits itself, a border, or if the snake has been alive for too long with nothing happening, then the game is over.
            game_over=True
            reward = -10
            return reward,game_over,self.score
        # 4. Place new Food or just move
        if(self.head == self.food):
            self.score+=1
            reward=10
            self._place__food()
            
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game Over and Display Score
        
        return reward,game_over,self.score

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display,BLUE1,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(self.display,BLUE2,pygame.Rect(pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render("Score: "+str(self.score),True,WHITE)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def _move(self,action): #need to adapt the move function to take in the action from the model #TODO
        #action comes from "final_move" in the agent file. 
        # Action
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn 
        # [0,0,1] -> Left Turn

        clockwise_directions = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP] #define all the possible directions in a clockwise order.
        idx = clockwise_directions.index(self.direction) #get the current index of the current direction.
        if np.array_equal(action,[1,0,0]):
            new_dir = clockwise_directions[idx] # stay on current direction
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clockwise_directions[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clockwise_directions[next_idx] # Left Turn
        self.direction = new_dir

        #reposition the head of the snake
        x = self.head.x     
        y = self.head.y
        if(self.direction == Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(self.direction == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(self.direction == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(self.direction == Direction.UP):
            y-=BLOCK_SIZE
        self.head = Point(x,y)

    def is_collision(self,pt=None): #adding to the function a pt arguement to check if the snake is hitting the boundary.
        if(pt is None):
            pt = self.head
        #hit boundary
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.snake[1:]):
            return True
        return False
