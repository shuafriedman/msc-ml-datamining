# -*- coding: utf-8 -*-
import Othelo
import alphaBetaPruning

class GameManager:
    def __init__(self):
        # Initialize the current player to the human.
        self.current_player = "human"
        self.othelo=Othelo.Othello(8,8)

    def switch_turn(self):
        # Switch the current player from the human to the computer or vice versa.
        if self.current_player == "human":
            self.current_player = "computer"
        else:
            self.current_player = "human"
    
    def turn_input(self):
        if self.current_player == "human":
            while not(self.othelo.read_input_and_update_board('O')):
                print("Try again")
        else:
            move=self.othelo.inputComputer()
            #print("next move is: ", move)
            
            
    def game_over(self):
        return self.othelo.check_game_over()
    
    def print_board(self):
        self.othelo.print_board()
    
    def check_winner(self):
        return self.othelo.check_winner()
            

            
if __name__ == "__main__":
    # Create a new Othello game with 8 rows and 8 columns
    game = GameManager()
    game.print_board() 

    # Place a piece on the board at row 0, column 0

    
    # Flip any pieces that need to be flipped
    x=0
    while not(game.game_over()) and x!=99:
        print("Turn of ", game.current_player)
        game.turn_input()
        game.switch_turn()
        print("Current Board:")
        game.print_board()
        #x=int(input("Enter a number (99 to exit)"))

    print("GAME OVER") 
    print("The winner is: ", game.check_winner()[0])