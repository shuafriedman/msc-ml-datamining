import alphaBetaPruning
import copy

VICTORY=10**20 #The value of a winning board (for max) 
LOSS = -VICTORY #The value of a losing board (for max)
TIE=0 #The value of a tie
COMPUTER='X' #Marks the computer's cells on the board
HUMAN='O' #Marks the human's cells on the board

class Othello:
  def __init__(self, rows, cols):
    self.rows = rows
    self.cols = cols
    self.board = [['-' for c in range(cols)] for r in range(rows)]
    self.place_initial_piece(4, 3, 'X')
    self.place_initial_piece(4, 4, 'O')
    self.place_initial_piece(3, 3, 'O')
    self.place_initial_piece(3, 4, 'X')
    
  def __str__(self):
     return str(self.board)

    
    

  def read_input_and_update_board(self, turn):
      # Get the current player's symbol (either "X" or "O")

      # Read input from the user in the format "X Y"
      x, y = map(int, input("Enter move coordinates (X Y): ").split())

      # Update the board with the current player's symbol at the specified coordinates
      if x >= 0 and x < self.rows and y >= 0 and y < self.cols:
          if self.place_piece(x, y, turn)==False:
              print("Ilegal move")  
              return False
          else:
              self.flip_pieces(x, y, turn)
              return True
      else:
          print("Ilegal move")
          return False

  def place_piece(self, row, col, color):
   #print("in place_piece. self board in place {},{} is {}".format(row,col,self.board[row][col]))
   if self.board[row][col] != '-':
     # The specified position is already occupied
     #print("place_piece returns false")
     return False
   elif self.is_near_occupied_place(row, col):
     # Place the piece on the board
     self.board[row][col] = color 
     #print("update board {},{} to {}".format(row,col,color))
     return True
   else:
     return False

  def place_initial_piece(self, row, col, color):
   if self.board[row][col] != '-':
     # The specified position is already occupied
     return False
   else:
     # Place the piece on the board
     self.board[row][col] = color
     return True
   

  def is_near_occupied_place(self, x, y):
   # Check if any of the adjacent locations on the board are occupied
    if (
      # Check above
      x > 0 and self.board[x - 1][y] != "-"
      # Check below
      or x < len(self.board) - 1 and self.board[x + 1][y] != "-"
      # Check left
      or y > 0 and self.board[x][y - 1] != "-"
      # Check right
      or y < len(self.board[0]) - 1 and self.board[x][y + 1] != "-"
      # Check upper left diagonal
      or x > 0 and y > 0 and self.board[x - 1][y - 1] != "-"
      # Check upper right diagonal
      or x > 0 and y < len(self.board[0]) - 1 and self.board[x - 1][y + 1] != "-"
      # Check lower left diagonal
      or x < len(self.board) - 1 and y > 0 and self.board[x + 1][y - 1] != "-"
      # Check lower right diagonal
      or x < len(self.board) - 1 and y < len(self.board[0]) - 1 and self.board[x + 1][y + 1] != "-"
    ):
        return True
    else:
        return False

  def flip_pieces(self, row, col, color):
   # Check all directions for pieces to flip
   directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
   for r_offset, c_offset in directions:
     r = row + r_offset
     c = col + c_offset
     flip_list = []
     while r >= 0 and r < self.rows and c >= 0 and c < self.cols:
      if self.board[r][c] == color:
        # Found a piece of the same color, so flip the pieces in flip_list
        for r_flip, c_flip in flip_list:
          self.board[r_flip][c_flip] = color
        break
      elif self.board[r][c] == '-':
        # Empty space, so no pieces can be flipped in this direction
        break
      else:
        # Found a piece of the opposite color, so add it to the flip list
        flip_list.append((r, c))
        r += r_offset
        c += c_offset


  def check_game_over(self):
   # Check if the board is full
   if all(self.board[r][c] != '-' for r in range(self.rows) for c in range(self.cols)):
     return True

   # Check if there are any valid moves for either player
   for r in range(self.rows):
    for c in range(self.cols):
      if self.board[r][c] == '-':
        # Check all directions to see if a move is valid
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for r_offset, c_offset in directions:
          r2 = r + r_offset
          c2 = c + c_offset
          if r2 >= 0 and r2 < self.rows and c2 >= 0 and c2 < self.cols and self.board[r2][c2] != '-':
            # There is a valid move, so the game is not over
            return False

   # No valid moves were found, so the game is over
   return True
  
  def check_winner(self): #****Should be completed****
        # Count the number of "O" and "X" pieces on the board.
        # print('here')
        # print(self.check_game_over())
        countO=0
        countX=0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c]=="O":
                    countO+= 1
                elif self.board[r][c]=="X":
                    countX+= 1
        if countO>countX:
            return ("O", countO)
        elif countX>countO:
            return ("X", countX)
        else:
          return ("Tie", 0)

  def count_pieces(self, color): #Shua-- Added this function
      count=0
      for r in range(self.rows):
          for c in range(self.cols):
              if self.board[r][c]==color:
                  count+= 1
      return count

  def value(self):  #****Should be completed****
      if self.check_game_over():
        cw=self.check_winner()
        return cw[1]
      else:
          score = 0
          # Material Advantage--- Heuristic 1
          score += 100 * (self.count_pieces("X") - self.count_pieces("O")) #X = opponent, o = user
          # Mobility -- Heruistic 2
          legal_moves = self.find_valid_moves("X")
          score += 50 * len(legal_moves)
          return score
                  
          

  def print_board(self):
    for r in range(self.rows):
      for c in range(self.cols):
        print(self.board[r][c], end=" ")
      print()

  def find_valid_moves(self, color):
    # Return a list of valid moves for the specified color
    valid_moves = []
    for r in range(self.rows):
      for c in range(self.cols):
        if self.board[r][c] == '-':
          # Check all directions to see if the move is valid
          directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
          for r_offset, c_offset in directions:
            r2 = r + r_offset
            c2 = c + c_offset
            if r2 >= 0 and r2 < self.rows and c2 >= 0 and c2 < self.cols  \
               and self.board[r2][c2] != '-':
              tmp=cpy(self)
              if tmp.place_piece(r, c, color):
                  tmp.flip_pieces(r, c, color)
                  #print("next possible move if moving ", r, ",", c)
                  #tmp.print_board()
                  valid_moves.append(tmp)
              break
    return valid_moves

  def inputComputer(self):
      newObj=alphaBetaPruning.go(self)
      self.board = newObj.board
      
      return newObj
  
  

def cpy(s1):
    # construct a parent DataFrame instance
    s2=Othello(s1.rows, s1.cols)
    s2.board=copy.deepcopy(s1.board)
    #print("board ", s2.board)
    return s2
  