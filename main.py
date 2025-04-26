import sys
from game_logic import VectorGame
from terminal_trainer import main as trainer_main

if __name__ == "__main__":
    # If '--train' provided, run terminal trainer and exit
    if '--train' in sys.argv:
        trainer_main()
        sys.exit()
    game = VectorGame()
    game.run()
