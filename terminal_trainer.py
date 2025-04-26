#!/usr/bin/env python
"""
Vector Game Terminal Trainer

This script allows training the Vector Game neural network via command line.

Example:
    python terminal_trainer.py --episodes 1000 --depth 3 --lr 0.0004 \
        --aggression 80 --hidden-dim 512 --num-layers 5 \
        --save-interval 20 --log-dir "training_logs"
"""

import os
import argparse
import torch
import time
from pathlib import Path
import json
import random
import numpy as np
from game_logic import VectorGame
from neural_networks import RegressionNetwork

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Vector Game AI via terminal')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=100, 
                        help='Number of training episodes (games) to run')
    parser.add_argument('--depth', type=int, default=3, choices=range(1, 7),
                        help='Game board complexity (1-6)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for neural network training')
    parser.add_argument('--aggression', type=float, default=80,
                        help='Trainer aggression percentage (0-100)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size of neural network')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in neural network')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for neural network training')
                        
    # Hardware options
    parser.add_argument('--cuda', action='store_true', 
                        help='Use CUDA if available')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save model every N episodes')
    
    # Output options
    parser.add_argument('--log-dir', type=str, default='training_logs',
                        help='Directory for saving training logs')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    return parser.parse_args()

def initialize_game(args):
    """Initialize game with command line parameters."""
    # Initialize game without UI (headless training)
    game = VectorGame(headless=True)
    
    # Override settings with command-line arguments
    game.depth = args.depth
    game.learning_rate = args.lr
    game.blue_ai_aggression = args.aggression  # Store aggression for blue player
    game.use_neural_net = True
    game.neural_net_ratio = 100  # Full neural network usage for red
    game.red_first_prob = 50     # Equal chance for first move
    game.hidden_dim = args.hidden_dim
    game.num_layers = args.num_layers
    
    # Set the device (CPU/CUDA)
    if args.cuda and torch.cuda.is_available():
        # Select the first CUDA device
        torch.cuda.set_device(0)
        game.device = torch.device("cuda:0")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        game.device = torch.device("cpu")
        if args.cuda:
            print("CUDA requested but not available. Using CPU instead.")
        else:
            print("Using CPU for training")
    
    # Reset game and initialize neural network
    game.reset_game_files()
    game.generate_triangle_tessellation()
    game.initialize_policy_network()
    
    # Make sure the regression network is on the correct device
    game.regression_net.to(game.device)
    
    # Set batch size based on available memory
    if torch.cuda.is_available() and args.cuda:
        # Allow custom batch size but use a larger default on GPU if not specified
        if args.batch_size == 32:  # If user didn't change the default CPU batch size
            game.regression_net.batch_size = 512  # Use larger default for GPU
        else:
            game.regression_net.batch_size = args.batch_size  # Use user-specified batch size
    else:
        # Use specified batch size for CPU
        game.regression_net.batch_size = args.batch_size
    
    # Initialize training stats
    game.stats['value_losses'] = []
    game.stats['training_red_wins'] = 0
    game.stats['training_blue_wins'] = 0
    game.stats['training_ties'] = 0
    
    return game

def run_episode(game, episode_num, total_episodes, args):
    """Run a single training episode (game)."""
    # Reset for this episode
    game.reset_game_files()
    
    # Play the game (AI vs AI)
    clicks = []
    current_player = 'red' if random.random() < 0.5 else 'blue'
    turn_number = 0
    
    # For performance tracking
    start_time = time.time()
    move_times = []
    
    # Pre-compute valid actions once before starting
    valid_actions = game.get_valid_actions(clicks)
    
    # Play until all vertices are filled
    while len(clicks) < len(game.all_vertices):
        move_start = time.time()
        
        # Get AI move for current player
        if current_player == 'red':
            move = game.get_red_move(clicks)
        else:
            # Blue uses heuristic logic with aggression control
            move = game.get_blue_move(clicks, args.aggression)
        
        move_end = time.time()
        move_times.append(move_end - move_start)
        
        if move:
            # Add the move
            turn_number += 1
            clicks.append({
                'turn': turn_number,
                'color': current_player,
                'address': move
            })
            
            # Update valid actions incrementally (much faster than recomputing)
            for i, vertex in enumerate(game.all_vertices):
                if game.is_same_vertex(vertex, move):
                    if i in valid_actions:
                        valid_actions.remove(i)
                    break
            
            # Switch players
            current_player = 'blue' if current_player == 'red' else 'red'
    
    # Force batch processing of any remaining training examples
    if hasattr(game.regression_net, 'batch_train_step') and game.regression_net.batch_states_indices:
        loss = game.regression_net.batch_train_step(game.regression_optimizer)
        if loss is not None:
            game.stats['value_losses'].append(loss)
    
    # Calculate final score
    valid_loops = game.load_valid_loops()
    formed_loops = game.find_formed_loops(clicks, valid_loops)
    loop_colors = game.get_loop_colors(formed_loops, clicks)
    red_score, blue_score = game.calculate_scores(loop_colors)
    
    # Update win counts
    if red_score > blue_score:
        game.stats['training_red_wins'] += 1
        outcome = "Red wins"
    elif blue_score > red_score:
        game.stats['training_blue_wins'] += 1
        outcome = "Blue wins"
    else:
        game.stats['training_ties'] += 1
        outcome = "Tie"
    
    # Performance statistics
    episode_time = time.time() - start_time
    avg_move_time = sum(move_times) / len(move_times) if move_times else 0
    
    # Log the results
    if not args.quiet:
        avg_loss = sum(game.stats['value_losses'][-100:]) / min(100, len(game.stats['value_losses'])) if game.stats['value_losses'] else 0
        progress = episode_num / total_episodes * 100
        
        print(f"Episode {episode_num}/{total_episodes} ({progress:.1f}%) - "
              f"Outcome: {outcome} (Red: {red_score}, Blue: {blue_score}) - "
              f"Avg Loss: {avg_loss:.6f} - Time: {episode_time:.2f}s")
    
    # Return the game results for logging
    return {
        'episode': episode_num,
        'red_score': red_score,
        'blue_score': blue_score,
        'outcome': outcome,
        'num_moves': turn_number,
        'avg_loss': avg_loss if 'avg_loss' in locals() else 0,
        'episode_time': episode_time,
        'avg_move_time': avg_move_time
    }

def save_model(game, episode, args, is_final=False):
    """Save the trained model."""
    try:
        # Create unique model filename with episode number and timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        if is_final:
            model_path = model_dir / f"vector_game_model_{len(game.all_vertices)}_final_{timestamp}.pt"
        else:
            model_path = model_dir / f"vector_game_model_{len(game.all_vertices)}_ep{episode}_{timestamp}.pt"
        
        # Save model data
        model_data = {
            'regression_state': game.regression_net.state_dict(),
            'value_state': game.value_net.state_dict(),  # Keeping for backwards compatibility
            'training_params': {
                'depth': args.depth,
                'learning_rate': args.lr,
                'aggression': args.aggression,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'episodes_completed': episode
            }
        }
        torch.save(model_data, model_path)
        
        # Also update the standard model file for continued gameplay
        torch.save(model_data, game.model_file)
        
        if not args.quiet:
            if is_final:
                print(f"Final model saved to {model_path}")
            else:
                print(f"Checkpoint model saved to {model_path}")
                
        return str(model_path)
    
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def save_training_log(game, args, episode_logs):
    """Save training statistics to a log file."""
    try:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_log_{timestamp}.json"
        
        # Compile training statistics
        training_data = {
            'parameters': {
                'depth': args.depth,
                'episodes': args.episodes,
                'learning_rate': args.lr,
                'aggression': args.aggression,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'device': str(game.device)
            },
            'results': {
                'red_wins': game.stats['training_red_wins'],
                'blue_wins': game.stats['training_blue_wins'],
                'ties': game.stats['training_ties'],
                'win_rate_red': game.stats['training_red_wins'] / args.episodes,
                'avg_loss': sum(game.stats['value_losses']) / max(1, len(game.stats['value_losses']))
            },
            'episode_logs': episode_logs
        }
        
        # Save to file
        with open(log_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        if not args.quiet:
            print(f"Training log saved to {log_file}")
            
        return str(log_file)
    
    except Exception as e:
        print(f"Error saving training log: {e}")
        return None

def main():
    """Main training function."""
    print("Vector Game Terminal Trainer")
    print("----------------------------")
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize game with arguments
    game = initialize_game(args)
    
    print(f"\nStarting training with {args.episodes} episodes...")
    print(f"Board depth: {args.depth}, Learning rate: {args.lr}, Aggression: {args.aggression}%")
    print(f"Neural network: {args.hidden_dim} hidden units, {args.num_layers} layers")
    
    # Track start time
    start_time = time.time()
    
    # Storage for episode logs
    episode_logs = []
    
    # Run training episodes
    for episode in range(1, args.episodes + 1):
        # Run one episode
        episode_log = run_episode(game, episode, args.episodes, args)
        episode_logs.append(episode_log)
        
        # Save model at intervals
        if episode % args.save_interval == 0:
            save_model(game, episode, args)
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\nTraining complete!")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results - Red wins: {game.stats['training_red_wins']}, "
          f"Blue wins: {game.stats['training_blue_wins']}, "
          f"Ties: {game.stats['training_ties']}")
    
    if game.stats['value_losses']:
        avg_loss = sum(game.stats['value_losses']) / len(game.stats['value_losses'])
        print(f"Average loss: {avg_loss:.6f}")
    
    # Save final model
    final_model_path = save_model(game, args.episodes, args, is_final=True)
    
    # Save training log
    log_path = save_training_log(game, args, episode_logs)
    
    print("\nTraining artifacts:")
    if final_model_path:
        print(f"- Final model: {final_model_path}")
    if log_path:
        print(f"- Training log: {log_path}")
    
    print("\nModel can now be used for gameplay.")

if __name__ == "__main__":
    main()