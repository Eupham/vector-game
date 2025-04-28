import random
import math
import torch
import time  # For performance tracking

class QuiescenceSearch:
    """Implementation of Quiescence Search for the Vector Game"""
    
    def __init__(self, game, max_depth=2, quiescence_depth=2):
        self.game = game
        self.max_depth = max_depth
        self.quiescence_depth = quiescence_depth
        self.nodes_evaluated = 0
        self.transposition_table = {}  # Add a transposition table for position caching
        self.start_time = 0
        self.time_limit = 0.5  # Time limit in seconds (500ms)
        self.tactical_threshold = 0.2  # Only consider tactical moves with significant score impact
        
    def search(self, clicks, is_red_turn=True, use_nn=True):
        """Run the quiescence search to find the best move"""
        self.nodes_evaluated = 0
        self.start_time = time.time()
        self.transposition_table = {}  # Reset transposition table for new search
        
        best_score = float('-inf') if is_red_turn else float('inf')
        best_move = None
        
        # Use the pre-computed valid actions if available, otherwise compute them
        if hasattr(self, 'valid_actions'):
            valid_actions = self.valid_actions
        else:
            valid_actions = self.game.get_valid_actions(clicks)
        
        if not valid_actions:
            return None
            
        # If only one valid move, return it immediately
        if len(valid_actions) == 1:
            return self.game.all_vertices[valid_actions[0]]
        
        # Preload valid loops for efficiency
        valid_loops = self.game.load_valid_loops()
        
        # Calculate the initial position score
        formed_loops = self.game.find_formed_loops(clicks, valid_loops)
        loop_colors = self.game.get_loop_colors(formed_loops, clicks)
        red_score_before, blue_score_before = self.game.calculate_scores(loop_colors)
        
        # Order moves by potential (tactical moves first)
        ordered_actions = self.order_moves(clicks, valid_actions, is_red_turn, valid_loops)
        
        # For each valid move, run quiescence search
        for action_idx in ordered_actions:
            # Time check - stop if we've exceeded the time limit
            if time.time() - self.start_time > self.time_limit:
                if best_move is not None:
                    return best_move
                # If we haven't found any moves yet, just return first valid move
                return self.game.all_vertices[valid_actions[0]]
                
            move = self.game.all_vertices[action_idx]
            
            # Make the move
            new_clicks = clicks.copy()
            new_clicks.append({
                'turn': 0,
                'color': 'red' if is_red_turn else 'blue',
                'address': move
            })
            
            # Evaluate with quiescence search
            if is_red_turn:
                score = self.alpha_beta(new_clicks, 1, self.max_depth, float('-inf'), float('inf'), 
                                       False, valid_loops, use_nn=use_nn)
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                score = self.alpha_beta(new_clicks, 1, self.max_depth, float('-inf'), float('inf'), 
                                       True, valid_loops, use_nn=use_nn)
                if score < best_score:
                    best_score = score
                    best_move = move
        
        # Always double check the move is valid before returning
        if best_move:
            # Validate that the move is actually in a valid position
            for click in clicks:
                if self.game.is_same_vertex(click['address'], best_move):
                    # This should never happen now, but if it does, choose a random move
                    print(f"Warning: Search returned already occupied position: {best_move}")
                    return self.game.all_vertices[random.choice(valid_actions)]
        
        return best_move
    
    def order_moves(self, clicks, valid_actions, is_red_turn, valid_loops):
        """Order moves to help alpha-beta pruning efficiency (tactical moves first)"""
        move_scores = []
        
        # Calculate the initial position score
        formed_loops_before = self.game.find_formed_loops(clicks, valid_loops)
        loop_colors_before = self.game.get_loop_colors(formed_loops_before, clicks)
        red_score_before, blue_score_before = self.game.calculate_scores(loop_colors_before)
        
        # Evaluate all moves
        for action_idx in valid_actions:
            move = self.game.all_vertices[action_idx]
            
            # Quick static evaluation
            temp_clicks = clicks.copy()
            temp_clicks.append({
                'turn': 0,
                'color': 'red' if is_red_turn else 'blue',
                'address': move
            })
            
            # Score after move
            formed_loops_after = self.game.find_formed_loops(temp_clicks, valid_loops)
            loop_colors_after = self.game.get_loop_colors(formed_loops_after, temp_clicks)
            red_score_after, blue_score_after = self.game.calculate_scores(loop_colors_after)
            
            # Calculate immediate score change
            score_diff = 0
            if is_red_turn:
                score_diff = (red_score_after - red_score_before) - (blue_score_after - blue_score_before)
            else:
                score_diff = (blue_score_after - blue_score_before) - (red_score_after - red_score_before)
                
            # Use potential triangle completions as secondary score
            potential_score = 0
            if hasattr(self.game, 'triangle_vertices'):
                red_potentials, blue_potentials = self.quick_potential_triangles(temp_clicks)
                if is_red_turn:
                    potential_score = red_potentials - blue_potentials
                else:
                    potential_score = blue_potentials - red_potentials
            
            # Combine immediate score and potential
            move_score = score_diff * 10 + potential_score
            move_scores.append((action_idx, move_score))
        
        # Sort moves by score (descending for red, ascending for blue)
        if is_red_turn:
            move_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            move_scores.sort(key=lambda x: x[1])
            
        # Return ordered action indices
        return [idx for idx, _ in move_scores]
        
    def alpha_beta(self, clicks, depth, max_depth, alpha, beta, is_red_turn, valid_loops, use_nn=True):
        """Alpha-beta pruning with quiescence search"""
        self.nodes_evaluated += 1
        
        # Check for timeout
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate_position(clicks, valid_loops, use_nn)
        
        # Generate position hash for transposition table
        pos_hash = self.hash_position(clicks)
        
        # Check transposition table
        if pos_hash in self.transposition_table:
            stored_depth, stored_value = self.transposition_table[pos_hash]
            if stored_depth >= max_depth - depth:
                return stored_value
        
        # Check if we've reached max depth
        if depth >= max_depth:
            # Use quiescence search for tactical positions
            return self.quiescence(clicks, 0, alpha, beta, is_red_turn, valid_loops, use_nn)
        
        # Get valid actions
        valid_actions = self.game.get_valid_actions(clicks)
        
        # Check for terminal state
        if not valid_actions:
            value = self.evaluate_position(clicks, valid_loops, use_nn)
            self.transposition_table[pos_hash] = (max_depth, value)
            return value
        
        # Order moves for better pruning
        ordered_actions = self.order_moves(clicks, valid_actions, is_red_turn, valid_loops)
        
        if is_red_turn:
            value = float('-inf')
            for action_idx in ordered_actions:
                move = self.game.all_vertices[action_idx]
                
                # Make the move
                new_clicks = clicks.copy()
                new_clicks.append({
                    'turn': 0,
                    'color': 'red',
                    'address': move
                })
                
                # Recursive call
                value = max(value, self.alpha_beta(new_clicks, depth + 1, max_depth, 
                                                  alpha, beta, False, valid_loops, use_nn))
                alpha = max(alpha, value)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
                    
            # Store in transposition table
            self.transposition_table[pos_hash] = (max_depth - depth, value)
            return value
        else:
            value = float('inf')
            for action_idx in ordered_actions:
                move = self.game.all_vertices[action_idx]
                
                # Make the move
                new_clicks = clicks.copy()
                new_clicks.append({
                    'turn': 0,
                    'color': 'blue',
                    'address': move
                })
                
                # Recursive call
                value = min(value, self.alpha_beta(new_clicks, depth + 1, max_depth, 
                                                  alpha, beta, True, valid_loops, use_nn))
                beta = min(beta, value)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            
            # Store in transposition table
            self.transposition_table[pos_hash] = (max_depth - depth, value)
            return value
    
    def hash_position(self, clicks):
        """Create a hash of the current board position"""
        # Simple hash: sort clicks by position and create a tuple of (position, color)
        position_data = []
        for click in clicks:
            r, theta = click['address']
            position_data.append((r, theta, click['color']))
            
        # Sort by position for consistent hashing
        position_data.sort()
        return tuple(position_data)
    
    def quiescence(self, clicks, depth, alpha, beta, is_red_turn, valid_loops, use_nn=True):
        """Quiescence search to handle tactical positions (optimized)"""
        # Check for timeout
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate_position(clicks, valid_loops, use_nn)
            
        # Base evaluation
        stand_pat = self.evaluate_position(clicks, valid_loops, use_nn)
        
        # Return evaluation at maximum quiescence depth
        if depth >= self.quiescence_depth:
            return stand_pat
        
        if is_red_turn:
            value = stand_pat
            alpha = max(alpha, value)
            if beta <= alpha:
                return value
                
            # Only consider captures/tactical moves
            tactical_moves = self.get_tactical_moves(clicks, is_red_turn, valid_loops)
            
            # If no tactical moves, return static evaluation
            if not tactical_moves:
                return value
                
            # Order tactical moves for better pruning
            tactical_moves.sort(key=lambda move: self.score_move(clicks, move, True, valid_loops), reverse=True)
            
            for move in tactical_moves:
                new_clicks = clicks.copy()
                new_clicks.append({
                    'turn': 0,
                    'color': 'red',
                    'address': move
                })
                
                score = self.quiescence(new_clicks, depth + 1, alpha, beta, False, valid_loops, use_nn)
                value = max(value, score)
                alpha = max(alpha, value)
                
                if beta <= alpha:
                    break
        else:
            value = stand_pat
            beta = min(beta, value)
            if beta <= alpha:
                return value
                
            # Only consider captures/tactical moves
            tactical_moves = self.get_tactical_moves(clicks, is_red_turn, valid_loops)
            
            # If no tactical moves, return static evaluation
            if not tactical_moves:
                return value
                
            # Order tactical moves for better pruning
            tactical_moves.sort(key=lambda move: self.score_move(clicks, move, False, valid_loops))
            
            for move in tactical_moves:
                new_clicks = clicks.copy()
                new_clicks.append({
                    'turn': 0,
                    'color': 'blue',
                    'address': move
                })
                
                score = self.quiescence(new_clicks, depth + 1, alpha, beta, True, valid_loops, use_nn)
                value = min(value, score)
                beta = min(beta, value)
                
                if beta <= alpha:
                    break
        
        return value
    
    def score_move(self, clicks, move, is_red, valid_loops):
        """Score a move for move ordering"""
        # Make the move
        temp_clicks = clicks.copy()
        temp_clicks.append({
            'turn': 0,
            'color': 'red' if is_red else 'blue',
            'address': move
        })
        
        # Calculate score change
        formed_loops_before = self.game.find_formed_loops(clicks, valid_loops)
        loop_colors_before = self.game.get_loop_colors(formed_loops_before, clicks)
        red_score_before, blue_score_before = self.game.calculate_scores(loop_colors_before)
        
        formed_loops_after = self.game.find_formed_loops(temp_clicks, valid_loops)
        loop_colors_after = self.game.get_loop_colors(formed_loops_after, temp_clicks)
        red_score_after, blue_score_after = self.game.calculate_scores(loop_colors_after)
        
        if is_red:
            return (red_score_after - red_score_before) - (blue_score_after - blue_score_before)
        else:
            return (blue_score_after - blue_score_before) - (red_score_after - red_score_before)
    
    def get_tactical_moves(self, clicks, is_red_turn, valid_loops):
        """Get tactical moves (moves that score points) - optimized version"""
        # Get only valid (unoccupied) positions
        valid_actions = self.game.get_valid_actions(clicks)
        if not valid_actions:
            return []
            
        tactical_moves = []
        
        # Create a set of all occupied positions for faster lookup
        occupied_positions = {(click['address'][0], click['address'][1]) for click in clicks}
        
        # Use spatial hashing to find potentially scoring vertices
        potentially_scoring_vertices = set()
        
        # For each placed vertex, find nearby vertices that might complete triangles
        for click in clicks:
            r, theta = click['address']
            
            # Get all vertex indices near this one using spatial hashing if available
            if hasattr(self.game, 'get_nearby_vertices'):
                nearby_indices = self.game.get_nearby_vertices(r, theta, radius=0.3)
                
                # Filter to only include valid (unplayed) actions
                nearby_indices = [idx for idx in nearby_indices if idx in valid_actions]
                
                # Add to the set of potentially scoring vertices
                potentially_scoring_vertices.update(nearby_indices)
            else:
                # Fallback if spatial hashing is not available
                potentially_scoring_vertices = set(valid_actions)
                break
        
        # Calculate the initial position score
        formed_loops_before = self.game.find_formed_loops(clicks, valid_loops)
        loop_colors_before = self.game.get_loop_colors(formed_loops_before, clicks)
        red_score_before, blue_score_before = self.game.calculate_scores(loop_colors_before)
        
        # For each potentially scoring move, check if it actually scores points
        for action_idx in potentially_scoring_vertices:
            move = self.game.all_vertices[action_idx]
            
            # Double check that this position is not occupied (belt and suspenders)
            move_r, move_theta = move
            if any(self.game.is_same_vertex(click['address'], move) for click in clicks):
                continue
                
            # Make the move
            temp_clicks = clicks.copy()
            temp_clicks.append({
                'turn': 0,
                'color': 'red' if is_red_turn else 'blue',
                'address': move
            })
            
            # Check if this move scores points
            formed_loops_after = self.game.find_formed_loops(temp_clicks, valid_loops)
            loop_colors_after = self.game.get_loop_colors(formed_loops_after, temp_clicks)
            red_score_after, blue_score_after = self.game.calculate_scores(loop_colors_after)
            
            score_diff = 0
            if is_red_turn:
                score_diff = red_score_after - red_score_before
            else:
                score_diff = blue_score_after - blue_score_before
                
            # Only include moves that make a significant score change
            if score_diff >= self.tactical_threshold:
                tactical_moves.append(move)
        
        # If we didn't find any tactical moves using nearby vertices, and we have the triangle information,
        # try a more targeted approach
        if not tactical_moves and hasattr(self.game, 'triangle_vertices') and len(potentially_scoring_vertices) < 5:
            # Check if we can complete any triangles with two same-color markers
            tactical_moves = self.find_strategic_completions(clicks, is_red_turn, valid_actions)
            
        # One final check to make sure all tactical moves are valid
        tactical_moves = [move for move in tactical_moves if 
                         not any(self.game.is_same_vertex(click['address'], move) for click in clicks)]
        
        return tactical_moves
    
    def find_strategic_completions(self, clicks, is_red_turn, valid_actions):
        """Find moves that complete triangles with opponent's colors for optimal scoring"""
        strategic_moves = []
        color = 'red' if is_red_turn else 'blue'
        opponent_color = 'blue' if is_red_turn else 'red'
        
        # Create a set of occupied vertex indices for fast lookup
        occupied_indices = set()
        for click in clicks:
            for i, vertex in enumerate(self.game.all_vertices):
                if self.game.is_same_vertex(vertex, click['address']):
                    occupied_indices.add(i)
                    break
        
        # Create a mapping of vertices to colors
        vertex_colors = {}
        for click in clicks:
            for i, vertex in enumerate(self.game.all_vertices):
                if self.game.is_same_vertex(vertex, click['address']):
                    vertex_colors[i] = click['color']
                    break
        
        # Identify potential triangles with strategic scoring opportunities
        for triangle_idx, triangle_verts in self.game.triangle_vertices.items():
            # Skip triangles that don't have any empty vertices
            if all(v in vertex_colors for v in triangle_verts):
                continue
                
            my_color_count = 0
            opponent_color_count = 0
            empty_vertices = []
            
            for v in triangle_verts:
                if v in vertex_colors:
                    if vertex_colors[v] == color:
                        my_color_count += 1
                    else:
                        opponent_color_count += 1
                else:
                    # Make sure vertex is really empty and in valid_actions
                    if v not in occupied_indices and v in valid_actions:
                        empty_vertices.append(v)
            
            # We want triangles with exactly one empty vertex
            if len(empty_vertices) == 1:
                empty_vertex = empty_vertices[0]
                
                # Double-check it's actually unoccupied with vertex-by-vertex comparison
                # (paranoid validation to avoid any floating point precision issues)
                move = self.game.all_vertices[empty_vertex]
                if any(self.game.is_same_vertex(click['address'], move) for click in clicks):
                    continue
                    
                # Calculate the score impact if we place here
                # Scoring rules from game:
                # - Three same color: -1 point to that color, +1 to opponent
                # - Two same + one opponent: +1 point to color with two
                # - One color + two opponent: +2 points to color with one
                
                # If completing would give us three of our color (penalty), avoid it
                if my_color_count == 2 and opponent_color_count == 0:
                    continue  # Avoid forming three of our color
                
                # If completing would give us one of our color and two opponent (bonus), prioritize it
                if my_color_count == 0 and opponent_color_count == 2:
                    strategic_moves.append((self.game.all_vertices[empty_vertex], 2))  # Highest priority
                
                # If completing would give us two of our color and one opponent (points), consider it
                elif my_color_count == 1 and opponent_color_count == 1:
                    strategic_moves.append((self.game.all_vertices[empty_vertex], 1))  # Medium priority
        
        # Sort by priority (highest first) and return just the moves
        strategic_moves.sort(key=lambda x: x[1], reverse=True)
        
        # One final filter to ensure all returned moves are valid and unoccupied
        return [move for move, _ in strategic_moves if 
                not any(self.game.is_same_vertex(click['address'], move) for click in clicks)]
    
    def quick_potential_triangles(self, clicks):
        """Quick count of potential triangles (optimization of count_potential_triangles)"""
        red_potentials = 0
        blue_potentials = 0
        
        # Create a mapping of vertices to colors
        vertex_colors = {}
        for click in clicks:
            for i, vertex in enumerate(self.game.all_vertices):
                if self.game.is_same_vertex(vertex, click['address']):
                    vertex_colors[i] = click['color']
                    break
        
        # Sample a subset of triangles for quick estimation
        triangle_count = len(self.game.triangle_vertices)
        triangles_to_check = min(50, triangle_count)  # Check at most 50 triangles
        
        # Create a list of triangle indices to sample
        triangle_indices = list(self.game.triangle_vertices.keys())
        sampled_indices = random.sample(triangle_indices, triangles_to_check) if triangle_count > triangles_to_check else triangle_indices
        
        # For each sampled triangle, check if it has exactly 2 marked vertices
        for triangle_idx in sampled_indices:
            triangle_verts = self.game.triangle_vertices[triangle_idx]
            red_count = 0
            blue_count = 0
            
            for v in triangle_verts:
                if v in vertex_colors:
                    if vertex_colors[v] == 'red':
                        red_count += 1
                    else:
                        blue_count += 1
            
            # If triangle has 2 reds and 1 empty, it's a red potential
            if red_count == 2 and blue_count == 0:
                red_potentials += 1
            # If triangle has 2 blues and 1 empty, it's a blue potential
            elif blue_count == 2 and red_count == 0:
                blue_potentials += 1
        
        # Scale the counts if we sampled
        if triangle_count > triangles_to_check:
            scale_factor = triangle_count / triangles_to_check
            red_potentials = int(red_potentials * scale_factor)
            blue_potentials = int(blue_potentials * scale_factor)
        
        return red_potentials, blue_potentials
    
    def evaluate_position(self, clicks, valid_loops, use_nn=True):
        """Evaluate the current position with combined heuristics and neural network"""
        # Try transposition table first
        pos_hash = self.hash_position(clicks)
        if pos_hash in self.transposition_table:
            return self.transposition_table[pos_hash][1]
        
        # Use neural network evaluation if enabled
        if use_nn and self.game.use_neural_net:
            try:
                # Get state tensors for neural network
                state_idx, state_occ = self.game.state_to_tokens(clicks)
                
                # Use neural network for evaluation
                with torch.no_grad():
                    value = self.game.regression_net(state_idx, state_occ).item()
                
                # Store in transposition table
                self.transposition_table[pos_hash] = (0, value)
                return value
            except Exception:
                # Fallback to heuristic if neural network fails
                pass
        
        # Heuristic evaluation
        formed_loops = self.game.find_formed_loops(clicks, valid_loops)
        loop_colors = self.game.get_loop_colors(formed_loops, clicks)
        red_score, blue_score = self.game.calculate_scores(loop_colors)
        
        # Additional strategic features - simplified for speed
        feature_score = 0
        if hasattr(self.game, 'triangle_vertices') and len(clicks) > 4:  # Only do this for non-trivial positions
            red_potentials, blue_potentials = self.quick_potential_triangles(clicks)
            feature_score += 0.3 * (red_potentials - blue_potentials)
        
        # Return the difference in scores plus strategic feature value
        value = (red_score - blue_score) + feature_score
        
        # Store in transposition table
        self.transposition_table[pos_hash] = (0, value)
        return value


def integrate_quiescence_search(game, clicks, is_red_turn=True, max_depth=2, use_nn=True):
    """Entry point function for quiescence search"""
    # Adjust max_depth and quiescence_depth based on game state
    # - Use deeper search for early game (many strategic options)
    # - Use shallower search for later game (fewer positions to evaluate)
    num_vertices = len(game.all_vertices)
    num_placed = len(clicks)
    game_progress = num_placed / num_vertices
    
    # Get valid actions first - ensure we only work with unoccupied positions
    valid_actions = game.get_valid_actions(clicks)
    if not valid_actions:
        return None
            
    # If only one valid move, return it immediately without search
    if len(valid_actions) == 1:
        return game.all_vertices[valid_actions[0]]
    
    # Adjust search parameters based on game progress
    if game_progress < 0.3:  # Early game
        adjusted_max_depth = max_depth
        quiescence_depth = 2
    elif game_progress < 0.7:  # Mid game
        adjusted_max_depth = max(2, max_depth - 1)
        quiescence_depth = 1
    else:  # Late game
        adjusted_max_depth = max(1, max_depth - 2)
        quiescence_depth = 1
        
    # Create and run search
    search = QuiescenceSearch(game, max_depth=adjusted_max_depth, quiescence_depth=quiescence_depth)
    search.valid_actions = valid_actions  # Pass valid actions to search to avoid re-computation
    move = search.search(clicks, is_red_turn=is_red_turn, use_nn=use_nn)
    
    # Final validation with extra safety checks
    if move:
        # Double-check that the position is valid and not already occupied
        is_occupied = False
        for click in clicks:
            if game.is_same_vertex(click['address'], move):
                is_occupied = True
                print(f"Debug: Quiescence search returned invalid move: {move}")
                break
        
        # If position is occupied, select a definitely valid move
        if is_occupied:
            # Return a random move from our pre-validated valid_actions
            if valid_actions:
                # This is guaranteed to be valid since valid_actions is created using
                # get_valid_actions which already filters out occupied positions
                random_action_idx = random.choice(valid_actions)
                return game.all_vertices[random_action_idx]
            return None
    
    return move