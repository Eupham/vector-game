import random
import math
import torch
from torch.distributions import Categorical

class CPU:
    def __init__(self, game):
        self.game = game

    def get_move(self, clicks):
        use_nn = self.game.use_neural_net and random.random() < (self.game.neural_net_ratio / 100)
        if use_nn:
            action = self.select_action(clicks)
            if action is None:
                return None
            # Build temporary click to evaluate reward
            temp_click = {'turn': 0, 'color': 'red', 'address': action}
            valid_loops = self.game.load_valid_loops()
            formed_loops_before = self.game.find_formed_loops(clicks, valid_loops)
            loop_colors_before = self.game.get_loop_colors(formed_loops_before, clicks)
            red_score_before, blue_score_before = self.game.calculate_scores(loop_colors_before)
            temp_clicks = clicks.copy()
            temp_clicks.append(temp_click)
            formed_loops_after = self.game.find_formed_loops(temp_clicks, valid_loops)
            loop_colors_after = self.game.get_loop_colors(formed_loops_after, temp_clicks)
            red_score_after, blue_score_after = self.game.calculate_scores(loop_colors_after)
            points_earned = red_score_after - red_score_before
            score_diff = (red_score_after - blue_score_after) - (red_score_before - blue_score_before)
            reward = 3.0 * points_earned + score_diff
            self.game.last_reward = reward
            self.game.policy_net.rewards.append(reward)
            return action
        else:
            return self.get_heuristic_move(clicks)

    def select_action(self, clicks):
        state = self.game.state_to_tensor(clicks)
        prev_state = self.game.policy_net.current_state
        valid_actions = self.game.get_valid_actions(clicks)
        if not valid_actions:
            return None
        with torch.no_grad():
            probs = self.game.policy_net(state)
        mask = torch.zeros_like(probs)
        mask[valid_actions] = 1
        masked_probs = probs * mask
        masked_probs = masked_probs / (masked_probs.sum() + 1e-10)
        m = Categorical(masked_probs)
        # Monte Carlo consensus: draw multiple samples and pick the mode at test time
        consensus_k = getattr(self.game, 'consensus_samples', 1)
        if consensus_k > 1:
            samples = m.sample((consensus_k,))
            # count occurrences to find mode
            vals = samples.tolist()
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            mode_idx = max(counts, key=counts.get)
            action_idx = torch.tensor(mode_idx, device=probs.device)
            log_prob = m.log_prob(action_idx)
        else:
            action_idx = m.sample()
            log_prob = m.log_prob(action_idx)
        self.game.policy_net.saved_log_probs.append(log_prob.detach().requires_grad_())
        self.game.policy_net.saved_states.append(state)
        self.game.policy_net.current_state = state
        self.game.policy_net.current_action = action_idx.item()
        self.game.policy_net.current_log_prob = log_prob
        if prev_state is not None and hasattr(self.game, 'last_reward'):
            self.game.td_update(prev_state, state, self.game.last_reward, is_terminal=False)
            del self.game.last_reward
        return self.game.all_vertices[action_idx.item()]

    def get_heuristic_move(self, clicks):
        make_aggressive_move = random.random() < (self.game.red_ai_aggression / 100)
        valid_actions = self.game.get_valid_actions(clicks)
        if not valid_actions:
            return None
        if make_aggressive_move:
            scoring_moves = []
            valid_loops = self.game.load_valid_loops()
            formed_loops_before = self.game.find_formed_loops(clicks, valid_loops)
            loop_colors_before = self.game.get_loop_colors(formed_loops_before, clicks)
            red_score_before, blue_score_before = self.game.calculate_scores(loop_colors_before)
            # Get all placed vertices
            placed_vertices = [click['address'] for click in clicks]
            potentially_scoring_vertices = set()
            for vertex in placed_vertices:
                r, theta = vertex
                nearby_indices = self.game.get_nearby_vertices(r, theta, radius=0.3)
                nearby_indices = [idx for idx in nearby_indices if idx in valid_actions]
                potentially_scoring_vertices.update(nearby_indices)
            potentially_scoring_list = list(potentially_scoring_vertices)
            if potentially_scoring_list:
                evaluation_indices = (random.sample(potentially_scoring_list, self.game.max_heuristic_evals)
                                        if len(potentially_scoring_list) > self.game.max_heuristic_evals
                                        else potentially_scoring_list)
            else:
                evaluation_indices = (random.sample(valid_actions, self.game.max_heuristic_evals)
                                        if len(valid_actions) > self.game.max_heuristic_evals
                                        else valid_actions)
            unplayed_moves = [self.game.all_vertices[i] for i in evaluation_indices]
            for move in unplayed_moves:
                temp_click = {'turn': 0, 'color': 'red', 'address': move}
                temp_clicks = clicks.copy()
                temp_clicks.append(temp_click)
                formed_loops_after = self.game.find_formed_loops(temp_clicks, valid_loops)
                loop_colors_after = self.game.get_loop_colors(formed_loops_after, temp_clicks)
                red_score_after, blue_score_after = self.game.calculate_scores(loop_colors_after)
                points_earned = red_score_after - red_score_before
                score_difference = (red_score_after - blue_score_after) - (red_score_before - blue_score_before)
                weighted_score = 3.0 * points_earned + score_difference
                if points_earned > 0:
                    scoring_moves.append((move, weighted_score))
                    if self.game.use_neural_net:
                        for i, vertex in enumerate(self.game.all_vertices):
                            if self.game.is_same_vertex(move, vertex):
                                vertex_idx = i
                                break
                        state = self.game.state_to_tensor(clicks)
                        probs = self.game.policy_net(state)
                        mask = torch.zeros_like(probs)
                        mask[valid_actions] = 1
                        masked_probs = probs * mask
                        masked_probs = masked_probs / (masked_probs.sum() + 1e-10)
                        m = Categorical(masked_probs)
                        action_tensor = torch.tensor(vertex_idx, device=self.game.device)
                        log_prob = m.log_prob(action_tensor)
                        self.game.policy_net.saved_log_probs.append(log_prob)
                        self.game.policy_net.rewards.append(weighted_score)
            if scoring_moves:
                scoring_moves.sort(key=lambda x: x[1], reverse=True)
                return scoring_moves[0][0]
            else:
                return random.choice([self.game.all_vertices[i] for i in valid_actions])
        else:
            return random.choice([self.game.all_vertices[i] for i in valid_actions])
