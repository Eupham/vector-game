import pygame
import numpy as np
import json
import random
import math
from pathlib import Path
from collections import defaultdict, deque  # for running normalization buffer
import shutil
# Import just what we need from ui_elements
from ui_elements import GameBoard, Button, Slider, Checkbox, Label, Panel, Scrollbar, FileDialog, ModelParamsDialog, MLMetricsPanel, HelpPanel, SettingsPanel, Dashboard, GameOverScreen
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import datetime  # Add this import
from neural_networks import RegressionNetwork  # Use RegressionNetwork alias
from training_config import TrainingConfigPanel, start_training
# Import quiescence search implementation
from ai import integrate_quiescence_search, QuiescenceSearch

class VectorGame:
    def __init__(self, headless=False):
        # Initialize game parameters, neural params, and state
        self.init_game_parameters()
        
        self.init_neural_parameters()
        
        self.init_game_state()
        
        # Setup device for neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize necessary files
        self.init_files()
        self.headless = headless
        if not self.headless:
            # Initialize pygame and UI
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
            pygame.display.set_caption("Vector Game (Pygame)")
            self.create_ui()
            self.choose_game_mode()

        # Common initialization: prepare game logic and neural nets
        self.reset_game_files()
        self.generate_triangle_tessellation()
        self.initialize_policy_network()
        
        if not self.headless:
            # Main game loop variables for UI mode
            self.clock = pygame.time.Clock()
            self.running = True
            self.show_settings = False
            self.show_dashboard = False
            self.background = None
            self.dirty_rects = []
            self.needs_full_redraw = True
        
    def init_game_parameters(self):
        """Initialize game parameters"""
        # Get screen resolution
        pygame.init()  # Ensure pygame is initialized before getting display info
        screen_info = pygame.display.Info()
        available_width = screen_info.current_w
        available_height = screen_info.current_h
        
        # Set window size to 80% of available screen space
        self.screen_width = min(1200, int(available_width * 0.8))
        self.screen_height = min(800, int(available_height * 0.8))
        
        # Game state variables
        self.global_turn_number = 0
        self.round_number = 0
        self.current_round_moves = 0
        self.round_player_moves = {'red': 0, 'blue': 0}
        self.ai_first = False
        self.depth = 3
        self.red_first_prob = 50  # Default 50% chance Red goes first
        self.red_ai_aggression = 80  # Default 80% aggressive moves
        
        # Performance optimization parameters
        self.max_heuristic_evals = 20  # Maximum number of moves to evaluate with heuristic
        
        # Game board parameters
        self.high_score = None
        self.center_x = int(self.screen_width * 0.4)
        self.center_y = int(self.screen_height * 0.4)
        self.scale = 200  # Initial scaling factor for triangle coordinates
        self.scale_multiplier = 1.0  # Visual scale multiplier (1.0 = 100%)
        self.all_vertices = []  # Store all triangle vertices
        self.vertex_tolerance = 8  # Tolerance for vertex clicks (in pixels)
        
        # UI colors
        self.colors = {
            'background': (240, 240, 240),
            'panel': (220, 220, 220),
            'button': (180, 180, 180),
            'button_hover': (150, 150, 150),
            'red': (255, 80, 80),
            'blue': (80, 80, 255),
            'text': (0, 0, 0),
            'triangle_fill': (240, 240, 240),
            'triangle_border': (100, 100, 100)
        }
        
        # Game status message
        self.status_message = "Click on vertices to place your markers"

    def init_neural_parameters(self):
        """Initialize neural network related parameters"""
        # Neural network policy parameters
        self.use_neural_net = False
        self.neural_net_ratio = 20  # Default 20% neural net, 80% heuristic
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor for rewards
        # Running buffer of TD targets for z-normalization
        self.target_buffer = deque(maxlen=1000)
        
        # Monte Carlo consensus parameters
        self.consensus_samples = 1  # Default to no consensus (just single sampling)
        
        # Neural network architecture parameters
        self.hidden_dim = 128  # Default hidden dimension
        self.num_layers = 2   # Default number of layers
        
        # Statistics tracking for dashboard
        self.stats = {
            'losses': [],            # Track policy loss per update
            'value_losses': [],      # Track value network loss per update
            'rewards': [],           # Track rewards per round
            'round_rewards': [],     # Rewards grouped by round
            'game_rewards': [],      # Rewards grouped by game
            'round_losses': [],      # Losses grouped by round
            'game_losses': [],       # Losses grouped by game
            'current_game_rewards': [],  # Rewards in the current game
            'current_game_losses': [],   # Losses in the current game
            'advantages': [],        # Store advantage values for analysis
            'td_errors': [],         # Store TD errors for analysis
            'red_game_scores': [],   # Red's scores in each game
            'blue_game_scores': []   # Blue's scores in each game
        }

    def init_game_state(self):
        """Initialize game state variables"""
        # Cache structures for optimization
        self.scores_cache = {}  # Cache for score calculations
        self.vertex_to_triangles = {}  # Maps vertex index to list of triangle indices containing it
        self.triangle_vertices = {}  # Maps triangle index to list of vertex indices
        self.vertex_set_to_triangle = {}  # Maps set of 3 vertex indices to triangle index
        
        # Spatial hashing for fast vertex lookups
        self.spatial_hash = {}  # Maps grid cells to vertex indices
        self.grid_size = 0.1  # Size of grid cells in polar coordinates
        
        # Game scores
        self.red_score = 0
        self.blue_score = 0

    def init_files(self):
        """Initialize necessary files"""
        # Create necessary files if they don't exist
        self.valid_spaces_file = Path("valid_spaces.json")
        self.valid_spaces_file.touch(exist_ok=True)
        self.valid_loops_file = Path("valid_loops.json")
        self.valid_loops_file.touch(exist_ok=True)
        self.clicks_file = Path("clicks.json")
        self.clicks_file.touch(exist_ok=True)
        
        # Neural network model persistence
        self.model_file = Path("vector_game_model.pt")
        
        # Load or initialize master game data
        self.master_game_data_file = Path('master_game_data.json')
        if self.master_game_data_file.exists():
            with open(self.master_game_data_file, 'r') as file:
                self.master_game_data = json.load(file)
        else:
            self.master_game_data = []

    def create_ui(self):
        """Create game UI elements"""
        # Main game area panel
        self.game_panel = Panel(10, 10, int(self.screen_width * 0.75), int(self.screen_height - 20))
        
        # Side panel for scores and controls
        self.side_panel = Panel(int(self.screen_width * 0.75) + 20, 10, 
                            int(self.screen_width * 0.25) - 30, int(self.screen_height - 20))
        
        # Title label
        self.title_label = Label(int(self.side_panel.rect.centerx), 30, 
                                "Vector Game", (0, 0, 0), 36, "center")
        
        # Score labels
        self.red_score_label = Label(self.side_panel.rect.x + 20, 100, 
                                    "Red: 0", self.colors['red'], 28)
        self.blue_score_label = Label(self.side_panel.rect.x + 20, 140, 
                                    "Blue: 0", self.colors['blue'], 28)
        
        # Round label
        self.round_label = Label(self.side_panel.rect.x + 20, 180, 
                                "Round: 0", self.colors['text'], 24)
        
        # Status message - with text wrapping enabled
        self.status_label = Label(self.side_panel.rect.x + 20, 230, 
                                self.status_message, self.colors['text'], 20)
        # Set max width for wrapping text (adjust for side panel width minus some padding)
        self.status_label.set_max_width(self.side_panel.rect.width - 40)
        
        # Buttons - calculate positions from bottom upward
        button_y_positions = []
        button_height = 50
        button_spacing = 10
        
        # Start positioning from bottom of the panel
        current_y = self.side_panel.rect.bottom - 60
        
        # Add positions from bottom to top (now 5 buttons instead of 4)
        for i in range(5):  # Now 5 buttons
            button_y_positions.append(current_y)
            current_y -= (button_height + button_spacing)
        
        # New Game button (bottom)
        self.new_game_button = Button(self.side_panel.rect.x + 20, 
                                    button_y_positions[0], 200, button_height, 
                                    "New Game", self.colors['button'], 
                                    self.colors['button_hover'])
        
        # Training Session button (second from bottom)
        self.training_button = Button(self.side_panel.rect.x + 20, 
                                    button_y_positions[1], 200, button_height, 
                                    "Training Config", self.colors['button'], 
                                    self.colors['button_hover'])
        
        # Dashboard button (third from bottom)
        self.dashboard_button = Button(self.side_panel.rect.x + 20, 
                                    button_y_positions[2], 200, button_height, 
                                    "Dashboard", self.colors['button'], 
                                    self.colors['button_hover'])
        
        # Help button (fourth from bottom)
        self.help_button = Button(self.side_panel.rect.x + 20, 
                                button_y_positions[3], 200, button_height, 
                                "Help", self.colors['button'], 
                                self.colors['button_hover'])
        
        # Settings button (top)
        self.settings_button = Button(self.side_panel.rect.x + 20, 
                                    button_y_positions[4], 200, button_height, 
                                    "Settings", self.colors['button'], 
                                    self.colors['button_hover'])
        
        # Create specialized UI components
        self.create_specialized_ui()
        
        # Create rules UI and settings UI
        self.create_rules_ui()
        self.create_settings_ui()
        self.training_config_ui = TrainingConfigPanel(self, self.screen_width, self.screen_height)
        # Initialize show_help flag
        self.show_help = False
        
        # Initialize dialog state variables
        self.show_file_dialog = False
        self.file_dialog = None
        self.show_model_params_dialog = False
        self.model_params_dialog = None

    def create_specialized_ui(self):
        """Create specialized UI components"""
        # Make sure rules_content_height is calculated before creating HelpPanel
        # Define the game rules text if not already defined
        if not hasattr(self, 'rules_text'):
            self.rules_text = """
    VECTOR GAME: OFFICIAL RULES

    OVERVIEW:
    Vector Game is a strategic two-player game where players place markers on vertices of triangles, 
    trying to create specific color patterns for points while avoiding others.

    DEFINITIONS:
    • Game: A series of rounds played until all vertices are filled
    • Round: Two moves, one by each player (Red and Blue)
    • Move: Placing a marker on an unoccupied vertex

    GAMEPLAY:
    1. Players take turns placing markers on vertices during each round
    2. Each player can only place one marker per round
    3. Red (AI) and Blue (Player) alternate who goes first in a round (controlled by probability setting)
    4. The game ends when all vertices are filled

    OBJECTIVE:
    Create triangles with strategic color combinations while avoiding making triangles of all your color.

    SCORING SYSTEM:
    When a triangle is completed, points are awarded based on color pattern:

    For RED player:
    • Three RED markers: -1 point to RED, +1 point to BLUE (penalty)
    • Two RED + One BLUE markers: +1 point to RED
    • One RED + Two BLUE markers: +2 points to RED (bonus)

    For BLUE player:
    • Three BLUE markers: -1 point to BLUE, +1 point to RED (penalty)
    • Two BLUE + One RED markers: +1 point to BLUE
    • One BLUE + Two RED markers: +2 points to BLUE (bonus)

    WINNING:
    The player with the highest score when all vertices are filled wins the game.

    SETTINGS:
    • Adjust game complexity using the Depth setting
    • Control Red's aggressiveness and first-move probability
    • Enable neural network AI for advanced play

    STRATEGIES:
    • Try to create triangles that have both red and blue markers
    • Avoid completing triangles with only your color
    • Look for opportunities to create multiple triangles with a single move
    • Block your opponent from completing high-scoring triangles
    • Sometimes it's better to create a triangle that gives your opponent 1 point 
    than to let them potentially score 2 points later

    ADVANCED TIPS:
    • The center of the board often provides more opportunities for triangle formation
    • Pay attention to the tessellation pattern to predict potential triangle closures
    • In the late game, count remaining vertices to plan your final moves strategically
    • When Red AI uses neural networks, it learns from your play style over time
    """

        # Calculate the total height needed to display all text
        font = pygame.font.SysFont(None, 18)
        section_font = pygame.font.SysFont(None, 22, bold=True)
        
        self.rules_content_height = 0
        lines = self.rules_text.strip().split('\n')
        for line in lines:
            if line.strip().isupper() and ":" in line:
                self.rules_content_height += 30  # Section headers get more space
            elif not line.strip():
                self.rules_content_height += 15  # Empty lines get less space
            else:
                self.rules_content_height += 22  # Regular lines
        
        # Add padding
        self.rules_content_height += 80  # Extra space for padding

        # Create the game board UI
        self.game_board_ui = GameBoard(self)
        
        # Create ML metrics panel
        self.ml_metrics_ui = MLMetricsPanel(self)
        
        # Create help panel (now rules_content_height is defined)
        self.help_ui = HelpPanel(self, self.screen_width, self.screen_height)
        
        # Create settings panel
        self.settings_ui = SettingsPanel(self, self.screen_width, self.screen_height)
        
        # Create dashboard
        self.dashboard_ui = Dashboard(self, self.screen_width, self.screen_height)
        
        # Create game over screen
        self.game_over_ui = GameOverScreen(self, self.screen_width, self.screen_height)
        self.show_training_config = False

    def create_rules_ui(self):
        """Create the rules UI panel with scrollable content"""
        # Create rules panel that covers most of the screen
        panel_width = int(self.screen_width * 0.8)
        panel_height = int(self.screen_height * 0.8)
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        self.rules_panel = Panel(panel_x, panel_y, panel_width, panel_height)
        
        # Rules title
        self.rules_title = Label(self.rules_panel.rect.centerx, self.rules_panel.rect.y + 20, 
                               "Vector Game Rules", self.colors['text'], 32, "center")
        
        # Create close button
        self.close_rules_button = Button(self.rules_panel.rect.right - 120, self.rules_panel.rect.bottom - 50, 
                                        100, 40, "Close", self.colors['button'], 
                                        self.colors['button_hover'])
                                        
        # Create scrollbar for rules panel
        # Use the previously calculated rules_content_height from create_specialized_ui
        scrollbar_x = self.rules_panel.rect.right - 20
        scrollbar_y = self.rules_panel.rect.y + 60
        scrollbar_height = self.rules_panel.rect.height - 120  # Leave space for title and button
        
        self.rules_scrollbar = Scrollbar(
            scrollbar_x, scrollbar_y, 15, scrollbar_height, 
            self.rules_content_height, scrollbar_height,
            bar_color=(200, 200, 200), handle_color=(150, 150, 150)
        )

    def create_settings_ui(self):
        """Create the settings UI panel with controls"""
        # Create settings panel that covers most of the screen
        panel_width = int(self.screen_width * 0.8)
        panel_height = int(self.screen_height * 0.8)
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        self.settings_panel = Panel(panel_x, panel_y, panel_width, panel_height)
        
        # Settings title
        self.settings_title = Label(self.settings_panel.rect.centerx, self.settings_panel.rect.y + 20, 
                                "Game Settings", self.colors['text'], 32, "center")
        
        # Calculate the total content height needed for all settings
        content_height = 650  # Base height including padding
        
        # Create scrollbar for settings panel
        scrollbar_x = self.settings_panel.rect.right - 20
        scrollbar_y = self.settings_panel.rect.y + 60
        scrollbar_height = self.settings_panel.rect.height - 120  # Leave space for title and buttons
        
        self.settings_scrollbar = Scrollbar(
            scrollbar_x, scrollbar_y, 15, scrollbar_height, 
            content_height, scrollbar_height,
            bar_color=(200, 200, 200), handle_color=(150, 150, 150)
        )
        
        # Settings sliders
        self.depth_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 80, 
                                500, 20, 1, 6, self.depth, text="Search Depth")
        
        self.scale_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 130, 
                                500, 20, 0.5, 2.0, self.scale_multiplier, text="Visual Scale")
        
        # Keep red_first_slider as percentage (0-100)
        self.red_first_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 180, 
                                    500, 20, 0, 100, self.red_first_prob, text="Red First Probability (%)")
        
        # Convert red_ai_aggression_slider to -1 to 1 scale
        self.red_ai_aggression_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 230, 
                                        500, 20, -1, 1, self.red_ai_aggression_normalized, text="Strategy (Defensive ← → Aggressive)")
        
        self.heuristic_evals_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 280, 
                                        500, 20, 5, 50, self.max_heuristic_evals, text="AI Search Breadth")
        
        # Neural network settings
        self.use_neural_net_checkbox = Checkbox(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 330, 
                                            20, "Use Neural Network", self.use_neural_net)
        
        # Convert neural_net_ratio_slider to -1 to 1 scale
        self.neural_net_ratio_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 370, 
                                            500, 20, -1, 1, self.neural_net_ratio_normalized, text="Decision Style (Heuristic ← → Neural)")
        
        self.learning_rate_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 420, 
                                        500, 20, 0.0001, 0.01, self.learning_rate, text="Learning Rate")
                                        
        # Neural network architecture settings
        self.hidden_dim_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 470, 
                                    500, 20, 32, 1024, self.hidden_dim, text="Network Width")
                                    
        self.num_layers_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 520, 
                                    500, 20, 1, 8, self.num_layers, text="Network Depth")
        
        # Apply and Close buttons
        self.apply_settings_button = Button(self.settings_panel.rect.right - 120, self.settings_panel.rect.bottom - 60, 
                                        100, 40, "Apply", self.colors['button'], self.colors['button_hover'])
        
        self.close_settings_button = Button(self.settings_panel.rect.right - 230, self.settings_panel.rect.bottom - 60, 
                                        100, 40, "Cancel", self.colors['button'], self.colors['button_hover'])

        # Add Save and Load Config buttons
        self.save_config_button = Button(self.settings_panel.rect.x + 50, 
                                         self.settings_panel.rect.bottom - 60, 
                                         100, 40, "Save Config", self.colors['button'], 
                                         self.colors['button_hover'])
        self.load_config_button = Button(self.settings_panel.rect.x + 160, 
                                         self.settings_panel.rect.bottom - 60, 
                                         100, 40, "Load Config", self.colors['button'], 
                                         self.colors['button_hover'])
                                         
        # Add Load Model button
        self.load_model_button = Button(self.settings_panel.rect.x + 270, 
                                         self.settings_panel.rect.bottom - 60, 
                                         100, 40, "Load Model", self.colors['button'], 
                                         self.colors['button_hover'])

    def run(self):
        """Main game loop"""
        while self.running:
            # Handle events
            events = pygame.event.get()
            self.handle_events(events)
            
            # Update game state
            self.update(events)
            
            # Render game
            self.render()
            
            # Cap the frame rate
            self.clock.tick(60)
        
        pygame.quit()
    
    def handle_events(self, events):
        """Handle pygame events"""
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.screen_width, self.screen_height = event.size
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                
                # Recalculate center positions
                self.center_x = int(self.screen_width * 0.4)
                self.center_y = int(self.screen_height * 0.4)
                
                # Recreate UI elements to match new size
                self.create_ui()
                
                # Need to redraw everything
                self.needs_full_redraw = True
                self.background = None
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Don't process game clicks if any panel is open
                    if self.show_settings or self.show_dashboard or self.show_help:
                        pass
                    else:
                        # Only process clicks if they are within the game panel
                        if self.game_panel.rect.collidepoint(event.pos):
                            self.on_canvas_click(event.pos)
    
    def update(self, events):
        """Update game state and UI elements"""
        # Update buttons and handle clicks
        if self.new_game_button.update(events):
            self.reset_game()
        
        if self.settings_button.update(events):
            self.show_settings = True
        
        if self.dashboard_button.update(events):
            self.show_dashboard = True
            
        if self.help_button.update(events):
            self.show_help = True
            
        if self.training_button.update(events):
            # Open the training config panel instead of immediately running a training session
            self.show_training_config = True            
        # Handle settings panel updates
        if self.show_settings:
            result = self.settings_ui.update(events)
            
            if result == 'apply':
                self.apply_settings()
                self.show_settings = False
            elif result == 'close':
                self.show_settings = False
            elif result == 'save_config':
                self.save_config()
            elif result == 'load_config':
                self.load_config()
            elif result == 'load_model':
                self.show_file_dialog = True
                # Create file dialog centered on the screen
                dialog_width, dialog_height = 600, 400
                dialog_x = (self.screen_width - dialog_width) // 2
                dialog_y = (self.screen_height - dialog_height) // 2
                self.file_dialog = FileDialog(
                    dialog_x, dialog_y, dialog_width, dialog_height,
                    title="Select Model File"
                )
                self.status_message = "Selecting model file..."
                self.status_label.update_text(self.status_message)
        
        # Handle training config panel updates
        if self.show_training_config:
            result = self.training_config_ui.update(events)
            
            if result == 'apply':
                # Just apply the settings without closing
                self.status_message = "Training configuration applied."
                self.status_label.update_text(self.status_message)
            elif result == 'close':
                self.show_training_config = False
            elif result == 'start_training':
                # Start training with the configured settings
                self.show_training_config = False
                success, message = start_training(self)
                self.status_message = message
                self.status_label.update_text(self.status_message)

        # Handle dashboard panel updates
        if self.show_dashboard:
            result = self.dashboard_ui.update(events)
            if result == 'close':
                self.show_dashboard = False
            
        # Handle help panel updates
        if self.show_help:
            result = self.help_ui.update(events)
            if result == 'close':
                self.show_help = False
            
            # Update scrollbar
            self.rules_scrollbar.update(events)
        
        # Handle file dialog updates
        if self.show_file_dialog and self.file_dialog:
            is_done, model_path = self.file_dialog.update(events)
            
            if is_done:
                self.show_file_dialog = False
                
                if model_path:  # User selected a file
                    # Extract model parameters and show the params dialog
                    try:
                        model_params = self.extract_model_parameters(model_path)
                        if model_params:
                            self.show_model_params_dialog = True
                            dialog_width = 500
                            dialog_height = 350
                            current_params = {
                                "depth": self.depth,
                                "hidden_dim": self.hidden_dim,
                                "num_layers": self.num_layers,
                                "learning_rate": self.learning_rate
                            }
                            self.model_params_dialog = ModelParamsDialog(
                                (self.screen_width - dialog_width) // 2,
                                (self.screen_height - dialog_height) // 2,
                                dialog_width,
                                dialog_height,
                                model_params,
                                current_params
                            )
                            # Store model path for later loading
                            self.selected_model_path = model_path
                    except Exception as e:
                        self.status_message = f"Error loading model: {e}"
                        self.status_label.update_text(self.status_message)
        
        # Handle model parameters dialog updates
        if self.show_model_params_dialog and self.model_params_dialog:
            is_done, params = self.model_params_dialog.update(events)
            
            if is_done:
                self.show_model_params_dialog = False
                
                if params:  # User confirmed parameter changes
                    # Update settings with model parameters
                    self.depth = int(params.get("depth", self.depth))
                    self.hidden_dim = int(params.get("hidden_dim", self.hidden_dim))
                    self.num_layers = int(params.get("num_layers", self.num_layers))
                    self.learning_rate = float(params.get("learning_rate", self.learning_rate))
                    
                    # Update sliders to reflect new values
                    self.depth_slider.current_val = self.depth
                    self.depth_slider.update_handle_position(self.depth)
                    self.hidden_dim_slider.current_val = self.hidden_dim
                    self.hidden_dim_slider.update_handle_position(self.hidden_dim)
                    self.num_layers_slider.current_val = self.num_layers
                    self.num_layers_slider.update_handle_position(self.num_layers)
                    self.learning_rate_slider.current_val = self.learning_rate
                    self.learning_rate_slider.update_handle_position(self.learning_rate)
                    
                    # Enable neural network usage
                    self.use_neural_net = True
                    self.use_neural_net_checkbox.checked = True
                    
                    # The board needs to be regenerated with the new depth value
                    self.reset_game()
                    
                    # Load the model with the updated parameters
                    self.load_external_model(self.selected_model_path)
                    
                    self.status_message = f"Model loaded with depth={self.depth}, hidden_dim={self.hidden_dim}, layers={self.num_layers}"
                    self.status_label.update_text(self.status_message)

    def render(self):
        """Render the game"""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw panels
        self.game_panel.draw(self.screen)
        self.side_panel.draw(self.screen)
        
        # Draw triangles and markers
        self.draw_game_board()
        
        # Draw UI elements
        self.title_label.draw(self.screen)
        self.red_score_label.draw(self.screen)
        self.blue_score_label.draw(self.screen)
        self.round_label.draw(self.screen)
        self.status_label.draw(self.screen)
        
        # Draw all buttons
        self.new_game_button.draw(self.screen)
        self.training_button.draw(self.screen)  # Draw the training session button
        self.dashboard_button.draw(self.screen)
        self.help_button.draw(self.screen)
        self.settings_button.draw(self.screen)
        
        # Draw ML metrics in the upper left corner
        self.draw_ml_metrics()
        
        # Draw settings panel if open
        if self.show_settings:
            self.render_settings_panel()
        
        # Draw training config panel if open
        if self.show_training_config:
            self.training_config_ui.draw(self.screen)
        
        # Draw dashboard if open
        if self.show_dashboard:
            self.render_dashboard()
            
        # Draw help panel if open
        if self.show_help:
            self.render_help_panel()
            
        # Draw file dialog if open
        if self.show_file_dialog and self.file_dialog:
            self.file_dialog.draw(self.screen)
            
        # Draw model params dialog if open
        if self.show_model_params_dialog and self.model_params_dialog:
            self.model_params_dialog.draw(self.screen)
        
        # Draw model params footer at the bottom of the screen with text wrapping
        footer_font = pygame.font.SysFont(None, 22)
        
        # Model path (show only filename if too long)
        model_path = getattr(self, 'selected_model_path', None)
        if model_path:
            model_display = str(model_path)
            if len(model_display) > 50:
                model_display = '...' + model_display[-47:]
        else:
            model_display = 'None'
            
        # Config summary
        config_str = (
            f"Depth-{self.depth}, Hidden-{self.hidden_dim}, "
            f"Layers-{self.num_layers}, Aggro {self.red_ai_aggression}%, ai-{self.neural_net_ratio}%"
        )
        
        # Create footer text (potentially split into multiple lines)
        model_line = f"Model: {model_display}"
        config_line = f"Config: {config_str}"
        
        # Calculate max width for text (80% of screen width)
        max_width = int(self.screen_width * 0.8)
        
        # Check if lines need to be wrapped
        if footer_font.size(model_line)[0] > max_width:
            # Shorten model path further if needed
            if len(model_display) > 30:
                model_display = '...' + model_display[-27:]
                model_line = f"Model: {model_display}"
        
        # Render the two lines
        model_surf = footer_font.render(model_line, True, (40, 40, 40))
        config_surf = footer_font.render(config_line, True, (40, 40, 40))
        
        # Create background for both lines
        footer_height = model_surf.get_height() + config_surf.get_height() + 6  # Extra padding
        footer_width = max(model_surf.get_width(), config_surf.get_width()) + 20  # Extra padding
        
        # Position the footer
        footer_x = (self.screen_width - footer_width) // 2
        footer_y = self.screen_height - footer_height - 5  # 5px from bottom
        
        # Draw the background
        footer_bg = pygame.Surface((footer_width, footer_height), pygame.SRCALPHA)
        footer_bg.fill((220, 220, 220, 200))
        self.screen.blit(footer_bg, (footer_x, footer_y))
        
        # Draw the text lines
        self.screen.blit(model_surf, (footer_x + 10, footer_y + 3))
        self.screen.blit(config_surf, (footer_x + 10, footer_y + model_surf.get_height() + 3))
        
        # Update display
        pygame.display.flip()

    def render_settings_panel(self):
        """Render the settings panel"""
        # Delegate to SettingsPanel class
        self.settings_ui.draw(self.screen)

    def render_dashboard(self):
        """Render the model dashboard with actual plots"""
        # Delegate to Dashboard class
        self.dashboard_ui.draw(self.screen)
  
    def draw_game_board(self):
        """Draw the game board with triangles and markers"""
        # Delegate to GameBoard class
        self.game_board_ui.draw(self.screen)

    def draw_ml_metrics(self):
        """Draw machine learning metrics in the upper left corner"""
        # Delegate to MLMetricsPanel class
        self.ml_metrics_ui.draw(self.screen)

    def render_help_panel(self):
        """Render the help panel with game rules"""
        # Delegate to HelpPanel class
        self.help_ui.draw(self.screen)
    
    def draw_game_board(self):
        """Draw the game board with triangles and markers"""
        # Draw triangles
        clicks = self.load_clicks()
        valid_loops = self.load_valid_loops()
        
        # Load formed loops and their colors
        formed_loops = self.find_formed_loops(clicks, valid_loops)
        loop_colors = self.get_loop_colors(formed_loops, clicks)
        
        # Draw triangles
        for loop_idx, loop in enumerate(valid_loops):
            vertices = [(vertex['r'], vertex['theta']) for vertex in loop['vertices']]
            self.draw_triangle(vertices, self.colors['triangle_fill'], self.colors['triangle_border'])
        
        # Draw score values in triangles
        loop_scores = self.get_loop_scores(loop_colors)
        for i, (loop, (score_value, score_color)) in enumerate(zip(formed_loops, loop_scores)):
            if score_value != 0 and score_color:
                center_x, center_y = self.calculate_triangle_center(loop)
                font = pygame.font.SysFont(None, 32)
                text = font.render(str(score_value), True, self.colors[score_color])
                text_rect = text.get_rect(center=(center_x, center_y))
                self.screen.blit(text, text_rect)
        
        # Draw markers for played moves
        for click in clicks:
            r, theta = click['address']
            x, y = self.polar_to_cartesian(r, theta)
            color = self.colors[click['color']]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)  # Match marker size
            pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 8, 1)
    
    def draw_triangle(self, vertices, fill_color, border_color):
        """Draw a triangle from polar coordinates"""
        points = []
        for r, theta in vertices:
            x, y = self.polar_to_cartesian(r, theta)
            points.append((int(x), int(y)))
            
            # Draw a small circle at each vertex
            pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 3)
        
        # Draw triangle
        pygame.draw.polygon(self.screen, fill_color, points)
        pygame.draw.polygon(self.screen, border_color, points, 1)

    def initialize_policy_network(self):
        """Initialize the regression network for score prediction"""
        # Use token-based regression: embed each vertex then pool
        num_v = len(self.all_vertices)
        self.regression_net = RegressionNetwork(
            num_vertices=num_v,
            embed_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Value network is the same as our regression network
        self.value_net = self.regression_net
        
        # Create optimizer
        self.regression_optimizer = optim.Adam(self.regression_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = self.regression_optimizer

        # Initialize Blue model from Red model weights for dual training
        self.blue_regression_net = RegressionNetwork(
            num_vertices=num_v,
            embed_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        self.blue_regression_net.load_state_dict(self.regression_net.state_dict())
        # Optimizer for Blue model
        self.blue_regression_optimizer = optim.Adam(self.blue_regression_net.parameters(), lr=self.learning_rate)
        
        # Generate model file name based on board size
        self.model_file = Path(f"vector_game_model_{len(self.all_vertices)}.pt")
        
        # Load saved models if they exist
        if self.model_file.exists():
            try:
                model_data = torch.load(self.model_file)
                if isinstance(model_data, dict):
                    # Try to load regression model if available
                    if 'regression_state' in model_data:
                        self.regression_net.load_state_dict(model_data['regression_state'])
                        print(f"Loaded regression model for board size {len(self.all_vertices)}")
                    # Otherwise try value network weights
                    elif 'value_state' in model_data:
                        self.regression_net.load_state_dict(model_data['value_state'])
                        print(f"Loaded value network for board size {len(self.all_vertices)}")
            except Exception as e:
                print(f"Error loading model for board size {len(self.all_vertices)}: {e}")

    def get_state_dim(self):
        """Calculate state dimension based on current game settings"""
        # Position encodings (r, theta for each vertex)
        position_dim = len(self.all_vertices) * 2  # Each vertex: [r, theta]
        # Ownership encoding (one-hot for each vertex: [empty, red, blue])
        ownership_dim = len(self.all_vertices) * 3
        # Available moves mask
        mask_dim = len(self.all_vertices)  # Binary mask of valid moves
        # Basic state info
        base_dim = 3  # Turn number, red score, blue score
        
        return position_dim + ownership_dim + mask_dim + base_dim

    def state_to_tensor(self, clicks):
        """Convert game state to a tensor for the neural network"""
        # Get basic game information
        valid_loops = self.load_valid_loops()
        formed_loops = self.find_formed_loops(clicks, valid_loops)
        loop_colors = self.get_loop_colors(formed_loops, clicks)
        red_score, blue_score = self.calculate_scores(loop_colors)
        
        # Create position encoding for all vertices
        position_state = np.zeros((len(self.all_vertices), 2))
        for i, (r, theta) in enumerate(self.all_vertices):
            position_state[i] = [r, theta]
        
        # Create ownership encoding [empty, red, blue]
        ownership_state = np.zeros((len(self.all_vertices), 3))
        # Set all vertices to empty initially
        ownership_state[:, 0] = 1
        
        # Update ownership based on clicks
        for click in clicks:
            for i, vertex in enumerate(self.all_vertices):
                if self.is_same_vertex(vertex, click['address']):
                    ownership_state[i, 0] = 0  # Not empty
                    if click['color'] == 'red':
                        ownership_state[i, 1] = 1
                    else:
                        ownership_state[i, 2] = 1
                    break
        
        # Create available moves mask
        valid_moves_mask = np.ones(len(self.all_vertices))
        for click in clicks:
            for i, vertex in enumerate(self.all_vertices):
                if self.is_same_vertex(vertex, click['address']):
                    valid_moves_mask[i] = 0
                    break
        
        # Game progress features
        max_possible_score = len(valid_loops) * 2
        progress_features = np.array([
            self.global_turn_number / 100.0,  # Normalized turn number
            red_score / max(1, max_possible_score),  # Normalized red score
            blue_score / max(1, max_possible_score)  # Normalized blue score
        ])
        
        # Combine all features
        combined_features = np.concatenate([
            position_state.flatten(),  # [r1, theta1, r2, theta2, ...]
            ownership_state.flatten(), # [empty1, red1, blue1, empty2, ...]
            valid_moves_mask,         # [1, 0, 1, ...] where 1 means available
            progress_features        
        ])
        
        # Convert to PyTorch tensor
        state_tensor = torch.FloatTensor(combined_features).to(self.device)
        return state_tensor

    def state_to_tokens(self, clicks):
        """Convert current click state into vertex indices and occupancy states for token embedding"""
        # Vertex indices: 0..V-1
        num_v = len(self.all_vertices)
        vertex_indices = torch.arange(num_v, device=self.device, dtype=torch.long)
        # Build occupancy: 0=empty,1=red,2=blue
        occ = torch.zeros(num_v, device=self.device, dtype=torch.long)
        for click in clicks:
            addr = click['address']
            color = click['color']
            # find vertex idx
            for i, v in enumerate(self.all_vertices):
                if self.is_same_vertex(v, addr):
                    occ[i] = 1 if color=='red' else 2
                    break
        return vertex_indices, occ

    def count_triangles(self):
        """Count the number of triangles in the game"""
        try:
            with open('valid_loops.json', 'r') as f:
                lines = f.readlines()
                return len(lines)
        except FileNotFoundError:
            return 0

    def get_valid_actions(self, clicks):
        """Get valid actions (unplayed vertices) as indices - optimized version"""
        # Using set operations for efficiency
        played_vertex_indices = set()
        
        for click in clicks:
            for i, vertex in enumerate(self.all_vertices):
                if self.is_same_vertex(click['address'], vertex):
                    played_vertex_indices.add(i)
                    break
        
        # All indices minus played indices
        all_indices = set(range(len(self.all_vertices)))
        valid_actions = list(all_indices - played_vertex_indices)
        
        return valid_actions
    
    def get_red_move(self, clicks):
        """AI logic for Red's move with a two-layer decision approach:
        1. Quiescence vs Greedy (based on red_ai_aggression slider)
        2. AI vs Heuristic evaluation (based on neural_net_ratio slider)
        """
        # Get valid actions first - common to all approaches
        valid_actions = self.get_valid_actions(clicks)
        if not valid_actions:
            return None
            
        # If there's only one valid move, just take it
        if len(valid_actions) == 1:
            return self.all_vertices[valid_actions[0]]
        
        # FIRST LAYER: Decide between Quiescence Search and Greedy approach
        use_quiescence = random.random() < (self.red_ai_aggression / 100)
        
        if use_quiescence:
            # Use the quiescence search approach
            # SECOND LAYER: Decide between AI and heuristic evaluation for quiescence
            use_ai_eval = self.use_neural_net and random.random() < (self.neural_net_ratio / 100)
            
            # Try quiescence search with selected evaluation method
            move = integrate_quiescence_search(
                self,
                clicks,
                is_red_turn=True,
                max_depth=self.depth,
                use_nn=use_ai_eval  # Use AI evaluation based on neural_net_ratio
            )
            
            # If quiescence search found a good move, return it
            if move:
                # Double-check the move is valid
                for existing_click in clicks:
                    if self.is_same_vertex(existing_click['address'], move):
                        print(f"Warning: Quiescence search returned already occupied position: {move}")
                        return self.get_random_valid_move(valid_actions)  # Fall back to random valid move
                
                # If using AI evaluation, do TD learning
                if use_ai_eval:
                    # Tokenize current state
                    state_idx, state_occ = self.state_to_tokens(clicks)
                    
                    # Compute reward for training
                    valid_loops = self.load_valid_loops()
                    formed_loops_before = self.find_formed_loops(clicks, valid_loops)
                    loop_colors_before = self.get_loop_colors(formed_loops_before, clicks)
                    red_score_before, blue_score_before = self.calculate_scores(loop_colors_before)
                    
                    # Score after move
                    temp_clicks = clicks.copy()
                    temp_clicks.append({'turn': 0, 'color': 'red', 'address': move})
                    formed_loops_after = self.find_formed_loops(temp_clicks, valid_loops)
                    loop_colors_after = self.get_loop_colors(formed_loops_after, temp_clicks)
                    red_score_after, blue_score_after = self.calculate_scores(loop_colors_after)
                    
                    # Calculate points earned by each player
                    points_earned = red_score_after - red_score_before
                    blue_points_earned = blue_score_after - blue_score_before
                    reward = points_earned - blue_points_earned
                    
                    # Perform TD update
                    next_idx, next_occ = self.state_to_tokens(temp_clicks)
                    loss = self.td_update(state_idx, state_occ, reward, next_idx, next_occ, is_terminal=False)
                    if loss is not None:
                        self.stats['value_losses'].append(loss)
                
                return move
        
        # If we're here, either we're using the greedy approach or quiescence didn't find a move
        # GREEDY APPROACH: Take highest scoring move
        # SECOND LAYER: Decide between AI and heuristic evaluation for greedy
        use_ai_eval = self.use_neural_net and random.random() < (self.neural_net_ratio / 100)
        
        if use_ai_eval:
            # Use neural network to evaluate moves
            # Tokenize current state
            state_idx, state_occ = self.state_to_tokens(clicks)
            
            # Create batch of all possible next states
            batch_indices = []
            batch_occs = []
            
            # For each potential move, create a new state
            for action_idx in valid_actions:
                move = self.all_vertices[action_idx]
                # Generate the occupancy state after this move
                temp_occ = state_occ.clone()
                temp_occ[action_idx] = 1  # Red = 1
                batch_indices.append(state_idx.clone())
                batch_occs.append(temp_occ)
            
            # Stack into batches for efficient evaluation
            batch_vertex_indices = torch.stack(batch_indices)
            batch_occs = torch.stack(batch_occs)
            
            # Evaluate all moves at once
            with torch.no_grad():
                predicted_scores = self.regression_net(batch_vertex_indices, batch_occs).squeeze()
            
            # Find the best move
            if predicted_scores.dim() == 0:  # Only one score (scalar tensor)
                best_idx = 0  # Only one move, so index is 0
                best_score = predicted_scores.item()
            else:  # Multiple scores (vector tensor)
                best_idx = torch.argmax(predicted_scores).item()
                best_score = predicted_scores[best_idx].item()
            
            best_action_idx = valid_actions[best_idx]
            best_move = self.all_vertices[best_action_idx]
            
            # Double-check the move is valid
            for existing_click in clicks:
                if self.is_same_vertex(existing_click['address'], best_move):
                    print(f"Warning: Neural network returned already occupied position: {best_move}")
                    return self.get_random_valid_move(valid_actions)  # Fall back to random valid move
            
            # Compute reward for training
            valid_loops = self.load_valid_loops()
            formed_loops_before = self.find_formed_loops(clicks, valid_loops)
            loop_colors_before = self.get_loop_colors(formed_loops_before, clicks)
            red_score_before, blue_score_before = self.calculate_scores(loop_colors_before)
            
            # Score after move
            temp_clicks = clicks.copy()
            temp_clicks.append({'turn': 0, 'color': 'red', 'address': best_move})
            formed_loops_after = self.find_formed_loops(temp_clicks, valid_loops)
            loop_colors_after = self.get_loop_colors(formed_loops_after, temp_clicks)
            red_score_after, blue_score_after = self.calculate_scores(loop_colors_after)
            
            # Calculate points earned by each player
            points_earned = red_score_after - red_score_before
            blue_points_earned = blue_score_after - blue_score_before
            reward = points_earned - blue_points_earned
            
            # Perform TD update
            next_idx, next_occ = self.state_to_tokens(temp_clicks)
            loss = self.td_update(state_idx, state_occ, reward, next_idx, next_occ, is_terminal=False)
            if loss is not None:
                self.stats['value_losses'].append(loss)
                
            return best_move
        else:
            # Use heuristic evaluation - find potential scoring moves
            scoring_moves = []
            
            # Calculate score before any moves (only once)
            valid_loops = self.load_valid_loops()
            formed_loops_before = self.find_formed_loops(clicks, valid_loops)
            loop_colors_before = self.get_loop_colors(formed_loops_before, clicks)
            red_score_before, blue_score_before = self.calculate_scores(loop_colors_before)
            
            # Find potentially scoring vertices using spatial hashing
            potentially_scoring_vertices = set()
            
            # Get all placed vertices
            placed_vertices = [click['address'] for click in clicks]
            
            # For each placed vertex, find nearby vertices that might complete triangles
            for vertex in placed_vertices:
                r, theta = vertex
                
                # Get all vertex indices near this one
                nearby_indices = self.get_nearby_vertices(r, theta, radius=0.3)
                
                # Filter to only include valid (unplayed) actions
                nearby_indices = [idx for idx in nearby_indices if idx in valid_actions]
                
                # Add to the set of potentially scoring vertices
                potentially_scoring_vertices.update(nearby_indices)
            
            # If we found potentially scoring vertices, evaluate those first
            potentially_scoring_list = list(potentially_scoring_vertices)
            
            if potentially_scoring_list:
                evaluation_indices = (random.sample(potentially_scoring_list, min(self.max_heuristic_evals, len(potentially_scoring_list))))
            else:
                # If no potential scoring vertices found, use regular sampling
                evaluation_indices = random.sample(valid_actions, min(self.max_heuristic_evals, len(valid_actions)))
            
            # Convert indices to actual vertex coordinates
            unplayed_moves = [self.all_vertices[i] for i in evaluation_indices]
            
            # Evaluate each potential move
            for move in unplayed_moves:
                # Double-check the move is valid (safety check)
                move_already_taken = False
                for existing_click in clicks:
                    # Use the same tolerance as in is_same_vertex function
                    if self.is_same_vertex(existing_click['address'], move):
                        move_already_taken = True
                        print(f"Warning: Heuristic evaluation attempted to evaluate already taken position: {move}")
                        break
                
                if move_already_taken:
                    continue
                
                # Temporarily add this move to evaluate scores
                temp_clicks = clicks.copy()
                temp_clicks.append({'turn': 0, 'color': 'red', 'address': move})
                formed_loops_after = self.find_formed_loops(temp_clicks, valid_loops)
                loop_colors_after = self.get_loop_colors(formed_loops_after, temp_clicks)
                red_score_after, blue_score_after = self.calculate_scores(loop_colors_after)
                
                # Calculate points earned and score difference
                points_earned = red_score_after - red_score_before
                score_difference = (red_score_after - blue_score_after) - (red_score_before - blue_score_before)
                weighted_score = 3.0 * points_earned + score_difference
                
                scoring_moves.append((move, weighted_score))
            
            # Sort all moves by score (even if points_earned == 0)
            scoring_moves.sort(key=lambda x: x[1], reverse=True)
            
            # If there are scoring moves, choose the highest scoring one
            if scoring_moves:
                return scoring_moves[0][0]
            else:
                # Fallback - should never happen as we evaluate all possible moves
                return self.get_random_valid_move(valid_actions)
    
    def get_random_valid_move(self, valid_actions):
        """Helper function to get a random valid move"""
        return self.all_vertices[random.choice(valid_actions)]

    def get_blue_move(self, clicks, aggression=None):
        """AI logic for Blue's move with neural net or heuristic approach"""
        # Neural net branch for Blue using swapped perspective
        use_nn = self.use_neural_net and random.random() < (self.neural_net_ratio / 100)
        if use_nn:
            valid_actions = self.get_valid_actions(clicks)
            if not valid_actions:
                return None
            # Tokenize and swap occupancy: red<->blue
            state_idx, occ_orig = self.state_to_tokens(clicks)
            occ = occ_orig.clone()
            red_mask = occ_orig == 1; blue_mask = occ_orig == 2
            occ[red_mask] = 2; occ[blue_mask] = 1
            # Evaluate candidates with Blue model
            if len(valid_actions)==1:
                best_idx=valid_actions[0]; best_move=self.all_vertices[best_idx]
            else:
                bidx, boccs = [],[]
                for ai in valid_actions:
                    tmp=occ.clone(); tmp[ai]=1
                    bidx.append(state_idx.clone()); boccs.append(tmp)
                bidx=torch.stack(bidx); boccs=torch.stack(boccs)
                with torch.no_grad():
                    scores=self.blue_regression_net(bidx,boccs).squeeze()
                if scores.dim()==0: sel=0
                else: sel=torch.argmax(scores).item()
                best_idx=valid_actions[sel]; best_move=self.all_vertices[best_idx]
            # compute swapped reward
            loops_before=self.find_formed_loops(clicks,self.load_valid_loops())
            lc_before=self.get_loop_colors(loops_before,clicks)
            rsb,bsb=self.calculate_scores(lc_before)
            temp=clicks.copy(); temp.append({'turn':0,'color':'blue','address':best_move})
            loops_after=self.find_formed_loops(temp,self.load_valid_loops())
            lc_after=self.get_loop_colors(loops_after,temp)
            rsa,bsa=self.calculate_scores(lc_after)
            # points from blue perspective as red: reward = (blue gain) - (red gain)
            rew=(bsa-bsb) - (rsa-rsb)
            # next state tokens swap
            nidx,nocc_orig=self.state_to_tokens(temp)
            nocc=nocc_orig.clone(); red_mask=nocc_orig==1; blue_mask=nocc_orig==2
            nocc[red_mask]=2; nocc[blue_mask]=1
            # train Blue model
            self.td_update_blue(state_idx,occ,rew,nidx,nocc,is_terminal=False)
            return best_move
        # quiescence search branch for Blue
        if random.random() < (self.red_ai_aggression / 100):
            move = integrate_quiescence_search(
                self,
                clicks,
                is_red_turn=False,
                max_depth=self.depth,
                use_nn=self.use_neural_net
            )
            if move:
                return move
        # Use passed aggression parameter or fallback
        if aggression is None:
            if hasattr(self, 'blue_ai_aggression'):
                aggression = self.blue_ai_aggression
            else:
                aggression = self.red_ai_aggression
        
        # Determine if this move should be aggressive
        make_aggressive_move = random.random() < (aggression / 100)
        valid_actions = self.get_valid_actions(clicks)
        
        if not valid_actions:
            return None
            
        if make_aggressive_move:
            scoring_moves = []
            valid_loops = self.load_valid_loops()
            formed_loops_before = self.find_formed_loops(clicks, valid_loops)
            loop_colors_before = self.get_loop_colors(formed_loops_before, clicks)
            red_score_before, blue_score_before = self.calculate_scores(loop_colors_before)
            
            # Find potentially scoring vertices
            potentially_scoring_vertices = set()
            placed_vertices = [click['address'] for click in clicks]
            
            for vertex in placed_vertices:
                r, theta = vertex
                nearby_indices = self.get_nearby_vertices(r, theta, radius=0.3)
                nearby_indices = [idx for idx in nearby_indices if idx in valid_actions]
                potentially_scoring_vertices.update(nearby_indices)
            
            potentially_scoring_list = list(potentially_scoring_vertices)
            
            if potentially_scoring_list:
                evaluation_indices = (random.sample(potentially_scoring_list, self.max_heuristic_evals)
                                    if len(potentially_scoring_list) > self.max_heuristic_evals
                                    else potentially_scoring_list)
            else:
                evaluation_indices = (random.sample(valid_actions, self.max_heuristic_evals)
                                    if len(valid_actions) > self.max_heuristic_evals
                                    else valid_actions)
            
            unplayed_moves = [self.all_vertices[i] for i in evaluation_indices]
            
            for move in unplayed_moves:
                temp_click = {'turn': 0, 'color': 'blue', 'address': move}
                temp_clicks = clicks.copy()
                temp_clicks.append(temp_click)
                formed_loops_after = self.find_formed_loops(temp_clicks, valid_loops)
                loop_colors_after = self.get_loop_colors(formed_loops_after, temp_clicks)
                red_score_after, blue_score_after = self.calculate_scores(loop_colors_after)
                
                points_earned = blue_score_after - blue_score_before
                score_difference = (blue_score_after - red_score_after) - (blue_score_before - red_score_before)
                weighted_score = 3.0 * points_earned + score_difference
                
                if points_earned > 0:
                    scoring_moves.append((move, weighted_score))
            
            if scoring_moves:
                scoring_moves.sort(key=lambda x: x[1], reverse=True)
                return scoring_moves[0][0]
            else:
                return random.choice([self.all_vertices[i] for i in valid_actions])
        else:
            # Make a random move
            return random.choice([self.all_vertices[i] for i in valid_actions])

    def get_training_red_move(self, clicks, valid_loops):
        """AI logic for Red's move during training - uses cached valid_loops for efficiency"""
        # Decide whether to use neural network or heuristic
        use_nn = self.use_neural_net and random.random() < (self.neural_net_ratio / 100)
        
        if use_nn:
            # Get valid actions
            valid_actions = self.get_valid_actions(clicks)
            
            if not valid_actions:
                return None
                
            # Tokenize current state
            state_idx, state_occ = self.state_to_tokens(clicks)
            
            # Handle case when there's only one valid move
            if len(valid_actions) == 1:
                # If there's only one valid move, just take it without evaluation
                best_action_idx = valid_actions[0]
                best_move = self.all_vertices[best_action_idx]
                best_score = 0  # Default score since we're not evaluating
            else:
                # Batch evaluate all valid moves at once
                # Create batch of all possible next states
                batch_indices = []
                batch_occs = []
                
                # For each potential move, create a new state
                for action_idx in valid_actions:
                    move = self.all_vertices[action_idx]
                    # Generate the occupancy state after this move
                    temp_occ = state_occ.clone()
                    temp_occ[action_idx] = 1  # Red = 1
                    batch_indices.append(state_idx.clone())
                    batch_occs.append(temp_occ)
                
                # Stack into batches for efficient evaluation
                batch_vertex_indices = torch.stack(batch_indices)
                batch_occs = torch.stack(batch_occs)
                
                # Evaluate all moves at once
                with torch.no_grad():
                    predicted_scores = self.regression_net(batch_vertex_indices, batch_occs).squeeze()
                
                # Find the best move
                if predicted_scores.dim() == 0:  # Only one score (scalar tensor)
                    best_idx = 0  # Only one move, so index is 0
                    best_score = predicted_scores.item()
                else:  # Multiple scores (vector tensor)
                    best_idx = torch.argmax(predicted_scores).item()
                    best_score = predicted_scores[best_idx].item()
                
                best_action_idx = valid_actions[best_idx]
                best_move = self.all_vertices[best_action_idx]
            
            # Compute actual reward for training
            # OPTIMIZATION: Use cached valid_loops instead of loading from disk
            formed_loops_before = self.find_formed_loops(clicks, valid_loops)
            loop_colors_before = self.get_loop_colors(formed_loops_before, clicks)
            red_score_before, blue_score_before = self.calculate_scores(loop_colors_before)
            
            # Score after move
            temp_clicks = clicks.copy()
            temp_clicks.append({'turn': 0, 'color': 'red', 'address': best_move})
            formed_loops_after = self.find_formed_loops(temp_clicks, valid_loops)
            loop_colors_after = self.get_loop_colors(formed_loops_after, temp_clicks)
            red_score_after, blue_score_after = self.calculate_scores(loop_colors_after)
            
            # Calculate points earned by each player
            points_earned = red_score_after - red_score_before
            blue_points_earned = blue_score_after - blue_score_before
            # Reward penalized by blue's points in the same round
            reward = points_earned - blue_points_earned
            
            # Perform a TD(0) update using next state
            next_idx, next_occ = self.state_to_tokens(temp_clicks)
            loss = self.td_update(state_idx, state_occ, reward, next_idx, next_occ, is_terminal=False)
            # Make sure we record the loss for tracking
            if loss is not None:
                self.stats['value_losses'].append(loss)
            
            return best_move
        else:
            # Use optimized heuristic approach with spatial hashing
            make_aggressive_move = random.random() < (self.red_ai_aggression / 100)
            
            # Fast method to get unplayed moves using set operations
            valid_actions = self.get_valid_actions(clicks)
            
            if not valid_actions:
                return None
            
            # If there's only one valid move, just take it
            if len(valid_actions) == 1:
                return self.all_vertices[valid_actions[0]]
                
            # Use spatial hashing to find promising moves
            if make_aggressive_move:
                # Find potential scoring moves with spatial hashing
                scoring_moves = []
                
                # Calculate score before any moves (only once)
                # OPTIMIZATION: Use cached valid_loops
                formed_loops_before = self.find_formed_loops(clicks, valid_loops)
                loop_colors_before = self.get_loop_colors(formed_loops_before, clicks)
                red_score_before, blue_score_before = self.calculate_scores(loop_colors_before)
                
                # Find potentially scoring vertices using spatial hashing
                potentially_scoring_vertices = set()
                
                # Get all placed vertices
                placed_vertices = []
                for click in clicks:
                    placed_vertices.append(click['address'])
                
                # For each placed vertex, find nearby vertices that might complete triangles
                for vertex in placed_vertices:
                    r, theta = vertex
                    
                    # Get all vertex indices near this one
                    nearby_indices = self.get_nearby_vertices(r, theta, radius=0.3)
                    
                    # Filter to only include valid (unplayed) actions
                    nearby_indices = [idx for idx in nearby_indices if idx in valid_actions]
                    
                    # Add to the set of potentially scoring vertices
                    potentially_scoring_vertices.update(nearby_indices)
                
                # If we found potentially scoring vertices, evaluate those first
                potentially_scoring_list = list(potentially_scoring_vertices)
                
                if potentially_scoring_list:
                    evaluation_indices = (random.sample(potentially_scoring_list, min(self.max_heuristic_evals, len(potentially_scoring_list))))
                else:
                    # If no potential scoring vertices found, use regular sampling
                    evaluation_indices = random.sample(valid_actions, min(self.max_heuristic_evals, len(valid_actions)))
                
                # Convert indices to actual vertex coordinates
                unplayed_moves = [self.all_vertices[i] for i in evaluation_indices]
                
                # Evaluate each potential move
                for move in unplayed_moves:
                    # Temporarily add this move to evaluate scores
                    temp_click = {
                        'turn': 0,
                        'color': 'red',
                        'address': move
                    }
                    
                    # Score after move
                    temp_clicks = clicks.copy()
                    temp_clicks.append(temp_click)
                    formed_loops_after = self.find_formed_loops(temp_clicks, valid_loops)
                    loop_colors_after = self.get_loop_colors(formed_loops_after, temp_clicks)
                    red_score_after, blue_score_after = self.calculate_scores(loop_colors_after)
                    
                    # Calculate points earned and score difference
                    points_earned = red_score_after - red_score_before
                    score_difference = (red_score_after - blue_score_after) - (red_score_before - blue_score_before)
                    weighted_score = 3.0 * points_earned + score_difference
                    
                    if points_earned > 0:
                        scoring_moves.append((move, weighted_score))
                
                # If there are scoring moves, choose the highest scoring one
                if scoring_moves:
                    scoring_moves.sort(key=lambda x: x[1], reverse=True)
                    return scoring_moves[0][0]
                else:
                    # No scoring moves, choose random
                    return random.choice([self.all_vertices[i] for i in valid_actions])
            else:
                # Make a random move from valid actions
                return random.choice([self.all_vertices[i] for i in valid_actions])

    def get_training_blue_move(self, clicks, valid_loops, aggression=None):
        """AI logic for Blue's move during training - uses cached valid_loops for efficiency"""
        # Neural net branch for Blue using swapped perspective
        use_nn = self.use_neural_net and random.random() < (self.neural_net_ratio / 100)
        if use_nn:
            valid_actions = self.get_valid_actions(clicks)
            if not valid_actions:
                return None
                
            # Tokenize and swap occupancy: red<->blue
            state_idx, occ_orig = self.state_to_tokens(clicks)
            occ = occ_orig.clone()
            red_mask = occ_orig == 1; blue_mask = occ_orig == 2
            occ[red_mask] = 2; occ[blue_mask] = 1
            
            # Evaluate candidates with Blue model
            if len(valid_actions)==1:
                best_idx=valid_actions[0]; best_move=self.all_vertices[best_idx]
            else:
                bidx, boccs = [],[]
                for ai in valid_actions:
                    tmp=occ.clone(); tmp[ai]=1
                    bidx.append(state_idx.clone()); boccs.append(tmp)
                bidx=torch.stack(bidx); boccs=torch.stack(boccs)
                with torch.no_grad():
                    scores=self.blue_regression_net(bidx,boccs).squeeze()
                if scores.dim()==0: sel=0
                else: sel=torch.argmax(scores).item()
                best_idx=valid_actions[sel]; best_move=self.all_vertices[best_idx]
                
            # compute swapped reward
            # OPTIMIZATION: Use cached valid_loops
            loops_before=self.find_formed_loops(clicks, valid_loops)
            lc_before=self.get_loop_colors(loops_before,clicks)
            rsb,bsb=self.calculate_scores(lc_before)
            temp=clicks.copy(); temp.append({'turn':0,'color':'blue','address':best_move})
            loops_after=self.find_formed_loops(temp, valid_loops)
            lc_after=self.get_loop_colors(loops_after,temp)
            rsa,bsa=self.calculate_scores(lc_after)
            
            # points from blue perspective as red: reward = (blue gain) - (red gain)
            rew=(bsa-bsb) - (rsa-rsb)
            
            # next state tokens swap
            nidx,nocc_orig=self.state_to_tokens(temp)
            nocc=nocc_orig.clone(); red_mask=nocc_orig==1; blue_mask=nocc_orig==2
            nocc[red_mask]=2; nocc[blue_mask]=1
            
            # train Blue model
            self.td_update_blue(state_idx,occ,rew,nidx,nocc,is_terminal=False)
            return best_move

        # Use heuristic approach for Blue
        # Use passed aggression parameter or fallback
        if aggression is None:
            if hasattr(self, 'blue_ai_aggression'):
                aggression = self.blue_ai_aggression
            else:
                aggression = self.red_ai_aggression
        
        # Determine if this move should be aggressive
        make_aggressive_move = random.random() < (aggression / 100)
        valid_actions = self.get_valid_actions(clicks)
        
        if not valid_actions:
            return None
            
        if make_aggressive_move:
            scoring_moves = []
            # OPTIMIZATION: Use cached valid_loops
            formed_loops_before = self.find_formed_loops(clicks, valid_loops)
            loop_colors_before = self.get_loop_colors(formed_loops_before, clicks)
            red_score_before, blue_score_before = self.calculate_scores(loop_colors_before)
            
            # Find potentially scoring vertices
            potentially_scoring_vertices = set()
            placed_vertices = [click['address'] for click in clicks]
            
            for vertex in placed_vertices:
                r, theta = vertex
                nearby_indices = self.get_nearby_vertices(r, theta, radius=0.3)
                nearby_indices = [idx for idx in nearby_indices if idx in valid_actions]
                potentially_scoring_vertices.update(nearby_indices)
            
            potentially_scoring_list = list(potentially_scoring_vertices)
            
            if potentially_scoring_list:
                evaluation_indices = (random.sample(potentially_scoring_list, self.max_heuristic_evals)
                                    if len(potentially_scoring_list) > self.max_heuristic_evals
                                    else potentially_scoring_list)
            else:
                evaluation_indices = (random.sample(valid_actions, self.max_heuristic_evals)
                                    if len(valid_actions) > self.max_heuristic_evals
                                    else valid_actions)
            
            unplayed_moves = [self.all_vertices[i] for i in evaluation_indices]
            
            for move in unplayed_moves:
                temp_click = {'turn': 0, 'color': 'blue', 'address': move}
                temp_clicks = clicks.copy()
                temp_clicks.append(temp_click)
                formed_loops_after = self.find_formed_loops(temp_clicks, valid_loops)
                loop_colors_after = self.get_loop_colors(formed_loops_after, temp_clicks)
                red_score_after, blue_score_after = self.calculate_scores(loop_colors_after)
                
                points_earned = blue_score_after - blue_score_before
                score_difference = (blue_score_after - red_score_after) - (blue_score_before - red_score_before)
                weighted_score = 3.0 * points_earned + score_difference
                
                if points_earned > 0:
                    scoring_moves.append((move, weighted_score))
            
            if scoring_moves:
                scoring_moves.sort(key=lambda x: x[1], reverse=True)
                return scoring_moves[0][0]
            else:
                return random.choice([self.all_vertices[i] for i in valid_actions])
        else:
            # Make a random move
            return random.choice([self.all_vertices[i] for i in valid_actions])

    def save_model(self):
        """Save the neural network model to a file"""
        try:
            model_data = {
                'regression_state': self.regression_net.state_dict(),
                'value_state': self.value_net.state_dict()
            }
            torch.save(model_data, self.model_file)
            self.status_message = f"Neural network model saved successfully for board size {len(self.all_vertices)}."
        except Exception as e:
            self.status_message = f"Error saving model for board size {len(self.all_vertices)}: {e}"
        self.status_label.update_text(self.status_message)
   
    def polar_to_cartesian(self, r, theta):
        """Convert polar coordinates to Cartesian coordinates on screen"""
        # Apply scale multiplier
        scale = self.scale * self.scale_multiplier
        x = self.center_x + r * scale * math.cos(theta)
        y = self.center_y - r * scale * math.sin(theta)
        return x, y

    def cartesian_to_polar(self, x, y):
        """Convert Cartesian coordinates to polar coordinates"""
        scale = self.scale * self.scale_multiplier
        dx = x - self.center_x
        dy = self.center_y - y  # Invert y because pygame y increases downward
        r = math.sqrt(dx**2 + dy**2) / scale
        theta = math.atan2(dy, dx)
        return r, theta
    
    def get_surrounding_triangles(self, triangle_vertices):
        """Get the surrounding triangles for a given triangle"""
        midpoints = np.zeros((3, 2))
        for i in range(3):
            midpoints[i] = (triangle_vertices[i] + triangle_vertices[(i + 1) % 3]) / 2
            
        surrounding_triangles = np.zeros((4, 3, 2))
        for i in range(3):
            surrounding_triangles[i] = np.vstack((triangle_vertices[i], midpoints[i], midpoints[(i - 1) % 3]))
        surrounding_triangles[3] = midpoints
        
        return surrounding_triangles

    def generate_triangle_tessellation(self):
        """Generate the triangle tessellation with efficient direct vertex generation"""
        # Clear existing files
        self.all_vertices = []
        with open('valid_spaces.json', 'w') as f:
            pass
        with open('valid_loops.json', 'w') as f:
            pass
            
        # Use a set for fast duplicate detection
        vertices_set = set()
        triangles = []
        
        # Initial triangle
        r = 1
        theta = np.linspace(0, 2 * np.pi, 4)[:-1] + np.pi / 6
        vertices = np.zeros((3, 2))
        vertices[:, 0] = r * np.cos(theta)
        vertices[:, 1] = r * np.sin(theta)
        
        # Convert to polar coordinates
        polar_vertices = []
        for i in range(3):
            r_val = np.sqrt(vertices[i, 0]**2 + vertices[i, 1]**2)
            theta_val = np.arctan2(vertices[i, 1], vertices[i, 0])
            polar_vertices.append((r_val, theta_val))
            
        # Initial triangle
        triangles = [polar_vertices.copy()]
        
        # Add vertices to the set with discretization for reliable duplicate detection
        tolerance = 0.01
        for r_val, theta_val in polar_vertices:
            r_key = round(r_val / tolerance) * tolerance
            theta_key = round(theta_val / tolerance) * tolerance
            vertices_set.add((r_key, theta_key))
        
        # Process each depth level
        for _ in range(self.depth):
            new_triangles = []
            
            # Dictionary to deduplicate triangles
            triangle_dict = {}
            
            for triangle in triangles:
                # Convert to cartesian for surrounding triangle calculation
                cart_triangle = np.zeros((3, 2))
                for i in range(3):
                    r, theta = triangle[i]
                    cart_triangle[i, 0] = r * math.cos(theta)
                    cart_triangle[i, 1] = r * math.sin(theta)
                
                # Get surrounding triangles
                surrounding = self.get_surrounding_triangles(cart_triangle)
                
                # Process each surrounding triangle
                for surr_triangle in surrounding:
                    polar_surr = []
                    triangle_key = []
                    
                    for i in range(3):
                        # Convert to polar coordinates
                        x, y = surr_triangle[i]
                        r = np.sqrt(x**2 + y**2)
                        theta = np.arctan2(y, x)
                        
                        # Discretize for consistent duplicate detection
                        r_key = round(r / tolerance) * tolerance
                        # Normalize theta to [0, 2π)
                        while theta < 0:
                            theta += 2 * np.pi
                        while theta >= 2 * np.pi:
                            theta -= 2 * np.pi
                        theta_key = round(theta / tolerance) * tolerance
                        
                        # Keep actual values for the triangle
                        polar_surr.append((r, theta))
                        # Use discretized key for triangle deduplication
                        triangle_key.append((r_key, theta_key))
                    
                    # Sort the triangle key to ensure consistent ordering
                    triangle_key = tuple(sorted(triangle_key))
                    if triangle_key not in triangle_dict:
                        triangle_dict[triangle_key] = polar_surr
                        new_triangles.append(polar_surr)
                        
                        # Add vertices to the global set
                        for r_val, theta_val in polar_surr:
                            r_key = round(r_val / tolerance) * tolerance
                            theta_key = round(theta_val / tolerance) * tolerance
                            vertices_set.add((r_key, theta_key))
            
            triangles = new_triangles
        
        # Convert vertex set to list and normalize theta to [0, 2π)
        self.all_vertices = []
        for r_key, theta_key in vertices_set:
            theta_norm = theta_key
            while theta_norm < 0:
                theta_norm += 2 * np.pi
            while theta_norm >= 2 * np.pi:
                theta_norm -= 2 * np.pi
            self.all_vertices.append((r_key, theta_norm))
        
        # Batch write vertices
        vertices_json = []
        for r, theta in self.all_vertices:
            vertices_json.append({'r': r, 'theta': theta})
        
        with open('valid_spaces.json', 'w') as f:
            for vertex in vertices_json:
                json.dump(vertex, f)
                f.write('\n')
        
        # Batch write triangles
        with open('valid_loops.json', 'w') as f:
            for triangle in triangles:
                vertices_json = [{'r': r, 'theta': theta} for r, theta in triangle]
                json.dump({'vertices': vertices_json}, f)
                f.write('\n')
        
        # Build lookup tables for optimization
        self.build_lookup_tables()

    def build_lookup_tables(self):
        """Build lookup tables for optimization"""
        # Clear any existing data
        self.vertex_to_triangles = defaultdict(list)
        self.triangle_vertices = {}
        self.vertex_set_to_triangle = {}
        self.spatial_hash = {}
        
        # Load valid loops (triangles)
        valid_loops = self.load_valid_loops()
        
        # Build mappings
        for triangle_idx, loop in enumerate(valid_loops):
            vertex_coords = [(vertex['r'], vertex['theta']) for vertex in loop['vertices']]
            vertex_indices = []
            
            # Find the indices of the vertices in self.all_vertices
            for vertex_coord in vertex_coords:
                for vertex_idx, v in enumerate(self.all_vertices):
                    if self.is_same_vertex(vertex_coord, v):
                        vertex_indices.append(vertex_idx)
                        # Add this triangle to the vertex's list
                        self.vertex_to_triangles[vertex_idx].append(triangle_idx)
                        break
            
            # Store vertex indices for this triangle
            self.triangle_vertices[triangle_idx] = vertex_indices
            # Store mapping from vertex set to triangle index
            self.vertex_set_to_triangle[frozenset(vertex_indices)] = triangle_idx
        
        # Build spatial hash
        self.build_spatial_hash()
    
    def build_spatial_hash(self):
        """Build a spatial hash for fast vertex lookups"""
        # Clear existing hash
        self.spatial_hash = {}
        
        # Add all vertices to the hash
        for vertex_idx, vertex in enumerate(self.all_vertices):
            r, theta = vertex
            cell_key = self.get_cell_key(r, theta)
            
            if cell_key not in self.spatial_hash:
                self.spatial_hash[cell_key] = []
            
            self.spatial_hash[cell_key].append(vertex_idx)
    
    def get_cell_key(self, r, theta):
        """Convert polar coordinates to a grid cell key"""
        # Normalize theta to [0, 2π)
        while theta < 0:
            theta += 2 * np.pi
        while theta >= 2 * np.pi:
            theta -= 2 * np.pi
        
        # Compute grid cell indices
        r_idx = int(r / self.grid_size)
        theta_idx = int(theta / (self.grid_size * np.pi))
        
        # Return composite key
        return (r_idx, theta_idx)
    
    def get_nearby_vertices(self, r, theta, radius=0.2):
        """Get vertex indices near the given coordinates"""
        nearby_indices = set()
        
        # Get the central cell
        center_cell = self.get_cell_key(r, theta)
        r_idx, theta_idx = center_cell
        
        # Determine cell radius based on search radius
        cell_radius = int(radius / self.grid_size) + 1
        
        # Search neighboring cells
        for dr in range(-cell_radius, cell_radius + 1):
            for dt in range(-cell_radius, cell_radius + 1):
                # Get neighboring cell key
                neighbor_cell = (r_idx + dr, theta_idx + dt)
                
                # Add vertices in this cell if it exists
                if neighbor_cell in self.spatial_hash:
                    nearby_indices.update(self.spatial_hash[neighbor_cell])
        
        return list(nearby_indices)

    def find_nearest_valid_space(self, pos):
        """Find the nearest valid space to the clicked position"""
        x, y = pos
        r, theta = self.cartesian_to_polar(x, y)
        min_distance = float('inf')
        nearest_vertex = None
        
        for vertex_r, vertex_theta in self.all_vertices:
            # Calculate distance in polar coordinates
            delta_theta = vertex_theta - theta
            if delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            elif delta_theta < -np.pi:
                delta_theta += 2 * np.pi
                
            distance = np.sqrt((r - vertex_r)**2 + delta_theta**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_vertex = (vertex_r, vertex_theta)
                
        return nearest_vertex

    def is_same_vertex(self, v1, v2):
        """Check if two vertices are the same within a small tolerance."""
        delta_theta = v1[1] - v2[1]
        if delta_theta > np.pi:
            delta_theta -= 2 * np.pi
        elif delta_theta < -np.pi:
            delta_theta += 2 * np.pi
            
        # Increased tolerance to better handle floating-point precision issues
        # Especially for position (1.0, 4.71) vs (1.0, 4.713185307179586)
        tolerance = 0.02  # Doubled from 0.01
        return np.sqrt((v1[0] - v2[0])**2 + delta_theta**2) < tolerance

    def calculate_triangle_center(self, triangle_vertices):
        """Calculate the center of a triangle from its vertices"""
        x_sum = 0
        y_sum = 0
        
        for r, theta in triangle_vertices:
            x, y = self.polar_to_cartesian(r, theta)
            x_sum += x
            y_sum += y
            
        return x_sum / 3, y_sum / 3

    #######################
    # GAME LOGIC METHODS
    #######################
    
    def choose_game_mode(self):
        """Ask user if they want to play for high scores or low scores"""
        # In Pygame, this would be a dialog with buttons
        # For now, we'll default to True (high scores)
        self.high_score = True
        self.status_message = "Game mode: Play for high scores"

    def reset_game(self):
        """Reset the game state and files"""
        self.global_turn_number = 0
        self.round_number = 0
        self.current_round_moves = 0
        self.round_player_moves = {'red': 0, 'blue': 0}
        self.red_score = 0
        self.blue_score = 0
        
        # Reset policy network memory
        if hasattr(self, 'policy_net'):
            # Reset for batch updates
            self.policy_net.saved_log_probs = []
            self.policy_net.saved_states = []
            self.policy_net.rewards = []
            
            # Reset for TD learning
            self.policy_net.current_state = None
            self.policy_net.current_action = None
            self.policy_net.current_log_prob = None
            
        # Clear caches
        self.scores_cache = {}
        self.vertex_to_triangles = {}
        self.triangle_vertices = {}
        self.vertex_set_to_triangle = {}
        self.spatial_hash = {}  # Clear spatial hash
        
        # Reset current game statistics
        self.stats['current_game_rewards'] = []
        self.stats['current_game_losses'] = []
            
        self.reset_game_files()
        self.generate_triangle_tessellation()
        self.initialize_policy_network()
        self.build_lookup_tables()  # Initialize lookup tables for optimization
        
        # Update UI labels
        self.red_score_label.update_text("Red: 0")
        self.blue_score_label.update_text("Blue: 0")
        self.round_label.update_text("Round: 0")
        self.status_message = f"Click on vertices to place your markers. Red first probability: {self.red_first_prob}%"
        self.status_label.update_text(self.status_message)
        
        # Force full redraw
        self.needs_full_redraw = True
        self.background = None
        self.dirty_rects = []
        
    def reset_game_files(self):
        """Reset the game files (clicks.json and game_data.json)"""
        with open('clicks.json', 'w') as f:
            pass
        with open('game_data.json', 'w') as f:
            pass

    def on_canvas_click(self, pos):
        """Handle click events on the game board"""
        # Get current game state
        clicks = self.load_clicks()
        
        # Find the nearest valid vertex
        nearest_vertex = self.find_nearest_valid_space(pos)
        if nearest_vertex is None:
            return
        
        # Only accept clicks sufficiently close to that vertex
        vx, vy = self.polar_to_cartesian(nearest_vertex[0], nearest_vertex[1])
        if math.hypot(pos[0] - vx, pos[1] - vy) > self.vertex_tolerance:
            self.status_message = "Click closer to a vertex!"
            self.status_label.update_text(self.status_message)
            return
        
        # Check if this vertex has already been clicked
        for click in clicks:
            if self.is_same_vertex(click['address'], nearest_vertex):
                self.status_message = "This vertex is already taken!"
                self.status_label.update_text(self.status_message)
                return
        
        # Check if the player has already moved in this round
        if self.round_player_moves['blue'] >= 1:
            self.status_message = "You've already played in this round!"
            self.status_label.update_text(self.status_message)
            return
            
        # If it's a new round (nobody has moved yet)
        if self.current_round_moves == 0:
            # Use the slider value to determine if Red goes first
            red_goes_first = random.random() < (self.red_first_prob / 100)
            
            if red_goes_first:
                # Red (AI) goes first - player's click is ignored this time
                self.status_message = "Red goes first this round!"
                self.status_label.update_text(self.status_message)
                red_move = self.get_red_move(clicks)
                
                if red_move:
                    # Update game state
                    with open('clicks.json', 'a') as f:
                        json.dump({
                            'turn': self.global_turn_number + 1, 
                            'color': 'red', 
                            'address': red_move
                        }, f)
                        f.write('\n')
                    
                    # Track this spot as dirty (needs redraw)
                    x, y = self.polar_to_cartesian(red_move[0], red_move[1])
                    marker_rect = pygame.Rect(int(x) - 8, int(y) - 8, 16, 16)  # Match marker size
                    self.dirty_rects.append(marker_rect)
                    
                    # Update round tracking
                    self.current_round_moves += 1
                    self.round_player_moves['red'] += 1
                    self.global_turn_number += 1
                    
                    # Check if this was the last move in the game
                    updated_clicks = self.load_clicks()
                    
                    # FIX: Check if Red's move was the final move and end the game if needed
                    if len(updated_clicks) >= len(self.all_vertices):
                        self.update_game_state()
                        return
                    
                    # Player needs to click again to make their move
                    return
            
            # If Red didn't go first, continue with player's move
        
        # Process player's move (Blue)
        # Update game state
        with open('clicks.json', 'a') as f:
            json.dump({
                'turn': self.global_turn_number + 1, 
                'color': 'blue', 
                'address': nearest_vertex
            }, f)
            f.write('\n')
        
        # Track this spot as dirty (needs redraw)
        x, y = self.polar_to_cartesian(nearest_vertex[0], nearest_vertex[1])
        marker_rect = pygame.Rect(int(x) - 8, int(y) - 8, 16, 16)  # Match marker size
        self.dirty_rects.append(marker_rect)
        
        # Update round tracking
        self.current_round_moves += 1
        self.round_player_moves['blue'] += 1
        self.global_turn_number += 1
        
        # Check if Red still needs to move in this round
        if self.round_player_moves['red'] == 0:
            # It's Red's turn to complete the round
            clicks = self.load_clicks()
            red_move = self.get_red_move(clicks)
            
            if red_move:
                # Update game state
                with open('clicks.json', 'a') as f:
                    json.dump({
                        'turn': self.global_turn_number + 1, 
                        'color': 'red', 
                        'address': red_move
                    }, f)
                    f.write('\n')
                
                # Track this spot as dirty (needs redraw)
                x, y = self.polar_to_cartesian(red_move[0], red_move[1])
                marker_rect = pygame.Rect(int(x) - 8, int(y) - 8, 16, 16)  # Match marker size
                self.dirty_rects.append(marker_rect)
                
                # Update round tracking
                self.current_round_moves += 1
                self.round_player_moves['red'] += 1
                self.global_turn_number += 1
                
                # Update game state and check for game end
                updated_clicks = self.load_clicks()
                if len(updated_clicks) >= len(self.all_vertices):
                    self.update_game_state()
                    return
        
        # Check if the round is complete (both players have moved)
        # FIX: Only advance round if BOTH players have moved in the current round
        if self.current_round_moves >= 2 and self.round_player_moves['red'] >= 1 and self.round_player_moves['blue'] >= 1:
            # Reset for next round
            self.round_number += 1
            self.round_label.update_text(f"Round: {self.round_number}")
            self.current_round_moves = 0
            self.round_player_moves = {'red': 0, 'blue': 0}
            self.status_message = f"Round {self.round_number} completed. Click to start the next round."
            self.status_label.update_text(self.status_message)
        
        # Update scoring
        self.update_game_state()

    def load_clicks(self):
        """Load clicks from the clicks.json file"""
        try:
            with open('clicks.json', 'r') as f:
                lines = f.readlines()
                if not lines:
                    return []
                return [json.loads(line) for line in lines]
        except FileNotFoundError:
            return []

    def load_valid_loops(self):
        """Load valid loops from the valid_loops.json file"""
        try:
            with open('valid_loops.json', 'r') as f:
                lines = f.readlines()
                if not lines:
                    return []
                return [json.loads(line) for line in lines]
        except FileNotFoundError:
            return []

    def find_formed_loops(self, clicks, valid_loops):
        """Find all formed loops in the current game state with optimized lookup"""
        # Fast path for empty clicks
        if not clicks:
            return []
            
        # Fast path for small number of clicks (can't form triangles)
        if len(clicks) < 3:
            return []
        
        # Use cached lookups when available
        if hasattr(self, 'triangle_vertices') and self.triangle_vertices:
            # Better approach: use index-based lookups
            click_points = []
            vertex_indices = set()
            
            # Map clicks to vertex indices for faster lookup
            for click in clicks:
                for i, vertex in enumerate(self.all_vertices):
                    if self.is_same_vertex(click['address'], vertex):
                        click_points.append(i)
                        vertex_indices.add(i)
                        break
                        
            # Find triangles that have all 3 vertices in the clicked set
            formed_loops = []
            
            # Use triangle_vertices lookup table to find formed triangles
            for triangle_idx, triangle_verts in self.triangle_vertices.items():
                if all(v in vertex_indices for v in triangle_verts):
                    # Convert back to coordinates for return value
                    loop_coords = [self.all_vertices[v] for v in triangle_verts]
                    formed_loops.append(loop_coords)
            
            return formed_loops
        
        # Fallback to original algorithm
        formed_loops = []
        click_points = [tuple(click['address']) for click in clicks]
        
        for i in range(len(click_points)):
            for j in range(i + 1, len(click_points)):
                for k in range(j + 1, len(click_points)):
                    points = [click_points[i], click_points[j], click_points[k]]
                    if self.is_valid_loop(points, valid_loops):
                        formed_loops.append(points)
                        
        return formed_loops

    def is_valid_loop(self, points, valid_loops):
        """Check if a set of points forms a valid loop"""
        for loop in valid_loops:
            loop_points = set((point['r'], point['theta']) for point in loop['vertices'])
            if set(points) == loop_points:
                return True
        return False

    def vector_color(self, r, theta, clicks):
        """Get the color of a vector at the specified coordinates"""
        for click in clicks:
            click_r, click_theta = click['address']
            if self.is_same_vertex((click_r, click_theta), (r, theta)):
                return click['color']
        return None

    def get_loop_colors(self, formed_loops, clicks):
        """
        Get the colors of each loop and who closed it
        Returns a list of dictionaries with:
        - pattern: color pattern string (e.g., "redredblue")
        - closer: who closed the triangle (last person to play on that triangle)
        """
        loop_data = []
        
        # Create a cache of vertex coordinates to colors for fast lookup
        color_cache = {}
        # Map vertices to indices and clicks for fast lookup
        vertex_to_click = {}
        
        for click in clicks:
            for i, vertex in enumerate(self.all_vertices):
                if self.is_same_vertex(vertex, click['address']):
                    color_cache[i] = click['color']
                    vertex_to_click[i] = click
                    break
        
        for loop in formed_loops:
            colors = []
            vertex_indices = []
            triangle_clicks = []
            
            # Find all vertices in this loop
            for vertex in loop:
                found_vertex_idx = None
                for i, v in enumerate(self.all_vertices):
                    if self.is_same_vertex(v, vertex):
                        found_vertex_idx = i
                        vertex_indices.append(i)
                        break
                
                if found_vertex_idx is not None:
                    # If we found the vertex index, check if it's in color_cache
                    if found_vertex_idx in color_cache:
                        colors.append(color_cache[found_vertex_idx])
                        # Store the click that created this vertex
                        if found_vertex_idx in vertex_to_click:
                            triangle_clicks.append(vertex_to_click[found_vertex_idx])
            
            # Find who closed the triangle (highest turn number)
            closer = None
            highest_turn = -1
            for click in triangle_clicks:
                if click['turn'] > highest_turn:
                    highest_turn = click['turn']
                    closer = click['color']
            
            # Store both the color pattern and who closed it
            loop_data.append({
                'pattern': "".join(colors),
                'closer': closer
            })
        
        return loop_data

    def get_loop_scores(self, loop_data):
        """Get score values and colors for each loop based on color patterns and who closed it"""
        loop_scores = []
        
        for loop_info in loop_data:
            pattern = loop_info['pattern']
            closer = loop_info['closer']
            score_value = 0
            score_color = ""
            
            # Count occurrences of each color
            red_count = pattern.count('red')
            blue_count = pattern.count('blue')
            
            # Determine score based on who closed the triangle
            if closer == 'red':
                # Red closed the triangle
                if red_count == 3:  # 3 red
                    score_value = -1
                    score_color = 'red'  # Show negative score in red, but it's a penalty
                elif red_count == 2 and blue_count == 1:  # 2 red, 1 blue
                    score_value = 1
                    score_color = 'red'
                elif red_count == 1 and blue_count == 2:  # 1 red, 2 blue
                    score_value = 2
                    score_color = 'red'
            elif closer == 'blue':
                # Blue closed the triangle
                if blue_count == 3:  # 3 blue
                    score_value = -1
                    score_color = 'blue'  # Show negative score in blue, but it's a penalty
                elif blue_count == 2 and red_count == 1:  # 2 blue, 1 red
                    score_value = 1
                    score_color = 'blue'
                elif blue_count == 1 and red_count == 2:  # 1 blue, 2 red
                    score_value = 2
                    score_color = 'blue'
            
            loop_scores.append((score_value, score_color))
        
        return loop_scores

    def calculate_scores(self, loop_data):
        """Calculate scores based on loop colors and who closed each triangle"""
        red_score = 0
        blue_score = 0
        
        for loop_info in loop_data:
            pattern = loop_info['pattern']
            closer = loop_info['closer']
            
            # Count occurrences of each color
            red_count = pattern.count('red')
            blue_count = pattern.count('blue')
            
            # Scoring based on who closed the triangle
            if closer == 'red':
                # Red closed the triangle
                if red_count == 3:  # 3 red
                    red_score -= 1
                    blue_score += 1
                elif red_count == 2 and blue_count == 1:  # 2 red, 1 blue
                    red_score += 1
                elif red_count == 1 and blue_count == 2:  # 1 red, 2 blue
                    red_score += 2
            elif closer == 'blue':
                # Blue closed the triangle
                if blue_count == 3:  # 3 blue
                    blue_score -= 1
                    red_score += 1
                elif blue_count == 2 and red_count == 1:  # 2 blue, 1 red
                    blue_score += 1
                elif blue_count == 1 and red_count == 2:  # 1 blue, 2 red
                    blue_score += 2
        
        return red_score, blue_score

    def write_to_json(self, turn_number, formed_loops, loop_colors, red_score, blue_score):
        """Write game data to the JSON file"""
        # Load previous data
        previous_data = []
        try:
            with open('game_data.json', 'r') as f:
                for line in f:
                    previous_data.append(json.loads(line))
        except FileNotFoundError:
            pass

        # Create a set of all vertex sets in the previous data for fast lookup
        prev_vertex_sets = set(frozenset(tuple(vertex) for vertex in loop["Vertices"]) for loop in previous_data)

        # Write new data
        with open('game_data.json', 'a') as f:
            for i, loop in enumerate(formed_loops):
                loop_vertices = [tuple(vertex) for vertex in loop]
                color_pattern = loop_colors[i]

                # Create a frozenset of the vertices of this loop
                this_vertex_set = frozenset(loop_vertices)

                # If this loop's vertices are already in a previous loop, skip this loop
                if this_vertex_set in prev_vertex_sets:
                    continue

                # Prepare the new data
                new_data = {
                    'Turn Number': turn_number,
                    'Vertices': loop_vertices,
                    'Combo': color_pattern,
                    'Red_Points': red_score,
                    'Blue_Points': blue_score,
                }

                # Write the new data
                f.write(json.dumps(new_data))
                f.write('\n')

                # Add this loop's vertices to the set of previous vertices
                prev_vertex_sets.add(this_vertex_set)

    def update_game_state(self):
        """Update the game state and check for end game"""
        clicks = self.load_clicks()
        valid_loops = self.load_valid_loops()
        formed_loops = self.find_formed_loops(clicks, valid_loops)
        loop_colors = self.get_loop_colors(formed_loops, clicks)
        red_score, blue_score = self.calculate_scores(loop_colors)
        
        # Update score displays
        self.red_score = red_score
        self.blue_score = blue_score
        self.red_score_label.update_text(f"Red: {red_score}")
        self.blue_score_label.update_text(f"Blue: {blue_score}")
        
        # Mark any triangles with scores as dirty for redraw
        for loop in formed_loops:
            center_x, center_y = self.calculate_triangle_center(loop)
            triangle_rect = pygame.Rect(int(center_x) - 20, int(center_y) - 20, 40, 40)
            self.dirty_rects.append(triangle_rect)
        
        # Write game data to JSON
        self.write_to_json(self.global_turn_number, formed_loops, loop_colors, red_score, blue_score)
        
        # Check if game is over
        num_clicks = len(clicks)
        unique_vectors_count = len(self.all_vertices)
        
        if num_clicks >= unique_vectors_count:
            self.end_game(red_score, blue_score)

    def end_game(self, red_score, blue_score):
        """End the game and show the winner"""
        # record this game's final scores for the text dashboard
        self.stats.setdefault('red_game_scores', []).append(red_score)
        self.stats.setdefault('blue_game_scores', []).append(blue_score)

        winner = ''
        if self.high_score:
            if red_score > blue_score:
                winner = 'Red'
            elif blue_score > red_score:
                winner = 'Blue'
            else:
                winner = 'Tie'
        else:
            if red_score < blue_score:
                winner = 'Red'
            elif blue_score < red_score:
                winner = 'Blue'
            else:
                winner = 'Tie'
        
        # Final TD update if there are still experiences to learn from
        if self.use_neural_net and hasattr(self, 'last_reward'):
            # No need to check for current_state since we removed policy_net
            delattr(self, 'last_reward')
        
        # Record game statistics
        if self.stats['current_game_rewards']:
            avg_game_reward = sum(self.stats['current_game_rewards']) / len(self.stats['current_game_rewards'])
            self.stats['game_rewards'].append(avg_game_reward)
        
        if self.stats['current_game_losses']:
            avg_game_loss = sum(self.stats['current_game_losses']) / len(self.stats['current_game_losses'])
            self.stats['game_losses'].append(avg_game_loss)
            
        # Save statistics to file
        self.save_statistics()
        
        # Save log files
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        logs_dir = Path('Logs')
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Make a copy of the current game clicks for replay
        self.replay_clicks = self.load_clicks()
        
        # Save files
        shutil.copyfile('clicks.json', logs_dir / f'clicks_{timestamp}.json')
        shutil.copyfile('game_data.json', logs_dir / f'game_data_{timestamp}.json')
        
        # Save the neural network model if it was used
        if self.use_neural_net:
            try:
                model_data = {
                    'regression_state': self.regression_net.state_dict(),
                    'value_state': self.value_net.state_dict()
                }
                torch.save(model_data, logs_dir / f'model_{timestamp}.pt')
                # Also update the current model
                torch.save(model_data, self.model_file)
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # Unify logs
        self.unify_logs('Logs', 'clicks')
        self.unify_logs('Logs', 'game_data')
        
        # Show game over screen with replay option
        self.show_game_over_screen(winner, red_score, blue_score)

    def save_statistics(self):
        """Save model training statistics to a file"""
        stats_file = Path('model_statistics.json')
        try:
            with open(stats_file, 'w') as f:
                # Convert stats to serializable format
                serializable_stats = {
                    'losses': self.stats['losses'],
                    'value_losses': self.stats['value_losses'],
                    'rewards': self.stats['rewards'],
                    'round_rewards': self.stats['round_rewards'],
                    'game_rewards': self.stats['game_rewards'],
                    'round_losses': self.stats['round_losses'],
                    'game_losses': self.stats['game_losses'],
                    'td_errors': self.stats['td_errors'][:1000] if len(self.stats['td_errors']) > 1000 else self.stats['td_errors'],
                    'learning_type': 'TD learning'
                }
                json.dump(serializable_stats, f, indent=4)
        except Exception as e:
            print(f"Error saving statistics: {e}")

    def unify_logs(self, logs_folder, log_type):
        """Unify log files into a master log file"""
        # Initialize an empty list to store all data
        master_data = []

        # Get all JSON files in the logs folder that match the log_type
        log_files = Path(logs_folder).rglob(f'{log_type}_*.json')

        # Iterate over each file
        for file in log_files:
            # Open the file
            with open(file, 'r') as f:
                # Read the file line by line
                for line in f:
                    try:
                        # Load the JSON data from the line
                        item = json.loads(line)

                        # Append the filename data to the item
                        item['source_game'] = str(file)

                        # Add the item to the master list
                        master_data.append(item)
                    except json.JSONDecodeError:
                        continue

        # Save the master data to a new JSON file
        with open(f'master_{log_type}.json', 'w') as f:
            json.dump(master_data, f, indent=4)

    def apply_settings(self):
        """Apply settings from the UI controls"""
        # Get values from sliders
        self.depth = int(self.settings_ui.depth_slider.current_val)
        self.scale_multiplier = self.settings_ui.scale_slider.current_val
        
        # Keep red_first_slider as percentage (0-100)
        self.red_first_prob = int(self.settings_ui.red_first_slider.current_val)
        
        # Convert red_ai_aggression_slider from -1 to 1 scale to 0-100%
        self.red_ai_aggression = self.denormalize_slider_value(self.settings_ui.red_aggression_slider.current_val)
        
        self.max_heuristic_evals = int(self.settings_ui.heuristic_evals_slider.current_val)
        
        # Get neural network settings
        self.use_neural_net = self.settings_ui.use_neural_net_checkbox.checked
        
        # Convert neural_net_ratio_slider from -1 to 1 scale to 0-100%
        self.neural_net_ratio = self.denormalize_slider_value(self.settings_ui.neural_net_ratio_slider.current_val)
        
        self.learning_rate = self.settings_ui.learning_rate_slider.current_val
        
        # Get neural network architecture settings
        self.hidden_dim = int(self.settings_ui.hidden_dim_slider.current_val)
        self.num_layers = int(self.settings_ui.num_layers_slider.current_val)
        
        # Update optimizer with new learning rate if it exists
        if hasattr(self, 'regression_optimizer'):
            for param_group in self.regression_optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        # Reset game with new settings
        self.reset_game()

    def save_config(self, filename="config.json"):
        """Save the current configuration to a file."""
        config = {
            "depth": self.depth,
            "scale_multiplier": self.scale_multiplier,
            "red_first_prob": self.red_first_prob,
            "red_ai_aggression": self.red_ai_aggression,
            "max_heuristic_evals": self.max_heuristic_evals,
            "use_neural_net": self.use_neural_net,
            "neural_net_ratio": self.neural_net_ratio,
            "learning_rate": self.learning_rate
        }
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
            self.status_message = "Configuration saved successfully."
        except Exception as e:
            self.status_message = f"Error saving configuration: {e}"
        self.status_label.update_text(self.status_message)

    def load_config(self, filename="config.json"):
        """Load the configuration from a file."""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            self.depth = config.get("depth", self.depth)
            self.scale_multiplier = config.get("scale_multiplier", self.scale_multiplier)
            self.red_first_prob = config.get("red_first_prob", self.red_first_prob)
            self.red_ai_aggression = config.get("red_ai_aggression", self.red_ai_aggression)
            self.max_heuristic_evals = config.get("max_heuristic_evals", self.max_heuristic_evals)
            self.use_neural_net = config.get("use_neural_net", self.use_neural_net)
            self.neural_net_ratio = config.get("neural_net_ratio", self.neural_net_ratio)
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            
            # Update sliders and checkboxes with loaded values
            if hasattr(self, 'depth_slider'):
                self.depth_slider.current_val = self.depth
                self.depth_slider.update_handle_position(self.depth)
                # Update text input as well
                self.depth_slider.text_input.text = f"{self.depth:.1f}"
                
            if hasattr(self, 'scale_slider'):
                self.scale_slider.current_val = self.scale_multiplier
                self.scale_slider.update_handle_position(self.scale_multiplier)
                self.scale_slider.text_input.text = f"{self.scale_multiplier:.1f}"
                
            if hasattr(self, 'red_first_slider'):
                self.red_first_slider.current_val = self.red_first_prob
                self.red_first_slider.update_handle_position(self.red_first_prob)
                self.red_first_slider.text_input.text = f"{self.red_first_prob:.1f}"
                
            if hasattr(self, 'red_aggression_slider'):
                self.red_aggression_slider.current_val = self.red_ai_aggression
                self.red_aggression_slider.update_handle_position(self.red_ai_aggression)
                self.red_aggression_slider.text_input.text = f"{self.red_ai_aggression:.1f}"
                
            if hasattr(self, 'heuristic_evals_slider'):
                self.heuristic_evals_slider.current_val = self.max_heuristic_evals
                self.heuristic_evals_slider.update_handle_position(self.max_heuristic_evals)
                self.heuristic_evals_slider.text_input.text = f"{self.max_heuristic_evals:.1f}"
                
            if hasattr(self, 'use_neural_net_checkbox'):
                self.use_neural_net_checkbox.checked = self.use_neural_net
                
            if hasattr(self, 'neural_net_ratio_slider'):
                self.neural_net_ratio_slider.current_val = self.neural_net_ratio
                self.neural_net_ratio_slider.update_handle_position(self.neural_net_ratio)
                self.neural_net_ratio_slider.text_input.text = f"{self.neural_net_ratio:.1f}"
                
            if hasattr(self, 'learning_rate_slider'):
                self.learning_rate_slider.current_val = self.learning_rate
                self.learning_rate_slider.update_handle_position(self.learning_rate)
                self.learning_rate_slider.text_input.text = f"{self.learning_rate:.2e}"
                
            # If we're in settings UI mode, update the SettingsPanel sliders too
            if hasattr(self, 'settings_ui'):
                self.settings_ui.depth_slider.current_val = self.depth
                self.settings_ui.depth_slider.update_handle_position(self.depth)
                self.settings_ui.depth_slider.text_input.text = f"{self.depth:.1f}"
                
                self.settings_ui.scale_slider.current_val = self.scale_multiplier
                self.settings_ui.scale_slider.update_handle_position(self.scale_multiplier)
                self.settings_ui.scale_slider.text_input.text = f"{self.scale_multiplier:.1f}"
                
                self.settings_ui.red_first_slider.current_val = self.red_first_prob
                self.settings_ui.red_first_slider.update_handle_position(self.red_first_prob)
                self.settings_ui.red_first_slider.text_input.text = f"{self.red_first_prob:.1f}"
                
                self.settings_ui.red_aggression_slider.current_val = self.red_ai_aggression
                self.settings_ui.red_aggression_slider.update_handle_position(self.red_ai_aggression)
                self.settings_ui.red_aggression_slider.text_input.text = f"{self.red_ai_aggression:.1f}"
                
                self.settings_ui.heuristic_evals_slider.current_val = self.max_heuristic_evals
                self.settings_ui.heuristic_evals_slider.update_handle_position(self.max_heuristic_evals)
                self.settings_ui.heuristic_evals_slider.text_input.text = f"{self.max_heuristic_evals:.1f}"
                
                self.settings_ui.use_neural_net_checkbox.checked = self.use_neural_net
                
                self.settings_ui.neural_net_ratio_slider.current_val = self.neural_net_ratio
                self.settings_ui.neural_net_ratio_slider.update_handle_position(self.neural_net_ratio)
                self.settings_ui.neural_net_ratio_slider.text_input.text = f"{self.neural_net_ratio:.1f}"
                
                self.settings_ui.learning_rate_slider.current_val = self.learning_rate
                self.settings_ui.learning_rate_slider.update_handle_position(self.learning_rate)
                self.settings_ui.learning_rate_slider.text_input.text = f"{self.learning_rate:.2e}"
            
            self.status_message = "Configuration loaded successfully."
        except Exception as e:
            self.status_message = f"Error loading configuration: {e}"
        self.status_label.update_text(self.status_message)

    def batch_update_regression(self):
        """Perform a batch update using stored examples for regression learning"""
        try:
            if not self.regression_net.input_states or not self.regression_net.target_values:
                return
                
            # Convert stored examples to tensors
            states = torch.stack(self.regression_net.input_states)
            targets = torch.stack(self.regression_net.target_values).to(self.device)
            
            # Predict values for all states
            predictions = self.regression_net(states)
            
            # Reshape tensors to ensure consistent dimensions
            predictions = predictions.view(-1, 1)
            targets = targets.view(-1, 1)
            
            # Compute MSE loss with correctly shaped tensors
            loss = F.mse_loss(predictions, targets)
            
            # Zero gradients
            self.regression_optimizer.zero_grad()
            
            # Backpropagate
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.regression_net.parameters(), 1.0)
            
            # Update weights
            self.regression_optimizer.step()
            
            # Record statistics
            self.stats['value_losses'].append(loss.item())
            
            # Clear stored examples after update
            self.regression_net.clear_training_data()
            
        except Exception as e:
            print(f"Error in batch regression update: {e}")

    def show_game_over_screen(self, winner, red_score, blue_score):
        """Show the game over screen with replay option"""
        # Variable to track if we're in replay mode
        self.replaying_game = False
        self.replay_index = 0
        self.replay_timer = 0
        
        # Use GameOverScreen class to show the game over screen
        self.game_over_ui.show(self.screen, winner, red_score, blue_score)
        
        # Wait for user to click a button
        waiting = True
        while waiting:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    waiting = False
                    self.running = False
            
            # Check for user interaction with game over screen
            result = self.game_over_ui.update(events)
            if result == 'replay':
                waiting = False
                self.start_replay()
            elif result == 'continue':
                waiting = False
                self.reset_game()
            
            # Keep the frame rate consistent
            self.clock.tick(60)

    def start_replay(self):
        """Start replaying the game"""
        # Set replay mode
        self.replaying_game = True
        self.replay_index = 0
        self.replay_timer = 0
        
        # Reset the game state but keep the replay clicks
        self.reset_game_files()
        self.red_score = 0
        self.blue_score = 0
        self.red_score_label.update_text("Red: 0")
        self.blue_score_label.update_text("Blue: 0")
        
        # Status message
        self.status_message = "Replaying game..."
        self.status_label.update_text(self.status_message)
        
        # Setup animation loop
        waiting = True
        while waiting and self.replaying_game and self.running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    waiting = False
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                        self.replaying_game = False
                        self.reset_game()
            
            # Update the replay
            self.update_replay()
            
            # Render the game
            self.render()
            
            # Keep the frame rate consistent
            self.clock.tick(30)  # Slower for replay
        
        # After replay finishes, reset for a new game
        self.reset_game()
    
    def update_replay(self):
        """Update the replay animation"""
        # Increase the timer
        self.replay_timer += 1
        
        # Every 30 frames (about 1 second at 30 FPS), add the next click
        if self.replay_timer >= 30 and self.replay_index < len(self.replay_clicks):
            # Add the next click to the current game state
            click = self.replay_clicks[self.replay_index]
            with open('clicks.json', 'a') as f:
                json.dump(click, f)
                f.write('\n')
            
            # Update the game state
            self.update_game_state()
            
            # Move to the next click
            self.replay_index += 1
            self.replay_timer = 0
            
            # Update status message
            self.status_message = f"Replaying move {self.replay_index}/{len(self.replay_clicks)}"
            self.status_label.update_text(self.status_message)
        
        # Check if replay is complete
        if self.replay_index >= len(self.replay_clicks):
            self.replaying_game = False
            self.status_message = "Replay finished. Starting new game..."
            self.status_label.update_text(self.status_message)
            pygame.time.wait(2000)  # Wait 2 seconds before resetting

    def run_training_session(self):
        """Run a training session with custom Red and Blue settings from the training config panel"""
        # Store original settings
        original_use_neural_net = self.use_neural_net
        original_neural_net_ratio = self.neural_net_ratio
        original_red_first_prob = self.red_first_prob
        original_status = self.status_message
        original_red_ai_aggression = self.red_ai_aggression
        original_blue_ai_aggression = getattr(self, 'blue_ai_aggression', self.red_ai_aggression)
        
        # Set training specific settings using the custom training configuration
        self.use_neural_net = True
        
        # Use settings from training config if they exist, otherwise use defaults
        red_episodes = getattr(self, 'red_episodes', 100)
        red_ai_heur_ratio = getattr(self, 'red_ai_heur_ratio', 50)
        blue_ai_heur_ratio = getattr(self, 'blue_ai_heur_ratio', 50)
        red_greedy_random_ratio = getattr(self, 'red_greedy_random_ratio', 50) 
        blue_greedy_random_ratio = getattr(self, 'blue_greedy_random_ratio', 50)
        red_locked = getattr(self, 'red_locked', False)
        blue_locked = getattr(self, 'blue_locked', False)
        red_end_ai_ratio = getattr(self, 'red_end_ai_ratio', 50)
        blue_end_ai_ratio = getattr(self, 'blue_end_ai_ratio', 50)
        
        # Convert AI/Heuristic ratio to neural_net_ratio for Red
        self.neural_net_ratio = red_ai_heur_ratio
        
        # Set Red's aggression based on greedy/random ratio
        self.red_ai_aggression = red_greedy_random_ratio
        
        # Set Blue's aggression based on greedy/random ratio
        self.blue_ai_aggression = blue_greedy_random_ratio
        
        # Equal chance for first move (you could make this configurable too)
        self.red_first_prob = 50
        
        # Clear previous training stats
        self.stats['training_value_losses'] = []
        self.stats['training_red_wins'] = 0
        self.stats['training_blue_wins'] = 0
        self.stats['training_ties'] = 0
        
        # Log training settings
        print(f"Starting training with settings:")
        print(f"Red: AI/Heur={red_ai_heur_ratio}%, Greedy/Random={red_greedy_random_ratio}%, " +
              f"Locked={red_locked}, End AI={red_end_ai_ratio}%, Episodes={red_episodes}")
        print(f"Blue: AI/Heur={blue_ai_heur_ratio}%, Greedy/Random={blue_greedy_random_ratio}%, " +
              f"Locked={blue_locked}, End AI={blue_end_ai_ratio}%, Episodes={getattr(self, 'blue_episodes', 100)}")
        
        # OPTIMIZATION: Load valid_loops once and keep in memory for the entire training session
        # This avoids repeatedly loading the same data from disk
        valid_loops = self.load_valid_loops()
        
        # Setup training progress display
        progress_panel = Panel(
            self.screen_width // 2 - 200, 
            self.screen_height // 2 - 150, 
            400, 300
        )
        progress_title = Label(
            progress_panel.rect.centerx, 
            progress_panel.rect.y + 30,
            "AI Training Session", 
            (0, 0, 0), 
            32, 
            "center"
        )
        progress_label = Label(
            progress_panel.rect.centerx, 
            progress_panel.rect.centery,
            f"Training in progress: Game 1/{red_episodes}", 
            (0, 0, 0), 
            24, 
            "center"
        )
        avg_loss_label = Label(
            progress_panel.rect.centerx, 
            progress_panel.rect.centery + 40,
            "Average Loss: 0.000000", 
            (0, 0, 0), 
            20, 
            "center"
        )
        stats_label = Label(
            progress_panel.rect.centerx, 
            progress_panel.rect.centery + 70,
            "Red wins: 0  Blue wins: 0  Ties: 0", 
            (0, 0, 0), 
            20, 
            "center"
        )
        # Federated learning indicator
        fed_label = Label(
            progress_panel.rect.centerx,
            progress_panel.rect.centery + 100,
            "Federated averaging: ON",
            (0, 0, 0),
            20,
            "center"
        )
        cancel_button = Button(
            progress_panel.rect.centerx - 50, 
            progress_panel.rect.bottom - 50,
            100, 40, 
            "Cancel", 
            self.colors['button'], 
            self.colors['button_hover']
        )

        # Create semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # 50% transparent black
        
        # Flag to track if training was cancelled
        training_cancelled = False
        
        # Get number of episodes from config (default to 100 if not set)
        num_episodes = red_episodes
        
        # Run training games
        for game_num in range(1, num_episodes + 1):
            if training_cancelled:
                break
                
            # Update progress display
            progress_label.update_text(f"Training in progress: Game {game_num}/{num_episodes}")
            
            if len(self.stats['value_losses']) > 0:
                avg_loss = sum(self.stats['value_losses'][-100:]) / min(len(self.stats['value_losses']), 100)
                avg_loss_label.update_text(f"Average Loss: {avg_loss:.6f}")
                
            stats_label.update_text(
                f"Red wins: {self.stats['training_red_wins']}  " +
                f"Blue wins: {self.stats['training_blue_wins']}  " +
                f"Ties: {self.stats['training_ties']}"
            )
            
            # Draw progress display
            self.screen.fill(self.colors['background'])
            self.screen.blit(overlay, (0, 0))
            progress_panel.draw(self.screen)
            progress_title.draw(self.screen)
            progress_label.draw(self.screen)
            avg_loss_label.draw(self.screen)
            stats_label.draw(self.screen)
            fed_label.draw(self.screen)
            cancel_button.draw(self.screen)
            pygame.display.flip()
            
            # Check for cancel button click or window close
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    training_cancelled = True
                    self.running = False
                    break
                    
            if cancel_button.update(events):
                training_cancelled = True
                break
            
            # Reset the game state for this episode but don't reload files
            self.reset_game_files()
            self.red_score = 0
            self.blue_score = 0
            
            # Completely AI-controlled game
            clicks = []
            
            # Play the game until all vertices are filled
            current_player = 'red' if random.random() < (self.red_first_prob / 100) else 'blue'
            turn_number = 0
            
            # For staged training (if enabled via locked mode)
            if red_locked and game_num > num_episodes / 2:
                # Second half: increase neural net influence for Red
                self.neural_net_ratio = red_end_ai_ratio
            
            if blue_locked and game_num > num_episodes / 2:
                # Second half: increase aggression for Blue
                self.blue_ai_aggression = blue_end_ai_ratio
                
            while len(clicks) < len(self.all_vertices):
                # Get AI move for current player
                if current_player == 'red':
                    # OPTIMIZATION: Pass clicks and valid_loops directly instead of loading from disk
                    move = self.get_training_red_move(clicks, valid_loops)
                else:
                    # OPTIMIZATION: Pass clicks and valid_loops directly instead of loading from disk
                    move = self.get_training_blue_move(clicks, valid_loops, self.blue_ai_aggression)
                
                if move:
                    # Add the move
                    turn_number += 1
                    clicks.append({
                        'turn': turn_number,
                        'color': current_player,
                        'address': move
                    })
                    
                    # Switch players
                    current_player = 'blue' if current_player == 'red' else 'red'
                    
                # Process events during gameplay to keep UI responsive
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        training_cancelled = True
                        self.running = False
                        break
                        
                if cancel_button.update(events):
                    training_cancelled = True
                    break
                    
                if training_cancelled:
                    break
                    
                # Draw progress display occasionally
                if turn_number % 5 == 0:
                    self.screen.fill(self.colors['background'])
                    self.screen.blit(overlay, (0, 0))
                    progress_panel.draw(self.screen)
                    progress_title.draw(self.screen)
                    progress_label.draw(self.screen)
                    avg_loss_label.draw(self.screen)
                    stats_label.draw(self.screen)
                    fed_label.draw(self.screen)
                    cancel_button.draw(self.screen)
                    pygame.display.flip()
                
            if training_cancelled:
                break
                
            # Calculate the final score
            # OPTIMIZATION: use the cached valid_loops instead of loading from disk
            formed_loops = self.find_formed_loops(clicks, valid_loops)
            loop_colors = self.get_loop_colors(formed_loops, clicks)
            red_score, blue_score = self.calculate_scores(loop_colors)
            
            # Update win counts
            if red_score > blue_score:
                self.stats['training_red_wins'] += 1
            elif blue_score > red_score:
                self.stats['training_blue_wins'] += 1
            else:
                self.stats['training_ties'] += 1
                
            # Save model after every 10 games
            if game_num % 10 == 0 and not training_cancelled:
                try:
                    model_data = {
                        'regression_state': self.regression_net.state_dict(),
                        'value_state': self.value_net.state_dict()
                    }
                    
                    # Save versioned model with episode count and timestamp
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    models_dir = Path('models')
                    models_dir.mkdir(parents=True, exist_ok=True)
                    model_path = models_dir / f"vector_game_model_{len(self.all_vertices)}_ep{game_num}_{timestamp}.pt"
                    torch.save(model_data, model_path)
                    
                    # Also update the current model file
                    torch.save(model_data, self.model_file)
                except Exception as e:
                    print(f"Error saving model: {e}")
            
            # Federated averaging: merge Red and Blue perspectives
            for rp,bp in zip(self.regression_net.parameters(), self.blue_regression_net.parameters()):
                avg=(rp.data+bp.data)*0.5
                rp.data.copy_(avg); bp.data.copy_(avg)
            
            # Process events to keep UI responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    training_cancelled = True
                    self.running = False
                    break
        
        # Final model save
        if not training_cancelled:
            try:
                model_data = {
                    'regression_state': self.regression_net.state_dict(),
                    'value_state': self.value_net.state_dict()
                }
                torch.save(model_data, self.model_file)
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # Reset original settings
        self.use_neural_net = original_use_neural_net
        self.neural_net_ratio = original_neural_net_ratio
        self.red_first_prob = original_red_first_prob
        self.red_ai_aggression = original_red_ai_aggression
        self.blue_ai_aggression = original_blue_ai_aggression
        
        # Reset the game for normal play
        self.reset_game()
        
        # Reset ML metrics
        self.draw_ml_metrics()
        
        # Show training results
        self.status_message = (
            f"Training {'completed' if not training_cancelled else 'cancelled'} - " +
            f"Red wins: {self.stats['training_red_wins']}, " +
            f"Blue wins: {self.stats['training_blue_wins']}, " +
            f"Ties: {self.stats['training_ties']}"
        )
        self.status_label.update_text(self.status_message)

    def td_update(self, state_idx, state_occ, reward, next_state_idx=None, next_state_occ=None, is_terminal=False):
        """Stream-optimized truly incremental TD update with adaptive learning rate"""
        # Initialize adaptive learning rate parameters if not present
        if not hasattr(self, 'lr_min'):
            self.lr_min = 0.0001  # Minimum learning rate
            self.lr_max = 0.01    # Maximum learning rate
            self.td_error_window = deque(maxlen=100)  # Track recent TD errors
            self.lr_decay = 0.99   # Learning rate decay factor
            self.lr_growth = 1.001  # Learning rate growth factor
            self.learning_rates = []  # For monitoring
        
        # Get current value estimate
        value = self.regression_net(state_idx, state_occ)
        
        # Compute raw TD target
        if is_terminal or next_state_idx is None:
            raw_target = float(reward)
        else:
            with torch.no_grad():
                next_value = self.regression_net(next_state_idx, next_state_occ).item()
            raw_target = reward + self.gamma * next_value
        
        # Calculate TD error (loss before optimization)
        td_error = raw_target - value.item()
        self.stats['td_errors'].append(td_error)
        self.td_error_window.append(abs(td_error))
        
        # Adaptive learning rate based on TD error magnitude
        if len(self.td_error_window) > 1:
            mean_error = sum(self.td_error_window) / len(self.td_error_window)
            if abs(td_error) > 1.5 * mean_error:
                # Surprising outcome - increase learning rate to adapt faster
                self.learning_rate = min(self.lr_max, self.learning_rate * self.lr_growth)
            else:
                # Expected outcome - decrease learning rate for fine-tuning
                self.learning_rate = max(self.lr_min, self.learning_rate * self.lr_decay)
            
            # Update optimizer with new learning rate
            for param_group in self.regression_optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        # Save learning rate for monitoring
        self.learning_rates.append(self.learning_rate)
        
        # Streaming normalization of TD targets
        self.target_buffer.append(raw_target)
        import numpy as _np
        # Use running statistics for normalization (streaming z-score)
        mean = _np.mean(self.target_buffer)
        std = _np.std(self.target_buffer) 
        std = max(std, 1e-6)  # Avoid divide by zero
        
        # Normalize target (helps with gradient scale)
        target_norm = (raw_target - mean) / std
        
        # Truly incremental single-step optimization with importance sampling
        loss = self.regression_net.train_step(state_idx, state_occ, target_norm, self.regression_optimizer)
        
        # Record stats for monitoring
        if loss is not None:
            self.stats['value_losses'].append(loss)
        
        return loss

    def td_update_blue(self, state_idx, state_occ, reward, next_state_idx=None, next_state_occ=None, is_terminal=False):
        """Perform a TD(0) update for the Blue value (regression) network."""
        # Get current value estimate
        value = self.blue_regression_net(state_idx, state_occ)
        
        # Compute raw TD target
        if is_terminal or next_state_idx is None:
            raw_target = float(reward)
        else:
            with torch.no_grad():
                next_value = self.blue_regression_net(next_state_idx, next_state_occ).item()
            raw_target = reward + self.gamma * next_value
        
        # Update running buffer and compute normalization stats
        self.target_buffer.append(raw_target)
        import numpy as _np
        mean = _np.mean(self.target_buffer)
        std = _np.std(self.target_buffer)
        if std < 1e-6:
            std = 1.0
        
        # Normalize target
        target_norm = (raw_target - mean) / std
        
        # Use batch training capability
        loss = self.blue_regression_net.train_step(state_idx, state_occ, target_norm, self.blue_regression_optimizer)
        
        # Record actual TD error (unnormalized) for monitoring
        td_error = raw_target - value.item()
        self.stats['td_errors'].append(td_error)
        
        return loss

    def extract_model_parameters(self, model_path):
        """Extract parameters from a saved model file"""
        try:
            # Load the model file
            model_data = torch.load(model_path, map_location=self.device)
            
            # Extract neural network parameters
            params = {}
            
            # Try to load in different formats
            if isinstance(model_data, dict):
                # Check if the model has policy and value states
                if 'policy_state' in model_data and 'value_state' in model_data:
                    policy_state = model_data['policy_state']
                    value_state = model_data['value_state']
                    
                    # Extract hidden dimension from the first layer weight matrix
                    for key, value in value_state.items():
                        if 'fc' in key and 'weight' in key:
                            # Store the shape of weight tensors to help determine hidden_dim
                            if key == 'fc1.weight':  # First layer in older models
                                params['hidden_dim'] = value.shape[0]
                            elif 'fc_layers.0.weight' in key:  # First layer in mid-era models
                                params['hidden_dim'] = value.shape[0]
                    
                    # For newer models with embedding layers
                    for key, value in value_state.items():
                        if 'embed.weight' in key:
                            params['hidden_dim'] = value.shape[1]  # Embedding dimension
                        
                    # Count the number of layers more accurately
                    layer_counts = []
                    # Look for patterns in both value and policy networks
                    for state_dict in [value_state, policy_state]:
                        layer_ids = []
                        # Look for layers in fc_layers
                        layer_ids.extend([int(k.split('.')[1]) for k in state_dict.keys() 
                                         if 'fc_layers' in k and 'weight' in k])
                        # Look for sequential layers
                        layer_ids.extend([int(k.split('.')[0]) for k in state_dict.keys() 
                                         if k[0].isdigit() and '.weight' in k])
                        # Look for layers
                        layer_ids.extend([int(k.split('.')[1]) for k in state_dict.keys() 
                                         if 'layers' in k and 'weight' in k and k.split('.')[1].isdigit()])
                        
                        if layer_ids:
                            layer_counts.append(max(layer_ids) + 1)
                    
                    if layer_counts:
                        params['num_layers'] = max(layer_counts)
                    else:
                        # Try counting unique layer prefixes
                        layer_prefixes = set()
                        for key in value_state.keys():
                            if 'weight' in key:
                                prefix = key.split('.weight')[0]
                                if 'fc' in prefix or prefix.isdigit() or 'layers' in prefix:
                                    layer_prefixes.add(prefix)
                        
                        if layer_prefixes:
                            params['num_layers'] = len(layer_prefixes)
                        else:
                            # Default if we can't determine
                            params['num_layers'] = 2
                    
                    # For newer models, check the policy network for hints
                    if not params.get('hidden_dim'):
                        for key, value in policy_state.items():
                            if key == 'fc1.weight':
                                params['hidden_dim'] = value.shape[1]  # Input dimension
            
            # Default values if we couldn't extract parameters
            if 'hidden_dim' not in params:
                params['hidden_dim'] = 128  # Default
            
            if 'num_layers' not in params:
                params['num_layers'] = 2  # Default
            
            # Add learning rate (use current value)
            params['learning_rate'] = self.learning_rate
            
            # Include current game depth
            params['depth'] = self.depth
            
            self.status_message = f"Model parameters extracted: hidden_dim={params['hidden_dim']}, layers={params['num_layers']}"
            self.status_label.update_text(self.status_message)
            
            return params
            
        except Exception as e:
            self.status_message = f"Error extracting model parameters: {e}"
            self.status_label.update_text(self.status_message)
            return None

    def load_external_model(self, model_path):
        """Load a model from external file with potentially different parameters"""
        try:
            # Reset networks with updated parameters
            num_v = len(self.all_vertices)
            
            # Create new networks with updated parameters
            self.regression_net = RegressionNetwork(
                num_vertices=num_v,
                embed_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            ).to(self.device)
            
            # Value network is our regression network
            self.value_net = self.regression_net
            
            # Update optimizers with new learning rate
            self.regression_optimizer = optim.Adam(self.regression_net.parameters(), lr=self.learning_rate)
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
            self.value_optimizer = self.regression_optimizer
            
            # Load the model weights
            model_data = torch.load(model_path, map_location=self.device)
            
            # Try to load the model weights safely with error checking
            if isinstance(model_data, dict):
                if 'policy_state' in model_data and 'value_state' in model_data:
                    # Try to load value/regression network weights
                    try:
                        # This may fail if the architectures don't match exactly
                        self.value_net.load_state_dict(model_data['value_state'], strict=False)
                        
                        # Also try to load policy network weights for compatibility
                        self.policy_net.load_state_dict(model_data['policy_state'], strict=False)
                        
                        # Display a more informative message about the loaded model
                        model_name = Path(model_path).name
                        hidden_info = f"hidden_dim={self.hidden_dim}"
                        layers_info = f"layers={self.num_layers}"
                        self.status_message = f"Model {model_name} loaded: {hidden_info}, {layers_info}"
                        
                        # Enable neural network usage
                        self.use_neural_net = True
                        self.use_neural_net_checkbox.checked = True
                    except Exception as e:
                        self.status_message = f"Error loading model weights: {e}. Created new networks with extracted parameters."
                else:
                    self.status_message = "Unknown model format. Created new networks with extracted parameters."
            
            self.status_label.update_text(self.status_message)
            
            # Force a redraw to show updated ML metrics
            self.needs_full_redraw = True
            return True
            
        except Exception as e:
            self.status_message = f"Error loading model: {e}"
            self.status_label.update_text(self.status_message)
            return False

    # Normalization helper methods for slider values
    @property
    def red_ai_aggression_normalized(self):
        """Convert 0-100 scale to -1 to 1 scale for red_ai_aggression slider"""
        return (self.red_ai_aggression / 50) - 1
        
    @property
    def neural_net_ratio_normalized(self):
        """Convert 0-100 scale to -1 to 1 scale for neural_net_ratio slider"""
        return (self.neural_net_ratio / 50) - 1
    
    def denormalize_slider_value(self, normalized_value):
        """Convert -1 to 1 scale back to 0-100 scale"""
        return int((normalized_value + 1) * 50)
