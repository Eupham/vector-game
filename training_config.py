import pygame
from ui_elements import Slider, Checkbox, Label, Panel

class TrainingConfigPanel:
    """Class for displaying and managing the training configuration panel"""
    
    def __init__(self, game_instance, screen_width, screen_height):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Training config panel
        self.panel = Panel(self.screen_width // 2 - 300, self.screen_height // 2 - 280, 600, 560)
        self.panel_border_color = (80, 80, 80)  # Dark gray border
        self.panel_border_width = 3
        
        # Title
        self.title = Label(self.panel.rect.centerx, self.panel.rect.y + 20, 
                           "Training Configuration", self.game.colors['text'], 32, "center")
        
        # Define column positions and spacing
        col1_x = self.panel.rect.x + 50  # Red column
        col2_x = self.panel.rect.x + 330  # Blue column
        start_y = self.panel.rect.y + 80
        spacing = 50
        
        # Column headers
        self.red_column_label = Label(col1_x + 100, start_y, 
                                "Red", self.game.colors['red'], 24, "center")
        self.blue_column_label = Label(col2_x + 100, start_y, 
                                "Blue", self.game.colors['blue'], 24, "center")
        
        # Red side controls
        self.red_ai_heur_ratio_slider = Slider(col1_x, start_y + spacing, 200, 20, 0, 100,
                                          getattr(self.game, 'red_ai_heur_ratio', 50),
                                          text="AI/Heur Ratio (%)")
        
        self.red_greedy_random_ratio_slider = Slider(col1_x, start_y + spacing * 2, 200, 20, 0, 100,
                                               getattr(self.game, 'red_greedy_random_ratio', 50),
                                               text="Greedy/Random (%)")
        
        self.red_fixed_locked_checkbox = Checkbox(col1_x, start_y + spacing * 3, 20,
                                            "Fixed vs Locked", 
                                            getattr(self.game, 'red_locked', False))
        
        self.red_end_ai_ratio_slider = Slider(col1_x, start_y + spacing * 4, 200, 20, 0, 100,
                                         getattr(self.game, 'red_end_ai_ratio', 50),
                                         text="End AI Ratio (%)")
        
        self.red_episodes_slider = Slider(col1_x, start_y + spacing * 5, 200, 20, 1, 10000,
                                     getattr(self.game, 'red_episodes', 1000),
                                     text="Episodes")
        
        # Blue side controls
        self.blue_ai_heur_ratio_slider = Slider(col2_x, start_y + spacing, 200, 20, 0, 100,
                                           getattr(self.game, 'blue_ai_heur_ratio', 50),
                                           text="AI/Heur Ratio (%)")
        
        self.blue_greedy_random_ratio_slider = Slider(col2_x, start_y + spacing * 2, 200, 20, 0, 100,
                                                getattr(self.game, 'blue_greedy_random_ratio', 50),
                                                text="Greedy/Random (%)")
        
        self.blue_fixed_locked_checkbox = Checkbox(col2_x, start_y + spacing * 3, 20,
                                             "Fixed vs Locked", 
                                             getattr(self.game, 'blue_locked', False))
        
        self.blue_end_ai_ratio_slider = Slider(col2_x, start_y + spacing * 4, 200, 20, 0, 100,
                                          getattr(self.game, 'blue_end_ai_ratio', 50),
                                          text="End AI Ratio (%)")
        
        self.blue_episodes_slider = Slider(col2_x, start_y + spacing * 5, 200, 20, 1, 10000,
                                      getattr(self.game, 'blue_episodes', 1000),
                                      text="Episodes")
        
        # Buttons
        self.apply_button = pygame.Rect(self.panel.rect.right - 120, self.panel.rect.bottom - 60, 100, 40)
        self.cancel_button = pygame.Rect(self.panel.rect.right - 230, self.panel.rect.bottom - 60, 100, 40)
        
        # Apply and Close buttons
        from ui_elements import Button
        self.apply_button = Button(self.panel.rect.right - 120, self.panel.rect.bottom - 60, 
                              100, 40, "Apply", self.game.colors['button'], self.game.colors['button_hover'])
        
        self.close_button = Button(self.panel.rect.right - 230, self.panel.rect.bottom - 60, 
                              100, 40, "Cancel", self.game.colors['button'], self.game.colors['button_hover'])
        
        # Start Training button
        self.start_training_button = Button(self.panel.rect.x + 50, 
                                       self.panel.rect.bottom - 60, 
                                       150, 40, "Start Training", self.game.colors['button'], 
                                       self.game.colors['button_hover'])

        # Status messages
        self.status_text = ""
        self.status_color = (0, 0, 0)  # Default color is black
    
    def draw(self, screen):
        """Render the training config panel"""
        # Draw panel with semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
        
        # Draw the panel
        self.panel.draw(screen)
        
        # Draw the panel border
        pygame.draw.rect(screen, self.panel_border_color, 
                         self.panel.rect, self.panel_border_width)
        
        # Draw the title
        self.title.draw(screen)
        
        # Draw column headers
        self.red_column_label.draw(screen)
        self.blue_column_label.draw(screen)
        
        # Draw Red side controls
        self.red_ai_heur_ratio_slider.draw(screen)
        self.red_greedy_random_ratio_slider.draw(screen)
        self.red_fixed_locked_checkbox.draw(screen)
        self.red_end_ai_ratio_slider.draw(screen)
        self.red_episodes_slider.draw(screen)
        
        # Draw Blue side controls
        self.blue_ai_heur_ratio_slider.draw(screen)
        self.blue_greedy_random_ratio_slider.draw(screen)
        self.blue_fixed_locked_checkbox.draw(screen)
        self.blue_end_ai_ratio_slider.draw(screen)
        self.blue_episodes_slider.draw(screen)
        
        # Draw buttons
        self.apply_button.draw(screen)
        self.close_button.draw(screen)
        self.start_training_button.draw(screen)
        
        # Draw status message if there is one
        if self.status_text:
            font = pygame.font.SysFont(None, 24)
            status_surf = font.render(self.status_text, True, self.status_color)
            status_rect = status_surf.get_rect(center=(self.panel.rect.centerx, self.panel.rect.bottom - 100))
            screen.blit(status_surf, status_rect)
    
    def update(self, events):
        """Update the training config panel and handle events"""
        # Check if buttons were clicked
        if self.apply_button.update(events):
            self.apply_settings()
            return 'apply'
        
        if self.close_button.update(events):
            return 'close'
        
        if self.start_training_button.update(events):
            self.apply_settings()
            return 'start_training'
        
        # Update Red side controls
        self.red_ai_heur_ratio_slider.update(events)
        self.red_greedy_random_ratio_slider.update(events)
        change_red, self.red_fixed_locked_checkbox.checked = self.red_fixed_locked_checkbox.update(events)
        self.red_end_ai_ratio_slider.update(events)
        self.red_episodes_slider.update(events)
        
        # Update Blue side controls
        self.blue_ai_heur_ratio_slider.update(events)
        self.blue_greedy_random_ratio_slider.update(events)
        change_blue, self.blue_fixed_locked_checkbox.checked = self.blue_fixed_locked_checkbox.update(events)
        self.blue_end_ai_ratio_slider.update(events)
        self.blue_episodes_slider.update(events)
        
        return None
    
    def apply_settings(self):
        """Apply the current settings to the game instance"""
        # Red settings
        self.game.red_ai_heur_ratio = self.red_ai_heur_ratio_slider.current_val
        self.game.red_greedy_random_ratio = self.red_greedy_random_ratio_slider.current_val
        self.game.red_locked = self.red_fixed_locked_checkbox.checked
        self.game.red_end_ai_ratio = self.red_end_ai_ratio_slider.current_val
        self.game.red_episodes = int(self.red_episodes_slider.current_val)
        
        # Blue settings
        self.game.blue_ai_heur_ratio = self.blue_ai_heur_ratio_slider.current_val
        self.game.blue_greedy_random_ratio = self.blue_greedy_random_ratio_slider.current_val
        self.game.blue_locked = self.blue_fixed_locked_checkbox.checked
        self.game.blue_end_ai_ratio = self.blue_end_ai_ratio_slider.current_val
        self.game.blue_episodes = int(self.blue_episodes_slider.current_val)
        
        self.status_text = "Settings applied successfully!"
        self.status_color = (0, 128, 0)  # Green color for success
    
    def set_status(self, message, color=(0, 0, 0)):
        """Set a status message to display in the panel"""
        self.status_text = message
        self.status_color = color

# Helper function to start the training process
def start_training(game):
    """Start the training process with the current configuration"""
    try:
        print(f"Starting training with settings:")
        print(f"Red: AI/Heur={game.red_ai_heur_ratio}%, Greedy/Random={game.red_greedy_random_ratio}%, " +
              f"Locked={game.red_locked}, End AI={game.red_end_ai_ratio}%, Episodes={game.red_episodes}")
        print(f"Blue: AI/Heur={game.blue_ai_heur_ratio}%, Greedy/Random={game.blue_greedy_random_ratio}%, " +
              f"Locked={game.blue_locked}, End AI={game.blue_end_ai_ratio}%, Episodes={game.blue_episodes}")
        
        # Call the game's actual training method
        game.run_training_session()
        
        return True, "Training completed successfully"
    except Exception as e:
        return False, f"Error during training: {str(e)}"