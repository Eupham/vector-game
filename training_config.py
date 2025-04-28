import pygame
from ui_elements import Slider, Checkbox, Label, Panel, Scrollbar, Button

class TrainingConfigPanel:
    """Class for displaying and managing the training configuration panel"""
    
    def __init__(self, game_instance, screen_width, screen_height):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Training config panel
        panel_width = 650
        panel_height = min(600, self.screen_height - 80)
        self.panel = Panel(self.screen_width // 2 - panel_width // 2, self.screen_height // 2 - panel_height // 2, panel_width, panel_height)
        self.panel_border_color = (80, 80, 80)  # Dark gray border
        self.panel_border_width = 3
        
        # Title
        self.title = Label(self.panel.rect.centerx, self.panel.rect.y + 20, 
                           "Training Configuration", self.game.colors['text'], 32, "center")
        
        # Calculate content area dimensions
        self.content_area = {
            'x': self.panel.rect.x + 20,
            'y': self.panel.rect.y + 60,
            'width': self.panel.rect.width - 40,
            'height': self.panel.rect.height - 120  # Leave room for title and bottom buttons
        }
        
        # Calculate total content height (to determine if scrollbar is needed)
        self.row_height = 70  # Increased height for each row
        self.content_height = 520  # Base height estimate
        
        # Create scrollbar
        scrollbar_x = self.panel.rect.right - 25
        scrollbar_y = self.content_area['y']
        scrollbar_height = self.content_area['height']
        
        self.scrollbar = Scrollbar(
            scrollbar_x, scrollbar_y, 15, scrollbar_height, 
            self.content_height, scrollbar_height,
            bar_color=(200, 200, 200), handle_color=(150, 150, 150)
        )
        
        # Track current scroll position
        self.scroll_y_offset = 0
        
        # Define column positions and spacing with additional padding
        col1_x = self.content_area['x'] + 20  # Red column
        col2_x = self.content_area['x'] + self.content_area['width'] // 2 + 10  # Blue column
        start_y = self.content_area['y'] + 30
        spacing = self.row_height  # Increased spacing between rows
        
        # Column headers
        self.red_column_label = Label(col1_x + 80, start_y, 
                                "Red", self.game.colors['red'], 24, "center")
        self.blue_column_label = Label(col2_x + 80, start_y, 
                                "Blue", self.game.colors['blue'], 24, "center")
        
        # Add MC consensus slider - centered across both columns
        self.consensus_samples_slider = Slider(
            self.content_area['x'] + self.content_area['width'] // 2 - 125,
            start_y + spacing,
            250, 20, 1, 21,
            getattr(self.game, 'consensus_samples', 1),
            text="MC Consensus k"
        )
        
        # Add Recursion Depth slider - positioned below consensus slider
        # This controls the game board size
        self.recursion_depth_slider = Slider(
            self.content_area['x'] + self.content_area['width'] // 2 - 125,
            start_y + spacing * 2,
            250, 20, 1, 6,
            getattr(self.game, 'depth', 3),
            text="Board Size (Recursion Depth)"
        )
        
        # Board size info label
        board_sizes = {
            1: "7 vertices", 
            2: "16 vertices", 
            3: "46 vertices", 
            4: "121 vertices", 
            5: "316 vertices", 
            6: "817 vertices"
        }
        current_depth = getattr(self.game, 'depth', 3)
        self.board_size_label = Label(
            self.content_area['x'] + self.content_area['width'] // 2,
            start_y + spacing * 2 + 30,  # More padding below the slider
            f"Current board: {board_sizes.get(current_depth, f'{current_depth} (custom)')}",
            self.game.colors['text'], 16, "center"
        )
        
        # Add Starting Learning Rate slider
        self.start_learning_rate_slider = Slider(
            self.content_area['x'] + self.content_area['width'] // 2 - 125,
            start_y + spacing * 3,
            250, 20, 0.0001, 0.01,
            getattr(self.game, 'learning_rate', 0.001),
            text="Start Learning Rate"
        )
        
        # Red side controls - adding extra padding to row_y calculations
        row_y = start_y + spacing * 4  # Start Red/Blue sections after the global settings
        
        self.red_ai_heur_ratio_slider = Slider(col1_x, row_y, 200, 20, 0, 100,
                                          getattr(self.game, 'red_ai_heur_ratio', 50),
                                          text="AI/Heur Ratio (%)")
        
        self.red_greedy_random_ratio_slider = Slider(col1_x, row_y + spacing, 200, 20, 0, 100,
                                               getattr(self.game, 'red_greedy_random_ratio', 50),
                                               text="Quiescence/Greedy (%)")
        
        self.red_fixed_locked_checkbox = Checkbox(col1_x, row_y + spacing * 2, 20,
                                            "Fixed vs Locked", 
                                            getattr(self.game, 'red_locked', False))
        
        self.red_end_ai_ratio_slider = Slider(col1_x, row_y + spacing * 3, 200, 20, 0, 100,
                                         getattr(self.game, 'red_end_ai_ratio', 50),
                                         text="End AI Ratio (%)")
        
        self.red_episodes_slider = Slider(col1_x, row_y + spacing * 4, 200, 20, 1, 10000,
                                     getattr(self.game, 'red_episodes', 1000),
                                     text="Episodes")
        
        # Blue side controls
        self.blue_ai_heur_ratio_slider = Slider(col2_x, row_y, 200, 20, 0, 100,
                                           getattr(self.game, 'blue_ai_heur_ratio', 50),
                                           text="AI/Heur Ratio (%)")
        
        self.blue_greedy_random_ratio_slider = Slider(col2_x, row_y + spacing, 200, 20, 0, 100,
                                                getattr(self.game, 'blue_greedy_random_ratio', 50),
                                                text="Quiescence/Greedy (%)")
        
        self.blue_fixed_locked_checkbox = Checkbox(col2_x, row_y + spacing * 2, 20,
                                             "Fixed vs Locked", 
                                             getattr(self.game, 'blue_locked', False))
        
        self.blue_end_ai_ratio_slider = Slider(col2_x, row_y + spacing * 3, 200, 20, 0, 100,
                                          getattr(self.game, 'blue_end_ai_ratio', 50),
                                          text="End AI Ratio (%)")
        
        self.blue_episodes_slider = Slider(col2_x, row_y + spacing * 4, 200, 20, 1, 10000,
                                      getattr(self.game, 'blue_episodes', 1000),
                                      text="Episodes")
                                      
        # Calculate total content height based on the last element's position
        self.content_height = max(self.content_height, row_y + spacing * 5 + 50)
        
        # Update scrollbar content height
        self.scrollbar.content_height = self.content_height
        # Fix: Don't pass a parameter to update_handle_position
        self.scrollbar.update_handle_position()
        
        # Buttons
        button_y = self.panel.rect.bottom - 60
        
        # Apply button
        self.apply_button = Button(
            self.panel.rect.right - 120,
            button_y, 
            100, 40,
            "Apply", self.game.colors['button'], self.game.colors['button_hover']
        )
        
        # Close button
        self.close_button = Button(
            self.panel.rect.right - 230,
            button_y, 
            100, 40,
            "Cancel", self.game.colors['button'], self.game.colors['button_hover']
        )
        
        # Start Training button
        self.start_training_button = Button(
            self.panel.rect.x + 50,
            button_y, 
            150, 40,
            "Start Training", self.game.colors['button'], self.game.colors['button_hover']
        )

        # Status messages
        self.status_text = ""
        self.status_color = (0, 0, 0)  # Default color is black
    
    def draw(self, screen):
        """Render the training config panel with scrolling support"""
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
        
        # Create a clipping mask for the content area
        content_rect = pygame.Rect(
            self.content_area['x'],
            self.content_area['y'],
            self.content_area['width'],
            self.content_area['height']
        )
        
        # Save original clipping rect
        original_clip = screen.get_clip()
        
        # Set clipping rect to content area
        screen.set_clip(content_rect)
        
        # Offset for scrolling
        scroll_offset = int(self.scroll_y_offset)
        
        # Draw column headers - fixed position at top
        self.red_column_label.y = self.red_column_label.original_y - scroll_offset
        self.blue_column_label.y = self.blue_column_label.original_y - scroll_offset
        
        if self.red_column_label.y >= self.content_area['y']:
            self.red_column_label.draw(screen)
        if self.blue_column_label.y >= self.content_area['y']:
            self.blue_column_label.draw(screen)
        
        # Draw sliders with scroll offset applied - use proper method to update both slider and handle positions
        for slider in [
            self.consensus_samples_slider,
            self.recursion_depth_slider,
            self.start_learning_rate_slider,
            self.red_ai_heur_ratio_slider,
            self.red_greedy_random_ratio_slider,
            self.red_end_ai_ratio_slider,
            self.red_episodes_slider,
            self.blue_ai_heur_ratio_slider,
            self.blue_greedy_random_ratio_slider,
            self.blue_end_ai_ratio_slider,
            self.blue_episodes_slider
        ]:
            # Use the set_y_position method which updates both slider and handle
            if hasattr(slider, 'original_y'):
                slider.set_y_position(slider.original_y - scroll_offset)
                
                # Only draw if within view
                if (slider.rect.bottom >= self.content_area['y'] and 
                    slider.rect.y <= self.content_area['y'] + self.content_area['height']):
                    slider.draw(screen)
        
        # Draw checkboxes with scroll offset
        for checkbox in [self.red_fixed_locked_checkbox, self.blue_fixed_locked_checkbox]:
            checkbox.rect.y = checkbox.original_y - scroll_offset
            if (checkbox.rect.bottom >= self.content_area['y'] and 
                checkbox.rect.y <= self.content_area['y'] + self.content_area['height']):
                checkbox.draw(screen)
        
        # Draw board size label
        self.board_size_label.y = self.board_size_label.original_y - scroll_offset
        if (self.board_size_label.y + 15 >= self.content_area['y'] and 
            self.board_size_label.y <= self.content_area['y'] + self.content_area['height']):
            self.board_size_label.draw(screen)
        
        # Reset clipping rect
        screen.set_clip(original_clip)
        
        # Draw the scrollbar (outside of clipped area)
        self.scrollbar.draw(screen)
        
        # Draw buttons (fixed at bottom)
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
        # Update scrollbar and get scroll position
        self.scrollbar.update(events)
        self.scroll_y_offset = self.scrollbar.scroll_position
        
        # Store original y positions if not already stored
        if not hasattr(self.red_column_label, 'original_y'):
            for item in [
                self.red_column_label, self.blue_column_label,
                self.consensus_samples_slider, self.recursion_depth_slider,
                self.start_learning_rate_slider,
                self.red_ai_heur_ratio_slider, self.red_greedy_random_ratio_slider,
                self.red_fixed_locked_checkbox, self.red_end_ai_ratio_slider,
                self.red_episodes_slider, self.blue_ai_heur_ratio_slider,
                self.blue_greedy_random_ratio_slider, self.blue_fixed_locked_checkbox,
                self.blue_end_ai_ratio_slider, self.blue_episodes_slider,
                self.board_size_label
            ]:
                item.original_y = item.y if hasattr(item, 'y') else item.rect.y
        
        # Check if buttons were clicked
        if self.apply_button.update(events):
            self.apply_settings()
            return 'apply'
        
        if self.close_button.update(events):
            return 'close'
        
        if self.start_training_button.update(events):
            self.apply_settings()
            return 'start_training'
        
        # Only update UI elements if they're in view
        view_rect = pygame.Rect(
            self.content_area['x'],
            self.content_area['y'],
            self.content_area['width'],
            self.content_area['height']
        )
        
        # Update sliders with proper handle positioning
        for slider in [
            self.consensus_samples_slider,
            self.recursion_depth_slider,
            self.start_learning_rate_slider,
            self.red_ai_heur_ratio_slider,
            self.red_greedy_random_ratio_slider,
            self.red_end_ai_ratio_slider,
            self.red_episodes_slider,
            self.blue_ai_heur_ratio_slider,
            self.blue_greedy_random_ratio_slider,
            self.blue_end_ai_ratio_slider,
            self.blue_episodes_slider
        ]:
            # Apply scroll offset for hit detection
            new_y = slider.original_y - int(self.scroll_y_offset)
            
            # Use set_y_position to update both slider and handle
            slider.set_y_position(new_y)
            
            # Make sure text input is also repositioned
            if hasattr(slider, 'text_input'):
                slider.text_input.rect.y = new_y
                
            # Only update if in view
            if view_rect.colliderect(slider.rect):
                slider.update(events)
        
        # Apply scrolling to checkboxes and update them
        self.red_fixed_locked_checkbox.rect.y = self.red_fixed_locked_checkbox.original_y - int(self.scroll_y_offset)
        self.blue_fixed_locked_checkbox.rect.y = self.blue_fixed_locked_checkbox.original_y - int(self.scroll_y_offset)
        
        if view_rect.colliderect(self.red_fixed_locked_checkbox.rect):
            change_red, self.red_fixed_locked_checkbox.checked = self.red_fixed_locked_checkbox.update(events)
            
        if view_rect.colliderect(self.blue_fixed_locked_checkbox.rect):
            change_blue, self.blue_fixed_locked_checkbox.checked = self.blue_fixed_locked_checkbox.update(events)
            
        # Update board size label based on recursion depth
        current_depth = int(self.recursion_depth_slider.current_val)
        board_sizes = {
            1: "7 vertices", 
            2: "16 vertices", 
            3: "46 vertices", 
            4: "121 vertices", 
            5: "316 vertices", 
            6: "817 vertices"
        }
        self.board_size_label.text = f"Current board: {board_sizes.get(current_depth, f'{current_depth} (custom)')}"
        
        return None
    
    def apply_settings(self):
        """Apply the current settings to the game instance"""
        # MC consensus samples
        self.game.consensus_samples = int(self.consensus_samples_slider.current_val)
        
        # Recursion Depth
        self.game.depth = int(self.recursion_depth_slider.current_val)
        
        # Starting Learning Rate
        self.game.learning_rate = self.start_learning_rate_slider.current_val
        
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
        print(f"MC Consensus k={game.consensus_samples}")
        print(f"Recursion Depth (Board Size)={game.depth}")
        print(f"Starting Learning Rate={game.learning_rate}")
        print(f"Red: AI/Heur={game.red_ai_heur_ratio}%, Quiescence/Greedy={game.red_greedy_random_ratio}%, " +
              f"Locked={game.red_locked}, End AI={game.red_end_ai_ratio}%, Episodes={game.red_episodes}")
        print(f"Blue: AI/Heur={game.blue_ai_heur_ratio}%, Quiescence/Greedy={game.blue_greedy_random_ratio}%, " +
              f"Locked={game.blue_locked}, End AI={game.blue_end_ai_ratio}%, Episodes={game.blue_episodes}")
        
        # Call the game's actual training method
        game.run_training_session()
        
        return True, "Training completed successfully"
    except Exception as e:
        return False, f"Error during training: {str(e)}"