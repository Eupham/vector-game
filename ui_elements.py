import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

class Button:
    def __init__(self, x, y, width, height, text, color=(200, 200, 200), 
                 hover_color=(150, 150, 150), text_color=(0, 0, 0), font_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.text_color = text_color
        self.font_size = font_size
        self.font = pygame.font.SysFont(None, font_size)
        self.hovered = False
        self.clicked = False
        
    def draw(self, surface):
        # Draw button rectangle
        if self.hovered:
            pygame.draw.rect(surface, self.hover_color, self.rect)
        else:
            pygame.draw.rect(surface, self.color, self.rect)
        
        # Draw button border
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        # Draw text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos)
        
        self.clicked = False
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.hovered:
                    self.clicked = True
        
        return self.clicked

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, 
                 color=(200, 200, 200), handle_color=(100, 100, 100), 
                 text="", text_color=(0, 0, 0), font_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_size = (height + 10, height + 10)
        self.handle_rect = pygame.Rect(0, 0, self.handle_size[0], self.handle_size[1])
        
        # Set these values BEFORE calling update_handle_position
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = initial_val
        
        # Now that min_val and max_val are defined, we can call this
        self.update_handle_position(initial_val)
        
        self.color = color
        self.handle_color = handle_color
        self.text = text
        self.text_color = text_color
        self.font_size = font_size
        self.font = pygame.font.SysFont(None, font_size)
        self.dragging = False
        
        # Create an associated text input box for direct value editing
        if "Learning Rate" in self.text:
            # For learning rate, use scientific notation format
            text_value = f"{self.current_val:.2e}"
        elif self.min_val == -1 and self.max_val == 1:
            # For -1 to 1 sliders, use decimal format without percentage
            text_value = f"{self.current_val:.1f}"
        else:
            # For other values, use decimal format
            text_value = f"{self.current_val:.1f}"
        
        # Position the text input to the right of the slider with more space
        # Reduce slider width to make space for text input
        adjusted_width = 80  # Width of text input box
        self.text_input = TextInput(self.rect.right - adjusted_width - 20, self.rect.y, adjusted_width, 20, 
                                   text_value, font_size=self.font_size)
    
    def update_handle_position(self, value):
        """Update the handle position based on the current value"""
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        x_pos = self.rect.x + (ratio * self.rect.width) - (self.handle_size[0] // 2)
        self.handle_rect.x = max(self.rect.x - (self.handle_size[0] // 4), 
                                min(x_pos, self.rect.right - (self.handle_size[0] * 3 // 4)))
        self.handle_rect.y = self.rect.y + (self.rect.height // 2) - (self.handle_size[1] // 2)
        
    # Add a setter for the y position that updates both the slider and handle
    def set_y_position(self, y):
        """Update the y position of the slider and its handle"""
        y_diff = y - self.rect.y
        self.rect.y = y
        self.handle_rect.y += y_diff
        if hasattr(self, 'text_input'):
            self.text_input.rect.y = y

    def get_value(self):
        """Get the current value based on handle position"""
        pos_ratio = (self.handle_rect.centerx - self.rect.x) / self.rect.width
        return self.min_val + pos_ratio * (self.max_val - self.min_val)
    
    def draw(self, surface):
        # Draw slider bar
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2, border_radius=3)
        
        # Draw handle
        pygame.draw.rect(surface, self.handle_color, self.handle_rect, border_radius=5)
        pygame.draw.rect(surface, (0, 0, 0), self.handle_rect, 2, border_radius=5)
        
        # Draw label text with appropriate formatting
        if self.text:
            if "Learning Rate" in self.text:
                # Use scientific notation for learning rate
                text_surf = self.font.render(f"{self.text}: {self.current_val:.2e}", True, self.text_color)
            elif self.min_val == -1 and self.max_val == 1:
                # For -1 to 1 sliders, display decimal without percentage
                text_surf = self.font.render(f"{self.text}: {self.current_val:.1f}", True, self.text_color)
            elif "Probability" in self.text or "Ratio" in self.text:
                # For percentage sliders, use percentage format
                text_surf = self.font.render(f"{self.text}: {self.current_val:.1f}", True, self.text_color)
            else:
                # Use decimal format for other sliders
                text_surf = self.font.render(f"{self.text}: {self.current_val:.1f}", True, self.text_color)
            text_rect = text_surf.get_rect(midleft=(self.rect.left, self.rect.top - 15))
            surface.blit(text_surf, text_rect)
        
        # Draw the text input box for direct value editing - draw AFTER the slider so it appears on top
        self.text_input.draw(surface)
    
    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        value_changed = False
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.handle_rect.collidepoint(mouse_pos):
                    self.dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
        
        if self.dragging:
            x_pos = max(self.rect.x, min(mouse_pos[0], self.rect.right))
            ratio = (x_pos - self.rect.x) / self.rect.width
            new_val = self.min_val + ratio * (self.max_val - self.min_val)
            if new_val != self.current_val:
                self.current_val = new_val
                self.update_handle_position(self.current_val)
                value_changed = True
                
                # Update the text input box to reflect the new value
                if "Learning Rate" in self.text:
                    self.text_input.text = f"{self.current_val:.2e}"
                elif self.min_val == -1 and self.max_val == 1:
                    # For -1 to 1 sliders, format without percentage
                    self.text_input.text = f"{self.current_val:.1f}"
                else:
                    self.text_input.text = f"{self.current_val:.1f}"
        
        # Update the text input box and check if value was changed
        text_input_result = self.text_input.update(events)
        
        # If text input has been submitted (Enter pressed), try to update the slider value
        if text_input_result and self.text_input.text:
            try:
                new_val = float(self.text_input.text)
                # Clamp the value to min/max range
                new_val = max(self.min_val, min(new_val, self.max_val))
                if new_val != self.current_val:
                    self.current_val = new_val
                    self.update_handle_position(self.current_val)
                    value_changed = True
            except ValueError:
                # Reset text input to current value if invalid input
                if "Learning Rate" in self.text:
                    self.text_input.text = f"{self.current_val:.2e}"
                elif self.min_val == -1 and self.max_val == 1:
                    self.text_input.text = f"{self.current_val:.1f}"
                else:
                    self.text_input.text = f"{self.current_val:.1f}"
        
        return value_changed or self.current_val

class Checkbox:
    def __init__(self, x, y, size, text="", checked=False, 
                 color=(200, 200, 200), check_color=(0, 0, 0), 
                 text_color=(0, 0, 0), font_size=20):
        self.rect = pygame.Rect(x, y, size, size)
        self.text = text
        self.checked = checked
        self.color = color
        self.check_color = check_color
        self.text_color = text_color
        self.font_size = font_size
        self.font = pygame.font.SysFont(None, font_size)
        self.clicked = False
        
    def draw(self, surface):
        # Draw checkbox background
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        # Draw check mark if checked
        if self.checked:
            inset = self.rect.width // 4
            pygame.draw.line(surface, self.check_color, 
                            (self.rect.left + inset, self.rect.centery), 
                            (self.rect.centerx, self.rect.bottom - inset), 
                            3)
            pygame.draw.line(surface, self.check_color, 
                            (self.rect.centerx, self.rect.bottom - inset), 
                            (self.rect.right - inset, self.rect.top + inset), 
                            3)
        
        # Draw label text
        if self.text:
            text_surf = self.font.render(self.text, True, self.text_color)
            text_rect = text_surf.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
            surface.blit(text_surf, text_rect)
    
    def update(self, events):
        changed = False
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.rect.collidepoint(event.pos):
                    self.checked = not self.checked
                    changed = True
        
        return changed, self.checked

class Label:
    def __init__(self, x, y, text, color=(0, 0, 0), font_size=20, align="left"):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.font_size = font_size
        self.font = pygame.font.SysFont(None, font_size)
        self.align = align
        self.max_width = None  # Default no wrapping
        self.line_spacing = 1.2  # Line spacing multiplier
        
    def draw(self, surface):
        if not self.max_width:  # No wrapping needed
            text_surf = self.font.render(self.text, True, self.color)
            
            if self.align == "left":
                text_rect = text_surf.get_rect(topleft=(self.x, self.y))
            elif self.align == "center":
                text_rect = text_surf.get_rect(midtop=(self.x, self.y))
            elif self.align == "right":
                text_rect = text_surf.get_rect(topright=(self.x, self.y))
                
            surface.blit(text_surf, text_rect)
        else:  # Text wrapping needed
            self.draw_wrapped_text(surface)
    
    def draw_wrapped_text(self, surface):
        words = self.text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Test if adding this word exceeds the width
            test_line = current_line + [word]
            test_text = ' '.join(test_line)
            test_width = self.font.size(test_text)[0]
            
            if test_width <= self.max_width:
                # Word fits, add it to current line
                current_line = test_line
            else:
                # Word doesn't fit, start a new line
                if current_line:  # If the line has words, add it to lines
                    lines.append(' '.join(current_line))
                current_line = [word]  # Start new line with this word
        
        # Add the last line if it exists
        if current_line:
            lines.append(' '.join(current_line))
        
        # Render each line
        line_height = self.font.get_linesize() * self.line_spacing
        for i, line in enumerate(lines):
            text_surf = self.font.render(line, True, self.color)
            
            if self.align == "left":
                text_rect = text_surf.get_rect(topleft=(self.x, self.y + i * line_height))
            elif self.align == "center":
                text_rect = text_surf.get_rect(midtop=(self.x, self.y + i * line_height))
            elif self.align == "right":
                text_rect = text_surf.get_rect(topright=(self.x, self.y + i * line_height))
                
            surface.blit(text_surf, text_rect)
    
    def update_text(self, new_text):
        self.text = new_text

    def set_max_width(self, width):
        """Set the maximum width for text wrapping"""
        self.max_width = width

class Panel:
    def __init__(self, x, y, width, height, color=(220, 220, 220), border_color=(0, 0, 0)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.border_color = border_color
    
    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, self.border_color, self.rect, 2)

class Scrollbar:
    def __init__(self, x, y, width, height, content_height, visible_height, 
                 bar_color=(180, 180, 180), handle_color=(100, 100, 100)):
        self.rect = pygame.Rect(x, y, width, height)
        self.content_height = content_height
        self.visible_height = visible_height
        self.bar_color = bar_color
        self.handle_color = handle_color
        
        # Calculate handle size and position
        self.handle_height = max(30, int(height * (visible_height / content_height)))
        self.handle_rect = pygame.Rect(x, y, width, self.handle_height)
        
        self.scroll_position = 0  # Scroll position in pixels
        self.dragging = False
        self.drag_offset = 0
        
    def draw(self, surface):
        # Draw scrollbar background
        pygame.draw.rect(surface, self.bar_color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 1)
        
        # Draw handle
        pygame.draw.rect(surface, self.handle_color, self.handle_rect)
        pygame.draw.rect(surface, (0, 0, 0), self.handle_rect, 1)
    
    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.handle_rect.collidepoint(mouse_pos):
                    self.dragging = True
                    self.drag_offset = mouse_pos[1] - self.handle_rect.y
                elif event.button == 1 and self.rect.collidepoint(mouse_pos):
                    # Click on scrollbar outside handle - jump to that position
                    self.handle_rect.y = min(max(mouse_pos[1] - self.handle_height // 2, 
                                               self.rect.y), 
                                           self.rect.bottom - self.handle_height)
                    # Update scroll position
                    scroll_ratio = (self.handle_rect.y - self.rect.y) / (self.rect.height - self.handle_height)
                    self.scroll_position = int(scroll_ratio * (self.content_height - self.visible_height))
                    
                # Handle mousewheel scrolling
                elif event.button == 4:  # Mouse wheel up
                    self.scroll_position = max(0, self.scroll_position - 20)
                    self.update_handle_position()
                elif event.button == 5:  # Mouse wheel down
                    self.scroll_position = min(self.content_height - self.visible_height, 
                                              self.scroll_position + 20)
                    self.update_handle_position()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            
            # Handle mousewheel events separately in case they're sent as MOUSEWHEEL instead of MOUSEBUTTONDOWN
            elif event.type == pygame.MOUSEWHEEL:
                self.scroll_position = max(0, min(self.content_height - self.visible_height,
                                                self.scroll_position - event.y * 20))
                self.update_handle_position()
        
        if self.dragging:
            # Update handle position based on mouse
            self.handle_rect.y = min(max(mouse_pos[1] - self.drag_offset, self.rect.y), 
                                    self.rect.bottom - self.handle_height)
            
            # Update scroll position based on handle
            scroll_ratio = (self.handle_rect.y - self.rect.y) / (self.rect.height - self.handle_height)
            self.scroll_position = int(scroll_ratio * (self.content_height - self.visible_height))
        
        return self.scroll_position
    
    def update_handle_position(self):
        """Update handle position based on scroll position"""
        if self.content_height <= self.visible_height:
            # All content is visible, place handle at top
            self.handle_rect.y = self.rect.y
        else:
            # Calculate ratio and position
            scroll_ratio = self.scroll_position / (self.content_height - self.visible_height)
            self.handle_rect.y = self.rect.y + int(scroll_ratio * (self.rect.height - self.handle_height))
    
    def update_content_height(self, new_content_height):
        """Update the content height and recalculate handle size"""
        self.content_height = new_content_height
        
        # Recalculate handle height
        self.handle_height = max(30, int(self.rect.height * (self.visible_height / self.content_height)))
        self.handle_rect.height = self.handle_height
        
        # Ensure scroll position is valid with new content height
        self.scroll_position = min(self.scroll_position, self.content_height - self.visible_height)
        
        # Update handle position
        self.update_handle_position()

class TextInput:
    def __init__(self, x, y, width, height, initial_text="", 
                 color=(255, 255, 255), text_color=(0, 0, 0), 
                 active_color=(220, 220, 255), font_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = initial_text
        self.color = color
        self.text_color = text_color
        self.active_color = active_color
        self.font_size = font_size
        self.font = pygame.font.SysFont(None, font_size)
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_blink_speed = 500  # milliseconds
        
    def draw(self, surface):
        # Determine background color based on active state
        bg_color = self.active_color if self.active else self.color
        
        # Draw text input background
        pygame.draw.rect(surface, bg_color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        # Draw text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 5, self.rect.centery))
        
        # Make sure text stays within the input box
        if text_rect.width > self.rect.width - 10:
            # If text is too long, render only the end portion that fits
            visible_text = self.text[-(self.rect.width // 8):]
            text_surf = self.font.render(visible_text, True, self.text_color)
            text_rect = text_surf.get_rect(midleft=(self.rect.x + 5, self.rect.centery))
        
        surface.blit(text_surf, text_rect)
        
        # Draw cursor when active and visible
        if self.active and self.cursor_visible:
            cursor_x = text_rect.right + 2 if text_rect.right < self.rect.right - 5 else self.rect.right - 5
            pygame.draw.line(
                surface, 
                self.text_color,
                (cursor_x, self.rect.y + 4),
                (cursor_x, self.rect.bottom - 4),
                2
            )
        
        # Blink cursor
        current_time = pygame.time.get_ticks()
        if current_time - self.cursor_timer > self.cursor_blink_speed:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = current_time
    
    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        text_changed = False
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Toggle active state based on click
                if event.button == 1:
                    self.active = self.rect.collidepoint(mouse_pos)
            
            elif event.type == pygame.KEYDOWN and self.active:
                if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    self.active = False
                    text_changed = True
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                    text_changed = True
                elif event.key == pygame.K_ESCAPE:
                    self.active = False
                else:
                    # Allow only numbers, decimal point, and 'e' for scientific notation
                    valid_chars = "0123456789.e-+"
                    if event.unicode in valid_chars:
                        self.text += event.unicode
                        text_changed = True
        
        return text_changed

class SettingsPanel:
    """Class for displaying and managing the settings panel"""
    
    def __init__(self, game_instance, screen_width, screen_height):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Settings panel - make it taller to fit additional controls
        self.settings_panel = Panel(self.screen_width // 2 - 300, self.screen_height // 2 - 280, 600, 560)
        self.settings_panel_border_color = (80, 80, 80)  # Dark gray border
        self.settings_panel_border_width = 3
        
        # Settings title
        self.settings_title = Label(self.settings_panel.rect.centerx, self.settings_panel.rect.y + 20, 
                                "Game Settings", self.game.colors['text'], 32, "center")
        
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
                                500, 20, 1, 6, self.game.depth, text="Depth")
        
        self.scale_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 130, 
                                500, 20, 0.5, 2.0, self.game.scale_multiplier, text="Visual Scale")
        
        self.red_first_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 180, 
                                    500, 20, 0, 100, self.game.red_first_prob, text="Red First Probability (%)")
        
        # Use -1 to 1 scale for red_aggression_slider
        self.red_aggression_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 230, 
                                        500, 20, -1, 1, self.game.red_ai_aggression_normalized, text="Strategy (Defensive ← → Aggressive)")
        
        self.heuristic_evals_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 280, 
                                        500, 20, 5, 50, self.game.max_heuristic_evals, text="Max Moves to Evaluate")
        
        # Neural network settings
        self.use_neural_net_checkbox = Checkbox(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 330, 
                                            20, "Use Neural Network", self.game.use_neural_net)
        
        # Use -1 to 1 scale for neural_net_ratio_slider
        self.neural_net_ratio_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 370, 
                                            500, 20, -1, 1, self.game.neural_net_ratio_normalized, text="Decision Style (Heuristic ← → Neural)")
        
        self.learning_rate_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 420, 
                                        500, 20, 0.0001, 0.01, self.game.learning_rate, text="Learning Rate")
                                        
        # Neural network architecture settings
        self.hidden_dim_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 470, 
                                    500, 20, 32, 1024, self.game.hidden_dim, text="Hidden Dimension")
                                    
        self.num_layers_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 520, 
                                    500, 20, 1, 8, self.game.num_layers, text="Number of Layers")
        
        # Apply and Close buttons
        self.apply_settings_button = Button(self.settings_panel.rect.right - 120, self.settings_panel.rect.bottom - 60, 
                                        100, 40, "Apply", self.game.colors['button'], self.game.colors['button_hover'])
        
        self.close_settings_button = Button(self.settings_panel.rect.right - 230, self.settings_panel.rect.bottom - 60, 
                                        100, 40, "Cancel", self.game.colors['button'], self.game.colors['button_hover'])

        # Add Save and Load Config buttons
        self.save_config_button = Button(self.settings_panel.rect.x + 50, 
                                         self.settings_panel.rect.bottom - 60, 
                                         100, 40, "Save Config", self.game.colors['button'], 
                                         self.game.colors['button_hover'])
        self.load_config_button = Button(self.settings_panel.rect.x + 160, 
                                         self.settings_panel.rect.bottom - 60, 
                                         100, 40, "Load Config", self.game.colors['button'], 
                                         self.game.colors['button_hover'])
        
        # Add Load Model button
        self.load_model_button = Button(self.settings_panel.rect.x + 270, 
                                         self.settings_panel.rect.bottom - 60, 
                                         100, 40, "Load Model", self.game.colors['button'], 
                                         self.game.colors['button_hover'])
    
    def draw(self, screen):
        """Render the settings panel"""
        # Draw panel with semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
        
        # Draw the panel
        self.settings_panel.draw(screen)
        
        # Draw the panel border
        pygame.draw.rect(screen, self.settings_panel_border_color, 
                         self.settings_panel.rect, self.settings_panel_border_width)
        
        # Draw the title (not affected by scrolling)
        self.settings_title.draw(screen)
        
        # Get the current scroll offset
        scroll_offset = self.settings_scrollbar.scroll_position
        
        # Create clipping rectangle for the scrollable area
        scroll_area = pygame.Rect(
            self.settings_panel.rect.x + 10, 
            self.settings_panel.rect.y + 60,
            self.settings_panel.rect.width - 40, 
            self.settings_panel.rect.height - 120  # Space for title and buttons at the bottom
        )
        
        # Set clipping to prevent drawing outside the scrollable area
        screen.set_clip(scroll_area)
        
        # Define the base positions for each UI element (without scroll offset)
        base_positions = {
            'depth_slider': self.settings_panel.rect.y + 80,
            'scale_slider': self.settings_panel.rect.y + 130,
            'red_first_slider': self.settings_panel.rect.y + 180,
            'red_aggression_slider': self.settings_panel.rect.y + 230,
            'heuristic_evals_slider': self.settings_panel.rect.y + 280,
            'use_neural_net_checkbox': self.settings_panel.rect.y + 330,
            'neural_net_ratio_slider': self.settings_panel.rect.y + 370,
            'learning_rate_slider': self.settings_panel.rect.y + 420,
            'hidden_dim_slider': self.settings_panel.rect.y + 470,
            'num_layers_slider': self.settings_panel.rect.y + 520
        }
        
        # Update UI element positions based on scroll offset using the set_y_position method
        self.depth_slider.set_y_position(base_positions['depth_slider'] - scroll_offset)
        self.scale_slider.set_y_position(base_positions['scale_slider'] - scroll_offset)
        self.red_first_slider.set_y_position(base_positions['red_first_slider'] - scroll_offset)
        self.red_aggression_slider.set_y_position(base_positions['red_aggression_slider'] - scroll_offset)
        self.heuristic_evals_slider.set_y_position(base_positions['heuristic_evals_slider'] - scroll_offset)
        self.use_neural_net_checkbox.rect.y = base_positions['use_neural_net_checkbox'] - scroll_offset
        self.neural_net_ratio_slider.set_y_position(base_positions['neural_net_ratio_slider'] - scroll_offset)
        self.learning_rate_slider.set_y_position(base_positions['learning_rate_slider'] - scroll_offset)
        self.hidden_dim_slider.set_y_position(base_positions['hidden_dim_slider'] - scroll_offset)
        self.num_layers_slider.set_y_position(base_positions['num_layers_slider'] - scroll_offset)
        
        # Draw UI elements if they're within the visible area
        ui_elements = [
            self.depth_slider,
            self.scale_slider,
            self.red_first_slider,
            self.red_aggression_slider,
            self.heuristic_evals_slider,
            self.use_neural_net_checkbox,
            self.neural_net_ratio_slider,
            self.learning_rate_slider,
            self.hidden_dim_slider,
            self.num_layers_slider
        ]
        
        for element in ui_elements:
            position_with_scroll = element.rect.y
            if scroll_area.y <= position_with_scroll <= scroll_area.bottom:
                element.draw(screen)
        
        # Reset clipping
        screen.set_clip(None)
        
        # Draw the scrollbar (not affected by scrolling)
        self.settings_scrollbar.draw(screen)
        
        # Draw buttons (not affected by scrolling)
        self.apply_settings_button.draw(screen)
        self.close_settings_button.draw(screen)
        self.save_config_button.draw(screen)
        self.load_config_button.draw(screen)
        self.load_model_button.draw(screen)
    
    def update(self, events):
        """Update the settings panel and handle events"""
        # Check if Apply or Close buttons were clicked
        if self.apply_settings_button.update(events):
            return 'apply'
        
        if self.close_settings_button.update(events):
            return 'close'
        
        # Check if Save Config or Load Config buttons were clicked
        if self.save_config_button.update(events):
            return 'save_config'
        
        if self.load_config_button.update(events):
            return 'load_config'
        
        # Check if Load Model button was clicked
        if self.load_model_button.update(events):
            return 'load_model'
        
        # Update the settings scrollbar
        self.settings_scrollbar.update(events)
        
        # Update sliders and checkboxes (including their text input boxes)
        self.depth_slider.update(events)
        self.scale_slider.update(events)
        self.red_first_slider.update(events)
        self.red_aggression_slider.update(events)
        self.heuristic_evals_slider.update(events)
        
        # Update neural network settings
        change_nn, self.use_neural_net_checkbox.checked = self.use_neural_net_checkbox.update(events)
        self.neural_net_ratio_slider.update(events)
        self.learning_rate_slider.update(events)
        self.hidden_dim_slider.update(events)
        self.num_layers_slider.update(events)
        
        return None