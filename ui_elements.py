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

class FileDialog:
    """Class for displaying a file selection dialog"""
    
    def __init__(self, x, y, width, height, title="Select a File", 
                 bg_color=(240, 240, 240), text_color=(0, 0, 0), 
                 button_color=(200, 200, 200), button_hover_color=(180, 180, 180)):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.bg_color = bg_color
        self.text_color = text_color
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 28)
        
        # Initialize file browsing state
        import os
        self.current_dir = os.path.abspath(os.path.curdir)
        self.files = []
        self.dirs = []
        self.update_file_list()
        
        # Scrollable file list
        self.list_panel = Panel(
            x + 10, 
            y + 50, 
            width - 20, 
            height - 120
        )
        
        # Scrollbar for file list
        self.scrollbar = Scrollbar(
            x + width - 30,
            y + 50,
            15,
            height - 120,
            1000,  # Will update with actual content height
            height - 120
        )
        
        # Buttons
        button_y = y + height - 50
        button_width = 100
        button_height = 40
        button_spacing = 20
        
        # Select button
        self.select_button = Button(
            x + width - 2 * button_width - button_spacing,
            button_y,
            button_width,
            button_height,
            "Select",
            button_color,
            button_hover_color
        )
        
        # Cancel button
        self.cancel_button = Button(
            x + width - button_width,
            button_y,
            button_width,
            button_height,
            "Cancel",
            button_color,
            button_hover_color
        )
        
        # Current path display
        self.path_label = Label(
            x + 10,
            y + 30,
            f"Path: {self.current_dir}",
            text_color,
            20
        )
        
        # Selected file name
        self.selected_file = None
        self.result = None  # Will be set to file path or None if canceled
        
    def update_file_list(self):
        """Update the file and directory listings"""
        import os
        try:
            # Get all files and directories in current directory
            all_entries = os.listdir(self.current_dir)
            
            # Separate into files and directories
            self.dirs = [".."] + [d for d in all_entries if os.path.isdir(os.path.join(self.current_dir, d))]
            self.files = [f for f in all_entries if os.path.isfile(os.path.join(self.current_dir, f)) 
                         and (f.endswith(".pt") or f.endswith(".pth"))]  # Only show PyTorch model files
            
            # Update path label
            if hasattr(self, 'path_label'):
                self.path_label.update_text(f"Path: {self.current_dir}")
                
            # Update scrollbar content height based on number of items
            if hasattr(self, 'scrollbar'):
                content_height = (len(self.dirs) + len(self.files)) * 30 + 20  # Item height + padding
                visible_height = self.list_panel.rect.height
                self.scrollbar.update_content_height(max(content_height, visible_height))
                self.scrollbar.scroll_position = 0  # Reset scroll position
                
        except Exception as e:
            print(f"Error updating file list: {e}")
            self.dirs = [".."]
            self.files = []
    
    def draw(self, surface):
        """Draw the file dialog"""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Black with 50% opacity
        surface.blit(overlay, (0, 0))
        
        # Draw dialog background
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        # Draw title
        title_surf = self.title_font.render(self.title, True, self.text_color)
        title_rect = title_surf.get_rect(midtop=(self.rect.centerx, self.rect.y + 10))
        surface.blit(title_surf, title_rect)
        
        # Draw current path
        self.path_label.draw(surface)
        
        # Draw file list panel
        self.list_panel.draw(surface)
        
        # Set clipping region to file list panel
        old_clip = surface.get_clip()
        surface.set_clip(self.list_panel.rect)
        
        # Draw files and directories
        y_offset = self.list_panel.rect.y + 10 - self.scrollbar.scroll_position
        item_height = 30
        
        # Draw directories first with a folder icon prefix
        for i, dir_name in enumerate(self.dirs):
            item_rect = pygame.Rect(self.list_panel.rect.x + 10, y_offset + i * item_height, 
                                   self.list_panel.rect.width - 30, 25)
            
            # Skip if item is not visible
            if item_rect.bottom < self.list_panel.rect.y or item_rect.y > self.list_panel.rect.bottom:
                continue
                
            # Highlight selected item
            if self.selected_file == (dir_name, True):  # (name, is_dir)
                pygame.draw.rect(surface, (180, 200, 220), item_rect)
            
            # Draw folder icon (simple yellow rectangle)
            folder_icon = pygame.Rect(item_rect.x, item_rect.y + 3, 16, 16)
            pygame.draw.rect(surface, (240, 220, 100), folder_icon)
            pygame.draw.rect(surface, (0, 0, 0), folder_icon, 1)
            
            # Draw directory name
            dir_text = self.font.render(f"📁 {dir_name}", True, self.text_color)
            surface.blit(dir_text, (item_rect.x + 20, item_rect.y + 2))
        
        # Draw files below directories
        file_y_offset = y_offset + len(self.dirs) * item_height
        for i, file_name in enumerate(self.files):
            item_rect = pygame.Rect(self.list_panel.rect.x + 10, file_y_offset + i * item_height, 
                                   self.list_panel.rect.width - 30, 25)
                
            # Skip if item is not visible
            if item_rect.bottom < self.list_panel.rect.y or item_rect.y > self.list_panel.rect.bottom:
                continue
                
            # Highlight selected item
            if self.selected_file == (file_name, False):  # (name, is_dir)
                pygame.draw.rect(surface, (180, 200, 220), item_rect)
            
            # Draw file icon (simple document shape)
            file_icon_rect = pygame.Rect(item_rect.x, item_rect.y + 3, 14, 18)
            pygame.draw.rect(surface, (255, 255, 255), file_icon_rect)
            pygame.draw.rect(surface, (0, 0, 0), file_icon_rect, 1)
            
            # Draw file name
            file_text = self.font.render(f"📄 {file_name}", True, self.text_color)
            surface.blit(file_text, (item_rect.x + 20, item_rect.y + 2))
        
        # Reset clipping region
        surface.set_clip(old_clip)
        
        # Draw scrollbar
        self.scrollbar.draw(surface)
        
        # Draw buttons
        self.select_button.draw(surface)
        self.cancel_button.draw(surface)
    
    def update(self, events):
        """Update dialog and check for result. Returns (is_done, result_path)"""
        import os
        
        # Update scrollbar
        self.scrollbar.update(events)
        
        # Check for button clicks
        if self.cancel_button.update(events):
            return True, None
        
        if self.select_button.update(events) and self.selected_file and not self.selected_file[1]:
            # Selected a file (not a directory), return the full path
            selected_path = os.path.join(self.current_dir, self.selected_file[0])
            return True, selected_path
        
        # Check for clicks on file list
        mouse_pos = pygame.mouse.get_pos()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.list_panel.rect.collidepoint(mouse_pos):
                    # Calculate which item was clicked
                    rel_y = mouse_pos[1] - (self.list_panel.rect.y - self.scrollbar.scroll_position)
                    item_height = 30
                    item_index = rel_y // item_height
                    
                    if 0 <= item_index < len(self.dirs):
                        # Clicked on a directory
                        dir_name = self.dirs[item_index]
                        if dir_name == "..":
                            # Navigate up one directory
                            self.current_dir = os.path.dirname(self.current_dir)
                            self.update_file_list()
                            self.selected_file = None
                        else:
                            # Either navigate into directory (on double click) or select it
                            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                                is_double_click = False
                                try:
                                    # Check if we're in same dir and same item for double click
                                    if (hasattr(self, 'last_click_time') and 
                                        pygame.time.get_ticks() - self.last_click_time < 500 and
                                        self.selected_file == (dir_name, True)):
                                        is_double_click = True
                                except:
                                    pass
                                
                                if is_double_click:
                                    # Double click - navigate into directory
                                    self.current_dir = os.path.join(self.current_dir, dir_name)
                                    self.update_file_list()
                                    self.selected_file = None
                                else:
                                    # Single click - select directory
                                    self.selected_file = (dir_name, True)  # (name, is_dir)
                                
                                self.last_click_time = pygame.time.get_ticks()
                                
                    elif 0 <= item_index - len(self.dirs) < len(self.files):
                        # Clicked on a file
                        file_index = item_index - len(self.dirs)
                        file_name = self.files[file_index]
                        self.selected_file = (file_name, False)  # (name, is_dir)
        
        # Still active, no result yet
        return False, None

class ModelParamsDialog:
    """A dialog for displaying and editing model parameters"""
    
    def __init__(self, x, y, width, height, model_params, current_params,
                 bg_color=(240, 240, 240), text_color=(0, 0, 0),
                 button_color=(200, 200, 200), button_hover_color=(180, 180, 180)):
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = bg_color
        self.text_color = text_color
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 28)
        self.title = "Model Parameters"
        
        # Store the model parameters
        self.model_params = model_params
        self.current_params = current_params
        
        # Initialize parameter input fields
        self.param_fields = {}
        
        # Text input fields for each parameter
        field_height = 30
        field_width = 120
        
        # Depth parameter
        depth_value = str(model_params.get('depth', current_params.get('depth', 3)))
        self.param_fields['depth'] = TextInput(
            self.rect.x + 300,
            self.rect.y + 85,
            field_width,
            field_height,
            depth_value
        )
        
        # Hidden dimension parameter
        hidden_dim_value = str(model_params.get('hidden_dim', current_params.get('hidden_dim', 128)))
        self.param_fields['hidden_dim'] = TextInput(
            self.rect.x + 300,
            self.rect.y + 125,
            field_width,
            field_height,
            hidden_dim_value
        )
        
        # Number of layers parameter
        num_layers_value = str(model_params.get('num_layers', current_params.get('num_layers', 2)))
        self.param_fields['num_layers'] = TextInput(
            self.rect.x + 300,
            self.rect.y + 165,
            field_width,
            field_height,
            num_layers_value
        )
        
        # Learning rate parameter
        learning_rate_value = str(model_params.get('learning_rate', current_params.get('learning_rate', 0.001)))
        self.param_fields['learning_rate'] = TextInput(
            self.rect.x + 300,
            self.rect.y + 205,
            field_width,
            field_height,
            learning_rate_value
        )
        
        # Apply and Cancel buttons
        button_y = self.rect.y + height - 50
        button_width = 100
        button_height = 40
        button_spacing = 20
        
        self.apply_button = Button(
            self.rect.x + width - 2 * button_width - button_spacing,
            button_y,
            button_width,
            button_height,
            "Apply",
            button_color,
            button_hover_color
        )
        
        self.cancel_button = Button(
            self.rect.x + width - button_width,
            button_y,
            button_width,
            button_height,
            "Cancel",
            button_color,
            button_hover_color
        )
        
        # Status message
        self.status_text = ""
        
    def draw(self, surface):
        # Draw semi-transparent overlay
        overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Black with 50% opacity
        surface.blit(overlay, (0, 0))
        
        # Draw dialog background
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        # Draw title
        title_surf = self.title_font.render(self.title, True, self.text_color)
        title_rect = title_surf.get_rect(midtop=(self.rect.centerx, self.rect.y + 10))
        surface.blit(title_surf, title_rect)
        
        # Draw information text
        info_text = "The model has the following parameters:"
        info_surf = self.font.render(info_text, True, self.text_color)
        info_rect = info_surf.get_rect(midtop=(self.rect.centerx, self.rect.y + 40))
        surface.blit(info_surf, info_rect)
        
        # Draw parameter labels and fields
        # Game depth/recursion
        depth_label = self.font.render("Game Depth:", True, self.text_color)
        surface.blit(depth_label, (self.rect.x + 30, self.rect.y + 85))
        self.param_fields['depth'].draw(surface)
        
        # Hidden dimension
        hidden_label = self.font.render("Hidden Dimension:", True, self.text_color)
        surface.blit(hidden_label, (self.rect.x + 30, self.rect.y + 125))
        self.param_fields['hidden_dim'].draw(surface)
        
        # Number of layers
        layers_label = self.font.render("Number of Layers:", True, self.text_color)
        surface.blit(layers_label, (self.rect.x + 30, self.rect.y + 165))
        self.param_fields['num_layers'].draw(surface)
        
        # Learning rate
        lr_label = self.font.render("Learning Rate:", True, self.text_color)
        surface.blit(lr_label, (self.rect.x + 30, self.rect.y + 205))
        self.param_fields['learning_rate'].draw(surface)
        
        # Draw status message if there is one
        if self.status_text:
            status_surf = self.font.render(self.status_text, True, (200, 0, 0))
            status_rect = status_surf.get_rect(midtop=(self.rect.centerx, self.rect.y + 240))
            surface.blit(status_surf, status_rect)
        
        # Draw buttons
        self.apply_button.draw(surface)
        self.cancel_button.draw(surface)
        
    def update(self, events):
        """Update dialog and check for result. Returns (is_done, params)"""
        # Update all text input fields
        for field_name, field in self.param_fields.items():
            field.update(events)
        
        # Check for button clicks
        if self.cancel_button.update(events):
            return True, None
        
        if self.apply_button.update(events):
            # Validate and convert parameter values
            try:
                params = {}
                
                # Get and validate depth
                depth = int(self.param_fields['depth'].text)
                if depth < 1 or depth > 10:
                    self.status_text = "Depth must be between 1 and 10"
                    return False, None
                params['depth'] = depth
                
                # Get and validate hidden_dim
                hidden_dim = int(self.param_fields['hidden_dim'].text)
                if hidden_dim < 16 or hidden_dim > 2048:
                    self.status_text = "Hidden dimension must be between 16 and 2048"
                    return False, None
                params['hidden_dim'] = hidden_dim
                
                # Get and validate num_layers
                num_layers = int(self.param_fields['num_layers'].text)
                if num_layers < 1 or num_layers > 10:
                    self.status_text = "Number of layers must be between 1 and 10"
                    return False, None
                params['num_layers'] = num_layers
                
                # Get and validate learning_rate
                learning_rate = float(self.param_fields['learning_rate'].text)
                if learning_rate <= 0 or learning_rate > 1.0:
                    self.status_text = "Learning rate must be between 0 and 1"
                    return False, None
                params['learning_rate'] = learning_rate
                
                # Return parameters if validation passes
                return True, params
                
            except ValueError:
                # Invalid number format
                self.status_text = "Invalid number format in parameters"
                return False, None
        
        # Still active, no result yet
        return False, None

class GameBoard:
    """Class for rendering the game board with triangles and markers"""
    
    def __init__(self, game_instance):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
    
    def draw(self, screen):
        """Draw the game board with triangles and markers"""
        # Draw triangles
        clicks = self.game.load_clicks()
        valid_loops = self.game.load_valid_loops()
        
        # Load formed loops and their colors
        formed_loops = self.game.find_formed_loops(clicks, valid_loops)
        loop_colors = self.game.get_loop_colors(formed_loops, clicks)
        
        # Draw triangles
        for loop_idx, loop in enumerate(valid_loops):
            vertices = [(vertex['r'], vertex['theta']) for vertex in loop['vertices']]
            self.draw_triangle(screen, vertices, self.game.colors['triangle_fill'], 
                              self.game.colors['triangle_border'])
        
        # Draw score values in triangles
        loop_scores = self.game.get_loop_scores(loop_colors)
        for i, (loop, (score_value, score_color)) in enumerate(zip(formed_loops, loop_scores)):
            if score_value != 0 and score_color:
                center_x, center_y = self.game.calculate_triangle_center(loop)
                font = pygame.font.SysFont(None, 32)
                text = font.render(str(score_value), True, self.game.colors[score_color])
                text_rect = text.get_rect(center=(center_x, center_y))
                screen.blit(text, text_rect)
        
        # Draw markers for played moves
        for click in clicks:
            r, theta = click['address']
            x, y = self.game.polar_to_cartesian(r, theta)
            color = self.game.colors[click['color']]
            pygame.draw.circle(screen, color, (int(x), int(y)), 8)  # Match marker size
            pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 8, 1)
    
    def draw_triangle(self, screen, vertices, fill_color, border_color):
        """Draw a triangle from polar coordinates"""
        points = []
        for r, theta in vertices:
            x, y = self.game.polar_to_cartesian(r, theta)
            points.append((int(x), int(y)))
            
            # Draw a small circle at each vertex
            pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 3)
        
        # Draw triangle
        pygame.draw.polygon(screen, fill_color, points)
        pygame.draw.polygon(screen, border_color, points, 1)

class MLMetricsPanel:
    """Class for displaying machine learning metrics in the corner"""
    
    def __init__(self, game_instance):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
    
    def draw(self, screen):
        """Draw machine learning metrics in the upper left corner"""
        # Calculate average loss over all games
        avg_loss = sum(self.game.stats['value_losses']) / max(1, len(self.game.stats['value_losses'])) if self.game.stats['value_losses'] else 0
        
        # Calculate loss trend over the last 500 moves (or fewer if not available)
        recent_losses = self.game.stats['value_losses'][-500:] if len(self.game.stats['value_losses']) > 500 else self.game.stats['value_losses']
        
        if len(recent_losses) > 1:
            # Calculate the slope of the trend line using linear regression
            x = list(range(len(recent_losses)))
            y = recent_losses
            
            # Simple linear regression formula: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
            n = len(recent_losses)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_xx = sum(i*i for i in x)
            
            # Calculate slope with protection against division by zero
            denominator = n * sum_xx - sum_x * sum_x
            if denominator != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            else:
                slope = 0
        else:
            slope = 0
            
        # Format metrics text
        metrics_text = [
            f"Avg Loss: {avg_loss:.6f}",
            f"Loss Trend: {slope:.6f}",
            f"Neural Net: {'ON' if self.game.use_neural_net else 'OFF'}",
            f"# Samples: {len(self.game.stats['value_losses'])}"
        ]
        
        # Use a fixed color for better visibility
        bg_color = (0, 0, 80)  # Dark blue background
        
        # Create a panel in the upper left corner
        margin = 10
        panel_width = 250
        panel_height = 100
        panel_x = margin
        panel_y = margin
        
        # Draw panel with high opacity for better visibility
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((bg_color[0], bg_color[1], bg_color[2], 220))  # More opaque
        screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw text with larger font
        font = pygame.font.SysFont(None, 24)  # Larger font size
        for i, text in enumerate(metrics_text):
            text_surf = font.render(text, True, (255, 255, 255))  # White text
            screen.blit(text_surf, (panel_x + 10, panel_y + 10 + i * 26))

class HelpPanel:
    """Class for displaying the help/rules panel"""
    
    def __init__(self, game_instance, screen_width, screen_height):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Rules panel
        self.rules_panel = Panel(self.screen_width // 2 - 350, self.screen_height // 2 - 300, 700, 600)
        
        # Rules title
        self.rules_title = Label(self.rules_panel.rect.centerx, self.rules_panel.rect.y + 20, 
                                "Vector Game Rules", self.game.colors['text'], 32, "center")
        
        # Scrollbar for rules panel
        scrollbar_x = self.rules_panel.rect.right - 20
        scrollbar_y = self.rules_panel.rect.y + 60
        scrollbar_height = self.rules_panel.rect.height - 120  # Leave space for title and close button
        
        self.rules_scrollbar = Scrollbar(
            scrollbar_x, scrollbar_y, 15, scrollbar_height, 
            self.game.rules_content_height, scrollbar_height,
            bar_color=(200, 200, 200), handle_color=(150, 150, 150)
        )
        
        # Close button
        self.close_rules_button = Button(self.rules_panel.rect.centerx - 50, 
                                    self.rules_panel.rect.bottom - 50, 
                                    100, 40, "Close", self.game.colors['button'], 
                                    self.game.colors['button_hover'])
    
    def draw(self, screen):
        """Render the help panel with game rules"""
        # Draw panel with semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
        
        # Draw the rules panel
        self.rules_panel.draw(screen)
        self.rules_title.draw(screen)
        
        # Draw rules text with scrolling
        scroll_offset = self.rules_scrollbar.scroll_position
        
        # Calculate visible area
        visible_height = self.rules_panel.rect.height - 120  # space for title and close button
        content_area = pygame.Rect(
            self.rules_panel.rect.x + 20, 
            self.rules_panel.rect.y + 60,
            self.rules_panel.rect.width - 40,
            visible_height
        )
        
        # Set up clipping area to avoid drawing outside the panel
        screen.set_clip(content_area)
        
        # Draw rules text with scrolling
        y_offset = content_area.y - scroll_offset
        font = pygame.font.SysFont(None, 18)
        section_font = pygame.font.SysFont(None, 22, bold=True)
        
        for line in self.game.rules_text.strip().split('\n'):
            if line.strip().isupper() and ":" in line:
                # Section header with spacing
                text_surf = section_font.render(line, True, self.game.colors['text'])
                text_rect = text_surf.get_rect(topleft=(content_area.x, y_offset))
                if content_area.y <= y_offset <= content_area.bottom:
                    screen.blit(text_surf, text_rect)
                y_offset += 30
            elif not line.strip():
                # Empty line
                y_offset += 15
            else:
                # Regular line
                text_surf = font.render(line, True, self.game.colors['text'])
                text_rect = text_surf.get_rect(topleft=(content_area.x, y_offset))
                if content_area.y <= y_offset <= content_area.bottom:
                    screen.blit(text_surf, text_rect)
                y_offset += 22
        
        # Reset clipping
        screen.set_clip(None)
        
        # Draw scrollbar
        self.rules_scrollbar.draw(screen)
        
        # Draw close button
        self.close_rules_button.draw(screen)
    
    def update(self, events):
        """Update the help panel and handle events"""
        if self.close_rules_button.update(events):
            return 'close'  # Signal to close the panel
        
        # Update scrollbar
        self.rules_scrollbar.update(events)
        return False

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
        
        self.red_aggression_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 230, 
                                        500, 20, 0, 100, self.game.red_ai_aggression, text="Red AI Aggression (%)")
        
        self.heuristic_evals_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 280, 
                                        500, 20, 5, 50, self.game.max_heuristic_evals, text="Max Moves to Evaluate")
        
        # Neural network settings
        self.use_neural_net_checkbox = Checkbox(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 330, 
                                            20, "Use Neural Network", self.game.use_neural_net)
        
        self.neural_net_ratio_slider = Slider(self.settings_panel.rect.x + 50, self.settings_panel.rect.y + 370, 
                                            500, 20, 0, 100, self.game.neural_net_ratio, text="Neural Net Ratio (%)")
        
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

class Dashboard:
    """Class for displaying the model dashboard"""
    
    def __init__(self, game_instance, screen_width, screen_height):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create the close button during initialization
        panel_width = int(self.screen_width * 0.75)
        panel_height = int(self.screen_height * 0.75)
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        self.close_button = Button(
            panel_x + panel_width - 120, 
            panel_y + panel_height - 60, 
            100, 40, 
            "Close", 
            self.game.colors['button'], 
            self.game.colors['button_hover']
        )
    
    def draw(self, screen):
        """Render the model dashboard with actual plots"""
        # Draw panel with semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (0, 0))
        
        # Size and position the dashboard panel (75% of screen size, centered)
        panel_width = int(self.screen_width * 0.75)
        panel_height = int(self.screen_height * 0.75)
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        dashboard_panel = Panel(panel_x, panel_y, panel_width, panel_height)
        dashboard_panel.draw(screen)
        
        # Draw title
        dashboard_title = Label(panel_x + panel_width // 2, panel_y + 20, 
                              "Model Dashboard", self.game.colors['text'], 32, "center")
        dashboard_title.draw(screen)
        
        # — Textual dashboard metrics for Red & Blue —

        # Gather per-game scores
        red_scores = self.game.stats.get('red_game_scores', [])
        blue_scores = self.game.stats.get('blue_game_scores', [])
        total_games = len(red_scores)

        # Win/Loss counts
        red_wins = sum(1 for r,b in zip(red_scores, blue_scores) if r > b)
        red_losses = sum(1 for r,b in zip(red_scores, blue_scores) if r < b)
        blue_wins = red_losses
        blue_losses = red_wins

        # High & average scores
        high_red = max(red_scores) if red_scores else 0
        high_blue = max(blue_scores) if blue_scores else 0
        avg_red = sum(red_scores)/total_games if total_games else 0
        avg_blue = sum(blue_scores)/total_games if total_games else 0

        # Reward statistics
        rewards = self.game.stats.get('rewards', [])
        reward_avg = sum(rewards)/len(rewards) if rewards else 0
        reward_var = np.var(rewards) if rewards else 0
        if len(rewards) > 1:
            reward_trend = (rewards[-1] - rewards[0])/(len(rewards)-1)
        else:
            reward_trend = 0

        # TD-error trend
        td_errors = self.game.stats.get('td_errors', [])
        if len(td_errors) > 1:
            td_trend = (td_errors[-1] - td_errors[0])/(len(td_errors)-1)
        else:
            td_trend = 0

        # Neural network parameter count
        param_count = sum(p.numel() for p in self.game.regression_net.parameters() if p.requires_grad)

        # Prepare lines of text
        lines = [
            f"Games played: {total_games}",
            f"Red wins: {red_wins}   Red losses: {red_losses}",
            f"Blue wins: {blue_wins}  Blue losses: {blue_losses}",
            f"High score — Red: {high_red}  Blue: {high_blue}",
            f"Avg score  — Red: {avg_red:.2f}  Blue: {avg_blue:.2f}",
            f"Reward avg: {reward_avg:.2f}",
            f"Reward trend (slope): {reward_trend:.3f} per action",
            f"Reward variance: {reward_var:.3f}",
            f"TD-error trend (slope): {td_trend:.3f}",
            f"NN Parameters: {param_count}"
        ]

        # Draw each line
        for idx, txt in enumerate(lines):
            Label(panel_x + 20,
                  panel_y + 60 + idx*30,
                  txt,
                  self.game.colors['text'],
                  20
            ).draw(screen)
        
        # Draw close button (now just drawing the previously initialized button)
        self.close_button.draw(screen)
    
    def update(self, events):
        """Update the dashboard panel and handle events"""
        # Check if close button was clicked
        if self.close_button.update(events):
            return 'close'
        return False
    
    def create_loss_plot(self, width, height):
        """Create a plot for policy loss"""
        if not self.game.stats['losses']:
            return None
            
        # Create figure
        dpi = 100
        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create subplot
        ax = fig.add_subplot(111)
        
        # Plot loss
        losses = self.game.stats['losses']
        x = range(1, len(losses) + 1)
        ax.plot(x, losses, 'r-')
        
        # Set labels
        ax.set_title("Policy Loss Over Time")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss Value")
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Create pygame surface
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
        return surf
    
    def create_reward_plot(self, width, height):
        """Create a plot for rewards"""
        if not self.game.stats['rewards']:
            return None
            
        # Create figure
        dpi = 100
        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create subplot
        ax = fig.add_subplot(111)
        
        # Plot rewards
        rewards = self.game.stats['rewards']
        x = range(1, len(rewards) + 1)
        ax.plot(x, rewards, 'g-')
        
        # Calculate and plot moving average if enough data
        if len(rewards) > 5:
            window_size = min(10, len(rewards) // 5)
            moving_avg = []
            for i in range(len(rewards) - window_size + 1):
                window_avg = sum(rewards[i:i+window_size]) / window_size
                moving_avg.append(window_avg)
            
            # Plot moving average
            x_ma = range(window_size, len(rewards) + 1)
            ax.plot(x_ma, moving_avg, 'b-', label=f'{window_size}-point Moving Avg')
        
        # Set labels
        ax.set_title("Rewards Over Time")
        ax.set_xlabel("Action")
        ax.set_ylabel("Reward Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Create pygame surface
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
        return surf
    
    def create_value_plot(self, width, height):
        """Create a plot for value network performance"""
        if not self.game.stats['value_losses']:
            return None
            
        # Create figure
        dpi = 100
        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create subplot
        ax = fig.add_subplot(111)
        
        # Plot value losses
        value_losses = self.game.stats['value_losses'][-1000:] if len(self.game.stats['value_losses']) > 1000 else self.game.stats['value_losses']
        x = range(1, len(value_losses) + 1)
        ax.plot(x, value_losses, 'b-')
        
        # Set labels
        ax.set_title("Value Network Loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss Value")
        ax.grid(True, alpha=0.3)
        
        # Calculate and plot trend line
        if len(value_losses) > 10:
            # Use polynomial fit
            z = np.polyfit(x, value_losses, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), 'r--', label=f'Trend (m={z[0]:.6f})')
            ax.legend(loc='upper right', fontsize='small')
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Create pygame surface
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
        return surf
    
    def create_game_plot(self, width, height):
        """Create a plot for game performance metrics"""
        # We'll show either game rewards or round rewards depending on what's available
        has_game_data = len(self.game.stats['game_rewards']) > 0
        has_round_data = len(self.game.stats['round_rewards']) > 0
        
        if not has_game_data and not has_round_data:
            return None
            
        # Create figure
        dpi = 100
        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Create subplot
        ax = fig.add_subplot(111)
        
        # Plot game data if available
        if has_game_data:
            game_rewards = self.game.stats['game_rewards']
            x = range(1, len(game_rewards) + 1)
            ax.plot(x, game_rewards, 'bo-', label='Game Rewards')
            
            # Add trend line
            if len(game_rewards) > 2:
                z = np.polyfit(x, game_rewards, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), 'b--', label=f'Trend (m={z[0]:.3f})')
            
            ax.set_title("Performance by Game")
            ax.set_xlabel("Game Number")
        
        # Plot round data if available and no game data
        elif has_round_data:
            round_data = self.game.stats['round_rewards']
            rounds, rewards = zip(*round_data)
            ax.plot(rounds, rewards, 'go-', label='Round Rewards')
            
            # Add trend line
            if len(rounds) > 2:
                z = np.polyfit(rounds, rewards, 1)
                p = np.poly1d(z)
                ax.plot(rounds, p(rounds), 'g--', label=f'Trend (m={z[0]:.3f})')
            
            ax.set_title("Performance by Round")
            ax.set_xlabel("Round Number")
        
        # Add y-axis label
        ax.set_ylabel("Average Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
        
        # Adjust layout
        fig.tight_layout()
        
        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Create pygame surface
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
        return surf

class GameOverScreen:
    """Class for displaying the game over screen"""
    
    def __init__(self, game_instance, screen_width, screen_height):
        """Initialize with a reference to the game instance"""
        self.game = game_instance
        self.screen_width = screen_width
        self.screen_height = screen_height
    
    def show(self, screen, winner, red_score, blue_score):
        """Show the game over screen with replay option"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # black at 70% opacity
        screen.blit(overlay, (0, 0))

        # Create panel for game over message
        panel_width = 500
        panel_height = 300
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        game_over_panel = Panel(panel_x, panel_y, panel_width, panel_height)
        game_over_panel.draw(screen)
        
        # Render the winner text
        font = pygame.font.SysFont(None, 64)
        msg = f"Game Over! {winner} has won!"
        text_surf = font.render(msg, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=(panel_x + panel_width//2, panel_y + 60))
        screen.blit(text_surf, text_rect)
        
        # Render scores
        score_font = pygame.font.SysFont(None, 40)
        red_text = score_font.render(f"Red Score: {red_score}", True, self.game.colors['red'])
        blue_text = score_font.render(f"Blue Score: {blue_score}", True, self.game.colors['blue'])
        
        red_rect = red_text.get_rect(center=(panel_x + panel_width//2, panel_y + 120))
        blue_rect = blue_text.get_rect(center=(panel_x + panel_width//2, panel_y + 170))
        
        screen.blit(red_text, red_rect)
        screen.blit(blue_text, blue_rect)
        
        # Create replay button
        self.replay_button = Button(panel_x + 75, panel_y + panel_height - 80, 
                              150, 50, "Replay Game", self.game.colors['button'], self.game.colors['button_hover'])
        self.replay_button.draw(screen)
        
        # Create continue button
        self.continue_button = Button(panel_x + panel_width - 225, panel_y + panel_height - 80, 
                                150, 50, "Continue", self.game.colors['button'], self.game.colors['button_hover'])
        self.continue_button.draw(screen)
        
        # Update display
        pygame.display.flip()
    
    def update(self, events):
        """Update the game over screen and handle events"""
        # Check if replay button is clicked
        if self.replay_button.update(events):
            return 'replay'
        
        # Check if continue button is clicked
        if self.continue_button.update(events):
            return 'continue'
        
        return None