import tkinter as tk
from tkinter import colorchooser, ttk
from PIL import Image, ImageDraw, ImageTk
import random
import math

class DotPatternGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Dot Pattern Generator")
        
        # Initialize default values
        self.canvas_width = 800
        self.canvas_height = 600
        self.dot_radius_min = 5
        self.dot_radius_max = 20
        self.dot_count = 300
        self.bg_color = "#0055AA"  # Default blue background
        self.dot_color = "#FFFFFF"  # Default white dots
        self.outline_color = None  # No outline by default
        self.outline_width = 1
        self.randomness = 0.5  # 0 = grid pattern, 1 = fully random
        self.density = 0.5  # Controls how many dots are drawn
        self.pattern_type = "scatter"  # Default pattern type
        
        # Create the image and preview
        self.image = None
        self.image_tk = None
        
        self.create_ui()
        self.generate_image()
        
 = tk.IntVar(value=self.canvas_height)
        ttk.Spinbox(size_frame, from_=100, to=2000, increment=100, width=5, textvariable=self.height_var).grid(row=0, column=3, sticky=tk.W)
        row += 1
        
        # Dot controls
        ttk.Label(control_frame, text="Dot Size:").grid(row=row, column=0, sticky=tk.W)
        dot_size_frame = ttk.Frame(control_frame)
        dot_size_frame.grid(row=row, column=1, sticky=tk.W)
        
        ttk.Label(dot_size_frame, text="Min:").grid(row=0, column=0, sticky=tk.W)
        self.min_radius_var = tk.IntVar(value=self.dot_radius_min)
        ttk.Spinbox(dot_size_frame, from_=1, to=50, width=3, textvariable=self.min_radius_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        ttk.Label(dot_size_frame, text="Max:").grid(row=0, column=2, sticky=tk.W)
        self.max_radius_var = tk.IntVar(value=self.dot_radius_max)
        ttk.Spinbox(dot_size_frame, from_=1, to=100, width=3, textvariable=self.max_radius_var).grid(row=0, column=3, sticky=tk.W)
        row += 1
        
        # Dot count
        ttk.Label(control_frame, text="Dot Count:").grid(row=row, column=0, sticky=tk.W)
        self.dot_count_var = tk.IntVar(value=self.dot_count)
        ttk.Spinbox(control_frame, from_=10, to=2000, increment=10, width=5, textvariable=self.dot_count_var).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Pattern type
        ttk.Label(control_frame, text="Pattern Type:").grid(row=row, column=0, sticky=tk.W)
        self.pattern_type_var = tk.StringVar(value=self.pattern_type)
        pattern_combo = ttk.Combobox(control_frame, textvariable=self.pattern_type_var, width=15)
        pattern_combo['values'] = ('scatter', 'grid', 'circular', 'spiral', 'wave')
        pattern_combo.grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Randomness slider
        ttk.Label(control_frame, text="Randomness:").grid(row=row, column=0, sticky=tk.W)
        self.randomness_var = tk.DoubleVar(value=self.randomness)
        ttk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                  variable=self.randomness_var, length=150).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Density slider
        ttk.Label(control_frame, text="Density:").grid(row=row, column=0, sticky=tk.W)
        self.density_var = tk.DoubleVar(value=self.density)
        ttk.Scale(control_frame, from_=0.1, to=1, orient=tk.HORIZONTAL, 
                  variable=self.density_var, length=150).grid(row=row, column=1, sticky=tk.W)
        row += 1
        
        # Color pickers
        ttk.Label(control_frame, text="Background Color:").grid(row=row, column=0, sticky=tk.W)
        color_frame = ttk.Frame(control_frame)
        color_frame.grid(row=row, column=1, sticky=tk.W)
        
        self.bg_color_btn = tk.Button(color_frame, bg=self.bg_color, width=3, height=1, 
                                    command=lambda: self.pick_color("bg"))
        self.bg_color_btn.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Button(color_frame, text="Reset", width=5, 
                   command=lambda: self.reset_color("bg")).grid(row=0, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(control_frame, text="Dot Color:").grid(row=row, column=0, sticky=tk.W)
        color_frame = ttk.Frame(control_frame)
        color_frame.grid(row=row, column=1, sticky=tk.W)
        
        self.dot_color_btn = tk.Button(color_frame, bg=self.dot_color, width=3, height=1, 
                                     command=lambda: self.pick_color("dot"))
        self.dot_color_btn.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Button(color_frame, text="Reset", width=5, 
                   command=lambda: self.reset_color("dot")).grid(row=0, column=1, sticky=tk.W)
        row += 1
        
        ttk.Label(control_frame, text="Outline:").grid(row=row, column=0, sticky=tk.W)
        outline_frame = ttk.Frame(control_frame)
        outline_frame.grid(row=row, column=1, sticky=tk.W)
        
        self.outline_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(outline_frame, text="Enable", variable=self.outline_enabled).grid(row=0, column=0, sticky=tk.W)
        
        self.outline_color_btn = tk.Button(outline_frame, bg='black', width=3, height=1, 
                                         command=lambda: self.pick_color("outline"))
        self.outline_color_btn.grid(row=0, column=1, sticky=tk.W, padx=(5, 5))
        
        ttk.Label(outline_frame, text="Width:").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        self.outline_width_var = tk.IntVar(value=self.outline_width)
        ttk.Spinbox(outline_frame, from_=1, to=10, width=2, textvariable=self.outline_width_var).grid(row=0, column=3, sticky=tk.W)
        row += 1
        
        # Random seed option
        ttk.Label(control_frame, text="Random Seed:").grid(row=row, column=0, sticky=tk.W)
        seed_frame = ttk.Frame(control_frame)
        seed_frame.grid(row=row, column=1, sticky=tk.W)
        
        self.use_seed = tk.BooleanVar(value=False)
        ttk.Checkbutton(seed_frame, text="Use Seed", variable=self.use_seed).grid(row=0, column=0, sticky=tk.W)
        
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(seed_frame, from_=0, to=9999, width=5, textvariable=self.seed_var).grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        row += 1
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Generate", command=self.generate_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Random Style", command=self.random_style).pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Configure resizing behavior
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def pick_color(self, color_type):
        initial_color = None
        if color_type == "bg":
            initial_color = self.bg_color
        elif color_type == "dot":
            initial_color = self.dot_color
        elif color_type == "outline":
            initial_color = self.outline_color if self.outline_color else "#000000"
            
        color = colorchooser.askcolor(initial_color)[1]
        if color:
            if color_type == "bg":
                self.bg_color = color
                self.bg_color_btn.configure(bg=color)
            elif color_type == "dot":
                self.dot_color = color
                self.dot_color_btn.configure(bg=color)
            elif color_type == "outline":
                self.outline_color = color
                self.outline_color_btn.configure(bg=color)
            
    def reset_color(self, color_type):
        if color_type == "bg":
            self.bg_color = "#0055AA"
            self.bg_color_btn.configure(bg=self.bg_color)
        elif color_type == "dot":
            self.dot_color = "#FFFFFF"
            self.dot_color_btn.configure(bg=self.dot_color)
            
    def generate_image(self):
        # Get current values from UI
        self.canvas_width = self.width_var.get()
        self.canvas_height = self.height_var.get()
        self.dot_radius_min = self.min_radius_var.get()
        self.dot_radius_max = self.max_radius_var.get()
        self.dot_count = self.dot_count_var.get()
        self.randomness = self.randomness_var.get()
        self.density = self.density_var.get()
        self.pattern_type = self.pattern_type_var.get()
        self.outline_width = self.outline_width_var.get()
        
        # Set random seed if specified
        if self.use_seed.get():
            random.seed(self.seed_var.get())
        else:
            random.seed(None)  # Use system time
            
        # Create new image
        self.image = Image.new('RGB', (self.canvas_width, self.canvas_height), self.bg_color)
        draw = ImageDraw.Draw(self.image)
        
        # Generate dot positions and properties based on pattern type
        dots = self.generate_pattern()
        
        # Draw dots with enhanced styling
        if hasattr(self, 'dotProps'):
            for x, y, radius, has_outline, custom_color in self.dotProps:
                # Use custom color or default dot color
                fill_color = custom_color if custom_color else self.dot_color
                
                # Determine if this dot has an outline
                outline = None
                if has_outline and self.outline_enabled.get():
                    outline = self.outline_color
                
                # Draw the dot
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), 
                            fill=fill_color, 
                            outline=outline, 
                            width=self.outline_width if outline else 0)
        else:
            # Fallback to basic drawing if dotProps isn't available
            for x, y, radius in dots:
                outline = self.outline_color if self.outline_enabled.get() else None
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), 
                            fill=self.dot_color, 
                            outline=outline, 
                            width=self.outline_width if outline else 0)
        
        # Update preview
        self.update_preview()
        
    def generate_pattern(self):
        dots = []
        
        # For storing dot properties: (x, y, radius, has_outline, custom_color)
        dotProps = []
        
        if self.pattern_type == "scatter":
            for _ in range(self.dot_count):
                x = random.randint(0, self.canvas_width)
                y = random.randint(0, self.canvas_height)
                radius = random.randint(self.dot_radius_min, self.dot_radius_max)
                
                # Randomly decide if this dot has an outline
                has_outline = random.random() < 0.5
                
                # Occasionally use a different color (20% chance)
                custom_color = None
                if random.random() < 0.2:
                    # Create a slightly varied color from the main dot color
                    r, g, b = [int(self.dot_color[1:3], 16), 
                              int(self.dot_color[3:5], 16), 
                              int(self.dot_color[5:7], 16)]
                    
                    # Vary the color slightly
                    var = 30  # Color variation amount
                    r = max(0, min(255, r + random.randint(-var, var)))
                    g = max(0, min(255, g + random.randint(-var, var)))
                    b = max(0, min(255, b + random.randint(-var, var)))
                    
                    custom_color = f"#{r:02x}{g:02x}{b:02x}"
                
                dotProps.append((x, y, radius, has_outline, custom_color))
                
        elif self.pattern_type == "grid":
            # Calculate grid dimensions - make it more precise for designers
            grid_size = max(self.dot_radius_max * 2.5, 15)  # Adjusted for better spacing
            cols = self.canvas_width // grid_size
            rows = self.canvas_height // grid_size
            
            # Calculate offsets to center the grid
            offset_x = (self.canvas_width - (cols * grid_size)) / 2
            offset_y = (self.canvas_height - (rows * grid_size)) / 2
            
            # Start with a more sophisticated grid pattern
            for row in range(rows+1):
                for col in range(cols+1):
                    # Base position - centered grid
                    x = offset_x + col * grid_size
                    y = offset_y + row * grid_size
                    
                    # Create special design patterns in the grid
                    pattern_factor = 1.0
                    
                    # Create a gradient effect - dots get smaller toward edges
                    center_x = self.canvas_width / 2
                    center_y = self.canvas_height / 2
                    distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_distance = math.sqrt((self.canvas_width/2)**2 + (self.canvas_height/2)**2)
                    gradient_factor = 1 - (distance_from_center / max_distance) * 0.6
                    
                    # Add randomness if needed - less randomness for design grids
                    rand_factor = self.randomness * 0.7  # Reduced randomness for grid
                    if rand_factor > 0:
                        x += random.uniform(-rand_factor * grid_size/3, rand_factor * grid_size/3)
                        y += random.uniform(-rand_factor * grid_size/3, rand_factor * grid_size/3)
                    
                    # Random dot size with gradient effect
                    base_size = random.randint(self.dot_radius_min, self.dot_radius_max)
                    radius = max(1, int(base_size * gradient_factor * pattern_factor))
                    
                    # More sophisticated density approach - keep corners less dense
                    edge_factor = 1.0
                    if distance_from_center > max_distance * 0.7:
                        edge_factor = 0.7
                    
                    # Apply density filter with edge consideration
                    if random.random() <= (self.density * edge_factor):
                        # Randomly decide if this dot has an outline (varies by position)
                        has_outline = random.random() < (0.3 + gradient_factor * 0.4)
                        
                        # For grids, let's add occasional color variations based on position
                        custom_color = None
                        if random.random() < 0.25:  # 25% chance of color variation
                            r, g, b = [int(self.dot_color[1:3], 16), 
                                      int(self.dot_color[3:5], 16), 
                                      int(self.dot_color[5:7], 16)]
                            
                            # Vary color based on position
                            position_factor = (x / self.canvas_width + y / self.canvas_height) / 2
                            hue_shift = int(position_factor * 30) - 15  # -15 to +15
                            
                            r = max(0, min(255, r + hue_shift))
                            g = max(0, min(255, g + hue_shift))
                            b = max(0, min(255, b + hue_shift))
                            
                            custom_color = f"#{r:02x}{g:02x}{b:02x}"
                        
                        dotProps.append((x, y, radius, has_outline, custom_color))
        
        elif self.pattern_type == "circular":
            # Generate concentric circles of dots
            center_x = self.canvas_width // 2
            center_y = self.canvas_height // 2
            max_radius = min(center_x, center_y)
            
            # Number of circles - more for better design
            num_circles = int(self.dot_count / 15)  # More circles, fewer dots per circle
            
            for circle in range(1, num_circles + 1):
                circle_radius = (circle / num_circles) * max_radius
                # Calculate number of dots for this circle
                circle_circumference = 2 * math.pi * circle_radius
                dots_in_circle = max(12, int((circle_circumference / 40) * self.density))
                
                # Every other circle has a special pattern
                special_circle = (circle % 3 == 0)
                
                for i in range(dots_in_circle):
                    angle = (i / dots_in_circle) * 2 * math.pi
                    
                    # Add designed randomness
                    angle_rand = 0
                    radius_rand = 0
                    if self.randomness > 0:
                        rand_factor = self.randomness * (0.8 if special_circle else 0.4)
                        angle_rand = random.uniform(-rand_factor * 0.15, rand_factor * 0.15)
                        radius_rand = random.uniform(
                            -rand_factor * max_radius * 0.08, 
                            rand_factor * max_radius * 0.08
                        )
                    
                    adjusted_angle = angle + angle_rand
                    circle_radius_adjusted = circle_radius + radius_rand
                    
                    x = center_x + circle_radius_adjusted * math.cos(adjusted_angle)
                    y = center_y + circle_radius_adjusted * math.sin(adjusted_angle)
                    
                    # Size varies based on circle
                    size_factor = 1.0
                    if special_circle:
                        size_factor = 1.3  # Larger dots on every third circle
                    
                    # Random dot size influenced by pattern
                    radius = random.randint(
                        max(1, int(self.dot_radius_min * size_factor * 0.9)),
                        max(2, int(self.dot_radius_max * size_factor))
                    )
                    
                    # Determine outline (special circles alternate with more outlines)
                    has_outline = (special_circle and i % 2 == 0) or (not special_circle and random.random() < 0.3)
                    
                    # Color variations
                    custom_color = None
                    if special_circle and random.random() < 0.4:
                        r, g, b = [int(self.dot_color[1:3], 16), 
                                  int(self.dot_color[3:5], 16), 
                                  int(self.dot_color[5:7], 16)]
                        
                        # Circle-specific colors
                        circle_factor = circle / num_circles
                        r = max(0, min(255, r + int((circle_factor - 0.5) * 40)))
                        g = max(0, min(255, g + int((0.5 - circle_factor) * 30)))
                        b = max(0, min(255, b + int(math.sin(circle_factor * math.pi) * 40)))
                        
                        custom_color = f"#{r:02x}{g:02x}{b:02x}"
                    
                    if 0 <= x <= self.canvas_width and 0 <= y <= self.canvas_height:
                        dotProps.append((x, y, radius, has_outline, custom_color))
        
        elif self.pattern_type == "spiral":
            # Generate a spiral pattern
            center_x = self.canvas_width // 2
            center_y = self.canvas_height // 2
            max_radius = min(center_x, center_y)
            
            # Spiral parameters - adjusted for design
            b = 0.15  # Controls how tightly wound the spiral is
            max_theta = 10 * math.pi  # More turns for design
            
            # Number of dots along the spiral
            dots_count = int(self.dot_count * self.density)
            
            # Create multiple intertwined spirals for a better design
            num_spirals = 2
            for spiral in range(num_spirals):
                spiral_phase = (spiral / num_spirals) * 2 * math.pi
                spiral_radial_offset = max_radius * 0.05  # Slight offset between spirals
                
                for i in range(dots_count // num_spirals):
                    # Parameter along the spiral (0 to 1)
                    t = i / (dots_count // num_spirals)
                    
                    # Calculate theta and radius
                    theta = t * max_theta + spiral_phase
                    spiral_radius = max_radius * t + spiral * spiral_radial_offset
                    
                    # Add controlled randomness
                    theta_rand = 0
                    radius_rand = 0
                    if self.randomness > 0:
                        # Less randomness at the beginning of the spiral
                        theta_rand = random.uniform(-self.randomness * 0.3, self.randomness * 0.3) * t
                        radius_rand = random.uniform(
                            -self.randomness * max_radius * 0.07, 
                            self.randomness * max_radius * 0.07
                        ) * t
                    
                    # Apply the randomness
                    theta_adjusted = theta + theta_rand
                    spiral_radius_adjusted = spiral_radius + radius_rand
                    
                    # Convert to cartesian coordinates
                    x = center_x + spiral_radius_adjusted * math.cos(theta_adjusted)
                    y = center_y + spiral_radius_adjusted * math.sin(theta_adjusted)
                    
                    # Dots get larger toward the outside
                    size_factor = 0.7 + t * 0.6
                    radius = random.randint(
                        max(1, int(self.dot_radius_min * size_factor)),
                        max(2, int(self.dot_radius_max * size_factor))
                    )
                    
                    # Alternate outlines along the spiral
                    has_outline = (i % 3 == spiral % 3)
                    
                    # Color varies along the spiral
                    custom_color = None
                    if random.random() < 0.3:
                        r, g, b = [int(self.dot_color[1:3], 16), 
                                  int(self.dot_color[3:5], 16), 
                                  int(self.dot_color[5:7], 16)]
                        
                        # Color changes with theta
                        theta_norm = (theta % (2 * math.pi)) / (2 * math.pi)
                        
                        r = max(0, min(255, r + int(math.sin(theta_norm * math.pi) * 30)))
                        g = max(0, min(255, g + int(math.cos(theta_norm * math.pi) * 30)))
                        b = max(0, min(255, b + int(math.sin(theta_norm * 2 * math.pi) * 30)))
                        
                        custom_color = f"#{r:02x}{g:02x}{b:02x}"
                    
                    if 0 <= x <= self.canvas_width and 0 <= y <= self.canvas_height:
                        dotProps.append((x, y, radius, has_outline, custom_color))
                    
        elif self.pattern_type == "wave":
            # Generate multiple overlapping wave patterns
            amplitude = self.canvas_height / 5
            frequency = 2 * math.pi / (self.canvas_width * 0.8)
            
            # Calculate grid dimensions (for horizontal spacing)
            spacing_x = max(15, int(self.canvas_width / (self.dot_count * self.density * 0.3)))
            num_waves = 4  # More waves for better design
            
            for wave in range(num_waves):
                # Each wave has slightly different parameters
                wave_amplitude = amplitude * (0.8 + 0.4 * (wave / num_waves))
                wave_frequency = frequency * (0.9 + 0.2 * wave / num_waves)
                wave_center = self.canvas_height * (wave + 1) / (num_waves + 1)
                phase = wave * math.pi / 2  # More varied phase shift
                
                # Find the right x-spacing for this wave
                wave_spacing = int(spacing_x * (0.8 + 0.4 * random.random()))
                
                for x in range(0, self.canvas_width, wave_spacing):
                    # Calculate several points along this x position to create a richer pattern
                    num_points = 2 if wave % 2 == 0 else 3  # Alternate between 2 and 3 points
                    
                    for point in range(num_points):
                        point_offset = (point / (num_points)) * (wave_amplitude * 0.4)
                        
                        # Calculate y position based on sine wave with offsets
                        base_y = wave_center + wave_amplitude * math.sin(wave_frequency * x + phase) + point_offset
                        
                        # Add designed randomness
                        x_pos = x
                        y_pos = base_y
                        if self.randomness > 0:
                            rand_factor = self.randomness * 0.7  # Controlled randomness
                            x_rand = random.uniform(-rand_factor * spacing_x/3, rand_factor * spacing_x/3)
                            y_rand = random.uniform(-rand_factor * wave_amplitude/4, rand_factor * wave_amplitude/4)
                            
                            # Apply randomness with wave-specific adjustments
                            x_pos += x_rand
                            y_pos += y_rand
                        
                        # Size varies based on vertical position
                        y_factor = y_pos / self.canvas_height
                        size_factor = 0.8 + y_factor * 0.4
                        
                        radius = random.randint(
                            max(1, int(self.dot_radius_min * size_factor)),
                            max(2, int(self.dot_radius_max * size_factor))
                        )
                        
                        # Determine if this dot has an outline (more outlines on certain waves)
                        has_outline = (wave % 2 == 0 and point % 2 == 0) or random.random() < 0.3
                        
                        # Add color variations based on wave and position
                        custom_color = None
                        if random.random() < 0.35:
                            r, g, b = [int(self.dot_color[1:3], 16), 
                                      int(self.dot_color[3:5], 16), 
                                      int(self.dot_color[5:7], 16)]
                            
                            # Color varies with wave and x position
                            wave_factor = wave / (num_waves - 1) if num_waves > 1 else 0.5
                            x_factor = x / self.canvas_width
                            
                            r = max(0, min(255, r + int((wave_factor - 0.5) * 30)))
                            g = max(0, min(255, g + int((x_factor - 0.5) * 30)))
                            b = max(0, min(255, b + int(math.sin(x_factor * math.pi) * 25)))
                            
                            custom_color = f"#{r:02x}{g:02x}{b:02x}"
                        
                        # Apply density filter with position-based adjustments
                        density_factor = 1.0
                        if x < spacing_x * 2 or x > self.canvas_width - spacing_x * 2:
                            density_factor = 0.7  # Less density at edges
                            
                        if random.random() <= (self.density * density_factor):
                            if 0 <= x_pos <= self.canvas_width and 0 <= y_pos <= self.canvas_height:
                                dotProps.append((x_pos, y_pos, radius, has_outline, custom_color))
        
        # Convert dot properties to simple x,y,radius tuples for compatibility
        dots = [(x, y, r) for x, y, r, _, _ in dotProps]
        
        # Store the enhanced properties for drawing
        self.dotProps = dotProps
        return dots
    
    def update_preview(self):
        # Resize for preview
        preview_width = self.preview_canvas.winfo_width()
        preview_height = self.preview_canvas.winfo_height()
        
        if preview_width <= 1:  # If canvas not yet sized
            preview_width = self.canvas_width // 2
            preview_height = self.canvas_height // 2
            
        # Keep aspect ratio
        aspect_ratio = self.canvas_width / self.canvas_height
        if preview_width / preview_height > aspect_ratio:
            new_width = int(preview_height * aspect_ratio)
            new_height = preview_height
        else:
            new_width = preview_width
            new_height = int(preview_width / aspect_ratio)
            
        preview_img = self.image.resize((new_width, new_height), Image.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(preview_img)
        
        # Update canvas
        self.preview_canvas.config(width=new_width, height=new_height)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)
    
    def save_image(self):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        
        if filename:
            self.image.save(filename)
            
    def random_style(self):
        # Set random values for all parameters to create a new style
        self.width_var.set(random.choice([800, 1000, 1200]))
        self.height_var.set(random.choice([600, 800, 1000]))
        
        self.min_radius_var.set(random.randint(2, 10))
        self.max_radius_var.set(random.randint(self.min_radius_var.get() + 5, 30))
        
        self.dot_count_var.set(random.randint(100, 1000))
        self.pattern_type_var.set(random.choice(['scatter', 'grid', 'circular', 'spiral', 'wave']))
        
        self.randomness_var.set(random.uniform(0.1, 0.9))
        self.density_var.set(random.uniform(0.3, 1.0))
        
        # Random colors
        self.bg_color = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
        self.bg_color_btn.configure(bg=self.bg_color)
        
        self.dot_color = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
        self.dot_color_btn.configure(bg=self.dot_color)
        
        self.outline_enabled.set(random.choice([True, False]))
        if self.outline_enabled.get():
            self.outline_color = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
            self.outline_color_btn.configure(bg=self.outline_color)
            self.outline_width_var.set(random.randint(1, 3))
            
        # Generate the new image
        self.generate_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = DotPatternGenerator(root)
    root.minsize(900, 600)
    root.mainloop()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = DotPatternGenerator(root)
#     root.minsize(900, 600)
#     root.mainloop()