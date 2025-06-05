import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QPushButton, QSizePolicy, QComboBox, 
                             QGroupBox, QRadioButton, QButtonGroup, QCheckBox)
from PyQt5.QtCore import Qt

class FractalPercolationExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Percolation Explorer")
        self.setGeometry(100, 100, 1200, 1000)
        
        # Parameters
        self.grid_sizes = {
            "128×128": 128,
            "256×256": 256,
            "512×512": 512,
            "1024×1024": 1024
        }
        self.L = 256  # Default grid size
        self.p = 0.5927  # Critical probability
        self.grid = None
        self.labels = None
        self.largest_cluster = None
        self.largest_cluster_size = 0
        self.percolating_cluster = None
        self.percolating_cluster_size = 0
        self.visualization_mode = "largest"  # "largest" or "percolating"
        self.show_fractal_box = False
        self.box_size = 64  # Size of fractal zoom box
        
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=(10, 10), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(f"Percolation Fractals (L={self.L}, p={self.p:.4f})", fontsize=16)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        
        # Create control panels
        control_layout = QHBoxLayout()
        
        # Left control panel (grid size and buttons)
        left_control = QVBoxLayout()
        
        # Grid size selector
        size_group = QGroupBox("Grid Settings")
        size_layout = QVBoxLayout()
        
        grid_size_layout = QHBoxLayout()
        size_label = QLabel("Grid Size:")
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(self.grid_sizes.keys()))
        self.size_combo.setCurrentText("256×256")
        self.size_combo.currentTextChanged.connect(self.change_grid_size)
        grid_size_layout.addWidget(size_label)
        grid_size_layout.addWidget(self.size_combo)
        
        # Fractal zoom box
        box_layout = QHBoxLayout()
        box_label = QLabel("Fractal Box Size:")
        self.box_size_combo = QComboBox()
        self.box_size_combo.addItems(["32", "64", "128"])
        self.box_size_combo.setCurrentText("64")
        self.box_size_combo.currentTextChanged.connect(self.update_box_size)
        box_layout.addWidget(box_label)
        box_layout.addWidget(self.box_size_combo)
        
        self.show_box_check = QCheckBox("Show Fractal Zoom Box")
        self.show_box_check.setChecked(False)
        self.show_box_check.stateChanged.connect(self.toggle_fractal_box)
        
        size_layout.addLayout(grid_size_layout)
        size_layout.addLayout(box_layout)
        size_layout.addWidget(self.show_box_check)
        size_group.setLayout(size_layout)
        
        # Buttons
        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout()
        self.regenerate_button = QPushButton("Generate New Configuration")
        self.regenerate_button.clicked.connect(self.generate_grid)
        self.regenerate_button.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        
        self.save_button = QPushButton("Save as PDF")
        self.save_button.clicked.connect(self.save_pdf)
        self.save_button.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        
        button_layout.addWidget(self.regenerate_button)
        button_layout.addWidget(self.save_button)
        button_group.setLayout(button_layout)
        
        left_control.addWidget(size_group)
        left_control.addWidget(button_group)
        left_control.addStretch()
        
        # Center control panel (visualization options)
        center_control = QVBoxLayout()
        vis_group = QGroupBox("Visualization Mode")
        vis_layout = QVBoxLayout()
        
        self.mode_group = QButtonGroup()
        self.largest_radio = QRadioButton("Show Largest Cluster")
        self.largest_radio.setChecked(True)
        self.percolating_radio = QRadioButton("Show Percolating Cluster")
        self.all_radio = QRadioButton("Show All Clusters (Grayscale)")
        
        self.mode_group.addButton(self.largest_radio)
        self.mode_group.addButton(self.percolating_radio)
        self.mode_group.addButton(self.all_radio)
        
        self.largest_radio.toggled.connect(lambda: self.set_visualization_mode("largest"))
        self.percolating_radio.toggled.connect(lambda: self.set_visualization_mode("percolating"))
        self.all_radio.toggled.connect(lambda: self.set_visualization_mode("all"))
        
        vis_layout.addWidget(self.largest_radio)
        vis_layout.addWidget(self.percolating_radio)
        vis_layout.addWidget(self.all_radio)
        vis_group.setLayout(vis_layout)
        
        # Fractal information
        info_group = QGroupBox("Fractal Properties")
        info_layout = QVBoxLayout()
        self.fractal_info = QLabel("Fractal dimension: N/A")
        self.fractal_info.setStyleSheet("font-style: italic;")
        info_layout.addWidget(self.fractal_info)
        info_group.setLayout(info_layout)
        
        center_control.addWidget(vis_group)
        center_control.addWidget(info_group)
        center_control.addStretch()
        
        # Right control panel (p slider)
        right_control = QVBoxLayout()
        p_group = QGroupBox("Probability Control")
        p_layout = QVBoxLayout()
        
        # Slider for p value
        slider_label = QLabel("Occupation Probability (p):")
        slider_label.setStyleSheet("font-weight: bold;")
        self.slider_value = QLabel(f"{self.p:.4f}")
        self.slider_value.setStyleSheet("font-size: 14px; font-weight: bold; color: #d32f2f;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(int(self.p * 1000))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.valueChanged.connect(self.update_p)
        
        # Critical p indicator
        p_c_label = QLabel("Critical Probability: p_c ≈ 0.5927")
        p_c_label.setStyleSheet("font-weight: bold; color: #1976d2;")
        
        # Tip for fractal exploration
        tip_label = QLabel("Tip: For best fractal visualization,\nset p = p_c ≈ 0.5927 and use large grid sizes")
        tip_label.setStyleSheet("font-style: italic; color: #388e3c;")
        
        p_layout.addWidget(slider_label)
        p_layout.addWidget(self.slider_value)
        p_layout.addWidget(self.slider)
        p_layout.addWidget(p_c_label)
        p_layout.addWidget(tip_label)
        p_layout.addStretch()
        p_group.setLayout(p_layout)
        
        right_control.addWidget(p_group)
        right_control.addStretch()
        
        # Add to control layout
        control_layout.addLayout(left_control, 25)
        control_layout.addLayout(center_control, 25)
        control_layout.addLayout(right_control, 50)
        
        # Add to main layout
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.canvas, 100)
        main_layout.addLayout(control_layout)
        
        # Generate initial grid
        self.generate_grid()

    def change_grid_size(self, size_text):
        """Update grid size when selection changes"""
        self.L = self.grid_sizes[size_text]
        self.generate_grid()

    def update_p(self, value):
        """Update the probability value when slider is moved"""
        self.p = value / 1000.0
        self.slider_value.setText(f"{self.p:.4f}")
        self.generate_grid()

    def set_visualization_mode(self, mode):
        """Set visualization mode"""
        self.visualization_mode = mode
        self.visualize_grid()

    def update_box_size(self, size_text):
        """Update fractal box size"""
        self.box_size = int(size_text)
        self.visualize_grid()

    def toggle_fractal_box(self, state):
        """Toggle fractal zoom box display"""
        self.show_fractal_box = (state == Qt.Checked)
        self.visualize_grid()

    def generate_grid(self):
        """Generate a new grid with the current p value and size"""
        # Create random grid
        self.grid = np.random.random((self.L, self.L)) < self.p
        
        # Find clusters
        self.find_clusters()
        
        # Check for percolation
        self.find_percolating_cluster()
        
        # Visualize the grid
        self.visualize_grid()

    def find_clusters(self):
        """Find clusters using iterative DFS"""
        # Initialize variables
        self.labels = np.zeros((self.L, self.L), dtype=int)
        visited = np.zeros((self.L, self.L), dtype=bool)
        cluster_sizes = {}
        current_label = 1
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 4-connectivity
        
        # Find all clusters
        for i in range(self.L):
            for j in range(self.L):
                if not self.grid[i, j] or visited[i, j]:
                    continue
                    
                # Perform DFS for this cluster
                stack = [(i, j)]
                visited[i, j] = True
                cluster_points = []
                
                while stack:
                    x, y = stack.pop()
                    cluster_points.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.L and 0 <= ny < self.L and 
                            self.grid[nx, ny] and not visited[nx, ny]):
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                
                # Assign label to this cluster
                size = len(cluster_points)
                for (x, y) in cluster_points:
                    self.labels[x, y] = current_label
                cluster_sizes[current_label] = size
                current_label += 1
        
        # Find largest cluster
        self.largest_cluster = None
        self.largest_cluster_size = 0
        if cluster_sizes:
            self.largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
            self.largest_cluster_size = cluster_sizes[self.largest_cluster]

    def find_percolating_cluster(self):
        """Find if any cluster percolates (spans from top to bottom)"""
        self.percolating_cluster = None
        self.percolating_cluster_size = 0
        
        # Find clusters that touch the top row
        top_clusters = set(self.labels[0, self.grid[0, :]])
        top_clusters.discard(0)  # Remove background
        
        # Find clusters that touch the bottom row
        bottom_clusters = set(self.labels[-1, self.grid[-1, :]])
        bottom_clusters.discard(0)
        
        # Find clusters that appear in both (span top to bottom)
        spanning_clusters = top_clusters & bottom_clusters
        
        if spanning_clusters:
            # Find the largest spanning cluster
            spanning_sizes = {}
            for i in range(self.L):
                for j in range(self.L):
                    label = self.labels[i, j]
                    if label in spanning_clusters:
                        spanning_sizes[label] = spanning_sizes.get(label, 0) + 1
            
            if spanning_sizes:
                self.percolating_cluster = max(spanning_sizes, key=spanning_sizes.get)
                self.percolating_cluster_size = spanning_sizes[self.percolating_cluster]
                return
        
        # If no spanning cluster, check for left-right percolation
        left_clusters = set(self.labels[self.grid[:, 0], 0])
        left_clusters.discard(0)
        
        right_clusters = set(self.labels[self.grid[:, -1], -1])
        right_clusters.discard(0)
        
        spanning_clusters = left_clusters & right_clusters
        
        if spanning_clusters:
            spanning_sizes = {}
            for i in range(self.L):
                for j in range(self.L):
                    label = self.labels[i, j]
                    if label in spanning_clusters:
                        spanning_sizes[label] = spanning_sizes.get(label, 0) + 1
            
            if spanning_sizes:
                self.percolating_cluster = max(spanning_sizes, key=spanning_sizes.get)
                self.percolating_cluster_size = spanning_sizes[self.percolating_cluster]

    def visualize_grid(self):
        """Create visualization of the grid with fractal properties"""
        # Create RGB image
        image = np.ones((self.L, self.L, 3))  # Start with white background
        
        # Set occupied sites to black by default
        occupied_mask = self.grid.copy()
        image[occupied_mask] = [0, 0, 0]
        
        # Apply visualization mode
        if self.visualization_mode == "largest" and self.largest_cluster is not None and self.largest_cluster_size > 0:
            # Color largest cluster red
            largest_mask = (self.labels == self.largest_cluster)
            image[largest_mask] = [1, 0, 0]
            
            # Calculate fractal dimension (for display only - actual calculation would be more rigorous)
            if self.p > 0.5 and self.p < 0.7:
                # Approximate fractal dimension for percolation cluster is about 91/48 ≈ 1.896
                fractal_dim = 91/48
                self.fractal_info.setText(f"Fractal dimension: ≈{fractal_dim:.3f} (theoretical 91/48)")
            else:
                self.fractal_info.setText("Fractal dimension: N/A (only near p_c ≈ 0.5927)")
        
        elif self.visualization_mode == "percolating" and self.percolating_cluster is not None:
            # Color percolating cluster blue
            percolating_mask = (self.labels == self.percolating_cluster)
            image[percolating_mask] = [0, 0, 1]
            
            # Calculate fractal dimension
            fractal_dim = 91/48
            self.fractal_info.setText(f"Fractal dimension: ≈{fractal_dim:.3f} (theoretical 91/48)")
        
        elif self.visualization_mode == "all":
            # Grayscale visualization based on cluster size
            unique_labels = np.unique(self.labels)
            cluster_sizes = {label: np.sum(self.labels == label) for label in unique_labels if label != 0}
            
            for label in unique_labels:
                if label == 0:  # Skip background
                    continue
                mask = (self.labels == label)
                size = cluster_sizes[label]
                
                # Normalize size to [0, 1] for grayscale
                # Use log scale to better see size differences
                if size > 0:
                    gray_value = np.log(size) / np.log(self.L * self.L)
                else:
                    gray_value = 0
                
                # Apply grayscale value
                image[mask] = [gray_value, gray_value, gray_value]
            
            self.fractal_info.setText("Grayscale by cluster size (darker = larger)")
        
        # Update the plot
        self.ax.clear()
        self.ax.imshow(image, interpolation='none', origin='lower', aspect='equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add fractal zoom box if enabled
        if self.show_fractal_box and self.L > self.box_size:
            # Center the box on the largest cluster if available
            if self.visualization_mode == "largest" and self.largest_cluster_size > 10:
                # Find a point in the largest cluster
                for i in range(self.L):
                    for j in range(self.L):
                        if self.labels[i, j] == self.largest_cluster:
                            start_i = max(0, min(i - self.box_size//2, self.L - self.box_size))
                            start_j = max(0, min(j - self.box_size//2, self.L - self.box_size))
                            break
                    else:
                        continue
                    break
            else:
                # Center the box
                start_i = (self.L - self.box_size) // 2
                start_j = (self.L - self.box_size) // 2
            
            # Draw rectangle
            rect = matplotlib.patches.Rectangle(
                (start_j - 0.5, start_i - 0.5), 
                self.box_size, self.box_size,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            self.ax.add_patch(rect)
            
            # Add inset for zoomed fractal view
            if self.visualization_mode != "all":  # Don't add for grayscale
                ax_inset = self.ax.inset_axes([0.65, 0.65, 0.3, 0.3])
                zoomed_view = image[start_i:start_i+self.box_size, start_j:start_j+self.box_size]
                ax_inset.imshow(zoomed_view, interpolation='none', origin='lower')
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                ax_inset.set_title(f"Zoomed {self.box_size}×{self.box_size}", fontsize=10)
        
        # Add informative title
        title = f"Percolation Fractals (L={self.L}, p={self.p:.4f})"
        
        if self.visualization_mode == "largest" and self.largest_cluster_size > 0:
            fraction = self.largest_cluster_size / (self.L * self.L)
            title += f"\nLargest Cluster: {self.largest_cluster_size} sites ({fraction:.2%})"
        elif self.visualization_mode == "percolating" and self.percolating_cluster_size > 0:
            fraction = self.percolating_cluster_size / (self.L * self.L)
            title += f"\nPercolating Cluster: {self.percolating_cluster_size} sites ({fraction:.2%})"
        elif self.visualization_mode == "percolating" and self.percolating_cluster_size == 0:
            title += "\nNo percolating cluster found"
        elif self.visualization_mode == "all":
            title += "\nAll clusters (grayscale by size)"
        
        self.ax.set_title(title, fontsize=14)
        self.canvas.draw()

    def save_pdf(self):
        """Save the current visualization as PDF"""
        self.fig.savefig(f"percolation_fractal_L{self.L}_p{self.p:.4f}.pdf", 
                         bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FractalPercolationExplorer()
    window.show()
    sys.exit(app.exec_())