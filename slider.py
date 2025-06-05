import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QPushButton, QSizePolicy, QComboBox)
from PyQt5.QtCore import Qt

class PercolationVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Site Percolation Explorer")
        self.setGeometry(100, 100, 1000, 900)
        
        # Parameters
        self.grid_sizes = {
            "64×64": 64,
            "128×128": 128,
            "256×256": 256
        }
        self.L = 128  # Default grid size
        self.p = 0.5927  # Initial p (critical point for square lattice)
        self.grid = None
        self.labels = None
        self.largest_cluster = None
        self.largest_cluster_size = 0
        
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(f"Site Percolation (L={self.L}, p={self.p:.4f})", fontsize=14)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        
        # Create controls
        control_layout = QHBoxLayout()
        
        # Left control panel (grid size and buttons)
        left_control = QVBoxLayout()
        
        # Grid size selector
        size_layout = QHBoxLayout()
        size_label = QLabel("Grid Size:")
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(self.grid_sizes.keys()))
        self.size_combo.setCurrentText("128×128")
        self.size_combo.currentTextChanged.connect(self.change_grid_size)
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_combo)
        
        # Buttons
        self.regenerate_button = QPushButton("Regenerate Grid")
        self.regenerate_button.clicked.connect(self.generate_grid)
        self.regenerate_button.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        
        self.save_button = QPushButton("Save as PDF")
        self.save_button.clicked.connect(self.save_pdf)
        self.save_button.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        
        left_control.addLayout(size_layout)
        left_control.addWidget(self.regenerate_button)
        left_control.addWidget(self.save_button)
        left_control.addStretch()
        
        # Right control panel (p slider)
        right_control = QVBoxLayout()
        
        # Slider for p value
        slider_label = QLabel("Occupation Probability (p):")
        slider_label.setStyleSheet("font-weight: bold;")
        self.slider_value = QLabel(f"{self.p:.4f}")
        self.slider_value.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(int(self.p * 1000))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.valueChanged.connect(self.update_p)
        
        # Critical p indicator
        p_c_label = QLabel(f"Critical p ≈ 0.5927")
        p_c_label.setStyleSheet("font-style: italic; color: #666;")
        
        right_control.addWidget(slider_label)
        right_control.addWidget(self.slider_value)
        right_control.addWidget(self.slider)
        right_control.addWidget(p_c_label)
        right_control.addStretch()
        
        # Add to control layout
        control_layout.addLayout(left_control, 30)
        control_layout.addLayout(right_control, 70)
        
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

    def generate_grid(self):
        """Generate a new grid with the current p value and size"""
        # Create random grid
        self.grid = np.random.random((self.L, self.L)) < self.p
        
        # Find clusters using corrected algorithm
        self.find_clusters()
        
        # Visualize the grid
        self.visualize_grid()

    def find_clusters(self):
        """Find clusters using a corrected DFS approach"""
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

    def visualize_grid(self):
        """Create visualization of the grid with largest cluster in red"""
        # Create RGB image: white=empty, black=occupied, red=largest cluster
        image = np.ones((self.L, self.L, 3))  # Start with white background
        
        # Set occupied sites to black
        occupied_mask = self.grid.copy()
        image[occupied_mask] = [0, 0, 0]
        
        # Color largest cluster red if exists
        if self.largest_cluster is not None and self.largest_cluster_size > 0:
            largest_mask = (self.labels == self.largest_cluster)
            image[largest_mask] = [1, 0, 0]
        
        # Update the plot
        self.ax.clear()
        self.ax.imshow(image, interpolation='none', origin='lower')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add informative title
        title = f"Site Percolation (L={self.L}, p={self.p:.4f})"
        if self.largest_cluster_size > 0:
            fraction = self.largest_cluster_size / (self.L * self.L)
            title += f"\nLargest Cluster: {self.largest_cluster_size} sites ({fraction:.2%})"
        else:
            title += "\nNo occupied sites"
            
        self.ax.set_title(title, fontsize=14)
        self.canvas.draw()

    def save_pdf(self):
        """Save the current visualization as PDF"""
        self.fig.savefig(f"percolation_L{self.L}_p{self.p:.4f}.pdf", 
                         bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PercolationVisualizer()
    window.show()
    sys.exit(app.exec_())