## Learning Percolation: Scale-Invariant Neural Networks
This repository contains the code for my Master's thesis: "Scale-Invariant Neural Networks for Percolation".

The project implements a physics-informed neural network inspired by the Renormalization Group (RG) to study 2D lattice percolation. 
The model performs iterative coarse-graining to classify percolating vs. non-percolating lattices and to estimate the percolation threshold.

## Key Findings
- Achieves 90–95% accuracy in classifying percolation across various lattice sizes

- Mixed-size training significantly enhances model stability and generalization

- Learned rules converge to sigmoidal RG-like transformations, yielding estimates of p_c ≈ 0.569–0.589

## Repository Structure

Scalar_Percolation/          # Direction of percolation not considered

  Fixed_Size/                # Models trained on fixed lattice sizes
  
    NFC/                     # No First Coarse-graining
    
    AFC/                     # Arithmetic First Coarse-graining
    
    PFC/                     # Percolating First Coarse-graining
    

  Mixed_Size/                # Models trained across multiple lattice sizes
  
    AFC/                     # Best-performing approach
    
      fine_tuning/           # Different sample to parameter ratios
      
      scaling_collapse/      # Critical exponent analysis
      
      learned_curves/        # Visualizations of learned rules
      

Directional_Percolation/     # Models that distinguish percolation directions

  fine_tuning/ 
  
  learned_curves/

## Getting Started
All code is implemented in Jupyter notebooks with inline instructions. To run the notebooks:

- Clone this repository

- Install dependencies: pip install jupyter torch numpy matplotlib

- Launch Jupyter: jupyter notebook

- Open and run the notebooks directly (parameters can be modified in-place)

## Key Notebooks
- Fixed Size Models: NFC/, AFC/, PFC/ directories contain training notebooks for specific lattice sizes

- Mixed Size Models: Mixed_Size/AFC/ contains the best-performing approach

- Visualizations: learned_curves/ directories contain notebooks for rule visualization
