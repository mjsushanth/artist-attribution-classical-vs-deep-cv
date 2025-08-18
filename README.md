# Artist-Classification - Project-Paper
This repo is for showcasing a project paper from the coursework 'Foundations of AI'.

This project investigates multiple machine learning paradigms for artist classification using the "Best Artworks of All Time" dataset (8,774 images across 50 renowned artists). While comparing traditional approaches like SVMs with SIFT descriptors and deep CNNs (ResNet-50, VGG-16), the primary research contribution centers on extensive Convolutional Autoencoder (CAE) experimentation and advanced unsupervised learning methodologies.


## Research Methodology 

### Advanced Preprocessing & Data Engineering

The foundation of this research involved developing sophisticated preprocessing pipelines that go beyond standard image resizing:
- **Custom Aspect-Ratio Preservation Algorithms**: Implemented three distinct padding methodologies to maintain artistic composition integrity while standardizing input dimensions
- **Adaptive Resizing Strategies**: Developed algorithms that calculate optimal padding for width/height centering, resize while maintaining aspect ratios, and perform intelligent center cropping
- **Domain-Specific Normalization**: Computed dataset-specific mean and standard deviation statistics alongside ImageNet comparisons for optimal feature representation

### Convolutional Autoencoder Architecture Exploration

The core research contribution lies in systematic CAE experimentation, representing the most technically intensive component of this work:

#### CAE 128 Architecture
- **Graduated Compression Scheme**: Progressive dimensionality reduction from 128×128 input to 4×4 latent representation
- **Channel Expansion Strategy**: Systematic increase from 32 to 256 channels through encoder layers
- **Strategic Batch Normalization**, **Dense Layer Integrations** (using final convolution mapping directly to latent space)

#### Compact CAE 224 Architecture  
- **High-Dimensional Feature Learning**: Channel expansion from 3 to 1024 across encoder layers
- **Latent Space Optimization**: 512-dimensional compressed representation through convolutional latent layers
- **Symmetric Decoder Design**: ConvTranspose2d layers with progressive upsampling and channel reduction

### Advanced Clustering & Unsupervised Analysis

Post-autoencoder feature extraction involved comprehensive clustering methodology exploration:

#### Multi-Algorithm Clustering Framework
- **K-Means Implementation**: Systematic cluster count optimization (5-10 clusters) with silhouette score analysis
- **Gaussian Mixture Model (GMM)**: Probabilistic clustering with covariance structure flexibility for artistic style modeling
- **Hierarchical Clustering**: Agglomerative clustering with linkage criterion optimization and dendrogram analysis

#### Feature Analysis & Dimensionality Reduction
- **UMAP Visualization**: High-dimensional feature space projection for cluster separability assessment
- **Silhouette Score Optimization**: Systematic evaluation across cluster counts (2-10) revealing optimal clustering configurations
- **Feature Map Activation Analysis**: Deep investigation of convolutional layer activations to understand representational learning patterns

### Experimental Designs

#### Architectural Experimentation
- **Multi-Scale CAE Training**: Experiments across 128×128 and 224×224 input resolutions
- **Loss Function Analysis**: Reconstruction loss minimization with detailed convergence pattern analysis
#### Class Imbalance Handling
- **SMOTE Implementation**: Synthetic minority oversampling for balanced representation
- **WeightedRandomSampler**: PyTorch-based inverse frequency weighting for training stability
- **Stratified Sampling**: Careful train/validation splits maintaining artist distribution integrity

### Feature Engineering & Representation Learning

#### Custom Feature Extraction
- **Latent Space Engineering**: 512-dimensional feature vectors optimized for downstream clustering
- **Feature Map Visualization**: Activation intensity analysis across convolutional layers (0.0-1.6 range)
- **Spatial Hierarchy Analysis**: Investigation of how CAEs capture and compress spatial artistic features

#### Advanced Visualization Framework
- **Confusion Matrix Analysis**: Detailed classification performance assessment across artist classes
- **Reconstruction Quality Assessment**: Visual and quantitative analysis of autoencoder reconstruction fidelity
- **Cluster Visualization**: Multi-dimensional plotting of learned feature representations and cluster boundaries

## Key Research Findings

Despite achieving modest classification accuracy (20-31% across clustering methods), this research revealed critical insights into unsupervised learning limitations for fine art classification:

- **Feature Compression Challenges**: CAEs struggled to encode discriminative artistic features into latent representations, often learning trivial solutions that minimized reconstruction loss without capturing meaningful artistic patterns
- **Spatial Detail Loss**: Autoencoder architectures inherently prioritized prominent, common features over subtle artistic nuances critical for artist differentiation
- **Clustering Separability Issues**: UMAP projections revealed significant feature overlap between artist classes, indicating insufficient discriminative power in learned representations

The most significant contribution lies not in achieving state-of-the-art accuracy, but in the exploration of why unsupervised deep learning approaches face fundamental limitations in artistic classification tasks. This work provided us insights into the trade-offs between reconstruction fidelity and discriminative feature learning.
