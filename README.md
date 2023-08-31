# Image-Processing
This repository contains an implementation of the GrabCut and Poisson Blending algorithms.

# Grab-Cut
An implementation of the GrabCut algorithm - an algorithm for extracting an object from the background.
GrabCut is an iterative image segmentation algorithm that aims to separate foreground object and background in an image. It starts by user selecting an initial bounding box around the object of interest. Then, it iteratively refines the segmentation by modeling the color distribution of foreground and background pixels using  Gaussians mixture model, optimizing a graph-cut energy function that incorporates both color and spatial information, resulting in an accurate segmentation mask.

# Poisson-Blending
The Poisson blending algorithm is a technique used to seamlessly merge two images while maintaining smooth transitions. It involves solving a partial differential equation that enforces a gradient constraint between the source and target images at the blending boundary. By enforcing the Laplacian of the blended image to match that of the source, Poisson blending achieves natural and artifact-free image composition.

