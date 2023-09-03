# Underwater Camera: Improving Visual Perception Via Adaptive Dark Pixel Prior and Color Correction

This repository contains our python implementation of the IJCV 2023 paper, ***Improving Visual Perception Via Adaptive Dark
Pixel Prior and Color Correction***. If you use any code or data from our work, please cite our paper.

@article{zhou2023underwater,
  title={Underwater camera: improving visual perception via adaptive dark pixel prior and color correction},
  author={Zhou, Jingchun and Liu, Qian and Jiang, Qiuping and Ren, Wenqi and Lam, Kin-Man and Zhang, Weishi},
  journal={International Journal of Computer Vision},
  pages={1--19},
  year={2023},
  publisher={Springer}
}

## Code

### 1. Usage
	
	(1). We provide scripts that make it easier to test data. The following are the steps:
	(2). Download code and comile.
		You need to install the dependencies in require.txt.
	(3). Download dataset to "InputImages" folder.
	(4). Change "file=." to corresponding path.
	(5). Run project.


### 2 Notice:
* If you get an error using the eng.niqe() function, configure the Python call matlab environment and create the niqe.m file that uses the matlab built-in function niqe().
* If the output is incorrect, try using the colorCheck() function.
	
You can find results in folder "OutputImages".

  




## Introduction
We present a novel method for underwater image restoration, which combines a Comprehensive Imaging Formation Model with 
prior knowledge and unsupervised techniques. Our approach has two main components: depth map estimation using a
Channel Intensity Prior (CIP) and backscatter elimination through Adaptive Dark Pixels (ADP). The CIP effectively mitigates
issues caused by solid-colored objects and highlighted regions in underwater scenarios. The ADP, utilizing a dynamic depth
conversion, addresses issues associated with narrow depth ranges and backscatter. Furthermore, an unsupervised method is
employed to enhance the accuracy of monocular depth estimation and reduce artificial illumination influence. The final output
is refined via color compensation and a blue-green channel color balance factor, delivering artifact-free images. Experimental
results show that our approach outperforms state-of-the-art methods, demonstrating its efficacy in dealing with uneven lighting
and diverse underwater environments.
