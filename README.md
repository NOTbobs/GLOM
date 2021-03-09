# GLOM
Attempt at implementing GLOM


## Bottom-Up implementation: 
## Functions: 
### conv_ind_strides (image_dim, kernel,strides) :
- generates indices to segment an image by kernel shape with a defined stride 
- example: segment a 600x600x3 image into 50x50 kernels with stride [25,25]: conv_ind_strides([600,600,3],[50,50][25,25]) 
- Returns matrix containing indices to perform convolution of shape [number_of strides, kernelW,kernelH]
### bottom_convolution(image,indices,parameters): 
 - generates a given shaped embedding for each image segment by performing a dot product on that patch with given parameters. 
 - Parameters should be shaped [desired embedding size, kernelwidth * kernelheight * image depth] 
 - Returns matrix shaped [segments,desired embedding size]
### BT_convolution(embeddings,indices,parameters): 
 - generates a given shaped embedding after convolving previous layer embeddings. 
 - Parameters should be shaped [desired_embedding_size, kerelwidth * kernelheight * previous_layer_embedding_size] 
 - Returns matrix shaped [number of segments, embedding_size]
### trace_convolution(interested_embedding,ind,starting_layer,total_layers): 
 - generates indices at the level of the original image representing a stride or embedding that you are interested in
TODO: 
- [ ] Top-Down activation. 
- [X] Clustering of top layer embeddings 
- [X] Clustering of intermediate layer embeddings. 
- [X ] Marking pixels for segmentation.

