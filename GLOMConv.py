
#will stop if kernel step/stride is > width/height of image. 
def conv_ind_strides (image_dim, kernel,strides=[1,1]): 
  imW,imH,imD=image_dim
  sW,sH=strides
  kw,kh = kernel
  all_values = np.array(range(0,imW*imH)).reshape(imW,imH)
  ind=[] 
  hselect=- sH
  wselect=0 
  while hselect+kh+sH<=imH: 
    hselect+=sH
    wselect=0
    while wselect+kw<=imW:: 
      ind.append(all_values[hselect:hselect+kh,wselect:wselect+kw])
      wselect+=sW
  return ind
      
      
def bottom_convolution(image,indices,parameters): 
  h,w,c=image.shape
  image=image.reshape(-1,c)
  split=image[indices,:] #generates [segments, kernelh, kernelw,channels] 
  split=split.reshape(len(split),-1) #flatten to (segments,kernelh*kernelw*channels)
  #for each stride matmul with parameters of shape [kernelw*kernel*h*channels]
  #parameter should be shape of [next layer,kernelw*kernelh*channels] ,
  #when mat mul with split will result in matrix of shape [strides,kernelh,kernelw,channels]

  #Can be substituted for any sized network. 
  convolution = np.dot(split,parameters.T) 

  return convolution

#def BT_convolution

#takes in a shaped series of embeddings (12,12,1,10). 
#embeddings have to be shaped (embeddings,embedding_size). 

def BT_convolution(embeddings,indices,parameters): #Take some number of adjacent embeddings, and concatenate. 
  _,kh,kw=indices.shape
  aggregate=embeddings[indices,:].reshape(len(indices),kh*kw,-1)  #outputs (secctions, kernelw,kernelh,embedding size)
  aggregate = aggregate.reshape(len(indices),-1) #you want sections, embedding_size*kernelw*kernelh, this 'concatenates all the 9 embeddings together. 
  #parameters, shaped (third layer,concatenated embedding) 
  output=np.dot(aggregate,parameters.T) 
  return output


#final code: 
def trace_convolution(interested_stride,ind,starting_layer,total_layers): 
  #interested_stride #top level, interested in the third stride. 
  #layer=
  #total_layers #top level  (0 is bottom, 2 is top) 
  #total_layers=2
  #Top layer
  current_layer=ind[starting_layer][interested_stride].reshape(len(ind[starting_layer][interested_stride]),-1).flatten() #pulls the stride of interest.  # generates 9 indices. This will start
  #the entire process. 

  for layer in range(total_layers-1,-1,-1): #iterate backwards 
    next_layer=[] #store the next_layer indices give the above layer indices. 
    for indices in current_layer: 
      next_layer.append(ind[layer][indices]) #gather the indices, this will output a 9,10,10 shape as there were 9 blocks, with 10x10 indices. 
    #once this is done, reshape into a list of indices. 
    
    next_layer=np.array(next_layer).flatten() 
    current_layer=next_layer #hand off new indices as currnt layer, and allow the next layer to iterate. 
  return current_layer


