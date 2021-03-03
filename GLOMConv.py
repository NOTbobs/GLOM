
#will stop if kernel step/stride is > width/height of image. 
def conv_ind_strides (image_dim, kernel,strides=[1,1]): 
  imW,imH,imD=image_dim
  sW,sH=strides
  kw,kh = kernel
  all_values = np.array(range(0,imW*imH)).reshape(imW,imH)
  ind=[] 
  hselect=- sH
  wselect=0 
  while hselect+kh<=imH-kh: 
    hselect+=sH
    wselect=0
    while wselect<=imW-kw: 
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

