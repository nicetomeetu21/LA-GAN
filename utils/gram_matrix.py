import torch
#Initializing gram_matrix function for our tensor image 
def gram_matrix(tensor):
   #Unwrapping the tensor dimensions into respective variables i.e. batch size, distance, height and width 
  _, d, h, w=tensor.size() 
  #Reshaping data into a two dimensional of array or two dimensional of tensor
  tensor=tensor.view(d, h*w)
  #Multiplying the original tensor with its own transpose using torch.mm 
  #tensor.t() will return the transpose of original tensor
  gram=torch.mm(tensor, tensor.t())
  #Returning gram matrix 
  return gram