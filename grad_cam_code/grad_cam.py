from torch import nn
import torch
from timm import create_model
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

class GradCAM:
  def __init__(self,model: torch.nn.Module,
               img_path:str,
               layer_idx: int = 2,
               input_shape: tuple = (224,224),
               model_type: str = 'Normal',
               auto_find_classfier: bool = False,
               ):
    """
    params:
    layer_idx: the index of the layer where you want to visulize.
    input_shape: the image shape to put into the model
    model: the model you want to visualize
    img_path: the path of the tested image
    model_type: some special model need addtional method to process the activations
      in order to get Grad-CAM. Currently, there's only function to handel vision
      transformer.
    auto_find_classfier: automatically let gradcam find the classfier head by 'fier', 
      and visualize before this layer.
    """
    self.model = model
    self.layer_idx = layer_idx
    if auto_find_classfier == True:
      for i, child in enumerate(model.children()):
        if "fier" in child.__class__.__name__:
            self.layer_idx = i
            break
    self.img_path = img_path
    self.input_shape = input_shape
    self.model_type = model_type
    print(f'The model types you can select from are either\n \'Normal\' (CNN based), \'ViT\', \'SwinT\', currently is {self.model_type} mode.')

  def __call__(self, heatmap_threshold=8, transform: transforms.Compose=None):
    """
    Args:
        heatmap_threshold (int, optional): Defaults to 8. Must greater than 1, the bigger the value, the less highlights will be displayed. 
    """
    model = self.model

    extractor = nn.Sequential(*list(model.children())[:self.layer_idx]) #truncate the model from where your specified idx
    classifier = nn.Sequential(*list(model.children())[self.layer_idx:])
    #print(extractor)
    print(f'Try to visualize the layer before : \n {list(model.children())[self.layer_idx:]}')

    img = Image.open(self.img_path).convert('RGB')
    if transform == None:
      transform = transforms.Compose([transforms.ToTensor()])
      img = transform(img.resize(self.input_shape))
    else: 
      img = transform(img)
    img = torch.unsqueeze(img, 0) #preprocess the image to tensor: (1,C,H,W)
    img.requires_grad = True

    result = model(img)
    class_Idx = torch.argmax(result)  #get the prediction category
      #the grad-cam will visualize the prediction towards a specific category. (Here is the model's prediction)

    activations = extractor(img)  #get the activations (output features) from target layer
    activations.retain_grad() #pytorch will automatically free the parameters' gradients that are not provided by user
                                #so here it needs to be specified to keep the gradient of activation w.r.t prediction logits

    certainty = nn.functional.softmax(result[0], dim=-1)[int(class_Idx)]
    print(f'Activation Shape:{activations.shape}')

    prediction_logits = classifier(activations) #the activation is fed into rest layers to get the prediction tensor
    print(f'Prediction_logits Shape:{prediction_logits.shape}')
    print(f'The Grad-CAM will be plotted based on model prediction result: {class_Idx} with {certainty*100:.3}% certainty')

    if self.model_type in ['Normal','SwinT']:
      prediction_logits = prediction_logits[:,class_Idx]  #only use the specific class to back propagate
    elif self.model_type == 'ViT':
      prediction_logits = prediction_logits[:,:,class_Idx]

    grad_output = torch.ones_like(prediction_logits) # all gradient will be saved in the specified layer
    prediction_logits.backward(gradient = grad_output)  #according to pytorch, backward() should specify
                                                          #with a tensor which its length is the same as backward tensor
                                                            #when the tensor contains more than one number
    d_act = activations.grad  #get the gradient of activation from target layer w.r.t. the specified category
    if self.model_type == 'Normal':
      if len(d_act.shape) != 4:
        raise ValueError(f"""Input should be (B,C,H,W) dimension, but now only have {d_act.shape} dimension(s).\n This is usually due to the model ends with a pooling layer,\n please increase your output layer index by 1 (or to higher layer)""")
      d_act = d_act.permute(0,2,3,1)  #(1,C,H,W) -> (1,H,W,C)
      activations = activations.permute(0,2,3,1)
    elif self.model_type == 'SwinT':
      pass
    elif self.model_type == 'ViT':
      d_act = self.output_decompose_vit_grad_cam(d_act)
      activations = self.output_decompose_vit_grad_cam(activations[:,:,:])

    print(f'gradient shape (predictioin logti(s) w.r.t. feature logits): {d_act.shape}')
    pooled_grads = torch.mean(d_act,dim = (0,1,2))  #according to the paper, the pooling happens all axis except the channel dim

    heatmap = activations.detach().numpy()[0] #for tensors where its requires_grad = True, need detach() function to convert to ndarray
    pooled_grads = pooled_grads.numpy()

    #back propagate
    for  i in range(d_act.shape[-1]):
      heatmap[:,:,i] *= pooled_grads[i]
    print(f'Shape of weighted Combination between gradients and activations: {heatmap.shape}')
    heatmap = np.sum(heatmap, axis = -1) #here the heatmap shapes as (H,W,1)
    print(f'Shape after channel summation : {heatmap.shape}')

    print(f'Maximum pixel value of heatmap is {heatmap.max()}')
    
    ## Slightly increase the threshold:
    threshold = heatmap.max()/heatmap_threshold
    heatmap[heatmap < threshold] = 0
    #threshold = 0
    self.heatmap = np.uint8(255*heatmap/np.max(heatmap))  #keep the logits that are greater than zero
      #in the paper, that is to say, only keep the positive influence with the specific class.
        #then normalize the heatmap and recale its value range from 0 to 255.


  def origin_cam_visualization(self,save_path:str = None):
    #display the orignal size heatmap (H,W,1)
    plt.rcParams.update({'font.size': 14})
    plt.matshow(self.heatmap)
    plt.title('Original generated heatmap') 
    plt.show()
    if save_path != None:
      plt.savefig(save_path)


  def imposing_visualization(self,save_path:str = None):
    alpha = 0.8 #how much CAM will overlap on original image
    plt.figure(figsize = (20,20))
    plt.rcParams.update({'font.size': 18})

    jet = cm.get_cmap('jet')  #create the color map object
    jet_colors = jet(np.arange(256))[:,:3]
    print(f'jet_color shape: {jet_colors.shape}')
    jet_colors = (jet_colors*256).astype(np.uint8)  #generate a color (RGB) image which has small H and W
                                                      # and maps the intensity to color from red to blue

    ## Use Guided Grad-CAM for visualization by bilinear interpolation
    self.heatmap = Image.fromarray(self.heatmap).resize(self.input_shape, Image.Resampling.BILINEAR)
    jet_heatmap = (jet_colors[np.uint8(self.heatmap)] * alpha).astype(np.uint8)

    img = Image.open(self.img_path).convert('RGB').resize(self.input_shape)

    jet_heatmap = np.asarray(jet_heatmap)

    img_cam = np.asarray(img) + np.asarray(jet_heatmap)

    plt.subplot(2,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.title('Original Image')


    plt.subplot(2,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_cam) #print the overlapped image (origin + cam)
    plt.title('Overlapped Colormap Image')

    plt.subplot(2,2,3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(self.heatmap)
    plt.title('Heatmap (2-D Magnitude)')

    plt.subplot(2,2,4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(jet_heatmap/255) #print the cam
    plt.title('Projected Colormap (3-D)')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.05, hspace=0.1)

    if save_path != None:
      plt.savefig(save_path,bbox_inches = 'tight', pad_inches = 0.3)

      name = save_path.split('.')[0]

      img.save(name+'-origin'+'.png')
      Image.fromarray(img_cam).save(name+'-overlapped'+'.png')
      self.heatmap.save(name+'-heatmap'+'.png')
      Image.fromarray(jet_heatmap).save(name+'-colormap'+'.png')

      

  def output_decompose_vit_grad_cam(self, vit_input):
    #decompose on vit's sequence dimension
    _, HW, C = vit_input.shape
    #print(f"Debug: Embedding size channel is: {HW}")
    HW = int(math.sqrt(HW))
    #print(f"Debug: The unwrap H/W is: {print(HW)}")
    vit_output = torch.reshape(vit_input,(1,HW,HW,C))

    #vit_output = vit_output.permute(0,3,1,2)
    return vit_output
