from torch import nn
import torch
from timm import create_model
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import pdb

import torch
from collections import defaultdict


class GradCAM:
  def __init__(self,model: torch.nn.Module,
               img_path:str = None,
               img_value = None,
               layer_name: str = None,
               input_shape: tuple = (224,224),
               model_type: str = 'Normal',
               transform: transforms.Compose=None,
               verbose: bool = False
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
    _hooked = False
    self.hook = {'act':[], 'grad':[]}
    for n, p in model.named_modules():
        if n == layer_name:
            p.register_forward_hook(self._hook_act)
            p.register_full_backward_hook(self._hook_grad)
            _hooked = True
    if _hooked == False:
       raise ValueError("Please give a full valid layer name, e.g.: blocks.10.drop_path2. You can use function print_layername to check it.")
    self.model = model
    self.img_path = img_path
    self.im_value = img_value
    self.input_shape = input_shape
    self.model_type = model_type
    self.transform = transform
    self.verbose = verbose
    if self.verbose:
      print(f'The model types you can select from are either\n \'Normal\' (CNN based), \'ViT\', \'SwinT\', currently is {self.model_type} mode.')

  def __call__(self, heatmap_threshold=8,):
    """
    Args:
        heatmap_threshold (int, optional): Defaults to 8. Must greater than 1, the bigger the value, the less highlights will be displayed. 
    """
    model = self.model

    if self.im_value != None:
      img = self.im_value
    else:
      img = Image.open(self.img_path).convert('RGB')
      
    if self.transform == None:
      self.transform = transforms.Compose([transforms.ToTensor()])
      img = self.transform(img.resize(self.input_shape))
    else: 
      if self.verbose:
        print('Use the custom transform you provided to preprocess the image')
      img = self.transform(img)
    img = torch.unsqueeze(img, 0) #preprocess the image to tensor: (1,C,H,W)
    self.img = img
    img.requires_grad = True  # del
    # ---------------------------------------------------------------------------------------
    result = model(img)
    class_Idx = torch.argmax(result)  #get the prediction category
      #the grad-cam will visualize the prediction towards a specific category. (Here is the model's prediction)
    # ------------------------------------------------------------------------------------------
    prediction_logits = result[0]
    certainty = nn.functional.softmax(prediction_logits, dim=-1)[int(class_Idx)]
    if self.verbose:
      print(f'Output logits shape: {result.shape}')
      print(f'The Grad-CAM will be plotted based on model prediction result: {class_Idx} with {certainty*100:.3}% certainty')

    activations = self.hook['act'][0]
    if self.verbose:
      print(f'Activation Shape:{activations.shape}')
    
    grad_output = torch.ones_like(prediction_logits) # all gradient will be saved in the specified layer
    prediction_logits.backward(gradient = grad_output)  #according to pytorch, backward() should specify
                                                          #with a tensor which its length is the same as backward tensor
                                                          #when the tensor contains more than one number, in conclusion:
                                                          #gradient argument = dL/d(output_logit)
    d_act = self.hook['grad'][0][0]
    if self.model_type == 'Normal':
      if len(d_act.shape) != 4:
        raise ValueError(f"""Input should be (B,C,H,W) dimension, but now only have {d_act.shape} dimension(s).\n This is usually due to the model ends with a pooling layer,\n please increase your output layer index by 1 (or to higher layer)""")
      d_act = d_act.permute(0,2,3,1)  #(1,C,H,W) -> (1,H,W,C)
      activations = activations.permute(0,2,3,1)
    elif self.model_type == 'SwinT':
      d_act = self.output_decompose_vit_grad_cam(d_act[:,:,:])
      activations = self.output_decompose_vit_grad_cam(activations[:,:,:])
    elif self.model_type in ['ViT', 'SwinT']:
      d_act = self.output_decompose_vit_grad_cam(d_act[:,1:,:])
      activations = self.output_decompose_vit_grad_cam(activations[:,1:,:])

    if self.verbose:
      print(f'gradient shape (predictioin logti(s) w.r.t. feature logits): {d_act.shape}')
    pooled_grads = torch.mean(d_act,dim = (0,1,2))  #according to the paper, the pooling happens all axis except the channel dim

    heatmap = activations.detach().numpy()[0] #for tensors where its requires_grad = True, need detach() function to convert to ndarray
    pooled_grads = pooled_grads.numpy()

    # \alpha_k^c * A^k, k is i here: 
    for i in range(d_act.shape[-1]):
      heatmap[:,:,i] *= pooled_grads[i]
    if self.verbose:
      print(f'Shape of weighted Combination between gradients and activations: {heatmap.shape}')
    heatmap = np.sum(heatmap, axis = -1) #here the heatmap shapes as (H,W,1)
    if self.verbose:
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


  def imposing_visualization(self,save_path:str = None, denormalize:tuple[int, int] =None):
    alpha = 0.8 #how much CAM will overlap on original image
    plt.figure(figsize = (20,20))
    plt.rcParams.update({'font.size': 18})

    jet = cm.get_cmap('jet')  #create the color map object
    jet_colors = jet(np.arange(256))[:,:3]
    if self.verbose:
      print(f'jet_color shape: {jet_colors.shape}')
    jet_colors = (jet_colors*256).astype(np.uint8)  #generate a color (RGB) image which has small H and W
                                                      # and maps the intensity to color from red to blue

    ## Use Guided Grad-CAM for visualization by bilinear interpolation
    self.heatmap = Image.fromarray(self.heatmap).resize(self.input_shape, Image.Resampling.BILINEAR)
    jet_heatmap = (jet_colors[np.uint8(self.heatmap)] * alpha).astype(np.uint8)

    #if self.im_value != None:
    #  img = self.im_value
    #else:
    #  img = Image.open(self.img_path).convert('RGB').resize(self.input_shape)
    img = self.img.squeeze(0).permute(1,2,0).detach().numpy()  #(1,C,H,W) -> (H,W,C)
    if denormalize != None:
      img = self.denormalize(img, mean=denormalize[0], std=denormalize[1])
    else:
      img = (img - img.min()) / (img.max() - img.min()) * 255
    jet_heatmap = np.asarray(jet_heatmap)

    img_cam = np.asarray(img) + np.asarray(jet_heatmap)
    #print(img_cam)

    plt.subplot(2,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img/255)
    plt.title('Original Image')


    plt.subplot(2,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_cam/(300)) #print the overlapped image (origin + cam)
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
      pil_img = Image.fromarray(img.astype(np.uint8), 'RGB')
      pil_img.save(name+'-original'+'.png')
      #pdb.set_trace()
      Image.fromarray(img_cam.astype(np.uint8)).save(name+'-overlapped'+'.png')
      self.heatmap.save(name+'-heatmap'+'.png')
      Image.fromarray(jet_heatmap.astype(np.uint8)).save(name+'-colormap'+'.png')
      

  def output_decompose_vit_grad_cam(self, vit_input):
    #decompose on vit's sequence dimension
    _, HW, C = vit_input.shape
    #print(f"Debug: Embedding size channel is: {HW}")
    HW = int(math.sqrt(HW))
    #print(f"Debug: The unwrap H/W is: {print(HW)}")
    vit_output = torch.reshape(vit_input,(1,HW,HW,C))

    #vit_output = vit_output.permute(0,3,1,2)
    return vit_output

  def denormalize(self, np_array, mean, std):
      #denormalize the image for visualization
      denormalized = np_array * np.array(std) + np.array(mean)
      denormalized = np.clip(denormalized *255, 0, 255)
      return denormalized
    
  def _hook_act(self, module, input, output):
      self.hook['act'].append(output) # Detach and move to CPU to avoid memory issues
          
  def _hook_grad(self, module, input, output):
      self.hook['grad'].append(output)
    
def print_layername(model: nn.ModuleDict):
    for n, p in model.named_modules():
        print(n)