from grad_cam import *

model = create_model('timm/resnet34.a1_in1k', pretrained=True)
#model = create_model('timm/resnet14t.c3_in1k', pretrained=True)
#model = create_model('timm/resnet10t.c3_in1k', pretrained=True)

#img_path = '/content/drive/MyDrive/ImageNet1k-Test-images/test5-pomeranian.png'
img_path = '/content/drive/MyDrive/ImageNet1k-Test-images/test3-hornbill.png'
#img_path = '/content/drive/MyDrive/ImageNet1k-Test-images/test2-pug-dog.png'
#img_path = '/content/drive/MyDrive/ImageNet1k-Test-images/test1-n01443537_goldfish.png'

cam = GradCAM(model,img_path,layer_idx=-2)
cam()
cam.imposing_visualization()
cam.heatmap
