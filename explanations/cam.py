import torch
import cv2
import numpy as np
import torch.nn.functional as F

class Cam():
    def __init__(self, model, model_name, output_size=None):
        self.model_name = model_name
        self.model = model

        self.model.eval()
        self.output_size = output_size





        self.gradients = []
        self.activations = []

        if self.model_name == 'alexnet':
            self.model.features[11].register_forward_hook(self.save_activation)
            self.model.features[11].register_backward_hook(self.save_gradient)

        # elif (self.model_name == 'squeezenet'):
        #     self.model.classifier[2].register_forward_hook(self.save_activation)
        #     self.model.classifier[2].register_backward_hook(self.save_gradient)

        elif self.model_name == 'vgg16':
            self.model.features[29].register_forward_hook(self.save_activation)
            self.model.features[29].register_backward_hook(self.save_gradient)
        elif self.model_name == 'vgg16bn':
            self.model.features[42].register_forward_hook(self.save_activation)
            self.model.features[42].register_backward_hook(self.save_gradient)

        elif self.model_name == 'inception':
            self.model._modules.get('Mixed_7c').register_forward_hook(self.save_activation)
            self.model._modules.get('Mixed_7c').register_backward_hook(self.save_gradient)

        elif self.model_name == 'densenet':
            self.model._modules.get('features').register_forward_hook(self.save_activation)
            self.model._modules.get('features').register_backward_hook(self.save_gradient)

        elif self.model_name == 'shufflenet':
            self.model._modules.get('conv5').register_forward_hook(self.save_activation)
            self.model._modules.get('conv5').register_backward_hook(self.save_gradient)

        elif self.model_name == 'mobilenet':
            self.model._modules.get('features').register_forward_hook(self.save_activation)
            self.model._modules.get('features').register_backward_hook(self.save_gradient)


        elif (self.model_name == 'resnet50'):

            self.model.layer4.register_forward_hook(self.save_activation)
            self.model.layer4.register_backward_hook(self.save_gradient)


        elif (self.model_name == 'resnet50_conv'):
            self.model.layer4[2].conv3.register_forward_hook(self.save_activation)
            self.model.layer4[2].conv3.register_backward_hook(self.save_gradient)


    def save_gradient(self, module, input, output):
        self.gradients.append(output[0].detach())

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def create_cam(self, weights, activationMaps):

        # print('weights.shape', weights.shape, 'activationMaps', activationMaps.shape)
        # exit()

        cam = torch.zeros((activationMaps.shape[0],
                        activationMaps.shape[2],
                        activationMaps.shape[3],
                        )).cuda()



        # for im_id in range(cam.size()[0]):
        #     for i, w in enumerate(weights[im_id, ...]):
        #         cam[im_id, ...] += w * activationMaps[im_id, i, ...]
        # print(cam[63,...])

        for im_id in range(cam.size()[0]):
            # pass
            # print(weights[im_id].shape, activationMaps[im_id].shape, weights[im_id].view(-1,1,1).shape)
            cam[im_id, ...] = torch.sum(weights[im_id].view(-1,1,1) * activationMaps[im_id, ...], 0)
        # print(cam[im_id, ...])
        # exit()
        #     import matplotlib.pyplot as plt
        #     plt.imshow(cam[im_id].cpu().numpy())
        #     plt.show()

        # ReLU
        cam = F.relu(cam)
        # cam = np.maximum(cam, 0)
        # print(cam.shape)
        if self.output_size is not None:
            if len(cam.shape) == 3:
                # cam = cv2.resize(cam, self.output_size)
                cam = cv2.resize(cam.cpu().numpy().squeeze(), self.output_size)
                return cam

        # cam = np.expand_dims(cam, axis=0)
        return cam.cpu().numpy()

    def __call__(self, input_image, target_class):
        print('define call function')
        exit()

