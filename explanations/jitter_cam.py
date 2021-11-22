from .cam import Cam
import torch
import cv2
import numpy as np
import torch.nn.functional as F

class JitterCam(Cam):

    def get_activation_size(self):
        is_inception = False
        if self.model_name == 'resnet50':
            model_blocks = 7
            block_size = 32

        elif self.model_name == 'densenet':
            model_blocks = 7
            block_size = 32
        elif self.model_name == 'shufflenet':
            model_blocks = 7
            block_size = 32

        if self.model_name == 'mobilenet':
            model_blocks = 7
            block_size = 32

        elif self.model_name == 'vgg16':
            model_blocks = 14
            block_size = 16

        elif self.model_name == 'inception':
            # print(model)
            is_inception = True
            model_blocks = 8
            block_size = 37
        elif self.model_name == 'vgg16':
            model_blocks = 14
            block_size = 16

        return model_blocks, block_size

    def make_gcam(self, input_patch, target_class):
        self.activations = []
        self.gradients = []

        # print('input_patch.shape', input_patch.shape)
        if input_patch.get_device() == -1:
            input_patch = input_patch.cuda()
        x = self.model(input_patch)
        # print(target_class)
        # print(x.shape, x.view(-1).shape, x.view(-1)[target_class].item())
        # print(x.view(-1))
        # exit()


        # one_hot = np.zeros((x.size()[0], x.size()[-1]), dtype=np.float32)
        # one_hot[:, target_class] = 1
        # one_hot = torch.from_numpy(one_hot).cuda()#.to('cuda:1')#
        # one_hot = torch.sum(one_hot * x)

        if len(x.size())==4:
            one_hot = torch.zeros((x.size()[0], x.size()[1])).cuda()
            one_hot[:, target_class] = 1
            one_hot = torch.sum(one_hot.view(one_hot.size()[0],-1,1,1) * x)
        else:
            one_hot = torch.zeros((x.size()[0], x.size()[-1])).cuda()
            one_hot[:, target_class] = 1
            # one_hot = torch.from_numpy(one_hot).cuda()#.to('cuda:1')#
            one_hot = torch.sum(one_hot * x)


        self.model.zero_grad()


        # one_hot.backward(retain_graph=True)
        one_hot.backward()
        # Take the first gradient calculated
        grads_val = self.gradients[0]#.cpu().data.numpy()
        # Take the last activation calculated
        acts_val = self.activations[-1]
        acts_val = F.relu(acts_val)

        # print(len(self.activations))
        # print(acts_val.shape)
        # exit()



        # tmp = grads_val
        # if len(np.array(tmp).shape) == 4:
        # print(grads_val.shape)
        weights = torch.mean(grads_val, axis=(2, 3))

        activations = acts_val.detach()#.cpu().data.numpy()

        #
        # weights = weights.squeeze(axis=0)
        # activations = activations.squeeze(axis=0)

        # if len(np.array(tmp).shape) == 4:
        # print(weights)
        # print(activations)
            # print('input_patch.size()',  self.input_image.size()[3], self.input_image.size()[2])
        cam = self.create_cam(weights, activations)
        return cam

    def create_cam(self, weights, activationMaps):

        cam = torch.zeros((activationMaps.shape[0],
                        activationMaps.shape[2],
                        activationMaps.shape[3],
                        )).cuda()


        for im_id in range(cam.size()[0]):
            cam[im_id, ...] = torch.sum(weights[im_id].view(-1,1,1) * activationMaps[im_id, ...], 0)

        cam = F.relu(cam)

        return cam.cpu().numpy()

    def set_new_size(self, new_size):
        self.new_size = new_size



    def __call__(self, input_image, target_class):
        self.input_image = input_image

        jitter_stride = 1
        batch_size = 32
        smoothing = True

        model_blocks, block_size = self.get_activation_size()
        image_size = self.input_image.shape[-1]
        # print('image size', image_size)

        # print(image_size)
        number_of_patches = ((self.new_size - model_blocks)+1)**2
        # print('number_of_patches:', number_of_patches)
        image_patches = torch.empty((number_of_patches, 3, image_size, image_size)).cuda()

        x = np.zeros((self.new_size, self.new_size))
        counter = np.ones((self.new_size, self.new_size))
        resize_amount = (self.new_size - model_blocks) * block_size

        resized_input = F.interpolate(input_image, image_size+resize_amount).cuda()


        patch_count = 0
        for i in range(0, (self.new_size - model_blocks)+1, jitter_stride):
            for j in range(0, (self.new_size - model_blocks)+1, jitter_stride):
                image_patches[patch_count, ...] = resized_input[:,:,j*block_size:image_size+j*block_size,i*block_size:image_size+i*block_size]
                patch_count += 1


        cam_patches = np.zeros((number_of_patches, model_blocks, model_blocks))

        for batch_i in range(0, number_of_patches, batch_size):
            pixel_scores = self.make_gcam(image_patches[batch_i:batch_i+batch_size, ...], target_class)
            cam_patches[batch_i:batch_i+batch_size, ...] = pixel_scores



        patch_count = 0
        for i in range(0, (self.new_size - model_blocks)+1, jitter_stride):
            for j in range(0, (self.new_size - model_blocks)+1, jitter_stride):
                x[j:model_blocks+j, i:model_blocks+i] += cam_patches[patch_count]
                counter[j:model_blocks+j, i:model_blocks+i] += 1.0

                patch_count += 1


        pixel_scores = x / counter



        # if smoothing:


        pixel_scores = cv2.resize(pixel_scores, self.output_size)

        # else:
        #     pixel_scores = cv2.resize(pixel_scores, (input.shape[2],input.shape[3]), interpolation=cv2.INTER_NEAREST)

        return pixel_scores


