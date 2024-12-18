import math
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

import torch
import torch.nn as nn  
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Data_Preprocessing:
    """
    This class contains methods to reduce the number of channels in the image
    parameters:
    image: numpy array of shape (122, height, width)
    n_components: number of components to keep after PCA
    """
    def reduce_channels(self, image, n_components=30):
        num_channels, height, width= image.shape
        assert num_channels == 122, "The input image must have 122 channels"

        # Step 1: Flatten the image channels
        flattened_image = image.reshape(-1, num_channels)
        
        # Step 2: Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(flattened_image)
        
        # Step 3: Apply PCA
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(standardized_data)
        
    # Step 4: Reshape the data back to image shape
        reduced_image = reduced_data.reshape(n_components, height, width)

        return reduced_image


#     def normalize(img):
#         """this function normalize the pixels in range (0,1)"""
#         # minv = img.min()
#         maxv = img.max()
#         print(maxv)
#         return img / maxv


# """ Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Unet model implementation
    Parameters:
    n_channels: number of input channels
    n_classes: number of output channels
    bilinear: whether to use bilinear
    
    """
    
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.softmax(logits)
        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class Predict:
    """This class return the predicntions of the UNet model
    parameters:
    test_image_path: path to the test image
    model_path: path to the model
    """
    def __init__(self, test_image_path, model_path):
        self.model_path = model_path
        self.test_image_path = test_image_path

    def read_image(self):
        """This image from the given path (.tif for now)"""
        img = tiff.imread(self.test_image_path)
        return img
    
    def load_model(self):
        """This function loads the model from the given path (.pth for now)"""
        model = UNet(n_channels=30, n_classes=3)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model
    
    def get_patches(self, model, patch_size):
        """
        Divides the image into patches and returns the predictions
        parameters:
        model: the model to use for prediction
        patch_size: the size of the patch
        """
        
        data_prep = Data_Preprocessing()
        x = self.read_image()
        x = data_prep.reduce_channels(x)

        img_height = x.shape[1]
        img_width =  x.shape[2]
        nb_channels = x.shape[0]
        device = "cuda:0"
        
        nb_patches_vertical = math.ceil(img_height / patch_size)
        nb_patches_horizontal = math.ceil(img_width / patch_size)
        extended_height = patch_size * nb_patches_vertical
        extended_width = patch_size * nb_patches_horizontal
        ext_x = np.zeros((nb_channels, extended_height, extended_width), dtype=np.float32)

        ext_x[:, :img_height, :img_width] = x
        for i in range(img_height, extended_height):
            mirror_i = img_height - (i - img_height) % img_height - 1
            ext_x[:, i, :] = ext_x[:, mirror_i, :]

        for j in range(img_width, extended_width):
            mirror_j = img_width - (j - img_width) % img_width - 1
            ext_x[:, :, j] = ext_x[:, :, mirror_j]

        patches_list = []
        
        for i in range(nb_patches_vertical):
            for j in range(nb_patches_horizontal):
                x0, x1 = i * patch_size, (i + 1) * patch_size
                y0, y1 = j * patch_size, (j + 1) * patch_size
                patches_list.append(ext_x[:, x0:x1, y0:y1])

        patches = torch.Tensor(patches_list).to(device)
        # print(patches.shape)
        model.to(device)
        model.eval()
        prediction = model(patches)
    # predicted_class = torch.argmax(prediction)

        return prediction.detach().cpu().numpy()
    
    def picture_from_mask(self, mask, threshold=0):
        """
        returns an RGB image with color-coded classes based on mask
        mask: mask of shape (height, width, nb_classes)
        threshold: threshold for class
        """
        colors = {
            0: [0, 0, 255], 
            1: [255, 0, 0],   
            2: [0, 255, 0],     
        }
        z_order = {
            1: 0,
            2: 1,
            3: 2,
        }

        pict = 255 * np.ones((3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
        for i in range(1, 4):
            cl = z_order[i]
            for ch in range(3):
                pict[ch, :, :][mask[cl, :, :] > threshold] = colors[cl][ch]
        return pict

    def get_color_map(self, mask, threshold):
        """
        returns the color map of the mask
        mask: mask of shape (height, width, nb_classes)
        threshold: threshold for class
        """
        mask = np.abs(mask)
        deficient = np.where(mask[0] > threshold, 1, 0)
        sufficient = np.where(mask[1] > 0.5, 1, 0)
        excessive = np.where(mask[2] > 0.124, 1, 0)
        return deficient, sufficient, excessive
    
    # def predict(self, patch_size, threshold):
    #     model = self.load_model()
    #     prediction = self.get_patches(model, patch_size)
    #     deficient, sufficient, excessive = self.get_color_map(prediction, threshold)
    #     return deficient, sufficient, excessive
    
    def match_predictions(self):
        """
        This function matches the predictions to the original image
        """
        data_prep = Data_Preprocessing()
        model = self.load_model()
        img = self.read_image()
        img = data_prep.reduce_channels(img)
        print(img.shape)
        preds = self.get_patches(model, 224)
        # print(preds_class)
        img_height = img.shape[1]
        img_width =  img.shape[2]
        nb_channels = img.shape[0]

        nb_patches_vertical = math.ceil(img_height / 224)
        nb_patches_horizontal = math.ceil(img_width / 224)
        extended_height = 224 * nb_patches_vertical
        extended_width = 224 * nb_patches_horizontal

        np_predictions = np.zeros((3,extended_height, extended_width), dtype=np.float32)
        for k in range(preds.shape[0]):
            # print(preds[k].shape)
            # count += 1 
            i = k // nb_patches_horizontal  # Corrected: vertical index
            j = k % nb_patches_horizontal   # Corrected: horizontal index
            x0, x1 = i * 224, (i + 1) * 224
            y0, y1 = j * 224, (j + 1) * 224
            # print(x0, x1, y0, y1)
            # print(count)
            # print(np_predictions.shape)
            np_predictions[:, x0:x1, y0:y1] = preds[k]

        final_predictions = np_predictions[:, :img_height, :img_width]
        return final_predictions    
    
    def get_mask(self):
        """
        This function returns the mask of the image
        """
        
        final_predictions = self.match_predictions() 
        # print(final_predictions.shape)
        predictions = np.zeros([final_predictions.shape[0], final_predictions.shape[1], final_predictions.shape[2]], dtype=np.float32)
        
        for i in range(final_predictions.shape[2]):
            for j in range(final_predictions.shape[1]):
                max_value = max(final_predictions[0,j,i], final_predictions[1,j,i], final_predictions[2,j,i])

                if max_value == final_predictions[0,j,i]:
                    predictions[0,j,i] = 1
                    predictions[1,j,i] = 0
                    predictions[2,j,i] = 0

                elif max_value == final_predictions[1,j,i]:
                    predictions[1,j,i] = 1
                    predictions[0,j,i] = 0
                    predictions[2,j,i] = 0   

                elif max_value == final_predictions[2,j,i]:
                    predictions[2,j,i] = 1 
                    predictions[1,j,i] = 0
                    predictions[0,j,i] = 0
        
        return predictions*255

    def plot(self):
        """
        This function plots the mask
        """
        
        predictions = self.get_mask()
        tiff.imshow(255*(predictions).astype('uint8'))

def main():
    predict = Predict("x_test.tif", "unet_model.pth")
    image = predict.get_mask()
    print(image.shape)

if __name__ == "__main__":
    main()