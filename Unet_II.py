import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import vgg19
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def crop(image,new_shape):
    middle_height = image.shape[2]//2
    middle_width = image.shape[3]//2
    starting_height = middle_height-round(new_shape[2]/2)
    final_height = starting_height+new_shape[2]
    starting_width = middle_width-round(new_shape[3]/2)
    final_width = starting_width+new_shape[3]
    cropped_image = image[:,:,starting_height:final_height,starting_width:final_width]
    return cropped_image


class ContractingBlock(nn.Module):
    def __init__(self,input_channels,use_in=True,use_dropout=False):
        super(ContractingBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,input_channels*2,kernel_size=3,padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels*2)
        self.use_in = use_in
        if use_dropout:
            self.drop = nn.Dropout()
        self.use_dropout = use_dropout
    
    def forward(self,x):
        x = self.conv(x)
        if self.use_in:
            x = self.insnorm(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

    
class ExpandingBlock(nn.Module):
    def __init__(self,input_channels,use_in=True):
        super(ExpandingBlock,self).__init__()
        self.tconv = nn.ConvTranspose2d(input_channels,input_channels//2,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv2 = nn.Conv2d(input_channels,input_channels//2,kernel_size=3,padding=1)
        self.activation = nn.LeakyReLU(0.2)
        if use_in:
            self.insnorm = nn.InstanceNorm2d(input_channels//2)
        self.use_in = use_in
        
    def forward(self,x,skip_x):
        x = self.tconv(x)
        skip_x = crop(skip_x,x.shape)
        x = torch.cat([x,skip_x],axis=1)
        x = self.conv2(x)
        if self.use_in:
            x = self.insnorm(x)
        x = self.activation(x)
        return x
    
    
class FeatureMapBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(FeatureMapBlock,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels,kernel_size=1)
        
    def forward(self,x):
        x = self.conv(x)
        return x
    
    
class SE_Block(nn.Module):
    def __init__(self,channels,reduction=16):
        super(SE_Block,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels,channels//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction,channels,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b,c)
        y = self.excitation(y).view(b,c,1,1)
        return x * y.expand_as(x)
    
    
class AtrousConv(nn.Module):
    def __init__(self,input_channels):
        super(AtrousConv,self).__init__()
        self.aconv2 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=2,padding=2)
        self.aconv4 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=4,padding=4)
        self.aconv8 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=8,padding=8)
        self.aconv16 = nn.Conv2d(input_channels,input_channels,kernel_size=3,stride=1,dilation=16,padding=16)
        self.batchnorm = nn.BatchNorm2d(input_channels)
        self.activation = nn.ReLU()
        
    def forward(self,x):
        x = self.aconv2(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.aconv4(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.aconv8(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        x = self.aconv16(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        
        return x
    

class UNet_II(nn.Module):
    def __init__(self,input_channels,output_channels,hidden_channels=16):
        super(UNet_II,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False,use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels*2,use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels*4,use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels*8)
        # self.contract5 = ContractingBlock(hidden_channels*16)
        
        self.atrous_conv = AtrousConv(hidden_channels*16)
        
        # self.expand0 = ExpandingBlock(hidden_channels*32)
        self.expand1 = ExpandingBlock(hidden_channels*16)
        self.expand2 = ExpandingBlock(hidden_channels*8)
        self.expand3 = ExpandingBlock(hidden_channels*4)
        self.expand4 = ExpandingBlock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels,output_channels)
        
        self.se1 = SE_Block(hidden_channels*2)
        self.se2 = SE_Block(hidden_channels*4)
        self.se3 = SE_Block(hidden_channels*8)
        
        self.tanh = torch.nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x1 = self.se1(x1)
        x2 = self.contract2(x1)
        x2 = self.se2(x2)
        x3 = self.contract3(x2)
        x3 = self.se3(x3)
        x4 = self.contract4(x3)
        # x5 = self.contract5(x4)
        x5 = self.atrous_conv(x4)
        # x6 = self.expand0(x5,x4)
        x7 = self.expand1(x5,x3)
        x8 = self.expand2(x7,x2)
        x9 = self.expand3(x8,x1)
        x10 = self.expand4(x9,x0)
        xn = self.downfeature(x10)
        
        return self.tanh(xn)


class Discriminator_whole(nn.Module):
    def __init__(self,input_channels,hidden_channels=4):
        super(Discriminator_whole,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16,1,kernel_size=1)  
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
        
    def forward(self,x,y):
        x = torch.cat([x,y],axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn

class Discriminator_mask(nn.Module):
    def __init__(self,input_channels,hidden_channels=4):
        super(Discriminator_mask,self).__init__()
        self.upfeature = FeatureMapBlock(input_channels,hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16,1,kernel_size=1) 
        self.dropout = nn.Dropout()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
        
    def forward(self,x,y): 
        x = torch.cat([x,y],axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x2 = self.dropout(x2)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


class PerceptualNet(nn.Module):
    def __init__(self, name = "vgg19", resize=True):
        super(PerceptualNet, self).__init__()
        blocks = []
        blocks.append(vgg19(pretrained=True).features[:4].eval())
        blocks.append(vgg19(pretrained=True).features[4:9].eval())
        blocks.append(vgg19(pretrained=True).features[9:16].eval())
        blocks.append(vgg19(pretrained=True).features[16:23].eval())
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device)
        self.resize = resize
    
    def forward(self, inputs, targets):
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
        inputs = (inputs+1)/2
        targets = (targets+1)/2
        if self.resize:
            inputs = self.transform(inputs, mode='bilinear', size=(224, 224), align_corners=False)
            targets = self.transform(targets, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = inputs
        y = targets
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def inpaint_unet(masked,binary):
    transform = transforms.Compose([
            transforms.ToTensor()
            ])

    input_dim = 6
    output_dim = 3
    disc_dim = 9
    lr = 0.0003
    device = 'cpu'

    gen = UNet(input_dim,output_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(),lr=lr)
    disc_whole = Discriminator_whole(disc_dim).to(device)
    disc_whole_opt = torch.optim.Adam(disc_whole.parameters(),lr=0.0001)
    disc_mask = Discriminator_mask(disc_dim).to(device)
    disc_mask_opt = torch.optim.Adam(disc_mask.parameters(),lr=0.0001)

    model_path = "models/Inpaint_UNet.pth"
    loaded_state = torch.load(model_path,map_location=torch.device('cpu'))  
    gen.load_state_dict(loaded_state["gen"])
    gen_opt.load_state_dict(loaded_state["gen_opt"])
    disc_whole.load_state_dict(loaded_state["disc_whole"])
    disc_whole_opt.load_state_dict(loaded_state["disc_whole_opt"])
    disc_mask.load_state_dict(loaded_state["disc_mask"])
    disc_mask_opt.load_state_dict(loaded_state["disc_mask_opt"])

    masked = transform(masked)
    masked = masked.detach().cpu().view(-1,*(masked.shape))
    binary = np.stack((binary,)*3, axis=-1)
    binary = transform(binary)
    binary = binary.detach().cpu().view(-1,*(binary.shape))
    masked=(masked-0.5)*2
    binary=(binary-0.5)*2   
    input_img = torch.cat((masked,binary),1)

    prediction = gen(input_img)
    prediction = (prediction + 1) / 2
    prediction = make_grid(prediction[:1], nrow=5)
    prediction = prediction.permute(1, 2, 0).squeeze()
    prediction = np.array(prediction.detach())

    return prediction
