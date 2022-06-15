import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm



class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        # TODO
        self.num_features=num_features
        self.embed_features=embed_features
        
        self.map_gamma=torch.nn.Linear(self.embed_features,self.num_features)
        self.map_bias=torch.nn.Linear(self.embed_features,self.num_features)

        self.batch_norm1=torch.nn.BatchNorm2d(self.num_features,affine=False)
        

    def forward(self, inputs, embeds):
        
        gamma = self.map_gamma(embeds) # TODO
        bias = self.map_bias(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = self.batch_norm1(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.embed_channels=embed_channels
        self.batchnorm=batchnorm
        self.upsample=upsample
        self.downsample=downsample
        
        if self.batchnorm == False:
            self.bn1 = None
            self.bn2 = None
        else:
            self.bn1 = AdaptiveBatchNorm(self.in_channels,self.embed_channels)
            self.bn2 = AdaptiveBatchNorm(self.out_channels,self.embed_channels)
        
        self.activation0 = nn.ReLU(inplace=True)
        self.activation1 = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channels))=ъ/эхё
                                      
        
        
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
         # TODO
        if self.upsample == True:
            inputs=torch.nn.Upsample((inputs.shape[2],inputs.shape[3]),mode='nearest')(inputs)

        
        if self.batchnorm == False:
            outputs = inputs
        else :
            outputs=self.bn1(inputs,embeds)
        
        outputs = self.activation(outputs)
        outputs = self.conv1(outputs)
        outputs=torch.nn.utils.spectral_norm(outputs)
        
        if self.batchnorm == False:
            outputs = outputs
        else :
            outputs=self.bn2(outputs,embeds)
        outputs = self.bn2(inputs)
        outputs = self.activation(outputs)
        outputs = self.conv2(outputs)
        outputs=torch.nn.utils.spectral_norm(outputs)
        
        outputs = outputs + self.skip(x)
        outputs = self.activation(outputs)
        
        if self.downsample == True:
            outputs=nn.AvgPool2d(2)(outputs)
        
        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO
        self.min_channels=min_channels
        self.max_channels=max_channels
        self.noise_channels=noise_channels

        self.num_classes=num_classes
        self.num_blocks=num_blocks
        self.use_class_condition=use_class_condition
        
        if self.use_class_condition == True:
            self.condition_noise=2*noise_channels
        else:
            self.condition_noise=noise_channels
        
        self.class_embedding=nn.Embedding(self.num_classes,self.noise_channels)
        self.mapping=nn.Linear(self.condition_noise,self.max_channels * 4 * 4)
        
        self.blocks=nn.ModuleList([])
        for i in range(0,self.num_down_blocks):
          self.blocks.append(PreActResBlock(self.max_channels/(2**i),
                 out_channels=self.max_channels/(2**(i+1)),
                 embed_channels=self.condition_noise,
                 batchnorm=self.use_class_condition,
                 upsample = True)
        self.final_block=nn.Sequential(
                        torch.nn.BatchNorm2d(self.max_channels/(2**(self.num_down_blocks))),
                        torch.nn.ReLU(),
                        torch.nn.utils.spectral_norm(nn.Conv2d(self.min_channels),3,3,1)))
        self.sigmoid_=torch.nn.Sigmoid()
    def forward(self, noise, labels):
        # TODO
        if self.use_class_condition = True:
            labels_embed=self.class_embedding(labels)
            noise=torch.cat([noise,labels_embed],dim=-1)
        x=self.mapping(noise)
        x=torch.nn.utils.spectral_norm(x)
        x=x.view(-1, 64, 4, 4)
        for i in range(0,self.num_down_blocks):
            outputs=self.blocks[i](x,noise)
        outputs=self.final_block(outputs)
        outputs=self.sigmoid_(outputs)
    
        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO
        self.min_channels=min_channels
        self.max_channels=max_channels
        self.num_classes=num_classes
        self.num_blocks=num_blocks
        self.use_projection_head=use_projection_head
        self.first_block=nn.Sequential(
                        torch.nn.utils.spectral_norm(nn.Conv2d(3,self.min_channels,3,1)),
                        torch.nn.ReLU(),
                        torch.nn.BatchNorm2d(self.min_channels))
                        
                        
                        
                        
        self.down_blocks=nn.ModuleList([])
        for i in range(0,self.num_down_blocks):
          self.down_blocks.append(PreActResBlock(self.min_channels*(2**i),
                 out_channels=self.min_channels*(2**(i+1)),
                 downsample = True)
        self.activation=torch.nn.ReLU()
        self.psi=torch.nn.utils.spectral_norm(nn.Linear(self.max_channels, 1))
        self.embed = torch.nn.utils.spectral_norm(nn.Embedding(self.num_classes, self.max_channels))

    def forward(self, inputs, labels):
        # TODO
        inputs=self.first_block(inputs)
        for i in range(0,self.num_down_blocks):
            inputs=self.down_blocks[i](inputs)
        inputs=self.activation(inputs)
        inputs=torch.sum(inputs,dim=(2,3))
        
        if self.use_projection_head == True:
            scores = self.psi(inputs)+ torch.sum(self.embed(labels) * inputs, dim=1, keepdim=True)
        else:
            scores = self.psi(inputs) # TODO

        assert scores.shape = (inputs.shape[0],)
        return scores
