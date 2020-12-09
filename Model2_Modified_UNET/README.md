## Overview of the Implementation

Considering that we are utilising this model on a large factory containing multiple production units, its very important that the proposed technicque is fast, and memory efficient without comprimising too much on the performance. 

Having the UNet architecture in mind, we perform the following modifications:
- To improve the model's ability to capture local and global information, in the initial few layers we utilise a kernels of mulitple sizes which are fused before propagation to the later layers. (residual dilated convolutional block)
- Similarly all the encoder blocks are replaced with convolutional layers which capture information at various receptive fields (3, 5, 7). These are then concatenated and passed onto another layer and combined with a residual connection.
- To ensure we have a lighter model, average pooling is employed instead of strided convolutions in the encoder block.
- To ensure that primitive features like edges and corners play a vital role in the model, dropout is removed in the initial and last layers of the network which also increases performance (experimentally shown here [https://arxiv.org/abs/1511.02680](https://arxiv.org/abs/1511.02680))
- The decoder follows a similar strategy like the encoder and fuses information from various scales. Apart from this we utilise a simple Pixel shuffling operation which avoids the need for tranposed convolutional layers and reduces the number of parameters. Also note skip connections as used in the Unet architecture is followed here.

All these above changes ensured that the model as mentioned in the Unet experiment is made twice as deeper with less than half the number of parameters. However do note that the model's performance is lower than the Unet while being almost 20 times faster than the other experiments.

We train the model with an objective to minimize the lovasz softmax and a weighted BCE loss. All code relevant to this experiment can be found in this directory.

## Results Tabulation
![Results Tabulation](https://github.com/Manoj-152/Steel-Surface-Defect-Detection/blob/main/Model2_Modified_UNET/Results_Tabulation.png)
