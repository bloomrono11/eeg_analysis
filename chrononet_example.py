import torch.nn as nn
import torch
import chrononet as chr


def gru_layer(out):
    gru_layer1 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)
    x = out.permute(0, 2, 1)
    print(f'Permuted x.shape {x.shape}')
    gru_out1, hh_out = gru_layer1(x)
    print(f'Shape of gru out 1: {gru_out1.shape}, hidden_out: {hh_out.shape}')

    # Feed layer one to layer 2 directly
    gru_layer2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
    gru_out2, _ = gru_layer2(gru_out1)
    print(f'Shape of gru out 2: {gru_out2.shape}')

    # Feed joined layer one & layer 2 via tensor.cat to layer 3
    gru_layer3 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
    gru_cat_out_1_2 = torch.cat((gru_out1, gru_out2), 2)
    print(f'Shape of gru out cat of 1 & 2: {gru_cat_out_1_2.shape}')
    gru_out3, _ = gru_layer3(gru_cat_out_1_2)
    print(f'Shape of gru out 3: {gru_out3.shape}')

    # Feed joined layer 1 & layer 2 & layer 3 via tensor.cat to layer 4 but use a linear/dense function in between
    gru_layer4 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)
    gru_cat_out_1_2_3 = torch.cat((gru_out1, gru_out2, gru_out3), 2)
    print(f'Shape of gru out cat of 1,2,3 : {gru_cat_out_1_2_3.shape}')

    # need to pass to a linear layer to reduce the last 1875 to 1
    linear = nn.Linear(1875, 1)
    linear_out = linear(gru_cat_out_1_2_3.permute(0, 2, 1))
    linear_out = nn.ReLU()(linear_out)
    print(f'Shape of linear out shape: {linear_out.shape}')

    # Need to reduce the 1875 from the concatenated from previous layer for gru 4
    gru_out4, _ = gru_layer4(linear_out.permute(0, 2, 1))
    print(f'Shape of gru out 4: {gru_out4.shape}')

    # Use activation function similar to softmax
    flatten = nn.Flatten()
    flatten_out = flatten(gru_out4)
    fcl = nn.Linear(32, 1)
    fcl_out = fcl(flatten_out)
    return fcl_out


def block_example(init_out):
    block1 = chr.Block(22)
    block1_out = block1(init_out)
    print(block1_out.shape)
    # 96 - channel size, 7500 is sequence length and 3 is batch size
    # torch.Size([3, 96, 7500]) # reverse of Keras

    block2 = chr.Block(96)
    block2_out = block2(block1_out)
    print(block2_out.shape)

    block3 = chr.Block(96)
    block3_out = block3(block2_out)
    print(block3_out.shape)


if __name__ == '__main__':
    init_out = torch.randn(3, 22, 15000)
    block_example(init_out)
