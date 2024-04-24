from torch import nn

class DilatedConvBlock(nn.Module):
        def __init__(self,
                        in_channels:int,
                        out_channels:int,
                        kernel_size:int=3,
                        stride:int=1,
                        padding:int=1):
                super().__init__()
                self.diconv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=padding),
                nn.ReLU())

        def forward(self, x):
                return self.diconv(x)

class FCN(nn.Module):
        def __init__(self,
                        n_actions:int,
                        input_shape:int=1,
                        hidden_units:int=64,
                        output_shape:int=1):
                super().__init__()
                self.conv1 = nn.Sequential(
                                nn.Conv2d(in_channels=input_shape,
                                        out_channels=hidden_units,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                                nn.ReLU()
                                )
                self.diconv2 = DilatedConvBlock(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                self.diconv3 = DilatedConvBlock(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                stride=1,
                                                padding=3)
                self.diconv4 = DilatedConvBlock(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                stride=1,
                                                padding=4)
                self.diconv5_pi = DilatedConvBlock(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                stride=1,
                                                padding=3)
                self.diconv6_pi = DilatedConvBlock(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                stride=1,
                                                padding=2)
                self.conv7_pi = nn.Sequential(
                                nn.Conv2d(in_channels=hidden_units,
                                        out_channels=n_actions,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                                nn.Softmax(dim=1)
                                )
                self.diconv5_v = DilatedConvBlock(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                stride=1,
                                                padding=3)
                self.diconv6_v = DilatedConvBlock(in_channels=hidden_units,
                                                out_channels=hidden_units,
                                                kernel_size=3,
                                                stride=1,
                                                padding=2)
                self.conv7_v = nn.Conv2d(in_channels=hidden_units,
                                        out_channels=output_shape,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

                self.train_average_max_reward = 0.0
                self.test_average_max_reward = 0.0

        def forward(self, x):
                h = self.diconv4(self.diconv3(self.diconv2(self.conv1(x))))
                p_out = self.conv7_pi(self.diconv6_pi(self.diconv5_pi(h)))
                v_out = self.conv7_v(self.diconv6_v(self.diconv5_v(h)))

                return p_out, v_out

        def update_train_avg_reward(self, r):
                if r > self.train_average_max_reward:
                        self.train_average_max_reward = r

        def update_test_avg_reward(self, r):
                if r > self.test_average_max_reward:
                        self.test_average_max_reward = r