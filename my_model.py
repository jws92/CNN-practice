import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

torch.manual_seed(777)
torch.cuda.manual_seed(777)

class Layers(nn.Module):
  def __init__(self, in_channels, layers, norm_type):
    super().__init__()

    self.in_channels = in_channels
    self.norm_type = norm_type
    self.num_layers = layers

    self.layer_module = nn.ModuleList()
    self.cat_conv_module = nn.ModuleList()

    for i in range(self.num_layers):
      layer = nn.Sequential(
          nn.Conv2d(self.in_channels, self.in_channels//2, 
                    kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(self.in_channels//2) if self.norm_type == 'batch' else nn.GroupNorm(32, self.in_channels//2),
          nn.GELU(),

          nn.Conv2d(self.in_channels//2, self.in_channels*2, 
                    kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(self.in_channels*2) if self.norm_type == 'batch' else nn.GroupNorm(32, self.in_channels*2),
          nn.GELU(),

          nn.Conv2d(self.in_channels*2, self.in_channels, 
                    kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(self.in_channels) if self.norm_type == 'batch' else nn.GroupNorm(32, self.in_channels),
          nn.GELU(),
      )
      self.layer_module.append(layer)

      if i != self.num_layers - 1:
        cat_conv = nn.Sequential(
          nn.Conv2d(self.in_channels*2, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(self.in_channels) if self.norm_type == 'batch' else nn.GroupNorm(32, self.in_channels),
          nn.GELU()
        )
        self.cat_conv_module.append(cat_conv)

    self.fit_ch_conv = nn.Sequential(
      nn.Conv2d(self.in_channels*2, self.in_channels*2,
                kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(self.in_channels*2) if self.norm_type == 'batch' else nn.GroupNorm(32, self.in_channels*2),
      nn.GELU()
    )


  def forward(self, x):
    res_x = x

    for i, layer in enumerate(self.layer_module):
      feats = layer(x)
      feats += x
      x = torch.cat((feats, res_x), dim=1)

      if i != len(self.layer_module)-1:
        x = self.cat_conv_module[i](x)
      else:
        x = self.fit_ch_conv(x)

    return x
    

class myCNN(nn.Module):
  def __init__(self, in_channels=64, num_classes=10, layers_per_stage=[2, 2, 8, 2], norm_type='batch'):
      super().__init__()

      self.in_channels = in_channels
      self.num_classes = num_classes
      self.num_stages = len(layers_per_stage)
      self.norm_type = norm_type
      self.layers_per_stage = layers_per_stage
      
      assert self.norm_type == 'batch' or self.norm_type == 'group', "No support normalization in the model."

      self.stem_layer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(self.in_channels) if self.norm_type == "batch" else nn.GroupNorm(32, self.in_channels),
        nn.GELU(),
      )

      self.stages = nn.ModuleList()
      for i in range(self.num_stages):
          layers = Layers(self.in_channels, self.layers_per_stage[i], norm_type)
          self.stages.append(layers)

          self.in_channels *= 2

      self.classifier = nn.Sequential(
        nn.Linear(self.in_channels, 256, bias=True),
        nn.GELU(),
        nn.Linear(256, self.num_classes, bias=True),
      )

      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.constant_(m.bias, 0)

  def forward(self, x):
      x = F.max_pool2d(self.stem_layer(x), 2, 2)

      for i, stage in enumerate(self.stages):
          x = stage(x)

          if i != len(self.stages)-1:
              x = F.max_pool2d(x, 2, 2)
          else:
              x = F.adaptive_avg_pool2d(x, 1)

      x = x.view(x.size(0), -1)
      x = self.classifier(x)

      return x


if __name__ == "__main__":
  model = myCNN(in_channels=64, num_classes=10, layers_per_stage=[2, 2, 8, 2], norm_type='group')

  summary(model, input_size=(3, 32, 32), batch_size=1, device='cpu')

