# 用于深度学习算法的训练，输出结果到models
# 使用config文件夹下面的det_config.yaml文件作为训练参数

import os
from torchvison import transforms
from torchvison import models
import yaml
import torch