from imports import *
from Unet3d import UNet3D

class TargetNet(nn.Module):
    def __init__(self, base_model, n_classes=3):
        super(TargetNet, self).__init__()
        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_classes, bias=True)

    def forward(self, x):
        
        #self.base_model(x)
        #self.base_out = self.base_model.out512
        #self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0], -1)

        base_out = self.base_model(x)  # Capture the return value from base_model directly
        self.out_glb_avg_pool = F.avg_pool3d(base_out, kernel_size=base_out.size()[2:]).view(base_out.size()[0], -1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        probabilities = self.dense_2(F.relu(self.linear_out))
        #probabilities = F.softmax(final_out, dim=1)

        return probabilities