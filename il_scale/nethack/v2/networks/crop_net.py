
import torch
import torch.nn as nn
import torch.nn.functional as F

from il_scale.nethack.v2.utils.model import _step_to_range

class Crop(nn.Module):
    """Helper class for PolicyNet."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        # NOTE: Need to do -self.width/2 + 1/2 here to cancel things out correctly
        # with the width_grid below for both even and odd input dimensions.
        x_shift = 2 / (self.width - 1) * (x.float() - self.width/2 + 1/2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height/2 + 1/2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # NOTE: Location x, y in grid tells you the shift from the cursor 
        # coordinates. The reason we do all this 2/(self.width - 1) stuff is because
        # the inverse of this happens in the below F.grid_sample function. The F.grid_sample
        # implementation is here: https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c#L41

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )