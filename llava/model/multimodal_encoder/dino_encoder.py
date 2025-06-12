import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel, AutoConfig


class DinoVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args

        if not delay_load:
            self.load_model()
        # else:
        self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        if not hasattr(self.args, "image_size"):
            if self.vision_tower_name.split('/')[-1] == 'dinov2-large':
                pass
            elif self.vision_tower_name.split('/')[-1] == 'dino-vitb16':
                size = self.image_processor.size
                setattr(self.image_processor, 'crop_size', {'width': size, 'height': size})
            else:
                print("not supported vision tower: {}".format(self.vision_tower_name))

        else:
            setattr(self.image_processor, 'size', {'width': self.args.image_size, 'height': self.args.image_size})
            setattr(self.image_processor, 'crop_size', {'width': self.args.image_size, 'height': self.args.image_size})

        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, use_safetensors=False)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.cfg_only

    # @property
    # def hidden_size(self):
    #     return self.vision_tower.num_features


    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2