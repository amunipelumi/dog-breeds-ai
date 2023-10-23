import torch
import torchvision

from torch import nn


def create_model(num_classes=1000, seed=13):

  """Creates an EfficientNetB2 feature extractor model and transforms.

  Args:
      num_classes (int, optional): number of classes in the classifier head.
          Defaults to 1000.
      seed (int, optional): random seed value. Defaults to 13.

  Returns:
      model (torch.nn.Module): EffNetB2 feature extractor model.
      transforms (torchvision.transforms): EffNetB2 image transforms.
  """

  weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
  auto_transforms = weights.transforms()
  model = torchvision.models.efficientnet_b3(weights=weights)

  manual_transforms = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(),
    auto_transforms
  ]) 

  for param in model.parameters():
    param.requires_grad = False

  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1536, out_features=num_classes)
  )

  return model, auto_transforms, manual_transforms  
