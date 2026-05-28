# Note
You are not allowed to use the 3D ResNet (in this task specifically) available in PyTorch or its weights.

# Task
Implement the 3D equivalent of the ResNet-18 backbone.

Compare two initialization strategies:
1. Inflating 2D weights from the equivalent ResNet-18 model (as in Carreira & Zisserman, 2017)
2. Random initialization (default PyTorch)

To prevent overfitting, we need augmentation. Use spatial random crop augmentation and well  as temporal random cropping during training. At test time, perform multiview testing (4 temporal views are usually enough, more views result in higher  performance).
1. Training: Use spatial random cropping and temporal random cropping for data augmentation.
2. Testing: Perform multi-view testing with 4 temporal views (you can use more views for even better performance).

Deductions:
- -2 point – no 2D resnet initialization
- -1 point - data augmentation is not used
- -2 points - multi-view testing is not implemented.
- -1 points - mistake in multi-view testing
- -1 point – mistake 2D resnet initialization