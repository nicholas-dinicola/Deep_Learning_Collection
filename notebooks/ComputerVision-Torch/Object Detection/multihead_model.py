from torch import nn


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: this can be a custom network, or a
        # pre-trained network such as a resnet50 where the
        # original classification head has been removed. It computes
        # an embedding for the input image
        self.backbone = nn.Sequential(..., nn.Flatten())

        # Classification head: an MLP or some other neural network
        # ending with a fully-connected layer with an output vector
        # of size n_classes
        self.class_head = nn.Sequential(..., nn.Linear(out_feature, n_classes))

        # Localization head: an MLP or some other neural network
        # ending with a fully-connected layer with an output vector
        # of size 4 (the numbers defining the bounding box)
        self.loc_head = nn.Sequential(..., nn.Linear(out_feature, 4))

    def forward(self, x):
        x = self.backbone(x)

        class_scores = self.class_head(x)
        bounding_box = self.loc_head(x)

        return class_scores, bounding_box

#An object localization network, with its two heads, is an example of a multi-head network. 
# Multi-head models provide multiple inputs that are in general paired with multiple inputs.
#
#For example, in the case of object localization, we have 3 inputs as ground truth: the image, 
# the label of the object ("car") and the bounding box for that object (4 numbers defining a bounding box). 
# The object localization network processes the image through the backbone, then the two heads provide two outputs:
#
#The class scores, that need to be compared with the input label
#The predicted bounding box, that needs to be compared with the input bounding box.
#The comparison between the input and predicted labels is made using the usual Cross-entropy loss, while the comparison between 
# the input and the predicted bounding boxes is made using, for example, the mean squared error loss. The two losses are then summed to provide the total loss L.


class_loss = nn.CrossEntropyLoss()
loc_loss = nn.MSELoss()
alpha = 0.5

...
for images, labels in train_data_loader:
    ...

    # Get predictions
    class_scores, bounding_box = model(images)

    # Compute sum of the losses
    loss = class_loss(class_scores) + alpha * loc_loss(bounding_box)

    # Backpropagation
    loss.backward()


    optimizer.step()