# utilities.py

from imports import *

def compute_metrics(y_true, y_pred):
    num_classes = 3
    overall_accuracy = tmf.accuracy(y_pred, y_true, num_classes=num_classes, task='multiclass')
    #weighted_acc = tmf.accuracy(y_pred, y_true, average='weighted', task='multiclass', num_classes=num_classes)
    #weighted_precision = tmf.precision(y_pred, y_true, average='weighted', task='multiclass', num_classes=num_classes)
    #weighted_recall = tmf.recall(y_pred, y_true, average='weighted', task='multiclass', num_classes=num_classes)
    f1_macro = tmf.f1_score(y_pred, y_true, average='macro', task='multiclass', num_classes=num_classes)
    #f1_weighted = tmf.f1_score(y_pred, y_true, average='weighted', task='multiclass', num_classes=num_classes)
    macro_precision = tmf.precision(y_pred, y_true, average='macro', task='multiclass', num_classes=num_classes)
    macro_recall = tmf.recall(y_pred, y_true, average='macro', task='multiclass', num_classes=num_classes)

    # Prepare the result dictionary
    return {
        'overall_accuracy': overall_accuracy.item(),
        'macro_precision': macro_precision.item(),
        'macro_recall': macro_recall.item(),
        'f1_macro': f1_macro.item(),
    }


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        """
        Implements Soft Cross-Entropy Loss with class weights.

        Args:
        - weights (torch.Tensor): Class weights tensor of shape (num_classes,).
        """
        super(SoftCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, y_hat, y):
        """
        Compute soft cross-entropy loss with class weights.

        Args:
        - y_hat (torch.Tensor): Logits from the model, shape (batch_size, num_classes).
        - y (torch.Tensor): Soft or one-hot encoded labels, shape (batch_size, num_classes).

        Returns:
        - loss (torch.Tensor): Scalar loss value.
        """
        # Log-softmax on model logits
        p = F.log_softmax(y_hat, dim=1)

        # Apply weights to the targets
        w_labels = self.weights.to(y_hat.device) * y

        # Compute weighted soft cross-entropy loss
        loss = -(w_labels * p).sum() / w_labels.sum()
        return loss