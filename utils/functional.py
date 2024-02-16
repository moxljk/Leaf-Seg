import torch

class Threshold():
    def __init__(self, threshold):
        if threshold > 1 or threshold < 0:
            raise ValueError(f"Threshold should be between 0 to 1, but got {threshold}")
        else:
            self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).type(x.dtype)

class IoU():
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold = Threshold(threshold)
        self.eps = eps
    
    def __call__(self, pr, gt):
        pr = self.threshold(pr)
        gt = self.threshold(gt)
        
        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + self.eps
        return (intersection + self.eps) / union