import torch
import torch.nn.functional as F
import torch.nn as nn

# It takes image and text embeddings as inputs and calculates the contrastive loss based on cosine similarities.
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        logits            = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity  = text_embeddings  @ text_embeddings.T
        targets           = F.softmax((images_similarity + texts_similarity) / (2 * self.temperature), dim=-1)

        texts_loss        = F.cross_entropy(logits, targets.argmax(dim=1), reduction='none')
        images_loss       = F.cross_entropy(logits.T, targets.argmax(dim=0), reduction='none')
        
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

# It takes the label to generate the mask of positive pairs directly (without using MLP).
class SupervisedLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupervisedLoss, self).__init__()
        self.temperature      = temperature
        self.contrast_mode    = contrast_mode  # allows to control whether you want to treat only 
                                            #  the first view as an anchor or consider all views as separate anchors
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        batch_size, contrast_count, _ = features.shape

        mask = (labels.view(-1, 1) == labels).float()

        contrast_feature = features.view(batch_size * contrast_count, -1)
        anchor_feature = features[:, 0].view(batch_size, -1) if self.contrast_mode == 'one' else contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(contrast_count, 1)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss