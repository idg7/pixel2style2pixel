import torch
import torchvision
from criteria.id_loss.concept_percept_forward import ConceptPerceptVGG


class PerceptConceptLoss(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, concept_loss_criterion: torch.nn.Module, percept_loss_criterion: torch.nn.Module, percept_lambda: float, concept_lambda: float):
        super(PerceptConceptLoss, self).__init__()
        if type(model) != torchvision.models.VGG:
            raise NotImplementedError("Percept-Concept loss is not implemented for models other than VGG")
        self.discriminator = ConceptPerceptVGG(model)
        self.concept_loss_criterion = concept_loss_criterion
        self.percept_loss_criterion = percept_loss_criterion
        self.percept_lambda = percept_lambda
        self.concept_lambda = concept_lambda
        self.discriminator.eval()
        self.concept_loss_criterion.eval()
        self.percept_loss_criterion.eval()

    def forward(self, input, output, target):
        input_concept, input_percept = self.discriminator(input)
        output_concept, output_percept = self.discriminator(output)
        concept_loss = self.concept_loss_criterion(output_concept, target)
        percept_loss = self.percept_loss_criterion(output_percept, input_percept)
        return self.concept_lambda * concept_loss + self.percept_lambda * percept_loss, {'perceptual loss': percept_loss, 'conceptual loss': concept_loss}

