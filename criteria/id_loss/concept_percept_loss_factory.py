from criteria.id_loss.concept_percept_loss import PerceptConceptLoss
from criteria.id_loss.local_model_store import LocalModelLoader
import torch
import torchvision
import os


class ConceptPerceptLossFactory(object):
    @staticmethod
    def get_cp_loss(opts):
        model_store = LocalModelLoader()

        model = torchvision.models.__dict__[opts.discriminator_arch](num_classes=opts.num_discrimiator_classes)
        if type(model) != torchvision.models.VGG:
            raise NotImplementedError("Percept-Concept loss is not implemented for models other than VGG")

        if os.path.isfile(opts.discriminator_path):
            model_store.load_model_and_optimizer_loc(model, model_location=opts.discriminator_path)

        concept_loss = torch.nn.__dict__[opts.concept_loss_criterion]()
        percept_loss = torch.nn.__dict__[opts.percept_loss_criterion]()

        if torch.cuda.is_available():
            # DataParallel will divide and allocate batch_size to all available GPUs
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            concept_loss.cuda()
            percept_loss.cuda()

        ConceptPerceptLossFactory.__freeze_discriminator(model)
        ConceptPerceptLossFactory.__freeze_discriminator(concept_loss)
        ConceptPerceptLossFactory.__freeze_discriminator(percept_loss)

        concept_loss.eval()
        percept_loss.eval()
        model.eval()
        return PerceptConceptLoss(model, concept_loss, percept_loss, opts.percept_loss_lambda, opts.concept_loss_lambda)

    @staticmethod
    def __freeze_discriminator(module: torch.nn.Module):
        for param in module.parameters():
            param.requires_grad=False


if __name__=="__main__":
    class Opts:
        def __init__(self):
            self.concept_loss_criterion = 'CrossEntropyLoss'
            self.percept_loss_criterion = 'PairwiseDistance'
            self.percept_loss_lambda = 1
            self.concept_loss_lambda = 1
            self.discriminator_arch = 'vgg16'
            self.num_discrimiator_classes = 1000
            self.discriminator_path = ''

    opts = Opts()
    input = torch.FloatTensor(size=[1,3,224,224])
    output = torch.FloatTensor(size=[1,3,224,224])
    target = torch.LongTensor(size=[1])
    print(input)
    print(output)
    print(target)
    target[0] = 1
    print(ConceptPerceptLossFactory.get_cp_loss(opts)(input,output,target))