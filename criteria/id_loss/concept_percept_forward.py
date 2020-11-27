import torch
import torchvision


class ConceptPerceptVGG(torch.nn.Module):
    def __init__(self, model):
        super(ConceptPerceptVGG, self).__init__()
        self.inner = model

    def forward(self, x):
        x = self.inner.features(x)
        x = self.inner.avgpool(x)
        x = torch.flatten(x, 1)
        last_fc = None
        curr_fc = None
        for m in self.inner.classifier.modules():
            if type(m) != torch.nn.Sequential:
                x = m(x)
                if type(m) == torch.nn.Linear:
                    last_fc = curr_fc
                    curr_fc = x
        return curr_fc, last_fc


if __name__ == '__main__':
    m = torchvision.models.vgg16()
    model = ConceptPerceptVGG(m)
    print(m.forward(torch.FloatTensor(size=[1, 3, 224, 224])))
    concept, percept = model.forward(torch.FloatTensor(size=[1, 3, 224, 224]))
    print("concept", concept.size())
    print("percept", percept.size())
    torch.nn.PairwiseDistance()

