import torch


class LocalModelLoader(object):

    def load_model_and_optimizer_loc(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, model_location=None):
        with open(model_location, 'br') as f:
            print("Loading model from: ", model_location)
            model_checkpoint = torch.load(f)
            model.load_state_dict(model_checkpoint['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(model_checkpoint['optimizer'])

        return model, optimizer, model_checkpoint['acc'], model_checkpoint['epoch']
