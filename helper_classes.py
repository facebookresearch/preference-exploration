from botorch.acquisition import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from torch import Tensor


class PosteriorMeanDummySampler(MCSampler):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def _construct_base_samples(self):
        pass

    def forward(self, posterior: Posterior):
        return posterior.mean.unsqueeze(0)


class LearnedPrefereceObjective(MCAcquisitionObjective):
    def __init__(self, pref_model, sampler=None, use_mean=False):
        super().__init__()
        self.use_mean = use_mean
        self.sampler = None
        if not self.use_mean:
            if sampler is None:
                self.sampler = SobolQMCNormalSampler(num_samples=1)
            else:
                self.sampler = sampler

        self.pref_model = pref_model

    def forward(self, samples: Tensor, X: Tensor = None):
        if self.use_mean:
            output_sample = self.pref_model.posterior(samples).mean.squeeze(-1)
        else:
            output_sample = self.sampler(self.pref_model.posterior(samples)).squeeze(-1)
            output_sample = output_sample.reshape((-1,) + output_sample.shape[2:])
        return output_sample
