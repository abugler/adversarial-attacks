import torch


class BaseAttack():
    """
    Base class for adversarial attacks. Any class that
    implements BaseAttack must fill perturb.

    Args:
        network (nn.Module): A differentiable network to attack.
    """
    def __init__(self, network: torch.nn.Module):
        self.network = network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def perturb(self, x):
        """
        Perturbs x to be an adversarial example of `self.network`.

        Not implemented, must be implemented by a class that inheirits this.
        """
        raise NotImplementedError


class FastGraidentSign(BaseAttack):
    """
    Fast Gradient Sign Attack from:
    Ian J. Goodfellow, Jonathon Shlens, & Christian Szegedy. (2014).
    Explaining and Harnessing Adversarial Examples.
    """
    def __init__(self,
                 network: torch.nn.Module,
                 loss: torch.nn.Module,
                 epsilon: float
                 ):
        self.loss = loss()
        self.epsilon = epsilon
        super().__init__(network)

    def perturb(self, x: torch.Tensor, label: torch.Tensor):
        """
        Perturbs sample `x` according to the Gradient Sign Attack
        """
        if x.grad is not None:
            x.grad.data.zero_()
        x.requires_grad = True
        out = self.network(x)
        loss_val = self.loss(out, label)
        loss_val.backward()
        x_p = x + self.epsilon * torch.sign(x.grad)
        return x_p


class SaliencyAttack(BaseAttack):
    """
    Saliency Map Attack from:
    Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson,
    Z. Berkay Celik, & Ananthram Swami. (2015). The Limitations of Deep
    Learning in Adversarial Settings.
    """
    def __init__(self,
                 network: torch.nn.Module,
                 max_pert: float,
                 feature_var: float,
                 num_classes: int
                 ):
        self.max_pert = max_pert
        self.feature_var = feature_var
        self.num_classes = num_classes
        super().__init__(network)

    def saliency_map(self, x_k, probabilities, target):
        gradients = []

        for prob in probabilities[0]:
            if x_k.grad is not None:
                x_k.grad.data.zero_()

            prob.backward(retain_graph=True)
            gradients.append(x_k.grad[0].clone())

        gradients = torch.stack(gradients, dim=0) * 1e10
        non_targets = [i for i in range(self.num_classes) if i != target]
        non_target_gradient_sum = torch.sum(gradients[non_targets], 0)[0] * 1e7
        smap = torch.where(
            (gradients[target] >= 0)
            & (non_target_gradient_sum <= 0),
            gradients[target] * torch.abs(non_target_gradient_sum),
            torch.zeros_like(gradients[target])
        )
        return torch.unsqueeze(smap, 0)

    def perturb(self, x: torch.Tensor, target: int):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        x_k = x.clone().to(device)
        x_k.requires_grad = True

        if target >= self.num_classes:
            raise ValueError
        height = x_k.size(-2)
        delta = 0
        self.network.to(device)
        while (torch.argmax(out := self.network(x_k), dim=-1)[0] != target
               and delta < self.max_pert):
            S = self.saliency_map(x_k, out, target)
            while True:
                max_idx = torch.argmax(S)
                loc = (0, 0, max_idx // height, max_idx % height)
                if x_k[loc] + self.feature_var > 1:
                    S[loc] = -1
                    # print(f"pixel saturated. trying a new one")
                    continue
                delta += self.feature_var
                x_k[loc] += self.feature_var
                break
            x_k = x_k.detach().clone()

            x_k.requires_grad = True
        return x_k
