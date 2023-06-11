import torch
from torch.optim import Adam

class ManifoldAdam(Adam):
    """
    Implement Manifold Adam algorighm
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            learning_rate = group["lr"]
            amsgrad = group["amsgrad"]

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("sparse gradients not supported")

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                state['step'] += 1
                # learning_rate = group['lr'] / np.sqrt(state['step'])

                # make local variables for easy access
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                # actual step
                grad.add_(p, alpha=weight_decay)

                bias_correction1 = 1 - betas[0] ** state["step"]
                bias_correction2 = 1 - betas[1] ** state["step"]

                if not hasattr(p, 'manifold') or p.manifold is None:

                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(grad*grad, alpha=1 - betas[1])

                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        if max_exp_avg_sq.abs() < exp_avg_sq.abs():
                            max_exp_avg_sq = exp_avg_sq
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()

                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # transport the exponential averaging to the new point
                    new_point = p-learning_rate * direction
                    exp_avg_new = exp_avg

                else:
                    manifold = p.manifold
                    grad = manifold.egrad2rgrad(p, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(manifold.inner(p, grad), alpha=1 - betas[1])

                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        if max_exp_avg_sq.abs() < exp_avg_sq.abs():
                            max_exp_avg_sq = exp_avg_sq
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # transport the exponential averaging to the new point
                    new_point = manifold.retr(p, -learning_rate * direction)
                    exp_avg_new = manifold.transp(p, new_point, exp_avg)

                p.copy_(new_point)
                exp_avg.copy_(exp_avg_new)

        return loss
