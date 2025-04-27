import torch

# === Adaptive Entropy Controller ===
class AdaptiveEntropyController:
    def __init__(self, start_lambda=0.01, min_lambda=0.002, decay_epochs=(15, 30, 50)):
        """
        Args:
            start_lambda (float): Initial entropy regularization weight.
            min_lambda (float): Minimum entropy weight after decay.
            decay_epochs (tuple): Epochs where decay happens (epoch boundaries).
        """
        self.start_lambda = start_lambda
        self.min_lambda = min_lambda
        self.decay_epochs = decay_epochs

    def get_lambda(self, epoch, gate_scores=None):
        """
        Compute dynamic lambda_entropy based on epoch and (optionally) gate_scores.

        Args:
            epoch (int): Current training epoch.
            gate_scores (Tensor, optional): Tensor [B, 2] of PRA and APP softmax scores.

        Returns:
            lambda_entropy (float)
        """
        # Step 1: Epoch-Based Annealing
        if epoch < self.decay_epochs[0]:
            lambda_entropy = self.start_lambda
        elif epoch < self.decay_epochs[1]:
            lambda_entropy = self.start_lambda * 0.5
        elif epoch < self.decay_epochs[2]:
            lambda_entropy = self.start_lambda * 0.25
        else:
            lambda_entropy = self.min_lambda

        # Step 2: (Optional) Confidence Gap Adjustment
        if gate_scores is not None:
            with torch.no_grad():
                pra_conf = gate_scores[:, 0]
                app_conf = gate_scores[:, 1]
                conf_gap = (pra_conf - app_conf).abs().mean().item()

                if conf_gap > 0.5:
                    # Gate already confident, reduce entropy more aggressively
                    lambda_entropy *= 0.5
                elif conf_gap < 0.1:
                    # Gate confused, keep entropy weight
                    lambda_entropy *= 1.2  # slight boost

                lambda_entropy = max(lambda_entropy, self.min_lambda)

        return lambda_entropy

# === Usage Example inside Training Loop ===
# Initialize controller
entropy_controller = AdaptiveEntropyController(start_lambda=0.01, min_lambda=0.002, decay_epochs=(15, 30, 50))

# Inside training epoch
# lambda_entropy = entropy_controller.get_lambda(epoch, gate_scores=batch_gate_scores)  # Use if available
# or simply
# lambda_entropy = entropy_controller.get_lambda(epoch)  # No gate score adjustment
