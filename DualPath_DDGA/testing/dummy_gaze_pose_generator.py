import torch

def generate_dummy_gaze_pose_importance(batch_size, h, w, device="cuda"):
    """
    Generate dummy gaze and pose importance maps.

    Args:
        batch_size (int): Batch size.
        h (int): Height of the feature map (e.g., 7).
        w (int): Width of the feature map (e.g., 7).
        device (str): 'cuda' or 'cpu'.

    Returns:
        gaze_importance: [B, H*W]
        pose_importance: [B, H*W]
    """
    gaze_importance = torch.rand(batch_size, h * w, device=device)
    pose_importance = torch.rand(batch_size, h * w, device=device)

    # Optional: Normalize between 0 and 1
    gaze_importance = (gaze_importance - gaze_importance.min()) / (gaze_importance.max() - gaze_importance.min())
    pose_importance = (pose_importance - pose_importance.min()) / (pose_importance.max() - pose_importance.min())

    return gaze_importance, pose_importance
