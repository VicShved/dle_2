import torch

def save_to_file(model, optimizer, epoch, loss,) -> str:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }
    pth = "savepoints/savepoint.pth"
    torch.save(state, pth)
    return pth

def load_from_file(pth, model, optimizer, epoch, loss,) -> tuple:
    checkpoint = torch.load(pth)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch, loss