def train_vae(train_loader, svi, use_cuda = False):
    epoch_loss = 0
    for x, _ in train_loader:
        if use_cuda:
            x = x.cuda()

        epoch_loss += svi.step(x)
    train_len = len(train_loader.dataset)
    avg_loss = epoch_loss/train_len

    return avg_loss

def evaluate_vae(test_loader, svi, use_cuda = False):
    test_loss = 0
    for x, _ in test_loader:
        if use_cuda:
            x = x.cuda()
        test_loss = svi.step(x)
    test_len = len(test_loader.dataset)
    avg_loss = test_loss/avg_loss

    return avg_loss
