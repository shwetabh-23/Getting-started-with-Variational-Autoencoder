import numpy as np
import matplotlib.pyplot as plt 

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

def compare_images(vae, test_loader, original_images, generated_images):

    examples = enumerate(test_loader)
    batch_indx, (test_images, test_labels) = next(examples)

    image_indx = np.random.choice(range(test_images.shape[0], original_images))
    fig = plt.figure()
    subfig = 1

    for i in range(original_images):

        indx_of_img = image_indx[i]
        image = test_images[indx_of_img][0]
        breakpoint()
        plt.subplot(original_images, generated_images+1, subfig)
        plt.title('Ground Truth : {}'.format(test_images[indx_of_img]))
        plt.imshow(image.detach().numpy(), cmap = 'grey', interpolation='none')
        subfig += 1

        for _ in range(generated_images):

            plt.subplot(original_images, generated_images+1, subfig)
            img = vae.reconstruct_img(image).detach().numpy().reshape(28, 28)
            if (subfig - 2) % (generated_images + 1) == 0:
                plt.title("Generated images")

            plt.imshow(img, cmap='grey', interpolation='none')
            subfig += 1
    plt.show()
