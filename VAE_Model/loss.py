import matplotlib.pyplot as plt

class MonitorLoss():
    def __init__(self):
        self.train_epoch = []
        self.test_epoch = []
        self.train_elbo = []
        self.test_elbo = []

    def append_loss(self, epoch, loss, set = 'train'):

        if set == 'train':
            self.train_epoch.append(epoch)
            self.train_elbo.append(loss)

        elif set == 'test':
            self.test_epoch.append(epoch)
            self.test_elbo.append(loss)

        else:
            raise ValueError("Something is wrong, I can feel it, it's the feeling I have got")
        
    def show_loss(self, title = 'Loss Curve'):
        plt.plot(self.test_epoch, self.test_elbo, '.-', label = 'Test Loss')
        plt.plot(self.train_epoch, self.train_elbo, '.-', label = 'Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.show()
