import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, save_name):
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.savefig('plots/loss_plot'+save_name+'.png', bbox_inches='tight', dpi=500)
    plt.show()