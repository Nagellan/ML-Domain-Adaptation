import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
from torch.autograd import Function
from torchvision import datasets, transforms

cudnn.benchmark = True


class Config(object):
    """
    Model's configuration data.
    """

    model_name = "svhn-mnist"

    # parameters for datasets and data loader
    batch_size = 128

    # parameters for source dataset
    src_dataset = "svhn"
    src_image_root = "data"

    # parameters for target dataset
    tgt_dataset = "mnist"
    tgt_image_root = "data"

    # parameters for training dann
    gpu_id = '0'
    num_epochs = 200
    log_step = 50
    save_step = 100
    eval_step = 1
    manual_seed = 228
    alpha = 0

    # parameters for optimizing models
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-6


class ReverseLayerF(Function):
    """
    Reverse layer architecture.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANNmodel(nn.Module):
    """
    SVHN model architecture.
    """

    def __init__(self):
        super(DANNmodel, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, input_data, alpha=1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output


def adjust_learning_rate(optimizer, p):
    """
    Adjust specific learning rate dependent on constant p to the optimizer's according parameter group.
    :param optimizer:  given optimizer
    :param p:          constant p
    :return:           nothing
    """

    # specify parameters
    lr_0 = 0.01
    alpha = 10
    beta = 0.75

    # calculate learning rate
    lr = lr_0 / (1 + alpha * p) ** beta

    # assign given learning rate to all optimizer's parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, params, src_data_loader, tgt_data_loader, device, optimizer, criterion, epoch):
    """
    Train the model using given source and target data loaders, and labels of source domain.
    :param model:            given model
    :param params:           model's configuration parameters
    :param src_data_loader:  source data
    :param tgt_data_loader:  target data
    :param device:           device used for model's training
    :param optimizer:        given optimizer
    :param criterion:        given loss function
    :param epoch:            current epoch
    :return:                 trained model
    """

    # set train state for Dropout and BN layers
    model.train()

    # zip source and target data pair
    len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
    data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

    for step, ((images_src, class_src), (images_tgt, _)) in data_zip:
        # specify constants
        p = float(step + epoch * len_dataloader) / params.num_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        adjust_learning_rate(optimizer, p)

        # prepare domain label
        size_src = len(images_src)
        size_tgt = len(images_tgt)
        label_src = torch.zeros(size_src).long().to(device)
        label_tgt = torch.ones(size_tgt).long().to(device)

        # make images variable
        class_src = class_src.to(device)
        images_src = images_src.to(device)
        images_tgt = images_tgt.to(device)

        # zero gradients for optimizer
        optimizer.zero_grad()

        # train on source domain
        src_class_output, src_domain_output = model(input_data=images_src, alpha=alpha)
        src_loss_class = criterion(src_class_output, class_src)
        src_loss_domain = criterion(src_domain_output, label_src)

        # train on target domain
        _, tgt_domain_output = model(input_data=images_tgt, alpha=alpha)
        tgt_loss_domain = criterion(tgt_domain_output, label_tgt)

        # calculate the loss
        loss = src_loss_class + src_loss_domain + tgt_loss_domain

        # optimize dann
        loss.backward()
        optimizer.step()

        if (step + 1) % params.log_step == 0:
            print(f"Epoch [{epoch + 1:4d}/{params.num_epochs}] Step [{step + 1:2d}/{len_dataloader}]: "
                  f"src_loss_class={src_loss_class.data.item():.6f}, src_loss_domain={src_loss_domain.data.item():.6f},"
                  f" tgt_loss_domain={tgt_loss_domain.data.item():.6f}, loss={loss.data.item():.6f}")

    return model


def test(model, data_loader, device, criterion, flag, title):
    """
    Evaluate model by given dataset.
    :param model:        given model
    :param data_loader:  dataloader of model to evaluate by
    :param device:       device used for model's training
    :param criterion:    loss function for testing
    :param flag:         label for target and source domains
    :param title:        model name title
    :return:             average loss, accuracy and domain accuracy
    """

    # set evaluation state for Dropout and BN layers
    model.eval()

    # initial loss and accuracy
    loss_ = 0.0
    acc_ = 0.0
    acc_domain_ = 0.0
    n_total = 0

    # evaluate the model
    for (images, labels) in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        size = len(labels)

        if flag == 'target':
            labels_domain = torch.ones(size).long().to(device)
        else:
            labels_domain = torch.zeros(size).long().to(device)

        # get predictions and domain
        preds, domain = model(images, alpha=0)

        # calculate data for evaluation criteria
        loss_ += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]
        acc_ += pred_cls.eq(labels.data).sum().item()
        acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
        n_total += size

    # calculate evaluation criteria
    loss = loss_ / n_total
    acc = acc_ / n_total
    acc_domain = acc_domain_ / n_total

    print(f"{title}: Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, "
          f"Avg Domain Accuracy = {acc_domain:2%}")

    return loss, acc, acc_domain


def get_svhn(dataset_root, batch_size, train):
    """
    Dataset loader for SVHN.
    :param dataset_root:  root folder for dataset
    :param batch_size:    size of one batch
    :param train:         label for specifying train & test sets
    :return:              dataloader for SVHN dataset
    """

    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                      )])

    # dataset initialization
    svhn_dataset = datasets.SVHN(root=os.path.join(dataset_root),
                                 split='train' if train else 'test',
                                 transform=pre_process,
                                 download=True)

    # dataloader configuration
    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    return svhn_data_loader


def get_mnist(dataset_root, batch_size, train):
    """
    Dataset loader for MNIST.
    :param dataset_root:  root folder for dataset
    :param batch_size:    size of one batch
    :param train:         label for specifying train & test sets
    :return:              dataloader for MNIST dataset
    """

    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32),  # different img size settings for mnist(28) and svhn(32).
                                      transforms.ToTensor()
                                      ])

    # dataset initialization
    mnist_dataset = datasets.MNIST(root=os.path.join(dataset_root),
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    # dataloader configuration
    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)

    return mnist_data_loader


def init_random_seed(manual_seed):
    """
    Set up seed manually or randomly.
    :param manual_seed:  manually set seed, if not None
    :return:             nothing
    """

    if manual_seed is None:
        seed = random.randint(1, 10000)
        print(f"Using random seed: {seed}")
    else:
        seed = manual_seed
        print(f"Using manual seed: {seed}")

    # set up the seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Main function containing all necessary for program work function calls.
    :return: nothing
    """

    params = Config()
    device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

    # set a seed
    init_random_seed(params.manual_seed)

    # load dataset
    svhn_train_loader = get_svhn(params.src_image_root, params.batch_size, train=True)
    svhn_test_loader = get_svhn(params.src_image_root, params.batch_size, train=False)
    mnist_train_loader = get_mnist(params.tgt_image_root, params.batch_size, train=True)
    mnist_test_loader = get_mnist(params.tgt_image_root, params.batch_size, train=False)

    # load dann model
    dann = DANNmodel()

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        dann.cuda()

    # setup optimizer and loss function
    optimizer = optim.SGD(dann.parameters(), lr=params.lr, momentum=params.momentum,
                          weight_decay=params.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(params.num_epochs):
        # train
        dann = train(dann, params, svhn_train_loader, mnist_train_loader, device, optimizer, criterion, epoch)

        if (epoch + 1) % params.eval_step == 0:
            # eval model
            test(dann, svhn_train_loader, device, criterion, flag='source', title="SVHN-train")
            test(dann, svhn_test_loader, device, criterion, flag='source', title="SVHN-test")
            test(dann, mnist_test_loader, device, criterion, flag='target', title="MNIST-test")
            print('\n')

            # save model parameters
            torch.save(dann.state_dict(), "svhn-mnist.pt")

    # save final model
    torch.save(dann.state_dict(), "svhn-mnist-final.pt")


if __name__ == '__main__':
    main()
