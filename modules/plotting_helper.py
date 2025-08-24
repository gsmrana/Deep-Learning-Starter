import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_feature2d(
    X_train, 
    y_train, 
    bias=None, 
    weights=None,
    x1_min=-20,
    x1_max=20,
    markersize=4,
):
    plt.plot(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        marker="o",
        markersize=markersize,
        linestyle="",
        label="Class 0",
    )

    plt.plot(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        marker="s",
        markersize=markersize,
        linestyle="",
        label="Class 1",
    )
    
    # plot 2d line
    if bias and len(weights) >= 2:
        x2_min = (-(weights[0] * x1_min) - bias) / weights[1]
        x2_max = (-(weights[0] * x1_max) - bias) / weights[1]
        plt.plot([x1_min, x1_max], [x2_min, x2_max], color="k")
    
    plt.xlabel("Feature $x_1$", fontsize=12)
    plt.ylabel("Feature $x_2$", fontsize=12)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.legend(loc=2)
    plt.grid()
    plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('D', '^', 'x', 's', 'v')
    colors = ('C0', 'C1', 'C2', 'C3', 'C4')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    logits = classifier.forward(tensor)
    Z = np.argmax(logits.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    #edgecolor='black',
                    marker=markers[idx], 
                    label=cl)


def plot_training_loss(
    minibatch_loss_list,
    num_epochs,
    iter_per_epoch,
    results_dir=None,
    averaging_iterations=100,
):
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        range(len(minibatch_loss_list)), (minibatch_loss_list), label="Minibatch Loss"
    )

    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([0, np.max(minibatch_loss_list[1000:]) * 1.5])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")

    ax1.plot(
        np.convolve(
            minibatch_loss_list,
            np.ones(
                averaging_iterations,
            )
            / averaging_iterations,
            mode="valid",
        ),
        label="Running Average",
    )
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, "plot_training_loss.pdf")
        plt.savefig(image_path)


def plot_accuracy(train_acc_list, valid_acc_list, results_dir=None):

    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs + 1), train_acc_list, label="Training")
    plt.plot(np.arange(1, num_epochs + 1), valid_acc_list, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, "plot_acc_training_validation.pdf")
        plt.savefig(image_path)


def show_examples(model, data_loader, unnormalizer=None, class_dict=None):

    fail_features, fail_targets, fail_predicted = [], [], []
    for _, (features, targets) in enumerate(data_loader):
        with torch.inference_mode():
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
            mask = targets != predictions
            fail_features.extend(features[mask])
            fail_targets.extend(targets[mask])
            fail_predicted.extend(predictions[mask])

        if len(fail_targets) > 15:
            break

    fail_features = torch.cat(fail_features)
    fail_targets = torch.tensor(fail_targets)
    fail_predicted = torch.tensor(fail_predicted)

    fig, axes = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)

    if unnormalizer is not None:
        for idx in range(fail_features.shape[0]):
            features[idx] = unnormalizer(fail_features[idx])

    if fail_features.ndim == 4:
        nhwc_img = np.transpose(fail_features, axes=(0, 2, 3, 1))
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap="binary")
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[fail_predicted[idx].item()]}"
                    f"\nA: {class_dict[fail_targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {fail_predicted[idx]} | T: {fail_targets[idx]}")
            ax.axison = False
    else:
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(fail_features[idx], cmap="binary")
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[fail_predicted[idx].item()]}"
                    f"\nA: {class_dict[fail_targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {fail_predicted[idx]} | A: {targets[idx]}")
            ax.axison = False
    plt.tight_layout()
    plt.show()


def show_failures(model, data_loader, unnormalizer=None, class_dict=None, 
                  nrows=3, ncols=5, figsize=None):
    failure_features = []
    failure_pred_labels = []
    failure_true_labels = []

    for _, (features, targets) in enumerate(data_loader):
        with torch.inference_mode():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

        for i in range(features.shape[0]):
            if targets[i] != predictions[i]:
                failure_features.append(features[i])
                failure_pred_labels.append(predictions[i])
                failure_true_labels.append(targets[i])

        if len(failure_true_labels) >= nrows * ncols:
            break

    features = torch.stack(failure_features, dim=0)
    targets = torch.tensor(failure_true_labels)
    predictions = torch.tensor(failure_pred_labels)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize
    )

    if unnormalizer is not None:
        for idx in range(features.shape[0]):
            features[idx] = unnormalizer(features[idx])
    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))

    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap="binary")
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[predictions[idx].item()]}"
                    f"\nA: {class_dict[targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {predictions[idx]} | A: {targets[idx]}")
            ax.axison = False
    else:
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[predictions[idx].item()]}"
                    f"\nA: {class_dict[targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {predictions[idx]} | A: {targets[idx]}")
            ax.axison = False
    return fig, axes
