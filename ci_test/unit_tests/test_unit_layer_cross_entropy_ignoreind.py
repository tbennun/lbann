"""
Tests ignoring out-of-bounds indices in cross-entropy.
"""
import lbann
import numpy as np
import test_util


def _cross_entropy_ref(logits, labels, num_labels):
    # Equivalent PyTorch implementation:
    # torch.nn.CrossEntropyLoss(reduction='sum')
    result = 0
    count = 0
    for i in range(logits.shape[0]):
        if labels[i] < 0 or labels[i] >= num_labels:
            continue
        result += -np.log(
            np.exp(logits[i, labels[i]]) / np.sum(np.exp(logits[i])))
        count += 1
    return result  # / count


@test_util.lbann_test(check_gradients=True)
def test_crossentropy_ignoreindex():
    # Prepare reference output
    np.random.seed(20240715)
    logits = np.random.rand(2, 6, 7)
    logits[1, :, :] = 0
    logits[1, -1, 1] = 2

    labels = np.array([
        [1, 2, -100, 3, -200, 200],
        [-100, -100, -100, -100, -100, 1],
    ])
    reference_numpy = np.array([
        _cross_entropy_ref(logits[i], labels[i], 7)
        for i in range(labels.shape[0])
    ])

    tester = test_util.ModelTester()

    # Do not compute gradients for labels
    x1, x2 = tester.inputs_like(logits, labels, nograd=[False, True])
    reference = tester.make_reference(reference_numpy)

    # Test layer
    flat_labels = lbann.Reshape(x2, dims=[1, logits.shape[1]])
    preds = lbann.ChannelwiseSoftmax(x1)
    preds = lbann.TensorPermute(preds, axes=[1, 0])
    y = lbann.CrossEntropy(preds, flat_labels, use_labels=True)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(y)
    return tester
