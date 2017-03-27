import six

from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class MClassifier(link.Chain):
    """A simple multiple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (list of ~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        target_num (int): target number of this model, it can handle more than
            1 taget in the model.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        assert hasattr(predictor, 'target_num')
        super(MClassifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.target_num = predictor.target_num

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.

        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = [None] * self.target_num
        self.accuracy = [None] * self.target_num
        self.y = self.predictor(*x)
        for i in six.moves.xrange(self.target_num):
            this_t = t[:, i].reshape(-1)
            self.loss[i] = self.lossfun(self.y[i], this_t)
            reporter.report({'loss_%d' % i: self.loss[i]}, self)
            if self.compute_accuracy:
                self.accuracy[i] = self.accfun(self.y[i], this_t)
                reporter.report({'accuracy_%d' % i: self.accuracy[i]}, self)
        return sum(self.loss)


DRNClassifier = MClassifier
