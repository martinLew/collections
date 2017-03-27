import six

from chainer.functions.loss import mean_squared_error
from chainer import link
from chainer import reporter


class MRegression(link.Chain):
    """A simple multiple regression model.

    This is an example of chain that wraps another chain. It computes the
    loss based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (list of ~chainer.Variable): Loss value for the last minibatch.
        target_num (int): target number of this model, it can handle more than
            1 taget in the model.

    """

    def __init__(self, predictor):
        assert hasattr(predictor, 'target_num')
        super(MRegression, self).__init__(predictor=predictor)
        self.lossfun = mean_squared_error.mean_squared_error
        self.y = None
        self.loss = None
        self.target_num = predictor.target_num

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

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
        self.y = self.predictor(*x)
        for i in six.moves.xrange(self.target_num):
            this_t = t[:, i].reshape(-1, 1)
            self.loss[i] = self.lossfun(self.y[i], this_t) / 2
            reporter.report({'loss_%d' % i: self.loss[i]}, self)
        return sum(self.loss)
