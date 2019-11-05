import numpy as np

class LinearRegression:
    """
    >>> import regression
    >>> model = regression.LinearRegression() 
    >>> model.x
    >>> #nothing
    None
    """
    x = None
    theta = None
    y = None

    def fit(self, x, y):
        """
        # testing (cont.)
        >>> import importlib
        >>> importlib.reload(regression)
        >>> model = regression.LinearRegression() 
        >>> model.fit(X, Y)
        >>> model.theta
        array([ 5.30412371, 0.49484536])
        """
        temp = np.linalg.inv(np.dot(x.T, x))
        self.theta = np.dot(np.dot(temp, x.T), y)

    def predict(self, x):
        """
        # testing (cont.)
        >>> importlib.reload(regression)
        >>> model = regression.LinearRegression()
        >>> model.fit(X, Y)
        >>> model.predict(X)
        array([ 7.28350515, 9.2628866 , 11.7371134 , 13.71649485])
        :param x: 
        :return: 
        """
        return np.dot(x, self.theta)

    def score(self, x, y):
        """
        # testing (cont.)
        >>> importlib.reload(regression)
        >>> model = regression.LinearRegression() >>> model.fit(X, Y)
        >>> model.score(X, Y)
        1.2474226804123705
        :param x: 
        :param y: 
        :return: 
        """
        error = self.predict(x) - y
        return (error ** 2).sum()