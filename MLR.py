class MulLR:
    def __init__(self, max_iter=1000, lr=0.1, lambda_=0.01, penalty = 'l2'):
        """
        Initialize a multinomial logistic regression object.

        Parameters
        ----------
        max_iter : int, optional (default=1000)
            Maximum number of iterations for gradient descent
        lr : float, optional (default=0.1)
            Learning rate for gradient descent
        lambda_ : float, optional (default=0.01)
            Regularization parameter
        penalty : {'l1', 'l2'}, optional (default='l2')
            Type of regularization penalty to use
        """
        self.max_iter = max_iter
        self.lr = lr
        self.lambda_ = lambda_
        self.penalty = penalty
        
    def softmax(self, X):
        """
        Compute the softmax activation function.

        Parameters
        ----------
        X : numpy array
            The input data of shape (n_samples, n_classes).

        Returns
        -------
        numpy array
            The output of the softmax function of shape (n_samples, n_classes).
        """
        return np.exp(X)/ np.sum(np.exp(X), axis = 1).reshape(-1,1)
    
    def fit(self, X, Y):
        """
        Fit the multinomial logistic regression model to the training data.

        Parameters
        ----------
        X : numpy array
            The input data of shape (n_samples, n_features).
        Y : numpy array
            The target values of shape (n_samples,).

        Returns
        -------
        None
        """
        self.theta, self.learning_curve = self.Gradient_descent(X, Y, self.max_iter,
                                                                self.lr, self.lambda_, self.penalty)

    def predict(self, X_new):
        """
        Predict the class labels for new data.

        Parameters
        ----------
        X_new : numpy array
            The input data of shape (n_samples, n_features).

        Returns
        -------
        numpy array
            The predicted class labels of shape (n_samples,).
        """
        Z = -X_new @ self.theta
        pr = self.softmax(Z)
        pred = np.argmax(pr, axis=1)
        return np.array([self.label_key[value] for value in pred])
    
    def Gradient_descent(self, X, Y, max_iter, lr, lambda_, penalty):
        """
        Perform gradient descent to optimize the cost function and obtain the weights.

        Parameters
        ----------
        X : numpy array
            The input data of shape (n_samples, n_features).
        Y : numpy array
            The target values of shape (n_samples,).
        max_iter : int
            Maximum number of iterations for gradient descent
        lr : float
            Learning rate for gradient descent
        lambda_ : float
            Regularization parameter
        penalty : {'l1', 'l2'}
            Type of regularization penalty to use

        Returns
        -------
        theta : numpy array
            The weights of the logistic regression model of shape (n_features, n_classes).
        learning_df : pandas dataframe
            A dataframe containing the iteration number and loss for each iteration of gradient descent.
        """
        def Cost(X, Y, theta, lambda_, penalty):
            Z = - X @ theta
            m = X.shape[0]
            loss = 1/m * (np.trace(X @ theta @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis = 1)))) 
            if penalty == 'l2':
                regularization = lambda_/(2*m) * np.sum(np.square(theta[1:]))  # exclude bias term from regularization
            elif penalty == 'l1':
                regularization = lambda_/(m) * np.sum(np.abs(theta[1:]))
            else:
                regularization = 0
            cost = loss + regularization
            return cost
        def Gradient(X, Y, theta, lambda_, penalty):
            Z = - X @ theta
            pr = self.softmax(Z)
            m = X.shape[0]
            d_loss = 1/m * (X.T @ (Y-pr))
            if penalty == 'l2':
                d_regul =  lambda_ * theta
            elif penalty == 'l1':
                d_regul = lambda_ * np.sign(theta)
            else:
                d_regul = np.zeros(theta.shape)
            d_regul[0, :] = 0
            gradient = d_loss + d_regul
            return gradient
        def OneHot(y):
            u = np.unique(y)
            n_unique = u.size
            n_sample = len(y)
            label_key = dict(zip(u, range(n_unique)))
            encoded_array = np.zeros((n_sample, n_unique), dtype=int)
            encoded_array[np.arange(n_sample), [label_key[value] for value in y]] = 1
            # switch label key for prediction access
            self.label_key = dict(zip(label_key.values(), label_key.keys()))
            return encoded_array
        Y_one = OneHot(Y)
        theta = np.zeros((X.shape[1], Y_one.shape[1]))
        iter_ = 0
        learning_curve = []
        while iter_ < max_iter:
            iter_ += 1
            theta = theta - lr * Gradient(X, Y_one, theta, lambda_, penalty)
            learning_curve.append(Cost(X, Y_one, theta, lambda_, penalty))
        learning_df = pd.DataFrame({
            'iter': range(iter_),
            'loss': learning_curve
        })
        return theta, learning_df

    def loss_plot(self):
    """
    Plots the learning curve.

    Returns:
    --------
    A matplotlib plot of the learning curve.
    """
        return self.learning_curve.plot(
            x='iter',
            y='loss',
            xlabel='iter',
            ylabel='loss'
        )
