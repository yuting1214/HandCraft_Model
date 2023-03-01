class MulLR:
    def fit(self, X, Y):
        self.theta, self.learning_curve = self.Gradient_descent(X, Y)
    def Gradient_descent(self, X, Y, max_iter=1000, lr=0.1, lambda_=0.01):
        def Cost(X, Y, theta, lambda_):
            Z = - X @ theta
            m = X.shape[0]
            cost = 1/m * (np.trace(X @ theta @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis = 1)))) + \
                    lambda_ * np.sum(theta**2)
            return cost
        def Gradient(X, Y, theta, lambda_):
            Z = - X @ theta
            pr = self.softmax(Z)
            m = X.shape[0]
            gradient = 1/m * (X.T @ (Y-pr)) + 2 * lambda_ * theta
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
            theta = theta - lr * Gradient(X, Y_one, theta, lambda_)
            learning_curve.append(Cost(X, Y_one, theta, lambda_))
        learning_df = pd.DataFrame({
            'iter': range(iter_),
            'loss': learning_curve
        })
        return theta, learning_df
