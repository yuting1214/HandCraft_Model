class MulLR:
    def fit(self, X, Y):
        self.theta, self.learning_curve = self.Gradient_descent(X, Y)
    def softmax(self, X):
        return np.exp(X)/ np.sum(np.exp(X), axis = 1).reshape(-1,1)
    def loss_plot(self):
        return self.learning_curve.plot(
            x='iter',
            y='loss',
            xlabel='iter',
            ylabel='loss'
        )
    def predict(self, X_new):
        Z = -X_new @ self.theta
        pr = self.softmax(Z)
        pred = np.argmax(pr, axis=1)
        return np.array([self.label_key[value] for value in pred])
    def Gradient_descent(self, X, Y, max_iter=1000, lr=0.1, lambda_=0.01, penalty = 'l1'):
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
