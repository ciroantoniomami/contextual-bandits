
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

class LogReg_TS(object):

    def __init__(self, lambda_: float, D: int, alpha: float) -> None:
        
        self.lambda_ = lambda_
        self.alpha = alpha
        self.D = D

        self.m = np.zeros(self.D)
        self.q = np.ones(self.D) * self.lambda_

        self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1), size=self.D)

    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m).dot(w - self.m)) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)

    def get_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q)**(-1), size=self.D)

    def update(self, X, y):
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':True}).x
        self.m = self.w

        P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
        try:
            self.q = self.q + (P*(1-P)).dot(X ** 2)
        except AttributeError:
            self.q = self.q + (P*(1-P)) * (X ** 2)
            
    def predict_proba(self, X):
        self.w = self.get_weights()
        proba = 1 / (1 + np.exp(-1 * X.dot(self.w)))
        return np.array([proba]).T

    def get_action(self, mtx_content):
        return np.argmax(self.predict_proba(mtx_content))