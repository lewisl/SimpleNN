# way to implement maxnorm regulization
def __call__(self, w):
    norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
    desired = K.clip(norms, 0, self.max_value)
    w *= (desired / (K.epsilon() + norms))
    return w


z2 = z .* (2.5/norm(z))