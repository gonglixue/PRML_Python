from mxnet import nd

def pure_batch_norm(X, gamma, beta, eps=1e-5):
    assert len(X.shape) in (2, 4)
    # fully connect: batch_size * feature
    if len(X.shape) == 2:
        mean = X.mean(axis=0)   # mean in batch_size-direction, and each feature has a mean
        variance = ((X-mean)**2).mean(axis=0)
    # 2D conv
    else:
        # mean in batch-direction, each channel has a mean
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        varaince = ((X-mean)**2).mean(axis=(0, 2, 3), keepdims=True)
        
    X_hat = (X-mean) / nd.sqrt(varaince + eps)
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)     # reshape?

