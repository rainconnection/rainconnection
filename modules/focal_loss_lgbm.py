class focal:
  def __init__(self, gamma, alpha):
    self.alpha = alpha
    self.gamma = gamma

  def loss(self, preds, true): # Focal loss 선정 - Imbalance에 강한 것으로 나타남 (정답에 대한 가중치)
    y = true.get_label()*2 - 1
    p = expit(preds)
    g = self.gamma

    # grad
    a = np.where(y, self.alpha, 1-self.alpha)
    p = np.clip(p,1e-15, 1 - 1e-15)
    p = np.where(y, p, 1-p)
    grad = (a*y*(1-p)**g) * (g*p*np.log(p)+p-1)

    # hess
    u = a*y*(1-p)**g
    du = -a*y*g*(1-p)**(g-1)
    v = g*p*np.log(p) + p - 1
    dv = g*np.log(p) + g + 1
    hess = (du*v*u*dv) * y * (p*(1-p))

    return grad, hess

  def eval(self, preds, true):
    y = true.get_label()
    p = expit(preds)

    g = self.gamma
    a = np.where(y, self.alpha, 1-self.alpha)
    p = np.clip(p,1e-15, 1 - 1e-15)
    p = np.where(y, p, 1-p)

    val = -a*(1-p)**g*np.log(p)

    return 'focal_loss', val.mean(), False
