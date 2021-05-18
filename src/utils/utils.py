class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()
        self.values = []

    def reset(self):
        self.values = []
        self.val = 0
        self.avg = 0
        self.min = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.values.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = val if val < self.min else self.min