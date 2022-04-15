class AverageMeter:
    def __init__(self, name=None, tag=None, train=True):

        self.tag_loss_mapper = {
            "phase1": "rmse",
            "phase2": "ce",
            "phase3": "conf",
            "reg": "rmse",
            "clf": "ce",
            "unl": "conf",
        }

        train = "train" if train else "valid"

        self.name = f"{train}_{self.tag_loss_mapper[tag]}" if name is None else name
        self.batch = []

    def append(self, data):

        self.batch.append(data)

    def extend(self, data):

        self.batch.extend(data)

    @property
    def average(self):

        average = sum(self.batch) / len(self.batch)
        self.batch = []
        return {self.name: average}

    @property
    def batch_dict(self):

        return {self.name: self.batch}


if __name__ == "__main__":

    losses = AverageMeter("loss")
    losses.append(1)
    losses.append(2)
    print(losses.batch_dict)
