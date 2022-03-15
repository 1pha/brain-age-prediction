from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Identity Scaler (returns X)
class Identity:
    def __init__(self):
        pass

    def fit_transform(self, X):
        return X


def get_scaler(scaler):

    return {"minmax": MinMaxScaler, "zscore": StandardScaler, "identity": Identity}[
        scaler
    ]()
