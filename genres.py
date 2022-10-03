import pandas as pd

class Genres:
    def __init__(self):
        self.data = None

    def get_genres(self):
        self.data = pd.read_csv('musicData.csv')
        self.data = self.data.dropna()
        for c in self.data.columns:
            self.data[c] = self.data[c].replace("?", -1)
        return self.data['genre'].unique()