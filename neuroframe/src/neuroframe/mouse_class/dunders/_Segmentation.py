class Dunders:
    def __repr__(self):
        return f'Segmentation(path="{self.path}")'
    def __str__(self):
        return f"Segmentation stored in {self.folder} and with filename {self.filename}"