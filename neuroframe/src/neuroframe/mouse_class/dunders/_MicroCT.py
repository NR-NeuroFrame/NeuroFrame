class Dunders:
    def __repr__(self):
        return f'MicroCT(path="{self.path}")'
    def __str__(self):
        return f"μCT stored in {self.folder} and with filename {self.filename}"