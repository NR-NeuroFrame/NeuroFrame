class Dunders:
    def __repr__(self):
        return f'MRI(path="{self.path}")'
    def __str__(self):
        return f"MRI stored in {self.folder} and with filename {self.filename}"