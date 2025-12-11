class Dunders:
    def __repr__(self):
        return f'MedicalImage(path="{self.path}")'
    def __str__(self):
        return f"Medical Image stored in {self.folder} and with filename {self.filename}"