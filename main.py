import shutil
from pathlib import Path
from threading import Thread
from tkinter import Tk, BOTH, TOP
from tkinter import filedialog
from tkinter import ttk

from models.classification.classify import predict_diagnosis
from models.unet.segment import predict_mask
from utils.extract_data.extract import extract_data
from utils.pdf_builder import create_pdf

DATA_PATH = "data/predict"


class Gui(Tk):
    def __init__(self):
        super().__init__()
        self.title("Glaucoma Screening")
        self.geometry("560x300")
        self.configure_ui()
        self.create_backup_directory()
        self.selected_file_name = None

    def configure_ui(self):
        style = ttk.Style(self)
        style.configure('TButton', font=('Arial', 12), padding=10)
        style.configure('TLabel', font=('Arial', 12), padding=5)

        labelFrame = ttk.LabelFrame(self, text="Select an Image")
        labelFrame.pack(fill=BOTH, expand=True, padx=20, pady=20)

        self.uploadButton = ttk.Button(labelFrame, text="Browse A File", command=self.fileDialog)
        self.uploadButton.pack(side=TOP, pady=10)

        self.label = ttk.Label(labelFrame, text="No file selected")
        self.label.pack(side=TOP, pady=10)

        self.predictButton = ttk.Button(labelFrame, text="Predict Diagnosis", command=self.predictDiagnosis)
        self.predictButton.pack(side=TOP, pady=10)
        self.predictButton.state(['disabled'])

        self.newlabel = ttk.Label(labelFrame)
        self.newlabel.pack(side=TOP, pady=10)

    def create_backup_directory(self):
        self.backup_directory = Path(DATA_PATH)
        self.backup_directory.mkdir(parents=True, exist_ok=True)

    def fileDialog(self):
        file_types = [("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=file_types)
        if file_path:
            self.process_file_selection(file_path)

    def process_file_selection(self, file_path):
        file_path = Path(file_path)
        destination_path = self.backup_directory / file_path.name
        shutil.copy(file_path, destination_path)
        self.selected_file_name = file_path.name
        self.label.configure(text=f"File successfully copied: {self.selected_file_name}")
        self.predictButton.state(['!disabled'])

    def predictDiagnosis(self):
        self.newlabel.configure(text=f"A PDF report will be generated for {self.selected_file_name}")
        # thread to prevent GUI from freezing
        prediction_thread = Thread(target=self.run_prediction)
        prediction_thread.start()

    def run_prediction(self):
        predict_mask(self.selected_file_name)
        extract_data(self.selected_file_name)
        diagnosis = predict_diagnosis(self.selected_file_name[:-4] + '.csv')
        create_pdf(self.selected_file_name, diagnosis)
        print("Done!")

        self.newlabel.configure(text="Done!")


if __name__ == '__main__':
    gui = Gui()
    gui.mainloop()
