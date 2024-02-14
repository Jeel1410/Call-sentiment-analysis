import sys
import os
import re
import speech_recognition as sr
from transformers import pipeline
import spacy
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QTextEdit, QGridLayout


class AudioAnalysis(QWidget):

    def __init__(self):
        super().__init__()

        # Set window properties
        self.title = 'Audio Analysis'
        self.left = 200
        self.top = 200
        self.width = 600
        self.height = 400
        self.initUI()

    def initUI(self):

        # Create grid layout
        grid = QGridLayout()

        # Create browse button
        self.browse_btn = QPushButton('Browse', self)
        self.browse_btn.setToolTip('Select an audio file')
        self.browse_btn.clicked.connect(self.get_audio_file)
        grid.addWidget(self.browse_btn, 0, 0)

        # Create analyse button
        self.analyse_btn = QPushButton('Analyse', self)
        self.analyse_btn.setToolTip('Analyse the selected audio file')
        self.analyse_btn.clicked.connect(self.audio_analysis)
        grid.addWidget(self.analyse_btn, 0, 1)

        # Create text editor to display the results
        self.results_editor = QTextEdit(self)
        grid.addWidget(self.results_editor, 1, 0, 1, 2)

        # Set window layout
        self.setLayout(grid)

        # Set window properties
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # Function to get the audio file
    def get_audio_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)",
                                                   options=options)
        if file_name:
            file_name = os.path.basename(file_name)
            self.browse_btn.setText(file_name)

    # Function to perform audio analysis
    def audio_analysis(self):
        file_name = self.browse_btn.text()
        if file_name == 'Browse':
            self.results_editor.setText('Please select an audio file to analyse.')
        else:
            spacy_model = spacy.load('en_core_web_sm')
            sentiment_analysis_pipeline = pipeline(model='federicopascual/finetuning-sentiment-model-3000-samples')
            r = sr.Recognizer()
            try:
                # Print the file name for debugging
                print("File name:", file_name)
                with sr.AudioFile(file_name) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
                    analysis, processed_text = self.preprocessing_text(text, spacy_model, sentiment_analysis_pipeline)
                    result_text = f'Results:\n\nText: {processed_text}\n\nAnalysis: {analysis[0]["label"]}'
                    self.results_editor.setText(result_text)
            except Exception as e:
                self.results_editor.setText(f'Error: {str(e)}')

    # Function to preprocess the text and perform sentiment analysis
    def preprocessing_text(self, sentence, spacy_model, sentiment_analysis_pipeline):
        analysis = sentiment_analysis_pipeline(sentence)
        ## data masking
        token = spacy_model(sentence)
        data_mask = []
        for i in token.ents:
            data_mask.append(i.text)
        for i in data_mask:
            sentence = re.sub(i, 'xxxxxxx', sentence)

        # Convert label 1 to positive and label 0 to negative
        if analysis[0]['label'] == 'LABEL_1':
            analysis[0]['label'] = 'positive'
        else:
            analysis[0]['label'] = 'negative'

        return analysis, sentence


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioAnalysis()
    sys.exit(app.exec_())
