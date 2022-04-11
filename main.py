from PyQt5 import QtWidgets, uic
from PyQt5 import *
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import pearsonr

i = 1
varX = []
varY = []
var = []
var_Oto = []
path = ''


class Ui(QtWidgets.QMainWindow):

    def __init__(self):

        super(Ui, self).__init__()
        uic.loadUi(
            'interface.ui', self)

        self.onlyInt = QtGui.QIntValidator()

        self.label_2 = self.findChild(
            QtWidgets.QLabel, 'label_2')
        self.label_3 = self.findChild(
            QtWidgets.QLabel, 'label_3')
        self.XEdit = self.findChild(
            QtWidgets.QLineEdit, 'XEdit')
        self.YEdit = self.findChild(
            QtWidgets.QLineEdit, 'YEdit')
        self.variabelButton = self.findChild(
            QtWidgets.QPushButton, 'variabelButton')
        self.variabelButton.setDisabled(True)
        self.XEdit.textChanged.connect(self.enablevariabelButton)
        self.YEdit.textChanged.connect(self.enablevariabelButton)
        self.resetButton = self.findChild(
            QtWidgets.QPushButton, 'resetButton')
        self.regresiButton = self.findChild(
            QtWidgets.QPushButton, 'regresiButton')

        self.About_Me.triggered.connect(lambda: self.aboutMe())

        self.XEdit.setValidator(self.onlyInt)
        self.YEdit.setValidator(self.onlyInt)
        self.variabelButton.clicked.connect(self.variabelButtonPressed)
        self.resetButton.clicked.connect(self.resetButtonPressed)
        self.label_2.setText("Variabel X-"+str(i))
        self.label_3.setText("Variabel Y-"+str(i))
        self.regresiButton.clicked.connect(self.regresiButtonPressed)
        self.importButton.clicked.connect(self.importButtonPressed)
        self.otoregresiButton.clicked.connect(self.otoregresiButtonPressed)
        self.otoresetButton.clicked.connect(self.otoresetButtonPressed)

        header = self.manualtableWidget.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        header2 = self.tableWidget_2.horizontalHeader()
        header2.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header2.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header2.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        self.show()

    def variabelButtonPressed(self):
        global i
        self.manualtableWidget.setRowCount(i)
        self.manualtableWidget.setColumnCount(2)
        self.label_2.setText("Variabel X-"+str(i+1))
        self.label_3.setText("Variabel Y-"+str(i+1))
        variabelX = (self.XEdit.text())
        variabelY = (self.YEdit.text())
        self.manualtableWidget.setItem(
            (i-1), 0, QtWidgets.QTableWidgetItem(variabelX))
        self.manualtableWidget.setItem(
            (i-1), 1, QtWidgets.QTableWidgetItem(variabelY))
        i = i + 1
        QtWidgets.QApplication.processEvents()

    def enablevariabelButton(self):
        if ((len(self.XEdit.text()) > 0) and (len(self.YEdit.text()) > 0)):
            self.variabelButton.setDisabled(False)
        else:
            self.variabelButton.setDisabled(True)

    def regresiButtonPressed(self):
        global arrayItem
        coloums = 2
        rows = i-1

        for row in range(rows):
            for coloum in range(coloums):
                val = (self.manualtableWidget.item(row, coloum).text())
                val = float(val)
                var.append(val)

        var_ = np.array(var)
        newVar = var_.reshape(rows, coloums)
        my_df = pd.DataFrame(newVar)
        my_df.to_csv('dataset.csv', index=False, header=True)

        df = pd.read_csv('dataset.csv')
        x = (df.iloc[:, 0]).values.reshape(-1, 1)
        y = (df.iloc[:, 1]).values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.4)
        lin_reg = LinearRegression()
        lin_reg.fit(x, y)

        plt.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter X vs Y')
        plt.show()

        x = (df.iloc[:, 0])
        y = (df.iloc[:, 1])
        corr, _ = pearsonr(x, y)
        self.korelasiEdit.setText(str(corr))

        slope = lin_reg.coef_
        intercept = lin_reg.intercept_

        formula = str(round(float(slope))) + \
            "x + ("+str(round(float(intercept)))+")"

        self.formulaEdit.setText(formula)
        self.koefisienEdit.setText(str(lin_reg.score(x_test, y_test)))

        keterangan = "KETERANGAN\nSignifikansi Korelasi Pearson yang digunakan untuk mengetahui tingkat hubungan antara variabel bebas dan terikat adalah sebesar" + \
            str(corr)

        self.textBrowser.append(keterangan)

        var.clear()

    def resetButtonPressed(self):
        global i
        global arrayItem
        global var
        i = 1
        varX.clear()
        varY.clear()
        var.clear()
        self.label_2.setText("Variabel X-"+str(i))
        self.label_3.setText("Variabel Y-"+str(i))
        self.manualtableWidget.setRowCount(0)
        self.textBrowser.clear()

    def aboutMe(self):
        print("Halo")
        QtWidgets.QMessageBox.about(
            self, "About Me", "Author. Ivan Pakpahan\nVersion.1.0.0\nUTS Pemrograman Simulasi")
        QtWidgets.QApplication.processEvents()

    def importButtonPressed(self):
        global path
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file', '',
                                                     'CSV Files (*.csv*)')
        if path != ('', ''):
            path = (path[0])
        self.dirEdit.setText(path)

        with open(path) as file_name:
            array = np.loadtxt(file_name, delimiter=",")

        rows, coloums = array[1:, :].shape

        self.tableWidget_2.setRowCount(rows)
        self.tableWidget_2.setColumnCount(coloums)
        for row in range(rows):
            for coloum in range(coloums):
                self.tableWidget_2.setItem(
                    row, coloum, QtWidgets.QTableWidgetItem(str(array[row+1, coloum])))

        QtWidgets.QApplication.processEvents()

    def otoregresiButtonPressed(self):
        global path
        global var_Oto
        with open(path) as file_name:
            array = np.loadtxt(file_name, delimiter=",")

        rows, coloums = array[1:, :].shape
        for row in range(rows):
            for coloum in range(coloums):
                val_Oto = (self.tableWidget_2.item(row, coloum).text())
                val_Oto = float(val_Oto)
                var_Oto.append(val_Oto)

        var_Oto_ = np.array(var_Oto)
        newVar_Oto = var_Oto_.reshape(rows, coloums)
        my_df_Oto = pd.DataFrame(newVar_Oto)
        my_df_Oto.to_csv('dataset_Oto.csv', index=False, header=True)

        df = pd.read_csv('dataset_Oto.csv')
        x = (df.iloc[:, 0]).values.reshape(-1, 1)
        y = (df.iloc[:, 1]).values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.4)
        lin_reg = LinearRegression()
        lin_reg.fit(x, y)

        plt.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter X vs Y')
        plt.show()

        x = (df.iloc[:, 0])
        y = (df.iloc[:, 1])
        corr, _ = pearsonr(x, y)
        self.otokorelasiEdit.setText(str(corr))

        slope = lin_reg.coef_
        intercept = lin_reg.intercept_

        formula = str(round(float(slope))) + \
            "x + ("+str(round(float(intercept)))+")"

        self.otoformulaEdit.setText(formula)
        self.otokoefisienEdit.setText(str(lin_reg.score(x_test, y_test)))

        keterangan = "KETERANGAN\nSignifikansi Korelasi Pearson yang digunakan untuk mengetahui tingkat hubungan antara variabel bebas dan terikat adalah sebesar" + \
            str(corr)

        self.textBrowser_2.append(keterangan)

        var_Oto.clear()

        QtWidgets.QApplication.processEvents()

    def otoresetButtonPressed(self):
        global var_Oto
        global path
        var_Oto.clear()
        path = ''
        self.dirEdit.setText(path)
        self.tableWidget_2.setRowCount(0)
        self.textBrowser_2.clear()


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
