import tkinter as tk
from tkinter import *
from tkinter import messagebox
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

global main_screen
main_screen = Tk()
X_train = pd.read_csv("Dataset Train/X_train.csv")
y_train = pd.read_csv("Dataset Train/y_train.csv")
X_test = pd.read_csv("Dataset test/x_test.csv")
y_test = pd.read_csv("Dataset test/y_test.csv")
X_train.drop(columns='Unnamed: 561', inplace=True)
train_size = int(X_train.shape[0] * 0.8)
X_tr = X_train[0:train_size]
y_tr = y_train[0:train_size]
X_val = X_train[train_size:]
y_val = y_train[train_size:]
def lr():
    msg=Toplevel(main_screen)
    msg.title("Report")
    msg.geometry("800x800")
    Label(msg,text="Detailed Report Logistic Regression", bg="black",fg="white", width="300", height="2", font=("Bold", 13)).pack(pady=10)
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    lr.fit(X_tr,y_tr)
    pred=lr.predict(X_test)
    Label(msg, text="Train Accuracy: ").pack()
    Label(msg, text=lr.score(X_tr, y_tr)).pack()
    Label(msg, text="Validation Accuracy: ").pack()
    Label(msg, text=lr.score(X_val, y_val)).pack()
    Label(msg, text="Test Accuracy: ").pack()
    Label(msg, text=lr.score(X_test, y_test)).pack()
    Label(msg,text="Classification Report").pack()
    Label(msg,text=classification_report(y_test, pred)).pack()
    Label(msg,text="Confusion Matrix").pack()
    Label(msg,text=confusion_matrix(y_test, pred)).pack()
    Label(msg,text="F1- Score").pack()
    Label(msg,text=f1_score(y_test, pred,average='weighted')).pack()
    Label(msg,text="Accuracy Score").pack()
    Label(msg,text=accuracy_score(y_test, pred)).pack()

def lsvm():
    msg = Toplevel(main_screen)
    msg.title("Report")
    msg.geometry("800x800")
    Label(msg, text="Detailed Report Linear Support Vector Machine", bg="black", fg="white", width="300", height="2", font=("Bold", 13)).pack(pady=10)
    from sklearn import svm
    lr = svm.SVC(C=1, kernel='linear')
    lr.fit(X_tr,y_tr)
    pred = lr.predict(X_test)
    Label(msg, text="Train Accuracy: ").pack()
    Label(msg, text=lr.score(X_tr, y_tr)).pack()
    Label(msg, text="Validation Accuracy: ").pack()
    Label(msg, text=lr.score(X_val, y_val)).pack()
    Label(msg, text="Test Accuracy: ").pack()
    Label(msg, text=lr.score(X_test, y_test)).pack()
    Label(msg, text="Classification Report").pack()
    Label(msg, text=classification_report(y_test, pred)).pack()
    Label(msg, text="Confusion Matrix").pack()
    Label(msg, text=confusion_matrix(y_test, pred)).pack()
    Label(msg, text="F1- Score").pack()
    Label(msg, text=f1_score(y_test, pred, average='weighted')).pack()
    Label(msg, text="Accuracy Score").pack()
    Label(msg, text=accuracy_score(y_test, pred)).pack()
def knn():
    msg = Toplevel(main_screen)
    msg.title("Report")
    msg.geometry("800x800")
    Label(msg, text="Detailed Report K Nearest Neighbour", bg="black", fg="white", width="300", height="2", font=("Bold", 13)).pack(pady=10)
    from sklearn.neighbors import KNeighborsClassifier
    lr = KNeighborsClassifier()
    lr.fit(X_tr, y_tr)
    pred = lr.predict(X_test)
    Label(msg, text="Train Accuracy: ").pack()
    Label(msg, text=lr.score(X_tr, y_tr)).pack()
    Label(msg, text="Validation Accuracy: ").pack()
    Label(msg, text=lr.score(X_val, y_val)).pack()
    Label(msg, text="Test Accuracy: ").pack()
    Label(msg, text=lr.score(X_test, y_test)).pack()
    Label(msg, text="Classification Report").pack()
    Label(msg, text=classification_report(y_test, pred)).pack()
    Label(msg, text="Confusion Matrix").pack()
    Label(msg, text=confusion_matrix(y_test, pred)).pack()
    Label(msg, text="F1- Score").pack()
    Label(msg, text=f1_score(y_test, pred, average='weighted')).pack()
    Label(msg, text="Accuracy Score").pack()
    Label(msg, text=accuracy_score(y_test, pred)).pack()
def gnb():
    msg = Toplevel(main_screen)
    msg.title("Report")
    msg.geometry("800x800")
    Label(msg, text="Detailed Report Gaussian Naive Bayes", bg="black", fg="white", width="300", height="2", font=("Bold", 13)).pack(pady=10)
    from sklearn.naive_bayes import GaussianNB
    lr = GaussianNB()
    lr.fit(X_tr, y_tr)
    pred = lr.predict(X_test)
    Label(msg, text="Train Accuracy: ").pack()
    Label(msg, text=lr.score(X_tr, y_tr)).pack()
    Label(msg, text="Validation Accuracy: ").pack()
    Label(msg, text=lr.score(X_val, y_val)).pack()
    Label(msg, text="Test Accuracy: ").pack()
    Label(msg, text=lr.score(X_test, y_test)).pack()
    Label(msg, text="Classification Report").pack()
    Label(msg, text=classification_report(y_test, pred)).pack()
    Label(msg, text="Confusion Matrix").pack()
    Label(msg, text=confusion_matrix(y_test, pred)).pack()
    Label(msg, text="F1- Score").pack()
    Label(msg, text=f1_score(y_test, pred, average='weighted')).pack()
    Label(msg, text="Accuracy Score").pack()
    Label(msg, text=accuracy_score(y_test, pred)).pack()
def dt():
    msg = Toplevel(main_screen)
    msg.title("Report")
    msg.geometry("800x800")
    Label(msg, text="Detailed Report Decision Tree", bg="black", fg="white", width="300", height="2", font=("Bold", 13)).pack(pady=10)
    from sklearn.tree import DecisionTreeClassifier
    lr = DecisionTreeClassifier()
    lr.fit(X_tr, y_tr)
    pred = lr.predict(X_test)
    Label(msg, text="Train Accuracy: ").pack()
    Label(msg, text=lr.score(X_tr, y_tr)).pack()
    Label(msg, text="Validation Accuracy: ").pack()
    Label(msg, text=lr.score(X_val, y_val)).pack()
    Label(msg, text="Test Accuracy: ").pack()
    Label(msg, text=lr.score(X_test, y_test)).pack()
    Label(msg, text="Classification Report").pack()
    Label(msg, text=classification_report(y_test, pred)).pack()
    Label(msg, text="Confusion Matrix").pack()
    Label(msg, text=confusion_matrix(y_test, pred)).pack()
    Label(msg, text="F1- Score").pack()
    Label(msg, text=f1_score(y_test, pred, average='weighted')).pack()
    Label(msg, text="Accuracy Score").pack()
    Label(msg, text=accuracy_score(y_test, pred)).pack()

def cc():
    msg = Toplevel(main_screen)
    msg.title("Comparative Study")
    msg.geometry("800x800")
    df_model = pd.DataFrame({'Model_Applied': ['GaussianNB', 'LR', 'Linear SVM', 'KNN', 'DT'], 'Test_Accuracy': [71.3789, 93.3270, 94.0860, 88.0455, 82.8906]})
    figure1 = plt.Figure(figsize=(5, 6), dpi=80)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, msg)
    bar1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, ipadx=10, ipady=100, padx=10, pady=10)
    df_model.plot(kind='bar', x='Model_Applied',legend=True, ax=ax1)
    ax1.set_title('Comparative Study of different models')

def main_account_screen():
    main_screen.geometry("600x500")
    main_screen.config(bg ="white")
    main_screen.title("Model Names")
    Label(text="SELECT YOUR CHOICE", bg="black",fg="white", width="300", height="2", font=("Bold", 13)).pack(pady=10)
    Button(main_screen,text="KNN", height="2", width="30", command=knn,font=("Bold",13)).pack(pady=10)
    Button(main_screen,text="Linear SVM", height="2", width="30", command=lsvm,font=("Bold",13)).pack(pady=10)
    Button(main_screen,text="Logistic Regression", height="2", width="30", command=lr,font=("Bold",13)).pack(pady=10)
    Button(main_screen,text="Gaussian Naive Bayes", height="2", width="30", command=gnb,font=("Bold",13)).pack(pady=10)
    Button(main_screen,text="Decision Tree", height="2", width="30", command=dt,font=("Bold",13)).pack(pady=10)
    Button(main_screen, text="Comparative Chart", height="2", width="30", command=cc, font=("Bold", 13)).pack(pady=10)
main_account_screen()      
main_screen.mainloop()
