from tkinter import ttk
import tkinter as tk

from keras.models import load_model
from PIL import ImageGrab, Image
import numpy as np


def predict_digit(img, model):
    # изменение рзмера изобржений на 28x28
    img = img.resize((28,28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    # изменение размерности для поддержки модели ввода и нормализации
    img = img.reshape(1,28,28,1)
    img = img/255.0
    # предсказание цифры
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.x = self.y = 0
        
        # Создание элементов
        self.variable_models = tk.StringVar()
        self.selector_models = ttk.Combobox(self, textvariable=self.variable_models, values=('mnist1.h5', 'mnist2.h5'))
        self.selector_models.bind('<<ComboboxSelected>>', self.set_model_by_selector)
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Распознать", command =         self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Очистить", command = self.clear_all)
        
        # Сетка окна
        self.selector_models.grid(row=0, column=0)
        self.canvas.grid(row=1, column=0, pady=2, sticky=tk.W, )
        self.label.grid(row=1, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=2, column=1, pady=2, padx=2)
        self.button_clear.grid(row=2, column=0, pady=2)
        
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        self.model = None

    def set_model(self, model_name):
        self.model = load_model(model_name)

    def set_model_by_selector(self, event):
        model_name = self.variable_models.get()
        print(f'Выбрана модель - {model_name}')
        self.set_model(model_name)

    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        if not self.model:
            print('Пожалуйста, выберите модель из списка.')
            return

        # получаем координату холста
        x, y = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        width, height = int(self.canvas.config('width')[-1]), int(self.canvas.config('height')[-1])
        rect = (x, y, x + width, y + height)
        im = ImageGrab.grab(rect)
        
        digit, acc = predict_digit(im, self.model)
        self.label.configure(text= str(digit)+', \n'+ str(int(acc*100))+'%')
        
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')


app = App()
tk.mainloop()
