from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.properties import StringProperty, ObjectProperty
from table_detect import recognize_table
from tt import jj
import cv2
import numpy as np
import pandas as pd
import pytesseract
try:
    from PIL import Image
except ImportError:
    import Image
import matplotlib.pyplot as plt
import os
Window.size = (600, 400)


class MainScreen(FloatLayout):
    filePath = StringProperty('')
    fileXlsxPath = StringProperty('')
    data = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        # self.size = (600, 400)
        self.add_widget(Label(text='Drag and drop image', color=(0,0,0,1), pos=(0,150)))

        self.preproc_btn = Button(text='Подготовить изображение', size_hint=(.25, .15), pos=(20, 0))
        self.recog_btn = Button(text='Распознать текст', size_hint=(.25, .15), pos=(520, 0))
        self.create_btn = Button(text='Сформировать xls', size_hint=(.25, .15), pos=(1020, 0))

        self.preproc_btn.bind(on_press=self.preproc)
        self.recog_btn.bind(on_press=self.recognize_table)
        self.create_btn.bind(on_press=self.data_to_excel)

        self.add_widget(self.preproc_btn)
        self.add_widget(self.recog_btn)
        self.add_widget(self.create_btn)
        # self.add_widget(self.norm_btn)

        Window.bind(on_dropfile=self._on_file_drop)

    def _on_file_drop(self, window, file_path):
        print(file_path)
        self.filePath = file_path.decode("utf-8")  # convert byte to string
        self.ids.img.source = self.filePath
        self.ids.img.reload()

    def get_tables(self, img, contours):
        # areaThr - константа обозначающая минимальную площадь контура
        areaThr = 10000
        i = 0
        contours1=[]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area > areaThr):
                i = i + 1
                contours1.append(cnt)
                x, y, width, height = cv2.boundingRect(cnt)
                filename = 'output' + str(i) + '.jpg'
                cv2.imwrite(filename, img[y:y + height - 1, x:x + width - 1])
        contour_sizes = [(cv2.contourArea(cnt), cnt) for cnt in contours1]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x,y,width,height=cv2.boundingRect(biggest_contour)
        filename = 'output_biggest_contour.jpg'
        cv2.imwrite(filename, img[y:y + height - 1, x:x + width - 1])
        self.filePath = filename
        self.ids.img.reload()

    def preproc(self, widget):
        img = cv2.imread(self.filePath)
        # cv2.imshow('result',img)
        print(self.filePath)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        canny = cv2.Canny(binary, 100, 200)

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('res',canny)
        self.get_tables(img, contours)
        cv2.imshow("res", img)
        cv2.waitKey()

    def sort_contours(self, cnts, method="left-to-right"):
        # инициализация обратного флага и индекса сортировки
        reverse = False
        i = 0
        # перебрать в обратном порядке
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # Если Сортировка по y-коодинате,а не по x-координате
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # Сортировка списка прямоугольных контуров сверху вниз
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # список отсортированных контуров и выделенных прямоугольников
        return (cnts, boundingBoxes)

    def recognize_table(self, widget):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # считывание файла
        file = self.filePath
        img = cv2.imread(file, 0)
        res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = res
        img.shape
        # пороговая бинаризация
        thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # инвертирование изображения
        img_bin = 255 - img_bin
        cv2.imwrite('cv_inverted.png', img_bin)
        # Вывод результата
        plotting = plt.imshow(img_bin, cmap='gray')
        plt.show()

        # Создание фильтра длинна, ядра которого равно сотой от общей ширины
        kernel_len = np.array(img).shape[1] // 100
        # Определение вертикального ядра для нахождения вертикальных линий
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        # Определение горизонтального ядра для нахождения горизонтальных линий
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        # Ядро 2x2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Использование вертикального ядра для нахождения вертикальных линий и сохранения в jpg
        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
        cv2.imwrite("vertical.jpg", vertical_lines)
        # Вывод результата
        plotting = plt.imshow(image_1, cmap='gray')
        plt.show()

        # Использование горизонтального ядра для нахождения горизонтальныъ линий и сохранения в jpg
        image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
        cv2.imwrite("horizontal.jpg", horizontal_lines)
        # Вывод результата
        plotting = plt.imshow(image_2, cmap='gray')
        plt.show()

        # Объединение горизонтальных и вертикальных линий в новом изображении
        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        # Эрозия и бинаризация
        img_vh = cv2.erode(~img_vh, kernel, iterations=3)
        thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite("img_vh.jpg", img_vh)
        # Вычисление побитового исключающего или и отрицания
        bitxor = cv2.bitwise_xor(img, img_vh)
        bitnot = cv2.bitwise_not(bitxor)
        # Вывод результата
        plotting = plt.imshow(bitnot, cmap='gray')
        plt.show()

        # Обнаружение контуров в пустой таблице
        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Сортировака контуров сверху вниз
        contours, boundingBoxes = self.sort_contours(contours, method="top-to-bottom")

        # Создание списка высот для всех обнаруженных рамок
        heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        # Вычисление среднего значения высоты
        mean = np.mean(heights)

        # Список для хранения всех рамок
        box = []
        # Получения x и y координата, высоты и ширины рамок
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w < 1000 and h < 500):
                image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box.append([x, y, w, h])
        plotting = plt.imshow(image, cmap="gray")
        plt.show()

        # Создание двух списков для определения строки и столбца, в которых находится ячейка
        row = []
        column = []
        j = 0
        # Сортировка рамок по соответствующей строке и столбцу
        for i in range(len(box)):
            if (i == 0):
                column.append(box[i])
                previous = box[i]
            else:
                if (box[i][1] <= previous[1] + mean / 2):
                    column.append(box[i])
                    previous = box[i]
                    if (i == len(box) - 1):
                        row.append(column)
                else:
                    row.append(column)
                    column = []
                    previous = box[i]
                    column.append(box[i])
        print(column)
        print(row)

        # расчет максимального количества ячеек
        countcol = 0
        for i in range(len(row)):
            countcol = len(row[i])
            if countcol > countcol:
                countcol = countcol

        # Получение центра каждого столбца
        center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
        center = np.array(center)
        center.sort()

        finalboxes = []
        for i in range(len(row)):
            lis = []
            for k in range(countcol):
                lis.append([])
            for j in range(len(row[i])):
                diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                lis[indexing].append(row[i][j])
            finalboxes.append(lis)

        outer = []
        for i in range(len(finalboxes)):
            for j in range(len(finalboxes[i])):
                inner = ""
                if (len(finalboxes[i][j]) == 0):
                    outer.append(' ')
                else:
                    for k in range(len(finalboxes[i][j])):
                        y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                     finalboxes[i][j][k][3]
                        finalimg = bitnot[x:x + h, y:y + w]
                        # Предобработка для каждой ячейки
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        dilation = cv2.dilate(resizing, kernel, iterations=1)
                        erosion = cv2.erode(dilation, kernel, iterations=1)
                        config = r'--oem 3 --psm 6'
                        out = pytesseract.image_to_string(erosion, lang='rus+eng')
                        if (len(out) == 0):
                            out = pytesseract.image_to_string(erosion, lang='rus+eng')
                        inner = inner + " " + out
                    outer.append(inner)

        # Создание фрейма данных сгенерированного списка результатов распознавания
        arr = np.array(outer)
        dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
        print(dataframe)
        self.data = dataframe.style.set_properties(align="left")
        # Конвертация в xls
        self.fileXlsxPath = "output.xlsx"

    def data_to_excel(self, widget):
        self.data.to_excel(self.fileXlsxPath)


class Recognize(App):
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    Recognize().run()

