import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from UI_329_up_up import Ui_MainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal,Qt, QPoint,QRect
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from transformers import SamConfig,SamModel,SamImageProcessor,SamProcessor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog


class MainWin(QMainWindow,Ui_MainWindow):


    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    send_img_path = pyqtSignal(str)
    send_checkpoint_path = pyqtSignal(str)

    boxSelected = pyqtSignal(int, int, int, int) 

    def __init__(self):
        super(MainWin,self).__init__()
        self.setupUi(self)
        self.img = None
        self.box = None
        self.res_img = None
        self.vit_path = None
        self.image_path = None
        self.image_paint = QRect(200, 178, 256, 256)


        self.keep = False
        self.drawing = False
        self.end_point = QPoint()
        self.start_point = QPoint()
        self.setMouseTracking(True)
        self.contain_in_label = False


        # self.pushButton.clicked.connect(self.show_image)
        self.ChooseCheckpoint.clicked.connect(self.choose_checkpoint)
        self.send_img_path.connect(self.get_img)
        self.send_checkpoint_path.connect(self.get_vit_path)
        self.pushButton_2.clicked.connect(self.run_model)
        self.ReloadButton.clicked.connect(self.Reload)
        self.SaveImageButton.clicked.connect(self.save_photo)
        self.boxSelected.connect(self.show_pos)
        # self.showGTButton.clicked.connect(self.show_GT)

        self.chooseImage.triggered.connect(self.show_image)
        self.chooseGT.triggered.connect(self.show_GT)
        self.org_checkpoint.triggered.connect(self.choose_orgcheckpoint)
        self.lung_chentpoint.triggered.connect(self.choose_lungcheckpoint)
        self.heart_checkpoint.triggered.connect(self.choose_heartcheckpoint)
        self.spolon_checkpoint.triggered.connect(self.choose_spleencheckpoint)
        self.colon_checkpoint.triggered.connect(self.choose_coloncheckpoint)




    def show_pos(self, a,b,c,d):
        self.box = [[[a-200, b-178, c-200, d-178]]]
        # print(f'{a}, {b}, {c}, {d}')
        print(f'box_prompt:{self.box}')
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if (self.drawing or self.keep) and self.image_path is not None:

            painter = QPainter(self)
            label_rect_global = QRect(200, 178, 256, 256)
            painter.drawImage(label_rect_global, QImage(self.image_path))
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(self.start_point.x(), self.start_point.y(),
                             self.end_point.x() - self.start_point.x(),
                             self.end_point.y() - self.start_point.y())
            
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.image_paint.contains(event.pos()):
            self.drawing = True
            self.start_point = event.pos()


    def mouseMoveEvent(self, event):
        if not self.image_paint.contains(event.pos()):
            self.drawing = False

        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        # label_rect = self.image.rect()
        if event.button() == Qt.LeftButton and self.image_paint.contains(event.pos()) and self.drawing:
            self.drawing = False
            self.keep = True
            self.end_point = event.pos()
            self.update()
            self.boxSelected.emit(self.start_point.x(), self.start_point.y(), self.end_point.x(), self.end_point.y())

    def run_model(self):
        assert self.img is not None,'choose a image first please'
        assert self.vit_path is not None,'choose the checkpoint first please'
        assert self.box is not None,'choose the box first please'

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'using {device}')

        config = SamConfig()
        model = SamModel(config).to(device)


        model.load_state_dict(torch.load(self.vit_path))
        img_processor = SamImageProcessor()
        processor = SamProcessor(img_processor)

        input_boxes = self.box
        inputs = processor(self.img, input_boxes=input_boxes, return_tensors="pt").to(device)
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])


        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        mask_list = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        mask = mask_list[0].squeeze()
        mask = np.array(mask)
        mask = mask.astype(np.uint8)
        mask_show = mask*255

        qimage = QImage(mask_show.data, mask.shape[1], mask.shape[0], QImage.Format_Grayscale8)
    
        # 从 QImage 创建 QPixmap
        self.res_img = qimage
        pixmap = QPixmap.fromImage(qimage)
        self.mask.setPixmap(pixmap)
        self.mask.setScaledContents(True)
        print(f'success! using: {self.vit_path}')

    def get_img(self,img_path):
        img = Image.open(img_path)
        self.img = img

    def get_vit_path(self,vit_path):
        # print(vit_path)
        self.vit_path = vit_path

    def show_image(self):
        # 选择图像文件
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg)")
        if not file_path:
            return
        # 读取图像
        print(f'loading image:{file_path}')
        if file_path:
            self.send_img_path.emit(file_path)
            # 显示选择的图片
            self.image_path = file_path

    def show_GT(self):
        # 选择图像文件
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg)")
        if not file_path:
            return

        print(f'loading GT:{file_path}')
        if file_path:
            pixmap = QPixmap(file_path)
            self.GT.setPixmap(pixmap)
            self.GT.setScaledContents(True)
            

    def choose_checkpoint(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self)
        if not file_path:
            return
        if file_path:
            self.send_checkpoint_path.emit(file_path)

    def Reload(self):
        self.vit_path = None
        self.image_path = None
        self.GT.setPixmap(QPixmap())
        self.mask.setPixmap(QPixmap())

    def choose_orgcheckpoint(self):
        self.vit_path = r'res_checkpoint\vit_b.pth'
    def choose_lungcheckpoint(self):
        self.vit_path = r'res_checkpoint\3_16_lung.pth'
    def choose_heartcheckpoint(self):
        self.vit_path = r'res_checkpoint\3_17_heart.pth'
    def choose_spleencheckpoint(self):
        self.vit_path = r'res_checkpoint\3_19_spleen.pth'
    def choose_coloncheckpoint(self):
        self.vit_path = r'res_checkpoint\3_31_colon.pth'

    def save_photo(self):
        file_dialog = QFileDialog()
        # file_path, _ = file_dialog.getOpenFileName(self)
        file_path, _ = QFileDialog.getSaveFileName(self, "save_image", "", "Images (*.png *.jpg *.bmp)")

        if not file_path:
            return
        
        if self.res_img is not None:
            pixmap = QPixmap.fromImage(self.res_img)
            pixmap.save(file_path)
            print("save_path:", file_path)
        else:
            print('Please generate the image first!')
        

if __name__ == '__main__':
    app = QApplication([])
    mywin = MainWin()
    mywin.show()
    app.exit(app.exec_())