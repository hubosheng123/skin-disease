import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QComboBox, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from model.Unet import Unet
from model.TinyUNet import TinyUNet
from model.Transunet import TransUNet
from model.SkinUnet import SkinUNet
from model.BiseNet import BiSeNet
from model.SegNet import SegNet
from model.Unext import UNeXt
from model.Deeplab import DeepLabV3
import torch
import logging
from PIL import Image
from PIL.ImageQt import ImageQt
import torchvision.transforms as T
logging.basicConfig(level=logging.INFO)
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QApplication
)
from PyQt5.QtCore import Qt
import sys

class SkinLesionSegmentationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化UI
        self.initUI()
        
        # 模型相关配置
        self.model_weights_path = "./trained_model/"
        self.model_def_path = "./model/"
        self.selected_model = None
        self.file_path = None
        self.label_path = None

    def initUI(self):
        self.setWindowTitle("皮肤病变分割小程序")
        self.setGeometry(100, 100, 1200, 600)
        logging.info('启动成功')
        # 主控件
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # 控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        # 添加按钮
        self.btn_load = QPushButton("加载图像")
        self.btn_predict = QPushButton("预测图像")
        self.btn_clear = QPushButton("清除图像")
        self.btn_load.setFixedSize(120, 40)
        self.btn_predict.setFixedSize(120, 40)
        self.btn_clear.setFixedSize(120, 40)

        
        # 模型选择下拉框
        self.model_selector = QComboBox()
        self.model_selector.addItems(["UNet", "DeepLabV3", "TransUNet", "UNeXt", "SkinUnet", "SegNet", "BiSeNet"])
        self.model_selector.setCurrentIndex(0)
        self.model_selector.setFixedSize(300,50)
        self.model_selector.setStyleSheet("""
    QComboBox {
        font-size: 18px;
        min-height: 35px;
        padding: 3px;
    }
    QComboBox QAbstractItemView {
        font-size: 18px;
    }
""")
        # 将控件添加到控制面板
        label_title = QLabel("选择模型:")
        label_title.setStyleSheet("font-size: 19px; font-weight: bold;margin-left: 1000px;")
        control_layout.addWidget(label_title)

        control_layout.addWidget(self.model_selector)
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_predict)
        control_layout.addWidget(self.btn_clear)
        control_panel.setLayout(control_layout)

        # 图像显示区域
        image_panel = QWidget()
        image_layout = QHBoxLayout()

        # 标题 + 图像布局组合
        def create_labeled_image(title_text):
            title = QLabel(title_text)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 16px;")
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(400, 400)
            image_label.setStyleSheet("""
                QLabel {
                    background-color: #FFF8DC;
                    border: 1px solid gray;
                }
            """)
            vbox = QVBoxLayout()
            vbox.addWidget(title)
            vbox.addWidget(image_label)
            return vbox, image_label

        # 分别创建带标题的图像显示区
        original_layout, self.original_image = create_labeled_image("原图")
        label_layout, self.label_image = create_labeled_image("真实标签")
        result_layout, self.result_image = create_labeled_image("预测结果")
        overlay_layout, self.overlay_image = create_labeled_image("叠加效果")

        # 加入主图像横向布局
        image_layout.addLayout(original_layout)
        image_layout.addLayout(label_layout)
        image_layout.addLayout(result_layout)
        image_layout.addLayout(overlay_layout)

        image_panel.setLayout(image_layout)

        # 组合主界面
        main_layout.addWidget(control_panel)
        main_layout.addWidget(image_panel)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 信号连接（待实现的函数需要你已有的 load_image, predict_image, clear_images, select_model）
        self.btn_load.clicked.connect(self.load_image)
        self.btn_predict.clicked.connect(self.predict_image)
        self.btn_clear.clicked.connect(self.clear_images)
        self.model_selector.currentIndexChanged.connect(self.select_model)
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择皮肤病变图像", "", "Images (*.png *.jpg *.jpeg)")
        self.file_path = file_path
        logging.info(f"file path is {self.file_path}")
        if file_path:
            # 加载原始图像
            self.original_img = cv2.imread(file_path)
            self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            
            # 等比例缩小（例如缩放到300x300像素）
            h, w = self.original_img.shape[:2]
            scale = min(350/w, 350/h)
            resized_img = cv2.resize(self.original_img, (int(w*scale), int(h*scale)))
            logging.info(f"加载图像大小{int(w*scale)}, {int(h*scale)}")
            # 转换为QImage并显示
            bytes_per_line = 3 * resized_img.shape[1]
            q_img = QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0], 
                        bytes_per_line, QImage.Format_RGB888)
            self.original_image.setPixmap(QPixmap.fromImage(q_img))
        # 自动匹配label路径的逻辑
        if "Test_Input" in file_path:
            label_path = file_path.replace('images', 'labels').replace("Test_Input", "Test_label").replace(".jpg", "_Segmentation.png")
        elif "Training_Input" in file_path:
            label_path = file_path.replace('images', 'labels').replace("Training_Input", "Training_label").replace(".jpg", "_Segmentation.png")
        elif "Val_Input" in file_path:
            label_path = file_path.replace('images', 'labels').replace("Val_Input", "Val_label").replace(".jpg", "_Segmentation.png")
        self.label_path=label_path
        try:
            self.label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if self.label_img is not None:
                # 对label图像应用相同的缩放比例
                resized_label = cv2.resize(self.label_img, (350, int(h * scale)))
                q_img = QImage(resized_label.data, resized_label.shape[1], resized_label.shape[0], 
                             resized_label.shape[1], QImage.Format_Grayscale8)
                self.label_image.setPixmap(QPixmap.fromImage(q_img))
            else:
                self.label_image.setText("Label图像加载失败")
        except Exception as e:
            self.label_image.setText(f"加载Label图像出错: {str(e)}")
        
    def predict_image(self):
        if not hasattr(self, 'original_img'):
            return

        # 根据选择的模型加载对应的权重
        model_name = self.model_selector.currentText()
        model_weights = {
            'UNet': r'./trained_model/Unet.pth',
            'TransUNet': r'./trained_model/transunet.pth',
            'BiSeNet': r'./trained_model/bisenet.pth',
            'DeepLabV3': r'./trained_model/deeplabv3.pth',
            'UNeXt': r'./trained_model/Unext.pth',
            'SkinUnet': r'./trained_model/SkinUnet.pth',
            'SegNet': r'./trained_model/SegNet.pth'
        }

        # 初始化模型
        if model_name == 'SegNet':
            model = SegNet(3, 2)
        elif model_name == 'UNeXt':
            model = UNeXt(in_channels=3, out_channels=2)
        elif model_name == 'SkinUnet':
            model = SkinUNet(in_channels=3, num_classes=2)
        elif model_name == 'UNet':
            model = Unet(2)
        elif model_name == 'TransUNet':
            model = TransUNet(img_dim=224, in_channels=3, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=2)
        elif model_name == 'BiSeNet':
            model = BiSeNet(num_classes=2)
        elif model_name == 'DeepLabV3':
            model = DeepLabV3(2)
        else:
            QMessageBox.warning(self, "错误", "未知模型")
            return

        # 加载模型权重
       # logging.info(f"模型权重加载未成功: {model_weights[model_name]}")
        checkpoint = torch.load(model_weights[model_name], map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=True)
        model.to('cuda')
        logging.info(f"模型权重加载成功: {model_weights[model_name]}")
        model.eval()
        image = Image.open(self.file_path).convert('RGB')
        w,h = image.size
        # 预处理输入图像
        transforms = T.Compose([
        T.Resize([224,224]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
        image = transforms(image).unsqueeze(0).to('cuda')

        # 模型预测
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred[pred == 1] = 255
            mask = Image.fromarray(pred)
            scale = min(350/w, 350/h)
            out = mask.resize([int(w*scale), int(h*scale)])
            logging.info(f"预测结果大小: {out.size}")
                # 将 PIL Image 转为 QImage
            out_rgb = out.convert("RGB")  # 转换为 RGB 图像以支持 QImage
            
            out_np = np.array(out_rgb)

# 获取尺寸信息
            h, w, ch = out_np.shape
            bytes_per_line = ch * w

# 转换为 QImage
            qimage = QImage(out_np.data, w, h, bytes_per_line, QImage.Format_RGB888)

# 转为 QPixmap 并显示
            pixmap = QPixmap.fromImage(qimage)
            self.result_image.setPixmap(pixmap)
            #self.result_image.setScaledContents(True)
            logging.info("预测结果已成功显示在结果图像区域（第三部分）。")
            # 加载原始图（用于叠加）
            original_img = cv2.imread(self.file_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转为RGB
            overlay_img = original_img.copy()

            # Resize 原图用于等比显示
            h_ori, w_ori = overlay_img.shape[:2]
            scale = min(350 / w_ori, 350 / h_ori)
            overlay_img = cv2.resize(overlay_img, (int(w_ori * scale), int(h_ori * scale)))

            # 预测掩膜转换为 numpy 并 resize
            pred_mask = np.array(mask.resize((overlay_img.shape[1], overlay_img.shape[0])))  # 预测掩膜resize

            # 假设你有真实标签路径：self.label_path
            if hasattr(self, 'label_path') and self.label_path:
                true_mask_pil = Image.open(self.label_path).convert('L')
                true_mask = np.array(true_mask_pil.resize((overlay_img.shape[1], overlay_img.shape[0])))
                true_mask_bin = (true_mask > 127).astype(np.uint8)
            else:
                true_mask_bin = np.zeros_like(pred_mask, dtype=np.uint8)  # 没有标签则显示空白

            # 提取轮廓
            contours_pred, _ = cv2.findContours((pred_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_true, _ = cv2.findContours(true_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 在原图上画轮廓
            cv2.drawContours(overlay_img, contours_true, -1, (255, 0, 0), 1)   # 蓝色，真实标签
            cv2.drawContours(overlay_img, contours_pred, -1, (0, 0, 255), 1) # 紫红色，预测掩膜

            # 转为 QImage 显示
            overlay_np = overlay_img
            h, w, ch = overlay_np.shape
            bytes_per_line = ch * w
            qimage_overlay = QImage(overlay_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap_overlay = QPixmap.fromImage(qimage_overlay)
            self.overlay_image.setPixmap(pixmap_overlay)
            # self.overlay_image.setScaledContents(True)  # 可选
            logging.info("原图+预测+标签叠加图已显示在第四部分")

        
    
    def clear_images(self):
        self.original_image.clear()
        self.label_image.clear()
        self.result_image.clear()
        self.overlay_image.clear()
    
    def select_model(self):
        self.selected_model = self.model_selector.currentText()
        print(f"已选择模型: {self.selected_model}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkinLesionSegmentationUI()
    window.show()
    sys.exit(app.exec_())