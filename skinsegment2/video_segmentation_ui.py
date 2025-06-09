import sys
import os
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QComboBox, QLabel, QFileDialog, QMessageBox,
                            QProgressBar, QSpinBox, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import logging
from PIL import Image
import torchvision.transforms as T

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessWorker(QThread):
    """视频处理工作线程"""
    progress_updated = pyqtSignal(int)  # 进度信号
    frame_processed = pyqtSignal(np.ndarray)  # 处理后的帧信号
    processing_finished = pyqtSignal(str)  # 处理完成信号
    error_occurred = pyqtSignal(str)  # 错误信号

    def __init__(self, video_path, model, output_path, batch_size=4):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.output_path = output_path
        self.batch_size = batch_size
        self.is_running = True

    def run(self):
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")

            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            frame_count = 0
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 写入处理后的帧
                out.write(processed_frame)
                
                # 更新进度
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)
                
                # 发送处理后的帧用于预览
                self.frame_processed.emit(processed_frame)

            # 释放资源
            cap.release()
            out.release()
            
            self.processing_finished.emit(self.output_path)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

    def process_frame(self, frame):
        """处理单个视频帧"""
        try:
            # 将 BGR 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为 PIL Image
            image = Image.fromarray(frame_rgb)
            
            # 保存原始尺寸
            original_size = image.size
            
            # 图像预处理
            transforms = T.Compose([
                T.Resize((224, 224)),  # 调整到模型输入尺寸
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])
            
            # 转换图像
            input_tensor = transforms(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 模型预测
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred[pred == 1] = 255
            
            # 将预测结果调整回原始尺寸
            pred_mask = Image.fromarray(pred)
            pred_mask = pred_mask.resize(original_size, Image.Resampling.NEAREST)
            
            # 转换为 numpy 数组
            pred_mask_np = np.array(pred_mask)
            
            # 创建彩色掩码
            colored_mask = np.zeros((*pred_mask_np.shape, 3), dtype=np.uint8)
            colored_mask[pred_mask_np == 255] = [0, 255, 0]  # 绿色掩码
            
            # 将掩码叠加到原始帧上
            alpha = 0.5  # 透明度
            overlay = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
            
            return overlay
            
        except Exception as e:
            logger.error(f"处理帧时发生错误: {str(e)}")
            return frame  # 发生错误时返回原始帧

    def stop(self):
        """停止处理"""
        self.is_running = False

class VideoSegmentationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_path = None
        self.output_path = None
        self.worker = None
        self.model = None

    def initUI(self):
        """初始化UI界面"""
        self.setWindowTitle('皮肤病变分割工具')
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()

        # 控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()

        # 添加按钮
        self.btn_load = QPushButton("加载视频")
        self.btn_process = QPushButton("开始处理")
        self.btn_stop = QPushButton("停止处理")
        self.btn_save = QPushButton("保存结果")

        # 设置按钮大小
        for btn in [self.btn_load, self.btn_process, self.btn_stop, self.btn_save]:
            btn.setFixedSize(120, 40)

        # 模型选择下拉框
        self.model_selector = QComboBox()
        self.model_selector.addItems(["UNet", "DeepLabV3", "TransUNet", "UNeXt", "SkinUnet", "SegNet", "BiSeNet"])
        self.model_selector.setFixedSize(200, 40)

        # 批处理大小设置
        self.batch_size_label = QLabel("批处理大小:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 16)
        self.batch_size_spin.setValue(4)
        self.batch_size_spin.setFixedSize(80, 40)

        # 添加控件到控制面板
        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_selector)
        control_layout.addWidget(self.batch_size_label)
        control_layout.addWidget(self.batch_size_spin)
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_process)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_save)
        control_panel.setLayout(control_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(30)

        # 视频预览区域
        preview_panel = QWidget()
        preview_layout = QHBoxLayout()

        # 原始视频预览
        self.original_preview = QLabel()
        self.original_preview.setFixedSize(560, 420)
        self.original_preview.setStyleSheet("border: 1px solid gray;")
        self.original_preview.setAlignment(Qt.AlignCenter)

        # 处理结果预览
        self.result_preview = QLabel()
        self.result_preview.setFixedSize(560, 420)
        self.result_preview.setStyleSheet("border: 1px solid gray;")
        self.result_preview.setAlignment(Qt.AlignCenter)

        # 添加预览标签
        preview_layout.addWidget(QLabel("原始视频"))
        preview_layout.addWidget(QLabel("处理结果"))
        preview_layout.addWidget(self.original_preview)
        preview_layout.addWidget(self.result_preview)
        preview_panel.setLayout(preview_layout)

        # 添加所有面板到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(preview_panel)
        main_widget.setLayout(main_layout)

        # 连接信号和槽
        self.btn_load.clicked.connect(self.load_video)
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_save.clicked.connect(self.save_result)
        self.model_selector.currentIndexChanged.connect(self.select_model)

        # 初始化按钮状态
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

    def load_video(self):
        """加载视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        
        if file_path:
            self.video_path = file_path
            self.btn_process.setEnabled(True)
            # 显示第一帧预览
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.display_frame(frame, self.original_preview)
            cap.release()

    def select_model(self):
        """选择模型"""
        try:
            model_name = self.model_selector.currentText()
            logger.info(f"正在加载模型: {model_name}")
            
            # 模型权重路径映射
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
                from model.SegNet import SegNet
                self.model = SegNet(3, 2)
            elif model_name == 'UNeXt':
                from model.Unext import UNeXt
                self.model = UNeXt(in_channels=3, out_channels=2)
            elif model_name == 'SkinUnet':
                from model.SkinUnet import SkinUNet
                self.model = SkinUNet(in_channels=3, num_classes=2)
            elif model_name == 'UNet':
                from model.Unet import Unet
                self.model = Unet(2)
            elif model_name == 'TransUNet':
                from model.Transunet import TransUNet
                self.model = TransUNet(img_dim=224, in_channels=3, out_channels=128, 
                                     head_num=4, mlp_dim=512, block_num=8, 
                                     patch_dim=16, class_num=2)
            elif model_name == 'BiSeNet':
                from model.BiseNet import BiSeNet
                self.model = BiSeNet(num_classes=2)
            elif model_name == 'DeepLabV3':
                from model.Deeplab import DeepLabV3
                self.model = DeepLabV3(2)
            
            # 加载模型权重
            if model_name in model_weights:
                weight_path = model_weights[model_name]
                if os.path.exists(weight_path):
                    checkpoint = torch.load(weight_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
                        self.model.load_state_dict(state_dict)
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    # 将模型移动到GPU（如果可用）
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model = self.model.to(device)
                    self.model.eval()
                    
                    logger.info(f"模型 {model_name} 加载成功")
                    QMessageBox.information(self, "成功", f"模型 {model_name} 加载成功！")
                else:
                    raise FileNotFoundError(f"找不到模型权重文件: {weight_path}")
            else:
                raise ValueError(f"未知模型: {model_name}")
                
        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.model = None

    def start_processing(self):
        """开始处理视频"""
        if not self.video_path:
            QMessageBox.warning(self, "警告", "请先加载视频文件！")
            return

        # 设置输出路径
        output_dir = os.path.dirname(self.video_path)
        output_name = os.path.splitext(os.path.basename(self.video_path))[0] + "_processed.mp4"
        self.output_path = os.path.join(output_dir, output_name)

        # 创建并启动工作线程
        self.worker = VideoProcessWorker(
            self.video_path,
            self.model,
            self.output_path,
            self.batch_size_spin.value()
        )

        # 连接信号
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.frame_processed.connect(self.update_preview)
        self.worker.processing_finished.connect(self.processing_completed)
        self.worker.error_occurred.connect(self.handle_error)

        # 更新按钮状态
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load.setEnabled(False)

        # 启动处理
        self.worker.start()

    def stop_processing(self):
        """停止处理"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.btn_process.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_load.setEnabled(True)

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_preview(self, frame):
        """更新预览图像"""
        self.display_frame(frame, self.result_preview)

    def display_frame(self, frame, label):
        """在标签上显示帧"""
        if frame is None:
            return
        
        # 调整图像大小以适应预览区域
        h, w = frame.shape[:2]
        scale = min(560/w, 420/h)
        new_size = (int(w*scale), int(h*scale))
        frame = cv2.resize(frame, new_size)
        
        # 转换颜色空间
        if len(frame.shape) == 2:  # 灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:  # BGR图
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # 转换为QImage并显示
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))

    def processing_completed(self, output_path):
        """处理完成回调"""
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_load.setEnabled(True)
        self.btn_save.setEnabled(True)
        QMessageBox.information(self, "完成", f"视频处理完成！\n保存路径: {output_path}")

    def handle_error(self, error_msg):
        """处理错误"""
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误：{error_msg}")

    def save_result(self):
        """保存处理结果"""
        if not self.output_path or not os.path.exists(self.output_path):
            QMessageBox.warning(self, "警告", "没有可保存的处理结果！")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存处理结果", "", "Video Files (*.mp4)")
        
        if save_path:
            try:
                # 复制文件到新位置
                import shutil
                shutil.copy2(self.output_path, save_path)
                QMessageBox.information(self, "成功", "结果已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoSegmentationUI()
    window.show()
    sys.exit(app.exec_()) 