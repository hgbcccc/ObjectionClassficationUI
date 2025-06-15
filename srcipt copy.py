import json
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QFileDialog, QShortcut, QProgressBar,
    QStyle, QStyleFactory
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt, QTimer
import sys


class COCOAnnotator(QMainWindow):
    def __init__(self, image_dir, annot_path, parent=None):
        super(COCOAnnotator, self).__init__(parent)
        self.image_dir = image_dir          # 图像文件夹路径（train2024 或 val2024）
        self.annot_path = annot_path        # 标注文件路径（instances_train2024.json 等）
        self.annot_data = None              # 存储 COCO 标注 JSON 数据
        self.image_info = []                # 存储图像信息：(image_id, file_name, annotations)
        self.current_index = 0              # 当前显示图像的索引
        self.classifications = {}           # 记录分类结果：{image_name: "密集" / "稀疏"}
        
        # 设置应用样式
        self.setStyle(QStyleFactory.create('Fusion'))
        
        # 解析 COCO 标注
        self.load_annotations()
        
        # 初始化 UI
        self.init_ui()
        
        # 设置快捷键
        self.setup_shortcuts()
        
        # 显示第一张图像
        self.show_image()

    def setup_shortcuts(self):
        """设置键盘快捷键"""
        # 左右方向键切换图片
        QShortcut(QKeySequence(Qt.Key_Left), self, self.show_prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.show_next_image)
        
        # 数字键1、2分别对应稀疏和密集
        QShortcut(QKeySequence("1"), self, lambda: self.quick_classify("稀疏"))
        QShortcut(QKeySequence("2"), self, lambda: self.quick_classify("密集"))
        
        # 空格键保存并前进
        QShortcut(QKeySequence(Qt.Key_Space), self, self.save_and_next)

    def quick_classify(self, classification):
        """快速分类并自动保存"""
        if classification == "稀疏":
            self.sparse_check.setChecked(True)
            self.dense_check.setChecked(False)
        else:
            self.sparse_check.setChecked(False)
            self.dense_check.setChecked(True)
        self.save_classification()

    def save_and_next(self):
        """保存当前分类并显示下一张图片"""
        self.save_classification()
        self.show_next_image()

    def load_annotations(self):
        """解析 COCO 标注 JSON 文件"""
        with open(self.annot_path, 'r', encoding='utf-8') as f:
            self.annot_data = json.load(f)
        
        # 建立图像 ID 到文件名的映射
        img_id_to_name = {img["id"]: img["file_name"] for img in self.annot_data["images"]}
        
        # 按图像分组标注
        anns_by_img = {}
        for ann in self.annot_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)
        
        # 构建图像信息列表
        for img_id, file_name in img_id_to_name.items():
            annotations = anns_by_img.get(img_id, [])
            self.image_info.append({
                "id": img_id,
                "file_name": file_name,
                "annotations": annotations
            })

    def init_ui(self):
        """初始化 GUI 界面"""
        self.setWindowTitle("COCO 数据集标注筛选工具")
        self.setGeometry(100, 100, 1024, 768)

        # 中心窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("进度: %v/%m (%p%)")
        main_layout.addWidget(self.progress_bar)
        self.update_progress()

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        main_layout.addWidget(self.image_label)

        # 分类选择区域
        self.sparse_check = QCheckBox("稀疏 (1)")
        self.dense_check = QCheckBox("密集 (2)")
        self.sparse_check.setStyleSheet("QCheckBox { font-size: 14px; }")
        self.dense_check.setStyleSheet("QCheckBox { font-size: 14px; }")
        
        # 确保只能选一个
        self.sparse_check.stateChanged.connect(lambda: self.single_check(self.sparse_check))
        self.dense_check.stateChanged.connect(lambda: self.single_check(self.dense_check))
        
        classification_layout = QHBoxLayout()
        classification_layout.addWidget(self.sparse_check)
        classification_layout.addWidget(self.dense_check)
        main_layout.addLayout(classification_layout)

        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("上一张 (←)")
        self.next_btn = QPushButton("下一张 (→)")
        self.save_btn = QPushButton("保存分类 (Space)")
        self.export_btn = QPushButton("导出结果")
        self.import_btn = QPushButton("导入进度")
        
        for btn in [self.prev_btn, self.next_btn, self.save_btn, self.export_btn, self.import_btn]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    padding: 5px 15px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
        
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.next_btn.clicked.connect(self.show_next_image)
        self.save_btn.clicked.connect(self.save_and_next)
        self.export_btn.clicked.connect(self.export_results)
        self.import_btn.clicked.connect(self.import_progress)
        
        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(button_layout)

    def single_check(self, checked_box):
        """确保只有一个 CheckBox 被选中"""
        if checked_box.isChecked():
            if checked_box == self.sparse_check:
                self.dense_check.setChecked(False)
            else:
                self.sparse_check.setChecked(False)

    def update_progress(self):
        """更新进度条"""
        total = len(self.image_info)
        classified = len(self.classifications)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(classified)

    def import_progress(self):
        """从文本文件导入之前的分类进度"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入进度", "", "Text Files (*.txt)", options=options
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        image_name, classification = line.strip().split(':')
                        self.classifications[image_name.strip()] = classification.strip()
            
            self.update_progress()
            self.show_image()  # 刷新显示
        except Exception as e:
            print(f"导入进度文件时出错: {str(e)}")

    def show_image(self):
        """显示当前索引的图像，并绘制标注"""
        if self.current_index < 0 or self.current_index >= len(self.image_info):
            return
        
        # 获取当前图像信息
        img_info = self.image_info[self.current_index]
        file_name = img_info["file_name"]
        image_path = os.path.join(self.image_dir, file_name)
        
        # 更新窗口标题显示当前图片信息
        self.setWindowTitle(f"COCO 数据集标注筛选工具 - {file_name} ({self.current_index + 1}/{len(self.image_info)})")
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            return
        
        # 绘制标注（目标框 + 类别）
        for ann in img_info["annotations"]:
            bbox = ann["bbox"]  # [x, y, w, h]
            category_id = ann["category_id"]
            # 根据 category_id 映射到类别名称（如果需要更友好的显示，可完善 self.annot_data["categories"] 的解析）
            category_name = f"类别 {category_id}"  
            
            # 绘制矩形框
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制类别名称
            cv2.putText(
                image, category_name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # 转换图像格式用于 PyQt 显示
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image.data, width, height, bytes_per_line,
            QImage.Format_RGB888
        ).rgbSwapped()  # OpenCV 是 BGR，转成 RGB 用于 PyQt
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # 自动勾选之前保存的分类
        image_name = file_name
        if image_name in self.classifications:
            classification = self.classifications[image_name]
            self.sparse_check.setChecked(classification == "稀疏")
            self.dense_check.setChecked(classification == "密集")
        else:
            self.sparse_check.setChecked(False)
            self.dense_check.setChecked(False)

    def show_prev_image(self):
        """显示上一张图像"""
        self.current_index = max(0, self.current_index - 1)
        self.show_image()

    def show_next_image(self):
        """显示下一张图像"""
        self.current_index = min(len(self.image_info) - 1, self.current_index + 1)
        self.show_image()

    def save_classification(self):
        """保存当前图像的分类结果"""
        img_info = self.image_info[self.current_index]
        file_name = img_info["file_name"]
        
        if self.sparse_check.isChecked():
            self.classifications[file_name] = "稀疏"
        elif self.dense_check.isChecked():
            self.classifications[file_name] = "密集"
        else:
            if file_name in self.classifications:
                del self.classifications[file_name]
        
        self.update_progress()

    def export_results(self):
        """导出分类结果到 TXT 文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "", "Text Files (*.txt)", options=options
        )
        
        if not file_path:
            return
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for image_name, classification in self.classifications.items():
                f.write(f"{image_name}: {classification}\n")


if __name__ == "__main__":
    # ---------------------------
    # 配置数据集路径
    # ---------------------------
    # 替换为实际的 train2024 或 val2024 图像文件夹路径 
    IMAGE_DIR = "data/ForestDamages/train2024"   # data/ForestDamages/train2024
    # 替换为实际的标注文件路径
    ANNOT_PATH = "data/ForestDamages/annotations/instances_train2024.json"  

    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle(QStyleFactory.create('Fusion'))
    
    window = COCOAnnotator(IMAGE_DIR, ANNOT_PATH)
    window.show()
    sys.exit(app.exec_())