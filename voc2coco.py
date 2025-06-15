import os
import xml.etree.ElementTree as ET
import json
import shutil


def voc_to_coco(voc_root, output_root):
    sets = ['train', 'val']
    for set_name in sets:
        images = []
        annotations = []
        categories = []
        category_id = 1
        annotation_id = 1

        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')
        set_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{set_name}.txt')

        with open(set_file, 'r') as f:
            image_ids = f.read().strip().splitlines()

        category_names = set()
        for image_id in image_ids:
            image_path = os.path.join(image_dir, f'{image_id}.jpg')
            annotation_path = os.path.join(annotation_dir, f'{image_id}.xml')

            if not os.path.exists(image_path) or not os.path.exists(annotation_path):
                continue

            tree = ET.parse(annotation_path)
            root = tree.getroot()

            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)

            image = {
                'id': len(images) + 1,
                'file_name': f'{image_id}.jpg',
                'width': width,
                'height': height
            }
            images.append(image)

            # 遍历当前标注文件里的每个 <object> 标签
            for obj in root.findall('object'):
                category_name = obj.find('name').text
                category_names.add(category_name)
                difficult = int(obj.find('difficult').text)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin

                annotation = {
                    'id': annotation_id,
                    'image_id': image['id'],
                    'category_id': None,
                    'bbox': [xmin, ymin, w, h],
                    'area': w * h,
                    'iscrowd': 0,
                    'difficult': difficult,
                    # 新增字段，临时保存当前目标的类别名称
                    'category_name': category_name  
                }
                annotations.append(annotation)
                annotation_id += 1

        # 构建类别信息
        for i, category_name in enumerate(sorted(list(category_names))):
            category = {
                'id': category_id,
                'name': category_name,
               'supercategory': 'none'
            }
            categories.append(category)
            category_id += 1

        # 填充 annotation 里的 category_id
        for ann in annotations:
            category_name = ann['category_name']
            # 根据类别名称找到对应的 category_id
            ann['category_id'] = [cat['id'] for cat in categories if cat['name'] == category_name][0]
            # 移除临时保存的类别名称字段（可选，若不想在最终 COCO 标注里保留）
            del ann['category_name']  

        output_images_dir = os.path.join(output_root, f'{set_name}2024')
        output_annotations_dir = os.path.join(output_root, 'annotations')
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_annotations_dir, exist_ok=True)

        # 复制图像文件
        for image in images:
            shutil.copy(os.path.join(image_dir, image['file_name']), os.path.join(output_images_dir, image['file_name']))

        # 构建并保存 COCO 格式数据
        coco_data = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        with open(os.path.join(output_annotations_dir, f'instances_{set_name}2024.json'), 'w') as f:
            json.dump(coco_data, f, indent=4)


if __name__ == "__main__":
    voc_root = 'data/VOC2012'
    output_root = 'data/voc2coco'
    voc_to_coco(voc_root, output_root)