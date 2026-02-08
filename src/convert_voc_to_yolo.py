import xml.etree.ElementTree as ET
import os
import glob
from pathlib import Path
import shutil
from tqdm import tqdm

# --- تنظیمات حیاتی پروژه ---
# ترتیب کلاس‌ها در فایل YOLO بسیار مهم است.
# 0: h0 (فرد غیرمسلح)
# 1: h1 (فرد مسلح)
# 2: w190 (اسلحه اصلی - CMMG Banshee)
# 3: w146 (اسلحه امتیازی - Lobaev DXL-5)
# 4: w0 (سایر اسلحه‌ها / اسلحه‌های عمومی در داده واقعی)
FINAL_CLASSES = ['h0', 'h1', 'w190', 'w146', 'w0']

def convert_box(size, box):
    """تبدیل مختصات باکس به فرمت نرمالایز شده YOLO"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_file, output_path, mapping_dict=None):
    """
    تبدیل فایل XML به TXT با قابلیت تغییر نام کلاس (Mapping)
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        txt_filename = Path(output_path) / Path(xml_file).with_suffix('.txt').name
        
        with open(txt_filename, 'w') as out_file:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                
                # --- لاجیک هوشمند برای داده‌های شبیه‌سازی ---
                # اگر نگاشت وجود داشت (مثلا w0 -> w190)، نام کلاس را عوض کن
                if mapping_dict and cls_name in mapping_dict:
                    cls_name = mapping_dict[cls_name]
                
                # اگر کلاس نهایی در لیست ما نبود، نادیده‌اش بگیر
                if cls_name not in FINAL_CLASSES:
                    continue

                cls_id = FINAL_CLASSES.index(cls_name)
                
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                out_file.write(f"{str(cls_id)} " + " ".join([str(a) for a in bb]) + '\n')
                
    except Exception as e:
        print(f"Error converting {xml_file}: {e}")

def prepare_dataset(source_dir, dest_root, split_type, mapping=None):
    """
    source_dir: مسیر ورودی
    dest_root: مسیر خروجی
    split_type: train (برای Sim) یا val (برای Real)
    """
    images_dir = Path(dest_root) / split_type / 'images'
    labels_dir = Path(dest_root) / split_type / 'labels'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # پیدا کردن فایل‌های XML (حساس به حروف کوچک و بزرگ نباشد بهتر است)
    xml_files = glob.glob(os.path.join(source_dir, '*.xml'))
    if not xml_files:
         xml_files = glob.glob(os.path.join(source_dir, '*.XML'))

    print(f"Processing {len(xml_files)} files from:\n  {source_dir}\n  To -> {split_type}/labels (Mapping: {mapping})")
    
    for xml_file in tqdm(xml_files):
        convert_annotation(xml_file, labels_dir, mapping)
        
        # کپی کردن تصویر متناظر
        base_name = os.path.splitext(xml_file)[0]
        image_found = False
        # فرمت‌های رایج تصویر را چک می‌کنیم
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            img_candidate = base_name + ext
            if os.path.exists(img_candidate):
                shutil.copy(img_candidate, images_dir)
                image_found = True
                break
        
        if not image_found:
            print(f"Warning: Image not found for {os.path.basename(xml_file)}")

if __name__ == '__main__':
    output_root = 'data/processed_yolo'
    
    # تمیزکاری: اگر پوشه از قبل بود پاکش کن تا دیتای تمیز ساخته شود
    if os.path.exists(output_root):
        print("Removing old processed data...")
        shutil.rmtree(output_root)

    print("--- STARTING CONVERSION ---")

    # 1. Main Dataset / Sim (CMMG) -> Train
    # اینجا w0 یعنی CMMG Banshee پس تبدیل میشه به w190
    prepare_dataset(
        'data/final_dataset_voc/Main_Dataset/Sim',
        output_root, 'train', mapping={'w0': 'w190'}
    )

    # 2. Main Dataset / Real -> Val
    # اینجا w0 یعنی اسلحه متفرقه، پس تغییری نمی‌کند
    prepare_dataset(
        'data/final_dataset_voc/Main_Dataset/Real',
        output_root, 'val', mapping={}
    )

    # 3. Bonus Dataset / Sim (Lobaev) -> Train
    # اینجا w0 یعنی Lobaev DXL-5 پس تبدیل میشه به w146
    prepare_dataset(
        'data/final_dataset_voc/Bonus_Dataset/Sim',
        output_root, 'train', mapping={'w0': 'w146'}
    )

    # 4. Bonus Dataset / Real -> Val
    # بدون تغییر
    prepare_dataset(
        'data/final_dataset_voc/Bonus_Dataset/Real',
        output_root, 'val', mapping={}
    )
    
    print("\n✅ Conversion Finished!")
    print(f"Classes Map: {FINAL_CLASSES}")