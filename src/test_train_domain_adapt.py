from ultralytics import YOLOWorld
import torch
import gc

def main():
    # خالی کردن حافظه گرافیک قبل از شروع
    torch.cuda.empty_cache()
    gc.collect()

    model = YOLOWorld('yolov8s-worldv2.pt')

    print("🚀 Starting Debug Training (Low Settings)...")
    
    try:
        results = model.train(
            data='data/data.yaml',
            epochs=3,                # فقط ۳ دور برای تست اینکه کد کار میکنه
            imgsz=640,
            
            # --- تنظیمات مخصوص سیستم ضعیف ---
            batch=2,                 # خیلی مهم: از 16 آوردیم روی 2
            workers=1,               # فشار روی CPU و رم کمتر شود
            device=0,
            # -------------------------------
            
            plots=True,
            save=True,
            name='debug_run',        # اسمش رو گذاشتیم debug
            close_mosaic=0,
            warmup_epochs=0
        )
        print("✅ Debug Training Finished! Code is ready for the Lab.")
        
    except Exception as e:
        print(f"❌ Still Error: {e}")
        print("Suggestion: Try setting device='cpu' just to verify the code logic.")

if __name__ == '__main__':
    main()