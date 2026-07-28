from ultralytics import YOLO
model = YOLO('/home/ps-ai/Desktop/manar/stampdetection/best.pt')

model.train(
    data='/home/ps-ai/Desktop/manar/yolofiles/data.yaml',
    epochs=100,
    imgsz=960,
    batch=-1,
    device=0,
    project='/home/ps-ai/Desktop/manar',
    name='retrained_model_for_crops',

    degrees=10,
    translate = 0.1,
    scale=0.2,
    fliplr=0.0,
    flipud=0.0,

    hsv_h=0,
    hsv_s=0,
    hsv_v=0.3,

    mosaic=0.0,

)