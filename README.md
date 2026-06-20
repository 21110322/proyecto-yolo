# proyecto-yolo

# ============================================
# INSTALACIÓN DE LIBRERÍAS
# ============================================

!pip install ultralytics

# ============================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================

from ultralytics import YOLO
import zipfile

# ============================================
# SUBIR DATASET
# ============================================

from google.colab import files
uploaded = files.upload()

# ============================================
# DESCOMPRIMIR DATASET
# ============================================

with zipfile.ZipFile('frutas.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

print("Dataset extraído correctamente")

# ============================================
# VERIFICAR DATASET
# ============================================

!ls /content/dataset

# ============================================
# CARGAR MODELO YOLOv8
# ============================================

model = YOLO('yolov8n.pt')

# ============================================
# ENTRENAMIENTO DEL MODELO
# ============================================

results = model.train(
    data='/content/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)

# ============================================
# PRUEBAS DEL MODELO
# ============================================

uploaded = files.upload()

modelo = YOLO('/content/runs/detect/train/weights/best.pt')

resultado = modelo('/content/prueba.jpg', save=True)

# ============================================
# MOSTRAR RESULTADO
# ============================================

from IPython.display import Image

Image('/content/runs/detect/predict/prueba.jpg')
