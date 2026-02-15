════════════════════════════════════════════════════════
  Com obtenir el fitxer yolov8n.onnx per a JHort
════════════════════════════════════════════════════════

El model YOLOv8 no es distribueix directament en format
ONNX. Cal obtenir-lo d'una d'aquestes dues maneres:

────────────────────────────────────────────────────────
OPCIÓ A — Descarregar-lo ja convertit (més fàcil)
────────────────────────────────────────────────────────
Busca "yolov8n.onnx" a Hugging Face:
  https://huggingface.co/models?search=yolov8n+onnx

Un repositori fiable és:
  https://huggingface.co/msprojectraspi/yolov8n-onnx

Descarrega el fitxer yolov8n.onnx i col·loca'l ací:
  jHort/models/yolov8n.onnx

────────────────────────────────────────────────────────
OPCIÓ B — Convertir-lo tu mateix amb Python (oficial)
────────────────────────────────────────────────────────
Requereix Python 3.8+ instal·lat.

1. Instal·la la llibreria d'Ultralytics:
   pip install ultralytics

2. Executa este codi Python:
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')   # es descarrega automàticament
   model.export(format='onnx', imgsz=640, simplify=True)

3. Es generarà un fitxer yolov8n.onnx al directori actual.
   Col·loca'l a: jHort/models/yolov8n.onnx

────────────────────────────────────────────────────────
Una vegada el fitxer estiga a la carpeta models/, l'opció
YOLOv8 de la configuració funcionarà sense Internet.
════════════════════════════════════════════════════════
