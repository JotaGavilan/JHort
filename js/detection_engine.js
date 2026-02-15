// ============================================================
//  detection_engine.js  –  Motor de detecció d'objectes
//  JHort – El guardià del teu bancal
//
//  Suporta tres models:
//    · lite     → COCO-SSD lite_mobilenet_v2  (TensorFlow.js)
//    · precise  → COCO-SSD mobilenet_v2       (TensorFlow.js)
//    · yolo     → YOLOv8-nano                 (ONNX Runtime Web)
//
//  Per adaptar a una altra categoria (residus, etc.) modifica:
//    1. CATEGORIES   → classes acceptades
//    2. TRANSLATIONS → noms en valencià
//    3. YOLO_CLASSES → índexs COCO de les categories (per a YOLO)
// ============================================================

// ── 1. CATEGORIES ACTIVES ────────────────────────────────────
const CATEGORIES = ['cat', 'bird', 'person'];

// ── 2. TRADUCCIONS ───────────────────────────────────────────
const TRANSLATIONS = {
  cat:    'gat',
  bird:   'ocell',
  person: 'persona',
  // Residus futurs:
  // bottle: 'ampolla',
  // cup:    'got',
};

// ── 3. ÍNDEXS YOLO (classes COCO80 que volem detectar) ───────
//  cat=15, bird=14, person=0  (índexs del dataset COCO80)
const YOLO_CLASS_MAP = {
  0:  'person',
  14: 'bird',
  15: 'cat',
};

// ── 4. MODELS DISPONIBLES ────────────────────────────────────
const MODELS = {
  lite: {
    type:            'cocossd',
    base:            'lite_mobilenet_v2',
    label:           '⚡ Ràpid (COCO-SSD lite)',
    description:     'El més ràpid. Recomanat per a mòbils antics.',
    score_threshold: 0.20,
  },
  precise: {
    type:            'cocossd',
    base:            'mobilenet_v2',
    label:           '🔍 Precís (COCO-SSD v2)',
    description:     'Més precís que el ràpid. Més lent.',
    score_threshold: 0.20,
  },
  yolo: {
    type:            'yolo',
    // ─────────────────────────────────────────────────────────
    //  El fitxer yolov8n.onnx ha d'estar a: jHort/models/yolov8n.onnx
    //  Descàrrega (~6MB):
    //  https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
    //  i col·loca'l a la carpeta models/ del projecte.
    // ─────────────────────────────────────────────────────────
    url:             './models/yolov8n.onnx',
    label:           '🚀 YOLOv8 (millor distància)',
    description:     'Detecta millor a distància. Cal tindre el fitxer yolov8n.onnx a models/',
    score_threshold: 0.25,
    input_size:      640,
  },
};

let currentModelKey = 'lite';

// ── Internals ─────────────────────────────────────────────────
let model        = null;
let yoloSession  = null;   // sessió ONNX (només per a YOLO)
let isRunning    = false;
let detectionLoop = null;

let onDetectionCallback  = null;
let onModelReadyCallback = null;
let onModelErrorCallback = null;

// ── Canvas intern per a preprocessar frames YOLO ─────────────
const yoloCanvas  = document.createElement('canvas');
const yoloCtx     = yoloCanvas.getContext('2d', { willReadFrequently: true });

// ─────────────────────────────────────────────────────────────
//  INICIALITZACIÓ
// ─────────────────────────────────────────────────────────────

async function initModel(modelKey) {
  if (modelKey) currentModelKey = modelKey;
  const cfg = MODELS[currentModelKey];
  stopDetection();
  model       = null;
  yoloSession = null;

  try {
    if (cfg.type === 'cocossd') {
      model = await cocoSsd.load({ base: cfg.base });

    } else if (cfg.type === 'yolo') {
      // Configurar ruta dels fitxers WASM auxiliars (mateix CDN que el script)
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
      ort.env.wasm.numThreads = 1;  // 1 thread per compatibilitat màxima

      let loaded = false;
      const backends = ['webgl', 'wasm'];
      for (const backend of backends) {
        try {
          console.log(`[YOLO] Provant backend: ${backend}`);
          yoloSession = await ort.InferenceSession.create(cfg.url, {
            executionProviders: [backend],
            graphOptimizationLevel: 'all',
          });
          console.log(`[YOLO] ✅ Carregat amb backend: ${backend}`);
          loaded = true;
          break;
        } catch (backendErr) {
          console.warn(`[YOLO] ❌ Backend ${backend} fallat:`, backendErr.message || backendErr);
        }
      }
      if (!loaded) throw new Error('YOLO_LOAD_FAILED');
    }
    if (onModelReadyCallback) onModelReadyCallback();
  } catch (e) {
    console.error('❌ Error carregant el model:', e);
    if (onModelErrorCallback) onModelErrorCallback(e);
  }
}

// ─────────────────────────────────────────────────────────────
//  BUCLE DE DETECCIÓ
// ─────────────────────────────────────────────────────────────

function startDetection(videoEl, intervalMs) {
  if (isRunning) stopDetection();
  isRunning = true;

  async function detect() {
    if (!isRunning) return;
    if (videoEl.readyState < 2) {
      detectionLoop = setTimeout(detect, 200);
      return;
    }
    try {
      const cfg = MODELS[currentModelKey];
      let results = [];

      if (cfg.type === 'cocossd' && model) {
        results = await detectCOCOSSD(videoEl, cfg);
      } else if (cfg.type === 'yolo' && yoloSession) {
        results = await detectYOLO(videoEl, cfg);
      }

      if (onDetectionCallback) onDetectionCallback(results);
    } catch (e) {
      console.error('❌ Error en detecció:', e);
    }
    detectionLoop = setTimeout(detect, intervalMs);
  }

  detect();
}

function stopDetection() {
  isRunning = false;
  if (detectionLoop) clearTimeout(detectionLoop);
}

// ─────────────────────────────────────────────────────────────
//  BACKENDS DE DETECCIÓ
// ─────────────────────────────────────────────────────────────

/** Backend COCO-SSD (TensorFlow.js) */
async function detectCOCOSSD(videoEl, cfg) {
  const predictions = await model.detect(videoEl);
  return predictions
    .filter(p => CATEGORIES.includes(p.class) && p.score >= cfg.score_threshold)
    .map(p => ({
      class: p.class,
      label: TRANSLATIONS[p.class] || p.class,
      score: Math.round(p.score * 100),
      bbox:  p.bbox,   // [x, y, w, h] en px originals
    }));
}

/** Backend YOLOv8-nano (ONNX Runtime Web) */
async function detectYOLO(videoEl, cfg) {
  const size = cfg.input_size;  // 640

  // 1. Preprocessar: redimensionar el frame a 640×640
  yoloCanvas.width  = size;
  yoloCanvas.height = size;

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;

  // Escala mantenint aspect ratio, amb letterboxing gris
  const scale = Math.min(size / vw, size / vh);
  const sw = Math.round(vw * scale);
  const sh = Math.round(vh * scale);
  const ox = Math.round((size - sw) / 2);
  const oy = Math.round((size - sh) / 2);

  yoloCtx.fillStyle = '#808080';
  yoloCtx.fillRect(0, 0, size, size);
  yoloCtx.drawImage(videoEl, ox, oy, sw, sh);

  const imgData = yoloCtx.getImageData(0, 0, size, size).data;

  // 2. Convertir a tensor float32 normalitzat [0,1] en format CHW
  const float32 = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    float32[i]                   = imgData[i * 4]     / 255;  // R
    float32[i + size * size]     = imgData[i * 4 + 1] / 255;  // G
    float32[i + size * size * 2] = imgData[i * 4 + 2] / 255;  // B
  }

  const tensor = new ort.Tensor('float32', float32, [1, 3, size, size]);
  const feeds  = { images: tensor };

  // 3. Inferència
  const output = await yoloSession.run(feeds);

  // YOLOv8 output shape: [1, 84, 8400]
  // 84 = 4 (bbox cx,cy,w,h) + 80 classes
  const raw    = output[Object.keys(output)[0]].data;
  const numDet = 8400;
  const numCls = 80;

  const detections = [];

  for (let i = 0; i < numDet; i++) {
    // Trobar classe amb màxima confiança
    let maxScore = 0;
    let maxCls   = -1;
    for (let c = 0; c < numCls; c++) {
      const score = raw[4 * numDet + c * numDet + i];
      if (score > maxScore) { maxScore = score; maxCls = c; }
    }

    if (maxScore < cfg.score_threshold) continue;
    if (!(maxCls in YOLO_CLASS_MAP))    continue;

    const className = YOLO_CLASS_MAP[maxCls];
    if (!CATEGORIES.includes(className)) continue;

    // Bbox en coordenades del canvas 640×640 (cx, cy, w, h)
    const cx = raw[0 * numDet + i];
    const cy = raw[1 * numDet + i];
    const bw = raw[2 * numDet + i];
    const bh = raw[3 * numDet + i];

    // Desfer letterboxing → coordenades del vídeo original
    const x1 = ((cx - bw / 2) - ox) / scale;
    const y1 = ((cy - bh / 2) - oy) / scale;
    const w  = bw / scale;
    const h  = bh / scale;

    detections.push({
      class: className,
      label: TRANSLATIONS[className] || className,
      score: Math.round(maxScore * 100),
      bbox:  [
        Math.max(0, x1),
        Math.max(0, y1),
        Math.min(w, vw - x1),
        Math.min(h, vh - y1),
      ],
      _raw_score: maxScore,
    });
  }

  // NMS simple: eliminar duplicats molt solapats
  return nms(detections, 0.45);
}

/** Non-Maximum Suppression simple per a YOLO */
function nms(dets, iouThreshold) {
  dets.sort((a, b) => b._raw_score - a._raw_score);
  const keep = [];
  const used = new Array(dets.length).fill(false);

  for (let i = 0; i < dets.length; i++) {
    if (used[i]) continue;
    keep.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (!used[j] && iou(dets[i].bbox, dets[j].bbox) > iouThreshold) {
        used[j] = true;
      }
    }
  }
  return keep;
}

function iou(a, b) {
  const ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const bx2 = b[0] + b[2], by2 = b[1] + b[3];
  const ix  = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
  const iy  = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = a[2]*a[3] + b[2]*b[3] - inter;
  return union > 0 ? inter / union : 0;
}

// ── API pública ───────────────────────────────────────────────
function onDetection(cb)    { onDetectionCallback   = cb; }
function onModelReady(cb)   { onModelReadyCallback  = cb; }
function onModelError(cb)   { onModelErrorCallback  = cb; }
function getCategories()    { return CATEGORIES; }
function getTranslations()  { return TRANSLATIONS; }
function getModels()        { return MODELS; }
function getCurrentModel()  { return currentModelKey; }
