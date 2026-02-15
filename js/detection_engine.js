// ============================================================
//  detection_engine.js  –  Motor de detecció d'objectes
//  JHort – El guardià del teu bancal
//
//  Suporta tres models (tots via TensorFlow.js):
//    · lite      → COCO-SSD lite_mobilenet_v2  (ràpid)
//    · precise   → COCO-SSD mobilenet_v2       (precís)
//    · efficient → EfficientDet-Lite0           (millor distància)
//
//  Per adaptar a una altra categoria (residus, etc.) modifica:
//    1. CATEGORIES   → classes acceptades
//    2. TRANSLATIONS → noms en valencià
// ============================================================

// ── 1. CATEGORIES ACTIVES ────────────────────────────────────
const CATEGORIES = ['cat', 'bird', 'person'];

// ── 2. TRADUCCIONS ───────────────────────────────────────────
const TRANSLATIONS = {
  cat:    'gat',
  bird:   'ocell',
  person: 'persona',
};

// ── 3. MODELS DISPONIBLES ────────────────────────────────────
const MODELS = {
  lite: {
    type:            'cocossd',
    base:            'lite_mobilenet_v2',
    label:           '⚡ Ràpid',
    description:     'Funciona bé fins a ~1,5m. Ideal per a mòbils antics o amb poca bateria.',
    score_threshold: 0.20,
  },
  precise: {
    type:            'cocossd',
    base:            'mobilenet_v2',
    label:           '🔍 Precís',
    description:     'Millor en angles difícils i moviment. Una mica més lent que el Ràpid.',
    score_threshold: 0.20,
  },
  efficient: {
    type:            'efficientdet',
    label:           '🚀 EfficientDet',
    description:     'Detecta fins a ~3-4m. Més lent; recomanat per a tauletes o mòbils potents.',
    score_threshold: 0.25,
  },
};

let currentModelKey = 'lite';

// ── Internals ─────────────────────────────────────────────────
let model         = null;
let isRunning     = false;
let detectionLoop = null;

let onDetectionCallback  = null;
let onModelReadyCallback = null;
let onModelErrorCallback = null;

// ─────────────────────────────────────────────────────────────
//  INICIALITZACIÓ
// ─────────────────────────────────────────────────────────────
async function initModel(modelKey) {
  if (modelKey) currentModelKey = modelKey;
  const cfg = MODELS[currentModelKey];
  stopDetection();
  model = null;

  try {
    if (cfg.type === 'cocossd') {
      model = await cocoSsd.load({ base: cfg.base });

    } else if (cfg.type === 'efficientdet') {
      // EfficientDet-Lite0 via tf.loadGraphModel (TF Hub, format SavedModel JS)
      model = await tf.loadGraphModel(
        'https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1',
        { fromTFHub: true }
      );
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
    if (!isRunning || !model) return;
    if (videoEl.readyState < 2) {
      detectionLoop = setTimeout(detect, 200);
      return;
    }
    try {
      const cfg = MODELS[currentModelKey];
      let results = [];

      if (cfg.type === 'cocossd') {
        results = await detectCOCOSSD(videoEl, cfg);
      } else if (cfg.type === 'efficientdet') {
        results = await detectEfficientDet(videoEl, cfg);
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
//  BACKENDS
// ─────────────────────────────────────────────────────────────

async function detectCOCOSSD(videoEl, cfg) {
  const predictions = await model.detect(videoEl);
  return predictions
    .filter(p => CATEGORIES.includes(p.class) && p.score >= cfg.score_threshold)
    .map(p => ({
      class: p.class,
      label: TRANSLATIONS[p.class] || p.class,
      score: Math.round(p.score * 100),
      bbox:  p.bbox,
    }));
}

async function detectEfficientDet(videoEl, cfg) {
  // EfficientDet-Lite0 espera un tensor [1, H, W, 3] uint8
  const vw = videoEl.videoWidth  || videoEl.width;
  const vh = videoEl.videoHeight || videoEl.height;

  const results = tf.tidy(() => {
    const imgTensor = tf.browser.fromPixels(videoEl);          // [H, W, 3]
    const batched   = imgTensor.expandDims(0);                  // [1, H, W, 3]
    return model.execute(batched);
  });

  // EfficientDet retorna un dict amb:
  //   'detection_boxes'   → [1, N, 4]  (ymin, xmin, ymax, xmax) normalitzat
  //   'detection_scores'  → [1, N]
  //   'detection_classes' → [1, N]     (índex COCO 1-based)
  const boxesTensor   = results['detection_boxes']   || results[Object.keys(results)[0]];
  const scoresTensor  = results['detection_scores']  || results[Object.keys(results)[1]];
  const classesTensor = results['detection_classes'] || results[Object.keys(results)[2]];

  const boxes   = await boxesTensor.array();
  const scores  = await scoresTensor.array();
  const classes = await classesTensor.array();

  // Alliberar tensors
  if (Array.isArray(results)) results.forEach(t => t.dispose());
  else Object.values(results).forEach(t => t.dispose());

  // Mapa índex COCO (1-based) → nom de classe
  const COCO_CLASSES = {
    1: 'person', 15: 'bird', 16: 'cat',
  };

  const detections = [];
  const n = scores[0].length;

  for (let i = 0; i < n; i++) {
    const score = scores[0][i];
    if (score < cfg.score_threshold) continue;

    const clsIdx = Math.round(classes[0][i]);
    const cls    = COCO_CLASSES[clsIdx];
    if (!cls || !CATEGORIES.includes(cls)) continue;

    const [ymin, xmin, ymax, xmax] = boxes[0][i];
    detections.push({
      class: cls,
      label: TRANSLATIONS[cls] || cls,
      score: Math.round(score * 100),
      bbox:  [xmin * vw, ymin * vh, (xmax - xmin) * vw, (ymax - ymin) * vh],
    });
  }

  return detections;
}

// ── API pública ───────────────────────────────────────────────
function onDetection(cb)    { onDetectionCallback   = cb; }
function onModelReady(cb)   { onModelReadyCallback  = cb; }
function onModelError(cb)   { onModelErrorCallback  = cb; }
function getCategories()    { return CATEGORIES; }
function getTranslations()  { return TRANSLATIONS; }
function getModels()        { return MODELS; }
function getCurrentModel()  { return currentModelKey; }
