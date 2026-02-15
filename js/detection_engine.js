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
      // EfficientDet via @tensorflow-models/object-detection
      // Usa automl-image-classification intern per a l'API
      model = await tf.automl.loadObjectDetection(
        'https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1'
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
  // EfficientDet retorna { boxes, scores, classes }
  // bbox format: [ymin, xmin, ymax, xmax] normalitzat 0-1
  const vw = videoEl.videoWidth  || videoEl.width;
  const vh = videoEl.videoHeight || videoEl.height;

  const predictions = await model.detect(videoEl, { score_threshold: cfg.score_threshold });

  return predictions
    .filter(p => {
      // L'API pot retornar el nom de la classe o l'índex
      const cls = normalizeClass(p.label || p.class);
      return CATEGORIES.includes(cls);
    })
    .map(p => {
      const cls  = normalizeClass(p.label || p.class);
      const box  = p.box || p.bbox;
      // Convertir [ymin,xmin,ymax,xmax] normalitzat → [x,y,w,h] en px
      let bbox;
      if (Array.isArray(box) && box.length === 4) {
        if (box[0] <= 1 && box[1] <= 1) {
          // Format normalitzat [ymin, xmin, ymax, xmax]
          const ymin = box[0], xmin = box[1], ymax = box[2], xmax = box[3];
          bbox = [xmin * vw, ymin * vh, (xmax - xmin) * vw, (ymax - ymin) * vh];
        } else {
          // Ja en píxels [x, y, w, h]
          bbox = box;
        }
      } else {
        bbox = [0, 0, vw, vh];
      }
      return {
        class: cls,
        label: TRANSLATIONS[cls] || cls,
        score: Math.round((p.score || p.probability || 0) * 100),
        bbox,
      };
    });
}

// Normalitza noms de classe (EfficientDet pot retornar noms llargs)
function normalizeClass(raw) {
  if (!raw) return '';
  const s = String(raw).toLowerCase().trim();
  if (s.includes('cat'))    return 'cat';
  if (s.includes('bird'))   return 'bird';
  if (s.includes('person') || s.includes('human')) return 'person';
  return s;
}

// ── API pública ───────────────────────────────────────────────
function onDetection(cb)    { onDetectionCallback   = cb; }
function onModelReady(cb)   { onModelReadyCallback  = cb; }
function onModelError(cb)   { onModelErrorCallback  = cb; }
function getCategories()    { return CATEGORIES; }
function getTranslations()  { return TRANSLATIONS; }
function getModels()        { return MODELS; }
function getCurrentModel()  { return currentModelKey; }
