// ============================================================
//  detection_engine.js  –  Motor de detecció d'objectes
//  JHort – Animals amb IA
//  Basat en TensorFlow.js + COCO-SSD
//
//  Per adaptar esta aplicació a una altra categoria (p.ex.
//  residus), cal modificar únicament:
//    1. CATEGORIES  →  llista de classes COCO acceptades
//    2. TRANSLATIONS →  nom traduït per a cada classe
//    3. El títol i els textos de la interfície (index.html)
// ============================================================

// ── 1. CATEGORIES ACTIVES ────────────────────────────────────
//  Llista de classes COCO-SSD que l'app mostrarà i enviarà.
//  Per canviar l'aplicació (residus, etc.) substituïx este array.
//  Classes COCO-SSD disponibles: https://github.com/nightrome/cocostuff
const CATEGORIES = ['cat', 'bird', 'person'];

// ── 2. TRADUCCIONS ──────────────────────────────────────────
//  Nom en valencià que es mostrarà en pantalla i s'enviarà per UART.
const TRANSLATIONS = {
  cat:    'gat',
  bird:   'ocell',
  person: 'persona',
  // ── Afig ací futures categories per a l'app de residus ──
  // bottle:      'ampolla',
  // cup:         'got',
  // bowl:        'bol',
  // book:        'paper',
  // chair:       'cadira',
};

// ── 3. MODELS DISPONIBLES ────────────────────────────────────
//  L'usuari pot triar en la pantalla de configuració.
const MODELS = {
  lite: {
    base:  'lite_mobilenet_v2',
    label: '⚡ Ràpid (recomanat per a mòbils)',
    score_threshold: 0.20,
  },
  precise: {
    base:  'mobilenet_v2',
    label: '🔍 Precís (més distància, més lent)',
    score_threshold: 0.20,
  },
};

let currentModelKey = 'lite';   // model actiu per defecte

// ────────────────────────────────────────────────────────────
//  Internals — no cal modificar per canviar de categoria
// ────────────────────────────────────────────────────────────
let model = null;
let isRunning = false;
let detectionLoop = null;

let onDetectionCallback  = null;
let onModelReadyCallback = null;
let onModelErrorCallback = null;

/**
 * Inicialitza (o reinicialitza) el model COCO-SSD.
 * @param {string} modelKey  — 'lite' | 'precise'
 */
async function initModel(modelKey) {
  if (modelKey) currentModelKey = modelKey;
  const cfg = MODELS[currentModelKey];
  try {
    stopDetection();
    model = null;
    model = await cocoSsd.load({ base: cfg.base });
    if (onModelReadyCallback) onModelReadyCallback();
  } catch (e) {
    console.error('❌ Error carregant el model:', e);
    if (onModelErrorCallback) onModelErrorCallback(e);
  }
}

/**
 * Comença el bucle de detecció sobre un element de vídeo.
 * @param {HTMLVideoElement} videoEl
 * @param {number} intervalMs
 */
function startDetection(videoEl, intervalMs) {
  if (isRunning) stopDetection();
  isRunning = true;

  const threshold = MODELS[currentModelKey].score_threshold;

  async function detect() {
    if (!isRunning || !model) return;
    if (videoEl.readyState < 2) {
      detectionLoop = setTimeout(detect, 200);
      return;
    }
    try {
      const predictions = await model.detect(videoEl);
      const filtered = predictions
        .filter(p => CATEGORIES.includes(p.class) && p.score >= threshold)
        .map(p => ({
          class: p.class,
          label: TRANSLATIONS[p.class] || p.class,
          score: Math.round(p.score * 100),
          bbox:  p.bbox,
        }));
      if (onDetectionCallback) onDetectionCallback(filtered);
    } catch (e) {
      console.error('❌ Error en detecció:', e);
    }
    detectionLoop = setTimeout(detect, intervalMs);
  }

  detect();
}

/** Atura el bucle de detecció. */
function stopDetection() {
  isRunning = false;
  if (detectionLoop) clearTimeout(detectionLoop);
}

// ── API pública ───────────────────────────────────────────────
function onDetection(cb)    { onDetectionCallback   = cb; }
function onModelReady(cb)   { onModelReadyCallback  = cb; }
function onModelError(cb)   { onModelErrorCallback  = cb; }
function getCategories()    { return CATEGORIES; }
function getTranslations()  { return TRANSLATIONS; }
function getModels()        { return MODELS; }
function getCurrentModel()  { return currentModelKey; }
