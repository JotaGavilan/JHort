// ============================================================
//  detection_engine.js  –  Motor de detecció d'objectes
//  JHort – Animals amb IA
//  Basat en TensorFlow.js + COCO-SSD
//
//  Per adaptar aquesta aplicació a una altra categoria (p.ex.
//  residus), cal modificar únicament:
//    1. CATEGORIES  →  llista de classes COCO acceptades
//    2. TRANSLATIONS →  nom traduït per a cada classe
//    3. El títol i els textos de la interfície (index.html)
// ============================================================

// ── 1. CATEGORIES ACTIVES ────────────────────────────────────
//  Llista de classes COCO-SSD que l'app mostrarà i enviarà.
//  Per canviar l'aplicació (residus, etc.) substitueix aquest array.
//  Classes COCO-SSD disponibles: https://github.com/nightrome/cocostuff
const CATEGORIES = ['cat', 'dog', 'bird', 'person'];

// ── 2. TRADUCCIONS ──────────────────────────────────────────
//  Nom en català que es mostrarà en pantalla i s'enviarà per UART.
const TRANSLATIONS = {
  cat:    'gat',
  dog:    'gos',
  bird:   'ocell',
  person: 'persona',
  // ── Afegeix aquí futures categories per a l'app de residus ──
  // bottle:      'ampolla',
  // cup:         'got',
  // bowl:        'bol',
  // book:        'paper',
  // chair:       'cadira',
};

// ── 3. CONFIGURACIÓ DEL MODEL ────────────────────────────────
const MODEL_CONFIG = {
  base:  'lite_mobilenet_v2',   // Ràpid i lleuger per a mòbils
  score_threshold: 0.35,        // Confiança mínima per mostrar detecció
};

// ────────────────────────────────────────────────────────────
//  Internals — no cal modificar per canviar de categoria
// ────────────────────────────────────────────────────────────
let model = null;
let isRunning = false;
let detectionLoop = null;

// Callbacks que el script principal pot registrar
let onDetectionCallback = null;
let onModelReadyCallback = null;
let onModelErrorCallback = null;

/**
 * Inicialitza el model COCO-SSD.
 * Crida onModelReady quan estigui llest, onModelError si falla.
 */
async function initModel() {
  try {
    model = await cocoSsd.load(MODEL_CONFIG);
    if (onModelReadyCallback) onModelReadyCallback();
  } catch (e) {
    console.error('❌ Error carregant el model:', e);
    if (onModelErrorCallback) onModelErrorCallback(e);
  }
}

/**
 * Comença el bucle de detecció sobre un element de vídeo o canvas.
 * @param {HTMLVideoElement} videoEl
 * @param {number} intervalMs  — interval en mil·lisegons entre deteccions
 */
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
      const predictions = await model.detect(videoEl);
      const filtered = predictions
        .filter(p => CATEGORIES.includes(p.class) && p.score >= MODEL_CONFIG.score_threshold)
        .map(p => ({
          class:       p.class,
          label:       TRANSLATIONS[p.class] || p.class,
          score:       Math.round(p.score * 100),
          bbox:        p.bbox,   // [x, y, width, height] en px de la imatge original
        }));

      if (onDetectionCallback) onDetectionCallback(filtered);
    } catch (e) {
      console.error('❌ Error en detecció:', e);
    }
    detectionLoop = setTimeout(detect, intervalMs);
  }

  detect();
}

/**
 * Atura el bucle de detecció.
 */
function stopDetection() {
  isRunning = false;
  if (detectionLoop) clearTimeout(detectionLoop);
}

/**
 * Canvia l'interval del bucle sense aturar-lo del tot.
 * @param {HTMLVideoElement} videoEl
 * @param {number} newIntervalMs
 */
function updateInterval(videoEl, newIntervalMs) {
  stopDetection();
  startDetection(videoEl, newIntervalMs);
}

// ── API pública ───────────────────────────────────────────────
function onDetection(cb)    { onDetectionCallback   = cb; }
function onModelReady(cb)   { onModelReadyCallback  = cb; }
function onModelError(cb)   { onModelErrorCallback  = cb; }
function getCategories()    { return CATEGORIES; }
function getTranslations()  { return TRANSLATIONS; }
