const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const labelEl = document.getElementById('label');
const scoreEl = document.getElementById('score');
const statusEl = document.getElementById('status');

let model;
let ultimoEnvio = 0;

// Lista de objetivos para filtrar lo que ve la cámara
const objetivosValidos = ["person", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"];

async function initIA() {
  statusEl.textContent = "⏳ Carregant IA...";
  // Carga del modelo COCO-SSD
  model = await cocoSsd.load();
  statusEl.textContent = "📷 Iniciant càmera...";
  startCamera();
}

async function startCamera() {
  // Usamos cámara trasera para detectar lo que tenemos delante
  const stream = await navigator.mediaDevices.getUserMedia({ 
    video: { facingMode: 'environment' } 
  });
  video.srcObject = stream;
  
  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detectFrame();
  };
}

async function detectFrame() {
  const predictions = await model.detect(video);
  
  // Limpiar y dibujar el frame de video actual
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  let detectado = false;

  predictions.forEach(p => {
    if (objetivosValidos.includes(p.class)) {
      detectado = true;
      const score = Math.round(p.score * 100);
      
      // Estilo visual similar al original (verde neón)
      ctx.strokeStyle = "#00FF00";
      ctx.lineWidth = 3;
      ctx.strokeRect(...p.bbox);
      
      // Actualizar el panel de datos superior
      labelEl.textContent = p.class;
      scoreEl.textContent = score;

      // Lógica de envío Bluetooth
      const ahora = Date.now();
      // Mantenemos el control de flujo para no saturar la micro:bit
      if (ahora - ultimoEnvio > 500) { 
        // ENVIAMOS: nombre del animal + score (Ejemplo: "dog:85")
        const mensaje = `${p.class}:${score}`;
        sendUARTData(mensaje); 
        ultimoEnvio = ahora;
      }
    }
  });

  if (!detectado) {
    labelEl.textContent = "--";
    scoreEl.textContent = "--";
  }

  requestAnimationFrame(detectFrame);
}

initIA();