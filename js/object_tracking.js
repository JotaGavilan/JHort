const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const labelEl = document.getElementById('label');
const scoreEl = document.getElementById('score');
const statusEl = document.getElementById('status');

let model;
let ultimoEnvio = 0;

async function setupApp() {
    statusEl.textContent = "Carregant IA...";
    model = await cocoSsd.load();
    
    // Configuración de cámara trasera
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
    });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        detect();
    };
}

async function detect() {
    const predictions = await model.detect(video);
    
    // Limpiar canvas y dibujar video
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    let encontrado = false;

    predictions.forEach(p => {
        // Filtramos por confianza > 60%
        if (p.score > 0.6) {
            encontrado = true;
            
            // Dibujar cuadro Verde Neón (Estética Jface)
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.strokeRect(...p.bbox);

            const ahora = Date.now();
            if (ahora - ultimoEnvio > 1000) { // 1 segundo
                labelEl.textContent = p.class.toUpperCase();
                scoreEl.textContent = Math.round(p.score * 100);
                
                // Enviar a micro:bit
                if (typeof sendUARTData === "function") {
                    sendUARTData(`${p.class}:${Math.round(p.score * 100)}`);
                }
                ultimoEnvio = ahora;
            }
        }
    });

    if (!encontrado) {
        labelEl.textContent = "--";
        scoreEl.textContent = "--";
    }

    requestAnimationFrame(detect);
}

setupApp();