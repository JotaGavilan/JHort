const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const labelEl = document.getElementById('label');
const scoreEl = document.getElementById('score');
const statusEl = document.getElementById('status');

let model;
let ultimoEnvio = 0;

// Objetivos de IA
const objetivosValidos = ["person", "dog", "cat", "bird", "sheep", "cow"];

async function init() {
    statusEl.textContent = "Carregant model...";
    model = await cocoSsd.load();
    statusEl.textContent = "Model a punt. Iniciant càmera...";
    
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment', width: 640, height: 480 } 
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
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach(p => {
        if (objetivosValidos.includes(p.class) && p.score > 0.6) {
            // Dibujamos el cuadro en ROJO (estética solicitada)
            ctx.strokeStyle = "#FF0000";
            ctx.lineWidth = 3;
            ctx.strokeRect(...p.bbox);

            const ahora = Date.now();
            if (ahora - ultimoEnvio > 1000) {
                labelEl.textContent = p.class.toUpperCase();
                scoreEl.textContent = Math.round(p.score * 100);
                
                // Envío a Microbit vía UART
                if (typeof sendUARTData === "function") {
                    sendUARTData(`${p.class}:${Math.round(p.score * 100)}`);
                }
                ultimoEnvio = ahora;
            }
        }
    });
    requestAnimationFrame(detect);
}

init();