// ============================================================
//  bluetooth_uart.js  –  Connexió Bluetooth amb micro:bit via UART
//  Mateix protocol que jFace. No cal modificar per canviar categoria.
// ============================================================

const UART_SERVICE_UUID      = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
const UART_TX_CHARACTERISTIC = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';
const UART_RX_CHARACTERISTIC = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';

let uBitDevice       = null;
let rxCharacteristic = null;
let uart             = null;
let queue            = Promise.resolve();
let isConnected      = false;

// Callback que script.js pot registrar per rebre canvis d'estat
let onStatusChangeCallback = null;

function queueGattOperation(operation) {
  queue = queue.then(operation, operation);
  return queue;
}

function onBTStatusChange(cb) {
  onStatusChangeCallback = cb;
}

function notifyStatus(connected, message) {
  isConnected = connected;
  if (onStatusChangeCallback) onStatusChangeCallback(connected, message);
}

async function connectBluetooth() {
  try {
    notifyStatus(false, '🔍 Cercant micro:bit...');

    uBitDevice = await navigator.bluetooth.requestDevice({
      filters:          [{ namePrefix: 'BBC micro:bit' }],
      optionalServices: [UART_SERVICE_UUID],
    });

    uBitDevice.addEventListener('gattserverdisconnected', onDisconnected);

    const server  = await uBitDevice.gatt.connect();
    const service = await server.getPrimaryService(UART_SERVICE_UUID);

    rxCharacteristic = await service.getCharacteristic(UART_TX_CHARACTERISTIC);
    await rxCharacteristic.startNotifications();
    rxCharacteristic.addEventListener('characteristicvaluechanged', onTxValueChanged);

    uart = await service.getCharacteristic(UART_RX_CHARACTERISTIC);

    notifyStatus(true, '✅ micro:bit connectada');
  } catch (e) {
    console.error('❌ Error BT:', e);
    uart = null;
    rxCharacteristic = null;
    notifyStatus(false, '❌ Error en la connexió');
  }
}

function onTxValueChanged(event) {
  const data = new Uint8Array(event.target.value.buffer);
  const str  = String.fromCharCode(...data);
  console.log('📥 Rebut de micro:bit:', str);
}

function onDisconnected(event) {
  console.log(`🔌 Desconnectat de ${event.target.name}`);
  uart             = null;
  rxCharacteristic = null;
  notifyStatus(false, '🔌 micro:bit desconnectada');
}

/**
 * Envia una cadena per UART.
 * Format esperat per a JHort: "gat:87,ocell:45"
 * @param {string} data
 */
function sendUARTData(data) {
  if (!uart) return;
  const encoded = new TextEncoder().encode(data + '\n');
  queueGattOperation(() =>
    uart.writeValue(encoded)
      .then(()  => console.log('📤 UART enviat:', data))
      .catch(e  => console.error('❌ Error UART:', e))
  );
}

function isBluetoothConnected() {
  return isConnected && uart !== null;
}
