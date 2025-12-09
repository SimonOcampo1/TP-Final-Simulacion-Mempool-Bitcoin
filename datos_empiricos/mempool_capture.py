"""
CAPTURA DE DATOS EMPÍRICOS - MEMPOOL DE BITCOIN
Universidad Tecnológica Nacional - FRLP
Autor: Simón Tadeo Ocampo
Año: 2025

Script de captura de transacciones no confirmadas en tiempo real
mediante WebSocket connection a blockchain.info.

Características:
- Timestamps de alta precisión al momento de recepción
- Cálculo automático de fee-rate (sats/vB)
- Reconexión automática con backoff exponencial
- Validación de datos antes de almacenamiento
- Duración configurable por línea de comandos

Uso:
    python mempool_capture.py           # Captura infinita (CTRL-C para detener)
    python mempool_capture.py 1.0       # Captura por 1 hora
    python mempool_capture.py 0.5       # Captura por 30 minutos
"""

import asyncio
import websockets
import json
import csv
import logging
import socket
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

WEBSOCKET_URL = "wss://ws.blockchain.info/inv"


def check_internet_connectivity() -> bool:
    """Verifica conectividad a internet antes de intentar conexión WebSocket"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False


def test_blockchain_connectivity() -> bool:
    """Verifica accesibilidad de blockchain.info"""
    try:
        socket.create_connection(("ws.blockchain.info", 443), timeout=10)
        return True
    except OSError:
        return False


def calculate_transaction_fee(tx: Dict[str, Any]) -> tuple[int, int]:
    """
    Calcula fee de transacción y retorna (fee_sats, size_bytes).
    Retorna (0, 0) si no se puede calcular.
    """
    try:
        inputs = tx.get('inputs', [])
        total_input = sum(
            inp.get('prev_out', {}).get('value', 0) 
            for inp in inputs 
            if isinstance(inp.get('prev_out', {}).get('value', 0), (int, float))
        )
        
        outputs = tx.get('out', [])
        total_output = sum(
            out.get('value', 0) 
            for out in outputs 
            if isinstance(out.get('value', 0), (int, float))
        )
        
        fee_sats = total_input - total_output
        size_bytes = tx.get('size', 0)
        
        if fee_sats < 0 or size_bytes <= 0:
            return 0, 0
        
        return fee_sats, size_bytes
        
    except Exception as e:
        logger.error(f"Error calculando fee: {e}")
        return 0, 0


async def process_transaction(data: Dict[Any, Any], writer: csv.DictWriter, counter: Dict[str, int]) -> bool:
    """
    Procesa transacción individual y escribe al CSV.
    Retorna True si fue procesada exitosamente.
    """
    try:
        capture_time = datetime.now(timezone.utc)
        
        if not (isinstance(data, dict) and data.get('op') == 'utx' and 'x' in data):
            return False
        
        tx = data['x']
        txid = tx.get('hash')
        if not txid:
            return False
        
        fee_sats, size_bytes = calculate_transaction_fee(tx)
        
        if fee_sats <= 0 or size_bytes <= 0:
            return False
        
        fee_rate = fee_sats / size_bytes
        
        writer.writerow({
            'timestamp_utc': capture_time.isoformat(),
            'txid': txid,
            'fee_sats': fee_sats,
            'size_bytes': size_bytes,
            'fee_rate_sats_per_byte': round(fee_rate, 4)
        })
        
        counter['processed'] += 1
        
        if counter['processed'] % 10 == 0:
            logger.info(f"✅ Procesadas {counter['processed']} transacciones | Última: {txid[:12]}... | Fee Rate: {fee_rate:.2f} sats/byte")
        
        return True
        
    except Exception as e:
        logger.error(f"Error procesando transacción: {e}")
        return False


async def listen_to_mempool_with_reconnect(max_retries: int = 5, duration_hours: float = None) -> None:
    """
    Conexión WebSocket con lógica de reconexión automática.
    
    Args:
        max_retries: Número máximo de reintentos
        duration_hours: Duración en horas (None = infinito)
    """
    if not check_internet_connectivity():
        logger.error("❌ Sin conectividad a internet.")
        return
    
    if not test_blockchain_connectivity():
        logger.error("❌ No se puede conectar a blockchain.info.")
        return
    
    logger.info("✅ Conectividad verificada correctamente.")
    
    start_time = datetime.now(timezone.utc)
    end_time = None
    if duration_hours:
        end_time = start_time + timedelta(hours=duration_hours)
        logger.info(f"⏱️  Captura programada por {duration_hours} horas")
        logger.info(f"🕐 Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"🕐 Fin programado: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mempool_data_{timestamp_str}.csv"
    
    logger.info("🚀 Iniciando captura de datos desde Blockchain.com")
    logger.info(f"📁 Guardando en: {filename}")
    if not duration_hours:
        logger.info("⏹️  Presiona CTRL-C para detener")
    
    counter = {'processed': 0, 'errors': 0, 'reconnects': 0}
    
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['timestamp_utc', 'txid', 'fee_sats', 'size_bytes', 'fee_rate_sats_per_byte']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.flush()
            
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info(f"🔗 Conectando a WebSocket... (Intento {retry_count + 1}/{max_retries})")
                    
                    async with websockets.connect(
                        WEBSOCKET_URL,
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=10,
                        max_size=10**6
                    ) as websocket:
                        
                        logger.info("✅ Conexión WebSocket establecida")
                        
                        subscribe_message = {"op": "unconfirmed_sub"}
                        await websocket.send(json.dumps(subscribe_message))
                        logger.info("📡 Suscripción enviada. Esperando transacciones...")
                        
                        retry_count = 0
                        
                        async for message in websocket:
                            try:
                                if end_time and datetime.now(timezone.utc) >= end_time:
                                    logger.info("⏰ Tiempo de captura completado.")
                                    await websocket.close()
                                    return
                                
                                data = json.loads(message)
                                
                                success = await process_transaction(data, writer, counter)
                                if success:
                                    csv_file.flush()
                                else:
                                    counter['errors'] += 1
                                    
                            except json.JSONDecodeError:
                                counter['errors'] += 1
                            except Exception as e:
                                logger.error(f"❌ Error procesando mensaje: {e}")
                                counter['errors'] += 1
                        
                        logger.warning("⚠️  Conexión WebSocket cerrada por el servidor")
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("⚠️  Conexión WebSocket cerrada inesperadamente")
                    retry_count += 1
                    counter['reconnects'] += 1
                except websockets.exceptions.WebSocketException as e:
                    logger.error(f"❌ Error de WebSocket: {e}")
                    retry_count += 1
                    counter['reconnects'] += 1
                except Exception as e:
                    logger.error(f"❌ Error inesperado: {e}")
                    retry_count += 1
                    counter['reconnects'] += 1
                
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 30)
                    logger.info(f"⏳ Esperando {wait_time} segundos antes de reconectar...")
                    await asyncio.sleep(wait_time)
            
            logger.error(f"❌ Máximo número de reintentos alcanzado ({max_retries})")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Captura detenida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error general: {e}")
    finally:
        logger.info("📊 ESTADÍSTICAS FINALES:")
        logger.info(f"   • Transacciones procesadas: {counter['processed']}")
        logger.info(f"   • Errores encontrados: {counter['errors']}")
        logger.info(f"   • Reconexiones realizadas: {counter['reconnects']}")
        logger.info(f"   • Archivo guardado: {filename}")


async def main(duration_hours: float = None):
    """Función principal que inicia captura de datos"""
    await listen_to_mempool_with_reconnect(duration_hours=duration_hours)


if __name__ == "__main__":
    import sys
    
    duration = None
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
            print(f"⏱️  Duración configurada: {duration} horas")
        except ValueError:
            print("❌ Error: La duración debe ser un número (ej: 1.5 para 1.5 horas)")
            sys.exit(1)
    
    asyncio.run(main(duration))
