"""
Punto de entrada del proyecto.

Orquesta el flujo completo:
  1. Comprobar si existe modelo guardado
     → SÍ: cargar modelo + vocabularios (sin tocar el dataset)
     → NO: cargar dataset → construir vocab → entrenar → guardar
  2. Ejecutar ejemplos de traducción
  3. Lanzar modo interactivo
"""
import os

from config import MODEL_PATH, SRC_VOCAB_PATH, TGT_VOCAB_PATH, device
from config import src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
from model import Transformer
from train import load_checkpoint, save_checkpoint, train
from data import load_data
from translate import translate, interactive_mode

print(f"Usando device: {device}", flush=True)

# Instanciar el modelo vacío (necesario tanto para cargar como para entrenar)
transformer = Transformer(
    src_vocab_size, tgt_vocab_size, d_model, num_heads,
    num_layers, d_ff, max_seq_length, dropout
).to(device)

# =============================================================================
# PASO 6: CARGAR O ENTRENAR
# El check va LO PRIMERO para no cargar el dataset innecesariamente.
# Cuando el modelo ya existe, no tiene sentido procesar 93k frases (~45 segundos)
# solo para descartarlas.
# =============================================================================
all_files_exist = all(os.path.exists(p) for p in [MODEL_PATH, SRC_VOCAB_PATH, TGT_VOCAB_PATH])

if all_files_exist:
    src_vocab, tgt_vocab = load_checkpoint(transformer)
else:
    src_vocab, tgt_vocab, dataloader = load_data()
    train(transformer, dataloader)
    save_checkpoint(transformer, src_vocab, tgt_vocab)

# =============================================================================
# PASO 7: TRADUCIR
# idx_to_word se construye UNA SOLA VEZ aquí y se pasa a translate().
# Construirlo dentro de translate() lo recrearía en cada llamada (ineficiente).
# =============================================================================
idx_to_word = {idx: word for word, idx in tgt_vocab.items()}

# Ejemplos automáticos para comprobar que el modelo funciona
print("\n--- Probando el modelo ---", flush=True)
test_sentences = [
    "el gato duerme en la cama",
    "buenos días, ¿cómo estás?",
    "me gusta el café por la mañana",
    "el libro está sobre la mesa",
    "hoy hace mucho calor",
]
for sentence in test_sentences:
    translation = translate(sentence, transformer, src_vocab, idx_to_word)
    print(f"ES: {sentence}", flush=True)
    print(f"EN: {translation}", flush=True)
    print(flush=True)

# Modo interactivo
interactive_mode(transformer, src_vocab, idx_to_word)
