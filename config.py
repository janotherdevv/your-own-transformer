import torch

# =============================================================================
# TOKENS ESPECIALES
# =============================================================================
PAD_IDX = 0  # padding: relleno para frases cortas (ignorado por CrossEntropyLoss)
SOS_IDX = 1  # start of sequence: marca el inicio de cada frase
EOS_IDX = 2  # end of sequence: marca el final de cada frase
UNK_IDX = 3  # unknown: cualquier palabra que no esté en el vocabulario

# =============================================================================
# HIPERPARÁMETROS DEL MODELO
# =============================================================================
# Los valores del paper original "Attention is All You Need" — diseñados para GPU.
# En CPU con estos valores el PC puede congelar (~2.5GB RAM en pico).
src_vocab_size = 5000   # Palabras distintas en el vocabulario español
tgt_vocab_size = 5000   # Palabras distintas en el vocabulario inglés
d_model        = 512    # Dimensión de los vectores de embedding
num_heads      = 8      # Cabezas de atención paralelas (d_model debe ser divisible)
num_layers     = 6      # Capas en encoder y decoder
d_ff           = 2048   # Dimensión interna de la red feed-forward (4 × d_model)
max_seq_length = 100    # Longitud máxima de secuencia (tokens)
dropout        = 0.1    # Fracción de neuronas apagadas durante entrenamiento

# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================
batch_size     = 32
epochs         = 10
learning_rate  = 0.0001

# =============================================================================
# RUTAS DE ARCHIVOS
# =============================================================================
MODEL_PATH     = "transformer_model.pt"  # Pesos del modelo entrenado
SRC_VOCAB_PATH = "src_vocab.json"        # Vocabulario español -> índice
TGT_VOCAB_PATH = "tgt_vocab.json"        # Vocabulario inglés  -> índice

# =============================================================================
# DEVICE (GPU si está disponible, CPU si no)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
