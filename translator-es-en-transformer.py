import torch  # Framework de deep learning: tensores, redes neuronales, autograd
import torch.nn as nn  # Módulo de redes neuronales: capas, funciones de activación
import torch.optim as optim  # Algoritmos de optimización: Adam, SGD, etc.
from torch.utils.data import Dataset, DataLoader  # Dataset y DataLoader para manejar datos
import math  # Funciones matemáticas: sqrt, log, etc.
import copy  # Copias profundas de objetos
from datasets import load_dataset  # HuggingFace datasets: carga datasets públicos en una línea
from collections import Counter  # Cuenta frecuencias de palabras para construir el vocabulario

class MultiHeadAttention(nn.Module):
    """
    Capa de atención multi-cabeza (Multi-Head Attention).
    
    Implementa el mecanismo de atención self-attention con múltiples heads paralelos.
    Permite al modelo capturar diferentes tipos de relaciones entre palabras simultáneamente.
    Cada head aprende patrones diferentes (sintácticos, semánticos, etc.).
    
    Args:
        d_model: Dimensionalidad de los vectores de entrada/salida (típicamente 512)
        num_heads: Número de heads de atención (típicamente 8)

    Ejemplo: En la frase "El gato persigue al ratón", un head podría enfocarse en la relación sujeto-verbo ("gato" -> "persigue"), mientras otro head podría enfocarse en la relación verbo-objeto ("persigue" -> "ratón"). La atención multi-cabeza permite que ambos patrones se aprendan simultáneamente.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Verifica que d_model sea divisible por num_heads para poder dividir evenly
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model  # Dimensionalidad del modelo
        self.num_heads = num_heads  # Número de heads
        self.d_k = d_model // num_heads  # Dimensionalidad de cada head (d_model / num_heads)
        
        # Matrices de proyección lineal para Q, K, V:
        # W_q: proyecta embeddings a Queries (qué busco)
        # W_k: proyecta embeddings a Keys (qué ofrezco)
        # W_v: proyectos embeddings a Values (qué inform tengo)
        # Cada una de dimensión d_model x d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # W_o: proyección de vuelta a dimensión d_model después de concatenar heads
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Implementa el mecanismo Scaled Dot-Product Attention.
        
        La escala por sqrt(d_k) evita que los productos punto sean demasiado grandes,
        lo que causaría gradientes muy pequeños durante el entrenamiento (softmax saturado).
        
        Fórmula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        
        Args:
            Q: Tensor de Queries [batch, num_heads, seq_len, d_k]
            K: Tensor de Keys [batch, num_heads, seq_len, d_k]
            V: Tensor de Values [batch, num_heads, seq_len, d_k]
            mask: Máscara opcional para padding o atención causal
            
        Returns:
            Tensor de salida [batch, num_heads, seq_len, d_k]
        """
        # Calcula los scores de atención: Q * K^T
        # Cada elemento representa la similitud entre query i y key j
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Aplica la máscara (si existe): marca posiciones inválidas con -1e9
        # Esto hace que el softmax devuelva ~0 en esas posiciones
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Aplica softmax para obtener probabilidades de atención (suman 1 por fila)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Pondera los valores (V) por las probabilidades de atención
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        """
        Divide el tensor en múltiples heads.
        
        Transforma [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        
        Args:
            x: Tensor de entrada [batch, seq_len, d_model]
            
        Returns:
            Tensor dividido [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_length, d_model = x.size()
        # reshape: agrupa las dimensiones del modelo en (num_heads, d_k)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """
        Combina múltiples heads de nuevo en un tensor único.
        
        Transforma [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        
        Args:
            x: Tensor de heads [batch, num_heads, seq_len, d_k]
            
        Returns:
            Tensor combinado [batch, seq_len, d_model]
        """
        batch_size, _, seq_length, d_k = x.size()
        # transpose + view: revierte la operación de split_heads
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass completo de Multi-Head Attention.
        
        Flujo:
        1. Proyectar entrada a Q, K, V mediante matrices lineales
        2. Dividir en múltiples heads
        3. Calcular atención con scaled dot-product
        4. Combinar heads
        5. Proyectar con W_o
        
        Args:
            Q: Queries [batch, seq_len, d_model]
            K: Keys [batch, seq_len, d_model]
            V: Values [batch, seq_len, d_model]
            mask: Máscara opcional
            
        Returns:
            Output attention [batch, seq_len, d_model]
        """
        # Paso 1: Proyecciones lineales
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Paso 2: Calcular atención
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Paso 3: Combinar heads y proyectar salida
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    """
    Red feed-forward posicionada (Position-wise Feed-Forward Network).
    
    Está compuesta por dos transformaciones lineales con una activación ReLU entre ellas.
    Se aplica la misma red a cada posición de la secuencia de forma independiente
    (los pesos son compartidos, pero el cálculo es independiente por posición).
    
    Añade no-linealidad y permite interacciones entre dimensiones diferentes a las de atención.
    
    Args:
        d_model: Dimensionalidad de entrada/salida (típicamente 512)
        d_ff: Dimensionalidad de la capa oculta (típicamente 2048 = 4 * d_model)

    Ejemplo: como una maquina expendedora: la maquina es la misma, pero dependiendo de que botones pulses (input) sale un producto distinto (output). La red feed-forward es la máquina, y cada posición de la secuencia es un botón diferente.
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # Primera capa lineal: expande de d_model a d_ff
        # Típicamente d_ff = 4 * d_model para dar más capacidad
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # Segunda capa lineal: comprime de d_ff de vuelta a d_model
        self.fc2 = nn.Linear(d_ff, d_model)
        
        # Función de activación ReLU: introduce no-linealidad
        # ReLU(x) = max(0, x)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    """
    Codificación posicional (Positional Encoding).
    
    El Transformer no tiene recurrencia ni convolución, por lo que necesita una forma
    de inyectar información sobre el orden de las palabras. Usa funciones senoidales
    con diferentes frecuencias:
    
    - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Ventajas de senos/cosenos sobre embeddings aprendidos:
    - Puedeposicionar información sobre distancias relativas
    - Generaliza a longitudes mayores que las vistas en entrenamiento
    - Es determinista y no requiere aprenderse
    
    Args:
        d_model: Dimensionalidad de los embeddings
        max_seq_length: Longitud máxima de secuencia a codificar
    
    Ejemplo: No es lo mismo "El gato persigue al ratón" que "El ratón persigue al gato". La codificación posicional ayuda al modelo a entender esta diferencia de orden.
    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Aplica seno a posiciones pares (índices 0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Aplica coseno a posiciones impares (índices 1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register_buffer: guarda 'pe' con el modelo pero no como parámetro entrenable
        # unsqueeze(0) añade dimensión de batch: [1, max_seq_length, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Añade la codificación posicional correspondiente a la longitud de secuencia
        return x + self.pe[:, :x.size(1)]

# Capa de codificador (Encoder Layer)
class EncoderLayer(nn.Module):
    """
    Una capa del Encoder del Transformer.
    
    El Encoder procesa la secuencia de entrada (español) y la transforma en una
    representación interna rica en información. Cada capa tiene dos subcapas principales:
    
    1. Self-Attention: la secuencia mira a toda ella misma para entender relaciones
    2. Feed-Forward: procesa cada posición independientemente
    
    Además usa:
    - Residual connections (Add): suma la entrada a la salida para preservar el gradiente
    - Layer Normalization: normaliza los valores para estabilizar el entrenamiento
    
    Ejemplo humano: Es como un equipo de lectura donde cada persona (capa) lee el texto
    y va subrayando información importante, pasando notas al siguiente lector.
    
    Args:
        d_model: Dimensionalidad del modelo (típicamente 512)
        num_heads: Número de heads de atención (típicamente 8)
        d_ff: Dimensionalidad de la capa oculta del FFN (típicamente 2048)
        dropout: Tasa de dropout para regularización (típicamente 0.1)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # Self-Attention: atención sobre la propia secuencia de entrada
        # "Cada palabra mira a todas las demás palabras"
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-Forward Network: procesamiento posicion por posicion
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # LayerNorm: normaliza los vectores para que tengan media 0 y varianza 1
        # Se aplica dos veces: despues de attention y despues de FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout: apalea aleatoriamente neuronas durante entrenamiento
        # ayuda a prevenir overfitting (memorizar en lugar de generalizar)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Paso 1: Self-Attention
        # Q = K = V = x (por eso se llama "self")
        # La secuencia entera mira a toda la secuencia para entender relaciones
        attn_output = self.self_attn(x, x, x, mask)
        
        # Paso 2: Residual Connection + Dropout + LayerNorm
        # x + attention: "recuerda" la entrada original mientras añade la nueva info
        x = self.norm1(x + self.dropout(attn_output))
        
        # Paso 3: Feed-Forward
        ff_output = self.feed_forward(x)
        
        # Paso 4: Residual Connection + Dropout + LayerNorm
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    """
    Una capa del Decoder del Transformer.
    
    El Decoder genera la secuencia de salida (inglés) token a token. A diferencia del
    Encoder, tiene TRES subcapas principales:
    
    1. Masked Self-Attention: solo mira las posiciones ya generadas (izquierda)
    2. Cross-Attention: mira la salida del Encoder (qué significa la entrada)
    3. Feed-Forward: procesa cada posición independientemente
    
    Ejemplo: Es como traducir sentence by sentence. Para traducir la segunda
    palabra, solo puedes ver la primera palabra traducida (no conoces el futuro).
    Pero SIEMPRE miras el texto original para saber qué estás traduciendo.
    
    Args:
        d_model: Dimensionalidad del modelo (típicamente 512)
        num_heads: Número de heads de atención (típicamente 8)
        d_ff: Dimensionalidad de la capa oculta del FFN (típicamente 2048)
        dropout: Tasa de dropout para regularización (típicamente 0.1)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # Self-Attention con máscara (masked): solo ve posiciones anteriores
        # "Solo puedo ver lo que ya he traducido, no el futuro"
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Cross-Attention: atención sobre la salida del Encoder
        # "Miro el texto original para saber qué traducir"
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-Forward Network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Tres LayerNorms: uno por cada subcapa
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Paso 1: Masked Self-Attention
        # Importante: usa tgt_mask para evitar que el modelo "haga trampa"
        # mirando palabras futuras que aún no ha generado
        attn_output = self.self_attn(x, x, x, tgt_mask)
        
        # Paso 2: Residual + Norm
        x = self.norm1(x + self.dropout(attn_output))
        
        # Paso 3: Cross-Attention
        # Q = output del decoder (lo que vamos traduciendo)
        # K = V = output del encoder (el texto original en español)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        
        # Paso 4: Residual + Norm
        x = self.norm2(x + self.dropout(attn_output))
        
        # Paso 5: Feed-Forward
        ff_output = self.feed_forward(x)
        
        # Paso 6: Residual + Norm
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
    
class Transformer(nn.Module):
    """
    Modelo Transformer completo para traducción seq2seq.
    
    El Transformer combina el Encoder y el Decoder para crear el modelo de traducción.
    
    Estructura general:
    1. Embeddings: convierte palabras a vectores densos
    2. Positional Encoding: añade información de posición
    3. Encoder (N capas): procesa la secuencia de entrada (español)
    4. Decoder (N capas): genera la secuencia de salida (inglés)
    5. Capa lineal + Softmax: convierte vectores a probabilidades de palabras
    
    Ejemplo humano completo:
    - Input (español): "Hola mundo"
    - El Encoder "entiende" cada palabra en contexto
    - El Decoder genera "Hello world" palabra por palabra
    - Cada palabra generada depende de: lo anterior + el contexto del Encoder
    
    Args:
        src_vocab_size: Tamaño del vocabulario fuente (español)
        tgt_vocab_size: Tamaño del vocabulario objetivo (inglés)
        d_model: Dimensionalidad de los embeddings (típicamente 512)
        num_heads: Número de heads de atención (típicamente 8)
        num_layers: Número de capas en Encoder y Decoder (típicamente 6)
        d_ff: Dimensionalidad de la capa oculta del FFN (típicamente 2048)
        max_seq_length: Longitud máxima de secuencia
        dropout: Tasa de dropout (típicamente 0.1)
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        
        # Embedding del Encoder: convierte tokens españoles a vectores de dimensión d_model
        # Cada palabra española se mapea a un vector de d_model dimensiones
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        
        # Embedding del Decoder: convierte tokens ingleses a vectores de dimensión d_model
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Codificación posicional: añade información de orden a los embeddings
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Pila de capas del Encoder (típicamente 6 capas)
        # ModuleList permite que cada capa tenga sus propios pesos entrenables
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Pila de capas del Decoder (típicamente 6 capas)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Capa final lineal: convierte los vectores d_model a probabilidades del vocabulario objetivo
        # Output: scores para cada palabra del vocabulario inglés
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Genera las máscaras necesarias para el modelo.
        
        Dos tipos de máscaras:
        1. src_mask: oculta los tokens de padding en el source
        2. tgt_mask: oculta los tokens de padding Y las posiciones futuras en el target
        
        Ejemplo de tgt_mask (máscara causal):
        Input: "Hello world"
        Máscara: 
        [[1, 0, 0],   # "Hello" solo ve "Hello"
         [1, 1, 0],   # "world" ve "Hello" y "world"  
         [1, 1, 1]]  # <EOS> ve todo
        
        Args:
            src: Tensores de entrada (español) [batch, src_len]
            tgt: Tensores objetivo (inglés) [batch, tgt_len]
            
        Returns:
            src_mask: Máscara para source
            tgt_mask: Máscara para target (causal + padding)
        """
        # src_mask: marca dónde hay padding (0) vs tokens válidos (1)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # tgt_mask: marca dónde hay padding
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        
        # nopeak_mask: evita que el modelo vea el futuro (máscara causal)
        # Crea una matriz triangular superior con 0s
        seq_length = tgt.size(1)
        # .to(tgt.device): nopeak_mask se crea siempre en CPU por defecto.
        # Si el modelo está en GPU, tgt_mask ya está en GPU y el & fallaría con
        # "Expected all tensors to be on the same device". Hay que moverlo explícitamente.
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        
        # Combina: padding mask AND causal mask
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Forward pass completo del Transformer.
        
        Flujo de datos:
        1. Embedding + Positional Encoding del source
        2. Pass por todas las capas del Encoder
        3. Embedding + Positional Encoding del target  
        4. Pass por todas las capas del Decoder (con output del Encoder)
        5. Capa lineal + Softmax para predicción
        
        Args:
            src: Secuencia de entrada (español) [batch, src_seq_len]
            tgt: Secuencia objetivo (inglés) [batch, tgt_seq_len]
            
        Returns:
            output: Predicciones de siguiente palabra [batch, tgt_seq_len, tgt_vocab_size]
        """
        # Paso 1: Generar máscaras
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Paso 2: Embedding + Positional Encoding del source (español)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        
        # Paso 3: Embedding + Positional Encoding del target (inglés)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        
        # Paso 4: Pasar por todas las capas del Encoder
        # Input: "Hola mundo" -> Output: representación "entendida"
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # Paso 5: Pasar por todas las capas del Decoder
        # Input: representación encoder + lo traducido hasta ahora
        # Output: siguiente palabra predicha
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Paso 6: Capa final: convertir vectores a probabilidades de vocabulario
        output = self.fc(dec_output)
        return output

# =============================================================================
# PASO 5: PREPARAR DATOS DE EJEMPLO
# =============================================================================
# Hiperparámetros del modelo: definen la arquitectura y capacidad
## NOTA: Los valores del paper original están diseñados para GPU.
## En CPU petan la RAM (~2.5GB en pico). Aquí usamos GPU si está disponible.
##
## Comparativa de memoria:
##   GPU (VRAM, este):  batch=64, seq=100, d=512, layers=6  → ~2.5GB VRAM  ✓ cabe en RTX 3060
##   CPU (RAM):         batch=64, seq=100, d=512, layers=6  → ~2.5GB RAM   ✗ puede congelar el PC

# Detectar si hay GPU disponible y usarla automáticamente.
# device = "cuda" si hay GPU con CUDA, "cpu" si no hay.
# Todo lo que vaya a la GPU (modelo, datos) debe estar en el mismo device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

src_vocab_size = 5000   # Tamaño del vocabulario español: 5000 palabras distintas conocidas
tgt_vocab_size = 5000   # Tamaño del vocabulario inglés: 5000 palabras distintas conocidas
d_model = 512           # Cada palabra se representa como un vector de 512 números
num_heads = 8           # La atención se divide en 8 "cabezas" paralelas
num_layers = 6          # El Encoder y el Decoder tienen 6 capas cada uno
d_ff = 2048             # La red feed-forward interna usa vectores de 2048 (4x d_model)
max_seq_length = 100    # Las frases pueden tener como máximo 100 tokens
dropout = 0.1           # Durante el entrenamiento, se apagan aleatoriamente el 10% de neuronas

# Instanciar el modelo y moverlo al device (GPU o CPU).
# .to(device) copia todos los pesos del modelo a la memoria del device elegido.
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)

# --- TOKENS ESPECIALES ---
# El vocabulario reserva los primeros 4 índices para tokens de control:
#   0 = <PAD>  padding: relleno para frases cortas (ignorado por CrossEntropyLoss)
#   1 = <SOS>  start of sequence: marca el inicio de cada frase
#   2 = <EOS>  end of sequence: marca el final de cada frase
#   3 = <UNK>  unknown: cualquier palabra que no esté en el vocabulario
# Las palabras reales empiezan desde el índice 4.
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# --- CARGAR DATASET REAL ---
# opus_books: traducciones de libros literarios español <-> inglés
# ~130k pares de frases, texto real de alta calidad.
# HuggingFace lo descarga y cachea automáticamente en ~/.cache/huggingface/
print("Cargando dataset opus_books es-en...")
dataset = load_dataset("opus_books", "es-en")
train_data = dataset["train"]

# Extraer las frases en cada idioma del dataset.
# La estructura de opus_books es: train_data["translation"] = [{"es": "...", "en": "..."}, ...]
src_sentences = [item["es"] for item in train_data["translation"]]
tgt_sentences = [item["en"] for item in train_data["translation"]]
print(f"Frases cargadas: {len(src_sentences)}")

# --- CONSTRUIR VOCABULARIOS ---
# Un vocabulario es un diccionario palabra -> índice.
# Ejemplo: {"<PAD>": 0, "<SOS>": 1, ..., "gato": 47, "perro": 83, ...}
#
# Usamos las src_vocab_size-4 palabras MÁS FRECUENTES del corpus.
# Las palabras raras (poco frecuentes) se mapean a <UNK>.
def build_vocab(sentences, max_size):
    # Tokenización simple: minúsculas + separar por espacios
    # En producción se usaría un tokenizador real (BPE, WordPiece, etc.)
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.lower().split())
    # Empezar con los tokens especiales, luego añadir las palabras más comunes
    vocab = {"<PAD>": PAD_IDX, "<SOS>": SOS_IDX, "<EOS>": EOS_IDX, "<UNK>": UNK_IDX}
    for word, _ in counter.most_common(max_size - 4):
        vocab[word] = len(vocab)
    return vocab

print("Construyendo vocabularios...")
src_vocab = build_vocab(src_sentences, src_vocab_size)
tgt_vocab = build_vocab(tgt_sentences, tgt_vocab_size)
print(f"Vocabulario español: {len(src_vocab)} palabras")
print(f"Vocabulario inglés:  {len(tgt_vocab)} palabras")

# --- CODIFICAR FRASES ---
# Convierte una frase de texto a una lista de índices enteros, con padding.
#
# Ejemplo humano:
#   "El gato duerme" con max_len=6
#   -> tokenizar:  ["el", "gato", "duerme"]
#   -> añadir SOS/EOS: [<SOS>, "el", "gato", "duerme", <EOS>]
#   -> convertir a IDs: [1, 47, 83, 291, 2]
#   -> padding hasta max_len: [1, 47, 83, 291, 2, 0]
def encode(sentence, vocab, max_len):
    tokens = sentence.lower().split()
    # Añadir SOS al inicio y EOS al final
    ids = [SOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    # Truncar si la frase es más larga que max_len
    ids = ids[:max_len]
    # Rellenar con PAD si la frase es más corta que max_len
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# --- CLASE DATASET ---
# PyTorch necesita que los datos estén envueltos en una clase Dataset.
# Esta clase implementa __len__ y __getitem__ para que DataLoader pueda
# iterar sobre ella y crear batches automáticamente.
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len):
        print("Codificando frases (esto puede tardar unos segundos)...")
        # Pre-codificar todas las frases al instanciar el dataset
        # para no tener que hacerlo en cada iteración del entrenamiento
        self.src = [encode(s, src_vocab, max_len) for s in src_sentences]
        self.tgt = [encode(s, tgt_vocab, max_len) for s in tgt_sentences]

    def __len__(self):
        # Cuántos pares de frases tiene el dataset
        return len(self.src)

    def __getitem__(self, idx):
        # Devuelve el par (frase española, frase inglesa) en el índice idx
        return self.src[idx], self.tgt[idx]

# Crear el dataset y el DataLoader.
# DataLoader divide el dataset en batches y los baraja en cada época.
#   batch_size=32: 32 pares de frases por paso de entrenamiento
#   shuffle=True: barajar el dataset en cada época para evitar que el modelo
#                 aprenda el orden de los datos
translation_dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_seq_length)
dataloader = DataLoader(translation_dataset, batch_size=32, shuffle=True)
print(f"Dataset listo: {len(translation_dataset)} pares | {len(dataloader)} batches por época")

# =============================================================================
# PASO 6: ENTRENAR EL MODELO
# =============================================================================

# CrossEntropyLoss: función de pérdida para clasificación multiclase.
# En cada posición, el modelo predice una distribución sobre 5000 palabras.
# ignore_index=0: ignora el token de padding al calcular el error,
# porque el padding no es una palabra real y no debería penalizarse.
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Optimizador Adam con los hiperparámetros del paper original "Attention is All You Need":
#   lr=0.0001:        tasa de aprendizaje (cuánto se ajustan los pesos en cada paso)
#   betas=(0.9, 0.98): coeficientes para las medias móviles de gradientes (momentum)
#   eps=1e-9:         término de estabilidad numérica para evitar división por cero
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Poner el modelo en modo entrenamiento: activa dropout y otras capas de regularización.
# (En modo evaluación/inferencia se usa transformer.eval(), que las desactiva)
transformer.train()

# Bucle de entrenamiento: 10 épocas
# Con datos reales, una época = ver los ~130k pares de frases una vez completa.
# Cada época tiene ~4000 batches de 32 frases cada uno.
for epoch in range(10):
    transformer.train()
    epoch_loss = 0  # Acumular el loss de todos los batches para hacer la media

    for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):

        # Mover el batch a la GPU (los datos salen del DataLoader en CPU por defecto)
        src_batch = src_batch.to(device)  # [batch=32, seq_len=100]
        tgt_batch = tgt_batch.to(device)  # [batch=32, seq_len=100]

        # Paso 1: Limpiar los gradientes del paso anterior.
        # Los gradientes se ACUMULAN en PyTorch por defecto, hay que resetearlos.
        optimizer.zero_grad()

        # Paso 2: Forward pass con TEACHER FORCING.
        #
        # Ejemplo humano: la frase objetivo es [<SOS>, "the", "cat", "sleeps", <EOS>, <PAD>, <PAD>]
        #
        # tgt_batch[:, :-1] = entrada al decoder (todo excepto el último token):
        #   [<SOS>, "the", "cat", "sleeps", <EOS>, <PAD>]
        #   "dado este contexto, predice el siguiente token"
        #
        # tgt_batch[:, 1:]  = respuesta esperada (todo excepto el primero):
        #   ["the", "cat", "sleeps", <EOS>, <PAD>, <PAD>]
        #   "esto es lo que deberías haber predicho"
        output = transformer(src_batch, tgt_batch[:, :-1])

        # Paso 3: Calcular el error (loss).
        # Aplanar para que CrossEntropyLoss pueda comparar:
        #   output: [32, 99, 5000] -> [3168, 5000]
        #   target: [32, 99]       -> [3168]
        # Los tokens PAD (índice 0) se ignoran gracias a ignore_index=0
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_batch[:, 1:].contiguous().view(-1)
        )

        # Paso 4: Backward pass + actualizar pesos
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Mostrar progreso cada 100 batches para no saturar la terminal
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    # Al final de cada época, mostrar el loss medio de toda la época
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} completada | Loss medio: {avg_loss:.4f}")
