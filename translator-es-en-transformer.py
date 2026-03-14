import torch  # Framework de deep learning: tensores, redes neuronales, autograd
import torch.nn as nn  # Módulo de redes neuronales: capas, funciones de activación
import torch.optim as optim  # Algoritmos de optimización: Adam, SGD, etc.
from torch.utils.data import Dataset, DataLoader  # Dataset y DataLoader para manejar datos
import math  # Funciones matemáticas: sqrt, log, etc.
import copy  # Copias profundas de objetos

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
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        
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

