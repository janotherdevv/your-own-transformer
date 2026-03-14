"""
Pasos 1-4: Arquitectura completa del Transformer.

Clases (en orden de dependencia):
  MultiHeadAttention        → mecanismo de atención multi-cabeza
  PositionWiseFeedForward   → red feed-forward posición a posición
  PositionalEncoding        → codificación posicional con senos/cosenos
  EncoderLayer              → una capa del encoder (self-attn + FFN)
  DecoderLayer              → una capa del decoder (self-attn + cross-attn + FFN)
  Transformer               → modelo completo (encoder + decoder)

Este archivo no tiene dependencias internas del proyecto — solo PyTorch y math.
Se puede importar y reutilizar en cualquier otro proyecto.
"""
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Capa de atención multi-cabeza (Multi-Head Attention).

    Implementa el mecanismo de atención self-attention con múltiples heads paralelos.
    Permite al modelo capturar diferentes tipos de relaciones entre palabras simultáneamente.
    Cada head aprende patrones diferentes (sintácticos, semánticos, etc.).

    Ejemplo: En la frase "El gato persigue al ratón", un head podría enfocarse en la
    relación sujeto-verbo ("gato" -> "persigue"), mientras otro head podría enfocarse
    en la relación verbo-objeto ("persigue" -> "ratón").
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # dimensión de cada head

        # Matrices de proyección lineal para Q, K, V y la salida
        self.W_q = nn.Linear(d_model, d_model)  # qué busco
        self.W_k = nn.Linear(d_model, d_model)  # qué ofrezco
        self.W_v = nn.Linear(d_model, d_model)  # qué información tengo
        self.W_o = nn.Linear(d_model, d_model)  # proyección de vuelta tras concatenar heads

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Fórmula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

        La escala por sqrt(d_k) evita que los productos punto sean tan grandes
        que el softmax se sature y los gradientes desaparezcan.
        """
        # Scores de atención: similitud entre cada query y cada key
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Aplicar máscara: posiciones inválidas → -1e9 → softmax devuelve ~0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        """[batch, seq_len, d_model] → [batch, num_heads, seq_len, d_k]"""
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """[batch, num_heads, seq_len, d_k] → [batch, seq_len, d_model]"""
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output))


class PositionWiseFeedForward(nn.Module):
    """
    Red feed-forward posición a posición (Position-wise Feed-Forward Network).

    Dos capas lineales con ReLU entre ellas, aplicadas independientemente
    a cada posición de la secuencia. Añade no-linealidad al modelo.

    Ejemplo: como una máquina expendedora — la máquina es la misma para todas
    las posiciones, pero cada posición (botón) produce una salida distinta.
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1  = nn.Linear(d_model, d_ff)   # expande: d_model → d_ff
        self.fc2  = nn.Linear(d_ff, d_model)   # comprime: d_ff → d_model
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """
    Codificación posicional con funciones senoidales.

    El Transformer no tiene recurrencia ni convolución, por lo que necesita
    inyectar el orden de las palabras. Usa senos/cosenos de distintas frecuencias:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Ejemplo: "El gato persigue al ratón" ≠ "El ratón persigue al gato".
    La codificación posicional ayuda al modelo a distinguir el orden.
    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe       = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # posiciones pares
        pe[:, 1::2] = torch.cos(position * div_term)  # posiciones impares

        # register_buffer: guarda 'pe' con el modelo pero no como parámetro entrenable
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_length, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    """
    Una capa del Encoder del Transformer.

    Dos subcapas:
      1. Self-Attention: la secuencia mira a toda ella misma para entender relaciones
      2. Feed-Forward: procesa cada posición independientemente

    Cada subcapa usa residual connection (Add) y Layer Normalization.

    Ejemplo: como un lector que subraya el texto (atención) y luego procesa
    la información subrayada (feed-forward), pasando notas al siguiente lector.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn    = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention: Q = K = V = x
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))   # residual + norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))     # residual + norm
        return x


class DecoderLayer(nn.Module):
    """
    Una capa del Decoder del Transformer.

    Tres subcapas:
      1. Masked Self-Attention: solo mira posiciones ya generadas (no el futuro)
      2. Cross-Attention: mira la salida del Encoder (el texto original)
      3. Feed-Forward: procesa cada posición independientemente

    Ejemplo: para traducir la segunda palabra, solo puedes ver la primera
    palabra que ya tradujiste (masked self-attn), pero SIEMPRE miras el texto
    original completo para saber qué estás traduciendo (cross-attn).
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn    = MultiHeadAttention(d_model, num_heads)  # masked
        self.cross_attn   = MultiHeadAttention(d_model, num_heads)  # sobre encoder
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)
        self.norm3        = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. Masked self-attention (tgt_mask evita ver el futuro)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-attention: Q = decoder, K = V = encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 3. Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    """
    Modelo Transformer completo para traducción seq2seq.

    Flujo:
      Embedding(src) + PosEnc → Encoder × N → enc_output
      Embedding(tgt) + PosEnc → Decoder × N (con enc_output) → Linear → logits

    Ejemplo:
      Input (español): "Hola mundo"
      → El Encoder "entiende" cada palabra en contexto
      → El Decoder genera "Hello world" token a token
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc      = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Genera las máscaras para padding y para evitar que el decoder vea el futuro.

        src_mask: marca dónde hay tokens reales (1) vs padding (0)
        tgt_mask: src_mask + máscara causal (triangular) para no ver posiciones futuras

        Ejemplo de máscara causal para "Hello world":
          [[1, 0, 0],   ← "Hello" solo ve "Hello"
           [1, 1, 0],   ← "world" ve "Hello" y "world"
           [1, 1, 1]]   ← <EOS> ve todo
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length  = tgt.size(1)
        # .to(tgt.device): torch.ones() crea en CPU por defecto; si la GPU está activa,
        # tgt_mask ya está en GPU y el & fallaría con "tensors on different devices"
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)
