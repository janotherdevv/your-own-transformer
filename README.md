# Transformer ES→EN desde cero

Implementación de un modelo Transformer para traducción español→inglés, construido desde cero con PyTorch siguiendo el paper original ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Proyecto de aprendizaje basado en la guía [Build your own Transformer from scratch using Pytorch](https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84850470dcb) de Arjun Sarkar.

---

## Qué hace este proyecto

Entrena un Transformer que aprende a traducir frases del español al inglés usando ~93.000 pares de frases reales de libros literarios (dataset `opus_books`). Una vez entrenado, el modelo se guarda en disco y puede usarse para traducir sin necesidad de reentrenar.

---

## Arquitectura

El modelo implementa la arquitectura Transformer completa tal como se describe en el paper original:

```
Entrada (español)
      ↓
 Embedding + Positional Encoding
      ↓
 Encoder (6 capas)
   └─ Multi-Head Self-Attention (8 heads)
   └─ Feed-Forward Network (d_ff=2048)
   └─ Add & Norm (residual connections)
      ↓
 Decoder (6 capas)
   └─ Masked Multi-Head Self-Attention
   └─ Cross-Attention (mira al Encoder)
   └─ Feed-Forward Network
   └─ Add & Norm
      ↓
 Capa lineal + Softmax
      ↓
Salida (inglés)
```

### Hiperparámetros

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `d_model` | 512 | Dimensión de los embeddings |
| `num_heads` | 8 | Cabezas de atención paralelas |
| `num_layers` | 6 | Capas en encoder y decoder |
| `d_ff` | 2048 | Dimensión interna de la FFN |
| `max_seq_length` | 100 | Longitud máxima de secuencia |
| `src_vocab_size` | 5000 | Palabras en vocabulario español |
| `tgt_vocab_size` | 5000 | Palabras en vocabulario inglés |
| `dropout` | 0.1 | Tasa de dropout |
| `batch_size` | 32 | Frases por paso de entrenamiento |
| `epochs` | 10 | Épocas de entrenamiento |

---

## Requisitos

```bash
pip install torch datasets
```

- Python 3.10+
- PyTorch (con soporte CUDA recomendado)
- GPU con ~3GB de VRAM (probado en RTX 3060 12GB)

> **Sin GPU:** el modelo es demasiado grande para entrenarse en CPU sin congelar el sistema. Si no tienes GPU, reduce `d_model=128`, `num_layers=2`, `d_ff=256` y `batch_size=8`.

---

## Uso

### Primera ejecución (entrenamiento)

```bash
python3 translator-es-en-transformer.py
```

El script:
1. Descarga el dataset `opus_books` (~93k pares de frases)
2. Construye los vocabularios español e inglés
3. Entrena el modelo durante 10 épocas (~1 hora en RTX 3060)
4. Guarda el modelo y vocabularios en disco
5. Lanza el modo interactivo para traducir

### Ejecuciones siguientes (sin reentrenar)

El script detecta automáticamente los archivos guardados y los carga directamente, saltándose el entrenamiento.

### Archivos generados

| Archivo | Tamaño aprox. | Contenido |
|---------|---------------|-----------|
| `transformer_model.pt` | ~200MB | Pesos del modelo entrenado |
| `src_vocab.json` | ~80KB | Diccionario español → índice |
| `tgt_vocab.json` | ~80KB | Diccionario inglés → índice |

---

## Modo interactivo

Al terminar el entrenamiento (o al cargar un modelo ya entrenado), el script muestra ejemplos automáticos y abre un modo interactivo:

```
ES > el gato duerme en la cama
EN > the cat sleeps on the bed

ES > buenos días
EN > good morning

ES > salir
```

Escribe `salir` para terminar.

---

## Estructura del código

```
translator-es-en-transformer.py
│
├── MultiHeadAttention       # Mecanismo de atención multi-cabeza
├── PositionWiseFeedForward  # Red feed-forward posición a posición
├── PositionalEncoding       # Codificación posicional con senos/cosenos
├── EncoderLayer             # Una capa del encoder
├── DecoderLayer             # Una capa del decoder
├── Transformer              # Modelo completo (encoder + decoder)
│
├── PASO 5: Preparar datos
│   ├── Carga opus_books
│   ├── Construye vocabularios
│   ├── TranslationDataset
│   └── DataLoader
│
├── PASO 6: Entrenar o cargar
│   ├── Si existe transformer_model.pt → carga
│   └── Si no existe → entrena y guarda
│
└── PASO 7: Traducir
    ├── translate()          # Greedy decoding
    ├── Ejemplos automáticos
    └── Modo interactivo
```

---

## Limitaciones

- **Vocabulario pequeño (5000 palabras):** palabras poco frecuentes se traducen como `<UNK>`. En producción se usaría BPE o WordPiece con vocabularios de 30k-50k tokens.
- **Greedy decoding:** en cada paso escoge el token más probable. Beam search produciría mejores traducciones.
- **Sin puntuación:** el tokenizador es simple (split por espacios), la puntuación pegada a las palabras puede no reconocerse.
- **Dominio literario:** el modelo ha aprendido de libros, funciona mejor con frases de ese estilo.

---

## Referencias

- Vaswani et al. (2017) — [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Arjun Sarkar — [Build your own Transformer from scratch using Pytorch](https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84850470dcb)
- Dataset: [opus_books en-es](https://huggingface.co/datasets/Helsinki-NLP/opus_books) via HuggingFace
