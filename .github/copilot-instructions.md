# 🤖 Transformer Translator (ES → EN)

**Objetivo:** Implementar un Transformer desde cero en PyTorch para traducción de español a inglés, como proyecto educativo.

**Referencia:** Basado en "Build your own Transformer from scratch using PyTorch" de Arjun Sarkar (Medium/TDS).

---

## 📋 Arquitectura del Proyecto

### Componentes Principales

1. **MultiHeadAttention**: Mecanismo de atención multi-cabeza
   - Permite capturar múltiples tipos de relaciones entre palabras
   - Implementa Scaled Dot-Product Attention
   - Métodos: `split_heads()`, `combine_heads()`, `scaled_dot_product_attention()`

2. **PositionWiseFeedForward**: Red feed-forward por posición
   - Dos capas lineales con ReLU entre ellas
   - Estructura: Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)

3. **PositionalEncoding**: Codificación posicional
   - Inyecta información del orden de palabras mediante funciones senoidales
   - Usa seno para dimensiones pares, coseno para impares
   - Permite al modelo generalizar a secuencias más largas que las vistas en entrenamiento

4. **EncoderLayer**: Capa del codificador
   - Combina atención + feed-forward con residual connections y layer normalization

### Próximos Pasos
- [ ] `DecoderLayer`: Capa del decodificador (atención cruzada + enmascaramiento)
- [ ] `Transformer`: Modelo completo (Encoder + Decoder stacks)
- [ ] Dataset y DataLoader para pares español-inglés
- [ ] Loop de entrenamiento
- [ ] Evaluación con BLEU score

---

## 🛠️ Convenciones de Código

### Dimensiones e Hiperparámetros

- **d_model**: Dimensionalidad interna del modelo (típicamente 512)
- **num_heads**: Número de heads de atención (típicamente 8)
- **d_k**: d_model / num_heads, dimensionalidad por head
- **d_ff**: Dimensionalidad de capas feed-forward (típicamente 4 * d_model = 2048)
- **max_seq_length**: Longitud máxima de secuencia que el modelo puede procesar

### Formas de Tensores

- **[batch, seq_len, d_model]**: Formato estándar para secuencias
- **[batch, num_heads, seq_len, d_k]**: Formato para heads cuantificados
- Todas las operaciones respetan este formato

### Comentarios

- Se usan comentarios extensos en español explicando **el qué** y **el por qué**
- Cada clase tiene docstring con ejemplo conceptual
- Se explican fórmulas matemáticas (ej: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V)

---

## 📚 Conceptos Clave

### Attention Mechanism
```
softmax(QK^T / sqrt(d_k)) * V
     ↓           ↓
Probabilidades Valores
 de atención  ponderados
```

### Multi-Head Attention
Permite múltiples subespacios de representación en paralelo. Cada head:
- Aprende diferentes patrones (sintácticos vs semánticos)
- Se ejecuta independientemente
- Se concatenan resultados y se proyectan

### Residual Connections + Layer Norm
```
LayerNorm(x + f(x))
```
Facilita el entrenamiento de redes profundas y estabiliza el aprendizaje.

---

## 🎯 Pasos para Completar el Proyecto

1. **Terminar arquitectura base**
   - Implementar `DecoderLayer` con cross-attention
   - Ensamblar `Transformer` completo

2. **Preparar datos**
   - Tokenizador para español e inglés
   - Corpus de entrenamiento (ej: TabéMT, Opus)

3. **Entrenamiento**
   - DataLoader con padding dinámico
   - Loss function (Cross-Entropy)
   - Optimizer (Adam)

4. **Evaluación**
   - BLEU score
   - Ejemplos de traducción

---

## 🔗 Referencias

- **Artículo original**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Guía del proyecto**: PDF en directorio
- **Documentación PyTorch**: torch.nn, torch.optim

