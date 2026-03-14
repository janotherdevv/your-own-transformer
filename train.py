"""
Paso 6: Bucle de entrenamiento y guardado/carga de checkpoints.

Funciones:
  train(transformer, dataloader)        → entrena el modelo in-place
  save_checkpoint(transformer, vocabs)  → guarda modelo + vocabularios en disco
  load_checkpoint(transformer)          → carga modelo + vocabularios desde disco
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    tgt_vocab_size, epochs, learning_rate,
    MODEL_PATH, SRC_VOCAB_PATH, TGT_VOCAB_PATH,
    device,
)


def train(transformer, dataloader):
    """
    Entrena el transformer con teacher forcing durante `epochs` épocas.

    Teacher forcing: en cada paso del decoder, se le da el token correcto anterior
    como contexto (no el que él mismo predijo). Acelera y estabiliza el entrenamiento.

    Ejemplo: la frase objetivo es [<SOS>, "the", "cat", "sleeps", <EOS>]
      - Entrada al decoder:   [<SOS>, "the", "cat", "sleeps"]  (tgt[:, :-1])
      - Respuesta esperada:   ["the", "cat", "sleeps", <EOS>]  (tgt[:, 1:])
    """
    # CrossEntropyLoss: mide cuán equivocada está la distribución de probabilidad
    # predicha respecto a la palabra correcta. ignore_index=0 ignora el padding.
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Adam con los hiperparámetros del paper original "Attention is All You Need"
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate,
                           betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(epochs):
        transformer.train()   # activar dropout
        epoch_loss = 0

        for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()  # limpiar gradientes acumulados del paso anterior

            # Forward pass
            output = transformer(src_batch, tgt_batch[:, :-1])

            # Aplanar para CrossEntropyLoss:
            #   output: [batch, seq-1, vocab] → [batch*(seq-1), vocab]
            #   target: [batch, seq-1]        → [batch*(seq-1)]
            loss = criterion(
                output.contiguous().view(-1, tgt_vocab_size),
                tgt_batch[:, 1:].contiguous().view(-1)
            )

            loss.backward()    # calcular gradientes
            optimizer.step()   # actualizar pesos

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} "
                      f"| Loss: {loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completada | Loss medio: {avg_loss:.4f}", flush=True)


def save_checkpoint(transformer, src_vocab, tgt_vocab):
    """
    Guarda los pesos del modelo y los vocabularios en disco.

    torch.save guarda el state_dict: solo los pesos (números aprendidos),
    no la arquitectura. Para cargarlos hay que crear primero un modelo vacío.

    Ejemplo humano: guardar las respuestas que memorizó el estudiante,
    no el cerebro completo. Para usarlas necesitas un cerebro vacío (la clase).
    """
    print(f"Guardando modelo en '{MODEL_PATH}'...", flush=True)
    torch.save(transformer.state_dict(), MODEL_PATH)

    with open(SRC_VOCAB_PATH, "w") as f:
        json.dump(src_vocab, f)
    with open(TGT_VOCAB_PATH, "w") as f:
        json.dump(tgt_vocab, f)

    print(f"Vocabularios guardados en '{SRC_VOCAB_PATH}' y '{TGT_VOCAB_PATH}'.", flush=True)
    print("La próxima vez se cargará el modelo directamente.", flush=True)


def load_checkpoint(transformer):
    """
    Carga los pesos del modelo y los vocabularios desde disco.

    map_location=device: si el modelo se guardó en GPU pero ahora solo hay CPU
    (o viceversa), PyTorch lo recoloca automáticamente en el device correcto.

    Returns:
        src_vocab : dict palabra→índice (español)
        tgt_vocab : dict palabra→índice (inglés)
    """
    print(f"Modelo encontrado en '{MODEL_PATH}'. Cargando pesos...", flush=True)

    with open(SRC_VOCAB_PATH, "r") as f:
        src_vocab = json.load(f)
    with open(TGT_VOCAB_PATH, "r") as f:
        tgt_vocab = json.load(f)

    transformer.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Modelo cargado. Listo para traducir sin reentrenar.", flush=True)

    return src_vocab, tgt_vocab
