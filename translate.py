"""
Paso 7: Inferencia — traducción de frases español → inglés.

Funciones:
  translate(sentence, transformer, src_vocab, idx_to_word) → str
  interactive_mode(transformer, src_vocab, idx_to_word)    → None (bucle interactivo)
"""
import torch

from config import PAD_IDX, SOS_IDX, EOS_IDX, max_seq_length, device
from data import encode


def translate(sentence, transformer, src_vocab, idx_to_word):
    """
    Traduce una frase español → inglés usando greedy decoding.

    Greedy decoding: en cada paso escoge el token con mayor probabilidad.
    No es el mejor método (beam search produce mejores resultados) pero es el más simple.

    Proceso:
      1. Codificar la frase española a índices
      2. Pasarla por el encoder
      3. Decoder genera tokens uno a uno hasta <EOS> o max_seq_length
      4. Convertir índices a palabras con idx_to_word

    Args:
      sentence    : frase en español (str)
      transformer : modelo entrenado
      src_vocab   : dict palabra→índice (español)
      idx_to_word : dict índice→palabra (inglés) — se construye UNA vez fuera

    Ejemplo humano: el modelo dicta la traducción en voz alta:
      "the... cat... sleeps... <fin>"
    """
    transformer.eval()  # desactivar dropout para inferencia

    # Codificar frase española: [max_seq_length] → [1, max_seq_length] (añadir dim batch)
    src_tensor = encode(sentence, src_vocab, max_seq_length).unsqueeze(0).to(device)

    # Empezar con solo <SOS> y añadir tokens uno a uno
    generated_tokens = [SOS_IDX]

    with torch.no_grad():  # sin gradientes en inferencia: ahorra memoria y tiempo
        for _ in range(max_seq_length):
            tgt_tensor = torch.tensor(generated_tokens).unsqueeze(0).to(device)

            # output: [1, len(generated_tokens), vocab_size]
            output = transformer(src_tensor, tgt_tensor)

            # Coger el token más probable en la ÚLTIMA posición
            next_token = output[0, -1, :].argmax().item()
            generated_tokens.append(next_token)

            if next_token == EOS_IDX:
                break

    # Convertir índices a palabras (saltando <SOS>, <EOS> y <PAD>)
    words = [
        idx_to_word.get(idx, "<UNK>")
        for idx in generated_tokens[1:]
        if idx not in (EOS_IDX, PAD_IDX)
    ]
    return " ".join(words)


def interactive_mode(transformer, src_vocab, idx_to_word):
    """
    Bucle interactivo: el usuario escribe frases en español y el modelo las traduce.
    Escribe 'salir' para terminar.
    """
    print("--- Modo interactivo (escribe 'salir' para terminar) ---", flush=True)
    while True:
        sentence = input("ES > ").strip()
        if sentence.lower() == "salir":
            break
        if sentence:
            print(f"EN > {translate(sentence, transformer, src_vocab, idx_to_word)}\n")
