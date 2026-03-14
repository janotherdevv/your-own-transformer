"""
Paso 5: Carga de datos, construcción de vocabularios y Dataset.

Funciones:
  build_vocab(sentences, max_size) → dict[str, int]
  encode(sentence, vocab, max_len) → torch.Tensor
  load_data()                      → (src_vocab, tgt_vocab, DataLoader)

Clase:
  TranslationDataset               → Dataset de PyTorch para pares ES/EN
"""
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset

from config import (
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    src_vocab_size, tgt_vocab_size, max_seq_length, batch_size,
)


def build_vocab(sentences, max_size):
    """
    Construye un vocabulario word→index con las max_size palabras más frecuentes.

    Los primeros 4 índices están reservados para tokens especiales:
      0=<PAD>, 1=<SOS>, 2=<EOS>, 3=<UNK>

    Tokenización: minúsculas + split por espacios (simple, suficiente para aprender).
    En producción se usaría BPE o WordPiece con vocabularios de 30k-50k tokens.
    """
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.lower().split())
    vocab = {"<PAD>": PAD_IDX, "<SOS>": SOS_IDX, "<EOS>": EOS_IDX, "<UNK>": UNK_IDX}
    for word, _ in counter.most_common(max_size - 4):
        vocab[word] = len(vocab)
    return vocab


def encode(sentence, vocab, max_len):
    """
    Convierte una frase de texto a un tensor de índices con padding.

    Ejemplo con max_len=6:
      "El gato duerme"
      → tokens:  ["el", "gato", "duerme"]
      → con SOS/EOS: [<SOS>, "el", "gato", "duerme", <EOS>]
      → IDs:  [1, 47, 83, 291, 2]
      → padding: [1, 47, 83, 291, 2, 0]
    """
    tokens = sentence.lower().split()
    ids = [SOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    ids = ids[:max_len]                             # truncar si es muy larga
    ids += [PAD_IDX] * (max_len - len(ids))        # rellenar si es corta
    return torch.tensor(ids, dtype=torch.long)


class TranslationDataset(Dataset):
    """
    Dataset de pares de frases (español, inglés) codificados como tensores.

    PyTorch necesita esta clase para que DataLoader pueda iterar y crear batches.
    Pre-codifica todas las frases al instanciar para no hacerlo en cada epoch.
    """
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len):
        self.src = self._encode_sentences(src_sentences, src_vocab, max_len, "ES")
        self.tgt = self._encode_sentences(tgt_sentences, tgt_vocab, max_len, "EN")

    @staticmethod
    def _encode_sentences(sentences, vocab, max_len, label):
        """Codifica una lista de frases mostrando progreso cada 10k."""
        total  = len(sentences)
        result = []
        print(f"Codificando {total} frases ({label})...", flush=True)
        for i, s in enumerate(sentences):
            result.append(encode(s, vocab, max_len))
            if (i + 1) % 10000 == 0:
                print(f"  {label}: {i+1}/{total} codificadas", flush=True)
        return result

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def load_data():
    """
    Descarga el dataset opus_books (ES↔EN), construye vocabularios y devuelve
    el DataLoader listo para el bucle de entrenamiento.

    Returns:
        src_vocab  : dict palabra→índice (español)
        tgt_vocab  : dict palabra→índice (inglés)
        dataloader : DataLoader con batches de (src_tensor, tgt_tensor)
    """
    print("Cargando dataset opus_books en-es...", flush=True)
    dataset    = load_dataset("opus_books", "en-es")
    train_data = dataset["train"]

    src_sentences = [item["es"] for item in train_data["translation"]]
    tgt_sentences = [item["en"] for item in train_data["translation"]]
    print(f"Frases cargadas: {len(src_sentences)}", flush=True)

    print("Construyendo vocabulario español...", flush=True)
    src_vocab = build_vocab(src_sentences, src_vocab_size)
    print(f"Vocabulario español: {len(src_vocab)} palabras", flush=True)

    print("Construyendo vocabulario inglés...", flush=True)
    tgt_vocab = build_vocab(tgt_sentences, tgt_vocab_size)
    print(f"Vocabulario inglés:  {len(tgt_vocab)} palabras", flush=True)

    translation_dataset = TranslationDataset(
        src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_seq_length
    )
    dataloader = DataLoader(translation_dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset listo: {len(translation_dataset)} pares | {len(dataloader)} batches/época", flush=True)

    return src_vocab, tgt_vocab, dataloader
