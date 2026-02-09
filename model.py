import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import re
from collections import defaultdict


# =========================
# TOKENIZER BPE
# =========================
class BPETokenizer:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<END>": 2}
        self.id_para_token = {}
        self.merges = []

    def treinar_com_arquivo(self, caminho_txt):
        with open(caminho_txt, "r", encoding="utf-8") as f:
            texto = f.read().lower()

        palavras = re.findall(r"[\w']+|[.,!?;]", texto)
        palavras_split = [list(p) + ["</w>"] for p in palavras]

        vocab_chars = set()
        for palavra in palavras_split:
            vocab_chars.update(palavra)

        idx = len(self.vocab)
        for char in sorted(vocab_chars):
            if char not in self.vocab:
                self.vocab[char] = idx
                idx += 1

        print(f"🔤 Executando BPE com {self.num_merges} merges...")

        for i in range(self.num_merges):
            pares = defaultdict(int)
            for palavra in palavras_split:
                for j in range(len(palavra) - 1):
                    pares[(palavra[j], palavra[j + 1])] += 1

            if not pares:
                break

            melhor_par = max(pares, key=pares.get)
            self.merges.append(melhor_par)

            novo_token = "".join(melhor_par)
            if novo_token not in self.vocab:
                self.vocab[novo_token] = idx
                idx += 1

            palavras_split = self._aplicar_merge(palavras_split, melhor_par)

            if (i + 1) % 20 == 0:
                print(f"  Merge {i+1}/{self.num_merges}: {melhor_par} → {novo_token}")

        self.id_para_token = {i: t for t, i in self.vocab.items()}
        print(f"✅ Vocabulário final: {len(self.vocab)} tokens")

    def _aplicar_merge(self, palavras_split, par):
        novas_palavras = []
        for palavra in palavras_split:
            nova = []
            i = 0
            while i < len(palavra):
                if i < len(palavra) - 1 and (palavra[i], palavra[i + 1]) == par:
                    nova.append("".join(par))
                    i += 2
                else:
                    nova.append(palavra[i])
                    i += 1
            novas_palavras.append(nova)
        return novas_palavras

    def _tokenizar_palavra(self, palavra):
        tokens = list(palavra) + ["</w>"]
        for par in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == par:
                    tokens = tokens[:i] + ["".join(par)] + tokens[i + 2 :]
                else:
                    i += 1
        return tokens

    def encode(self, texto):
        palavras = re.findall(r"[\w']+|[.,!?;]", texto.lower())
        ids = []
        for palavra in palavras:
            for token in self._tokenizar_palavra(palavra):
                ids.append(self.vocab.get(token, 1))
        return ids

    def decode(self, ids):
        tokens = [self.id_para_token.get(i, "<UNK>") for i in ids]
        return "".join(tokens).replace("</w>", " ").strip()

    def salvar(self, caminho_vocab):
        with open(caminho_vocab, "w", encoding="utf-8") as f:
            json.dump(
                {"vocab": self.vocab, "merges": self.merges, "num_merges": self.num_merges},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def carregar(self, caminho_vocab):
        with open(caminho_vocab, "r", encoding="utf-8") as f:
            dados = json.load(f)
        self.vocab = dados["vocab"]
        self.merges = [tuple(p) for p in dados["merges"]]
        self.num_merges = dados["num_merges"]
        self.id_para_token = {int(i): t for t, i in self.vocab.items()}


# =========================
# MODELO
# =========================
class ModeloTexto(nn.Module):
    def __init__(self, tam_vocab, dim_embed):
        super().__init__()
        self.embedding = nn.Embedding(tam_vocab, dim_embed)
        self.lstm = nn.LSTM(dim_embed, 48, batch_first=True)
        self.saida = nn.Linear(48, tam_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.saida(x[:, -1, :])


# =========================
# CONFIG
# =========================
ARQUIVO_TXT = "treino.txt"
ARQUIVO_MODELO = "ia_texto_bpe.pth"
ARQUIVO_VOCAB = "vocab_bpe.json"
seq_len = 20
NUM_MERGES = 150

if not os.path.exists(ARQUIVO_TXT):
    raise FileNotFoundError("Coloca um arquivo 'treino.txt' aí 👀")

tk = BPETokenizer(NUM_MERGES)

if os.path.exists(ARQUIVO_VOCAB):
    print("📖 Carregando vocabulário BPE...")
    tk.carregar(ARQUIVO_VOCAB)
else:
    print("🏗️ Construindo vocabulário BPE...")
    tk.treinar_com_arquivo(ARQUIVO_TXT)
    tk.salvar(ARQUIVO_VOCAB)

modelo = ModeloTexto(len(tk.vocab), 32)

print("📝 Tokenizando texto...")
with open(ARQUIVO_TXT, encoding="utf-8") as f:
    tokenized_text = tk.encode(f.read())

X_list, y_list = [], []
for i in range(len(tokenized_text) - seq_len):
    X_list.append(tokenized_text[i : i + seq_len])
    y_list.append(tokenized_text[i + seq_len])

X = torch.tensor(X_list)
y = torch.tensor(y_list)

if os.path.exists(ARQUIVO_MODELO):
    print("🧠 Carregando modelo existente...")
    modelo.load_state_dict(torch.load(ARQUIVO_MODELO))
else:
    print("🛠️ Treinando...")
    opt = optim.Adam(modelo.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    for epoca in range(200):
        opt.zero_grad()
        out = modelo(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if (epoca + 1) % 50 == 0:
            print(f"Época {epoca+1} - Erro {loss.item():.4f}")
    torch.save(modelo.state_dict(), ARQUIVO_MODELO)


# =========================
# GERAÇÃO
# =========================
def gerar_texto_longo(inicio, limite=30):
    tokens_gerados = tk.encode(inicio.lower())
    print(f"\n🤖 IA responde: {inicio}", end="")

    for step in range(limite):
        input_tokens = tokens_gerados[-seq_len:]

        while len(input_tokens) < seq_len:
            input_tokens.insert(0, tk.vocab["<PAD>"])

        input_ids = torch.tensor([input_tokens])

        with torch.no_grad():
            previsao = modelo(input_ids)

            #pra ia nao ficar burrinha no português
            temperatura = 1.1
            probs = torch.softmax(previsao[0] / temperatura, dim=0)

            # 🎲 Top-k sampling
            top_k = 10
            valores, indices = torch.topk(probs, top_k)
            valores = valores / valores.sum()
            escolhido = torch.multinomial(valores, 1).item()
            proximo_id = indices[escolhido].item()

            proximo_token = tk.id_para_token.get(proximo_id, "<UNK>")
            
            if len(tokens_gerados) < seq_len + 3 and proximo_token in [".</w>", "!</w>", "?</w>"]:
                continue

            if proximo_token in ["<PAD>", "<END>"]:
                break

            tokens_gerados.append(proximo_id)

            #mostra o token
            if "</w>" in proximo_token:
                print(proximo_token.replace("</w>", ""), end=" ", flush=True)
            else:
                print(proximo_token, end="", flush=True)

            if len(tokens_gerados) > seq_len + 5:
                if "." in proximo_token or "!" in proximo_token or "?" in proximo_token:
                    break

    print("\n")

#chat
modelo.eval()
print("\n✨ sofIA-BPE pronta!\n")

while True:
    seed = input("👤 Você: ")
    if seed.lower() == "sair":
        break
    gerar_texto_longo(seed)