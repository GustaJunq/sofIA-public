from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import json
import os

# ========================================
# API FLASK PARA MODELO DE IA
# Deploy no Render
# ========================================

app = Flask(__name__)
CORS(app)  # Permite requisições de qualquer origem

# Classe do modelo (mesma do treino)
class ModeloTexto(nn.Module):
    def __init__(self, tam_vocab, dim_embed):
        super(ModeloTexto, self).__init__()
        self.embedding = nn.Embedding(tam_vocab, dim_embed)
        self.lstm = nn.LSTM(dim_embed, 48, batch_first=True)
        self.saida = nn.Linear(48, tam_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.saida(x[:, -1, :])
        return x

# Classe BPE Tokenizer (simplificada pra inferência)
class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.id_para_token = {}
        self.merges = []
        
    def carregar(self, caminho_vocab):
        with open(caminho_vocab, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        self.vocab = dados['vocab']
        self.merges = [tuple(par) for par in dados['merges']]
        self.id_para_token = {int(i): t for t, i in self.vocab.items()}
    
    def _tokenizar_palavra(self, palavra):
        tokens = list(palavra) + ['</w>']
        for par in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == par:
                    tokens = tokens[:i] + [''.join(par)] + tokens[i + 2:]
                else:
                    i += 1
        return tokens
    
    def encode(self, texto):
        import re
        palavras = re.findall(r"[\w']+|[.,!?;]", texto.lower())
        ids = []
        for palavra in palavras:
            tokens = self._tokenizar_palavra(palavra)
            for token in tokens:
                ids.append(self.vocab.get(token, 1))
        return ids
    
    def decode(self, ids):
        tokens = [self.id_para_token.get(id, '<UNK>') for id in ids]
        texto = ''.join(tokens).replace('</w>', ' ').strip()
        return texto

# Carregar modelo e vocabulário
print("🔄 Carregando modelo e vocabulário...")

tk = BPETokenizer()
tk.carregar('vocab_bpe.json')

TAM_VOCAB = len(tk.vocab)
modelo = ModeloTexto(TAM_VOCAB, 24)
modelo.load_state_dict(torch.load('ia_texto_bpe.pth', map_location=torch.device('cpu')))
modelo.eval()

print("✅ Modelo carregado com sucesso!")

SEQ_LEN = 10

def gerar_texto(inicio, limite=30, temperatura=1.5, top_k=10):
    """Gera texto a partir de um início"""
    tokens_gerados = tk.encode(inicio.lower())
    texto_completo = inicio
    
    for _ in range(limite):
        input_tokens = tokens_gerados[-SEQ_LEN:]
        
        while len(input_tokens) < SEQ_LEN:
            input_tokens.insert(0, tk.vocab["<PAD>"])
        
        input_ids = torch.tensor([input_tokens])
        
        with torch.no_grad():
            previsao = modelo(input_ids)
            probs = torch.softmax(previsao[0] / temperatura, dim=0)
            
            valores, indices = torch.topk(probs, top_k)
            valores = valores / valores.sum()
            escolhido = torch.multinomial(valores, 1).item()
            proximo_id = indices[escolhido].item()
            
            proximo_token = tk.id_para_token.get(proximo_id, "<UNK>")
            
            if proximo_token in ["<PAD>", "<END>"]:
                break
            
            tokens_gerados.append(proximo_id)
            
            # Adicionar ao texto
            if '</w>' in proximo_token:
                texto_completo += proximo_token.replace('</w>', '') + " "
            else:
                texto_completo += proximo_token
            
            # Parar em pontuação
            if '.' in proximo_token or '!' in proximo_token or '?' in proximo_token:
                break
    
    return texto_completo.strip()

# ========================================
# ROTAS DA API
# ========================================

@app.route('/', methods=['GET'])
def home():
    """Rota raiz - informações da API"""
    return jsonify({
        'nome': 'sofIA-BPE API',
        'versao': '1.0',
        'status': 'online',
        'endpoints': {
            '/gerar': 'POST - Gera texto a partir de um início',
            '/health': 'GET - Verifica saúde da API'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check para o Render"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/gerar', methods=['POST'])
def gerar():
    """
    Gera texto a partir de um início
    
    Body (JSON):
    {
        "texto": "O gato",
        "limite": 30,  // opcional, padrão 30
        "temperatura": 1.5,  // opcional, padrão 1.5
        "top_k": 10  // opcional, padrão 10
    }
    """
    try:
        dados = request.get_json()
        
        if not dados or 'texto' not in dados:
            return jsonify({'erro': 'Campo "texto" é obrigatório'}), 400
        
        texto_inicial = dados['texto']
        limite = dados.get('limite', 30)
        temperatura = dados.get('temperatura', 1.5)
        top_k_param = dados.get('top_k', 10)
        
        # Validações
        if not isinstance(texto_inicial, str) or len(texto_inicial.strip()) == 0:
            return jsonify({'erro': 'Texto inicial inválido'}), 400
        
        if not (1 <= limite <= 100):
            return jsonify({'erro': 'Limite deve estar entre 1 e 100'}), 400
        
        if not (0.1 <= temperatura <= 5.0):
            return jsonify({'erro': 'Temperatura deve estar entre 0.1 e 5.0'}), 400
        
        if not (1 <= top_k_param <= 50):
            return jsonify({'erro': 'top_k deve estar entre 1 e 50'}), 400
        
        # Gerar texto
        resultado = gerar_texto(
            texto_inicial,
            limite=limite,
            temperatura=temperatura,
            top_k=top_k_param
        )
        
        return jsonify({
            'sucesso': True,
            'texto_gerado': resultado,
            'parametros': {
                'texto_inicial': texto_inicial,
                'limite': limite,
                'temperatura': temperatura,
                'top_k': top_k_param
            }
        })
        
    except Exception as e:
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500

# Porta padrão do Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
