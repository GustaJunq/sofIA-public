# sofIA-BPE API

API REST para geração de texto usando modelo LSTM com tokenização BPE.

## 🚀 Deploy no Render

### Arquivos necessários:
- `app.py` - Servidor Flask
- `requirements.txt` - Dependências
- `ia_texto_bpe.pth` - Modelo treinado
- `vocab_bpe.json` - Vocabulário BPE

### Passos para deploy:

1. **Criar repositório no GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin SEU_REPOSITORIO
   git push -u origin main
   ```

2. **Configurar no Render**
   - Acesse render.com
   - New → Web Service
   - Conecte seu repositório GitHub
   - Configurações:
     - **Name:** sofia-bpe-api (ou qualquer nome)
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app`
     - **Plan:** Free

3. **Aguardar deploy** (~5-10 min)

## 📡 Endpoints

### `GET /`
Informações da API

### `GET /health`
Health check

### `POST /gerar`
Gera texto a partir de um início

**Body (JSON):**
```json
{
  "texto": "O gato",
  "limite": 30,
  "temperatura": 1.5,
  "top_k": 10
}
```

**Resposta:**
```json
{
  "sucesso": true,
  "texto_gerado": "O gato dorme no sofá...",
  "parametros": {
    "texto_inicial": "O gato",
    "limite": 30,
    "temperatura": 1.5,
    "top_k": 10
  }
}
```

## 🧪 Testar localmente

```bash
python app.py
```

Acesse: `http://localhost:5000`

## 📝 Exemplo de uso (cURL)

```bash
curl -X POST https://sua-api.onrender.com/gerar \
  -H "Content-Type: application/json" \
  -d '{"texto": "O gato", "limite": 30}'
```

## 📝 Exemplo de uso (JavaScript)

```javascript
fetch('https://sua-api.onrender.com/gerar', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    texto: 'O gato',
    limite: 30,
    temperatura: 1.5,
    top_k: 10
  })
})
.then(response => response.json())
.then(data => console.log(data.texto_gerado));
```

## ⚙️ Parâmetros

- **texto** (obrigatório): Texto inicial para gerar
- **limite** (opcional, padrão 30): Número máximo de tokens a gerar
- **temperatura** (opcional, padrão 1.5): Controla criatividade (0.1-5.0)
- **top_k** (opcional, padrão 10): Número de opções consideradas (1-50)

## 📦 Tamanho dos arquivos

⚠️ **IMPORTANTE:** O plano grátis do Render tem limite de:
- 512MB RAM
- 1GB disco

Se seus arquivos forem muito grandes, considere:
- Hospedar modelo no Hugging Face
- Usar Render pago
- Alternativas: Railway, Fly.io
- 
