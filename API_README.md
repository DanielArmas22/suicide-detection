# FastAPI Suicide Detection API

## 🚀 Configuración y Uso

### Instalación

1. **Crear entorno virtual:**

```bash
python -m venv venv
```

2. **Activar entorno virtual:**

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

### Ejecutar la API

#### Opción 1: Script automatizado

```bash
# Windows
start_server.bat

# Linux/Mac
./start_server.sh
```

#### Opción 2: Comando directo

```bash
python fastapi_app.py
```

#### Opción 3: Con uvicorn (desarrollo)

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### 📋 Endpoints Disponibles

| Endpoint    | Método | Descripción                         |
| ----------- | ------ | ----------------------------------- |
| `/`         | GET    | Información básica de la API        |
| `/health`   | GET    | Estado de salud del servidor        |
| `/predict`  | POST   | Realizar predicción de texto        |
| `/api/info` | GET    | Información detallada de la API     |
| `/docs`     | GET    | Documentación interactiva (Swagger) |
| `/redoc`    | GET    | Documentación alternativa           |

### 🧪 Probar la API

#### 1. Usando el script de prueba:

```bash
python test_api.py
```

#### 2. Usando curl:

```bash
# Health check
curl http://localhost:8000/health

# Predicción
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel really sad today"}'
```

#### 3. Usando la documentación interactiva:

Visita `http://localhost:8000/docs` en tu navegador

### 📡 Ejemplo de Uso desde Frontend

#### JavaScript/Fetch

```javascript
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    text: "Tu texto aquí",
  }),
});

const result = await response.json();
console.log(result);
```

#### Python/Requests

```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'text': 'Tu texto aquí'}
)

result = response.json()
print(result)
```

### 📊 Formato de Respuesta

```json
{
    "prediction": "suicide" | "non-suicide",
    "confidence": 0.8542,
    "suicide_probability": 0.1458,
    "non_suicide_probability": 0.8542,
    "processed_text": "texto procesado..."
}
```

### 🔧 Configuración

#### Variables de Entorno

- `PORT`: Puerto del servidor (default: 8000)

#### Rutas de Modelo

La API busca el modelo en este orden:

1. `output/` (donde se guarda después del entrenamiento)
2. `models/suicide_detection_model/`
3. Modelo BERT pre-entrenado (fallback)

### 🐳 Docker

```bash
# Construir imagen
docker build -t suicide-detection-api .

# Ejecutar contenedor
docker run -p 8000:8000 suicide-detection-api
```

### 🔐 Consideraciones de Seguridad

Para producción:

1. Configurar CORS específicamente para tu dominio
2. Añadir autenticación si es necesario
3. Configurar HTTPS
4. Añadir rate limiting
5. Configurar logs apropiados

### ⚠️ Importante

Esta API es para fines educativos y de investigación. Para uso en producción con datos sensibles, se requieren medidas adicionales de seguridad y consideraciones éticas.
