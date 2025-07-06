# 🧠 Suicide Detection API

API basada en inteligencia artificial para detectar contenido con riesgo suicida usando BERT.

## 🚀 Despliegue en EasyPanel

### Configuración rápida:
1. **Tipo:** Docker
2. **Repositorio:** Este repositorio de GitHub
3. **Puerto:** 8000
4. **Variables de entorno:**
   ```
   ENVIRONMENT=production
   PYTHONUNBUFFERED=1
   TOKENIZERS_PARALLELISM=false
   ```
5. **Recursos mínimos:** 2 CPU cores, 4GB RAM

### Endpoints principales:
- `GET /health` - Estado del servicio
- `POST /predict` - Predicción de riesgo suicida
- `GET /docs` - Documentación interactiva

## 🐳 Despliegue manual con Docker

### Opción 1: Usando Docker Compose (Recomendado)
```bash
# Clonar el repositorio
git clone <tu-repositorio>
cd suicide-detection

# Construir y ejecutar
docker-compose up -d --build
```

### Opción 2: Docker directo
```bash
# Construir la imagen
docker build -t suicide-detection .

# Ejecutar el contenedor
docker run -d \
  --name suicide-detection \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  suicide-detection
```

## 📊 Estructura del proyecto

```
suicide-detection/
├── app.py                 # Aplicación web Flask
├── training.py           # Script de entrenamiento del modelo
├── requirements.txt      # Dependencias Python
├── Dockerfile           # Configuración Docker
├── docker-compose.yml   # Configuración Docker Compose
├── start.sh            # Script de inicio
├── .dockerignore       # Archivos a ignorar en Docker
├── input/
│   └── Suicide_Detection.csv  # Dataset
└── models/             # Modelos entrenados (generado)
```

## 🌐 Endpoints de la API

- **`GET /`** - Interfaz web principal
- **`GET /health`** - Health check
- **`POST /predict`** - Predicción de texto
  ```json
  {
    "text": "Texto a analizar"
  }
  ```
- **`GET /api/info`** - Información de la API

## 🔧 Configuración avanzada

### Variables de entorno
- `ENVIRONMENT`: `development` o `production`
- `PORT`: Puerto de la aplicación (default: 8000)
- `PYTHONUNBUFFERED`: Para logs en tiempo real
- `TOKENIZERS_PARALLELISM`: Desactiva paralelismo de tokenizers

### Monitoreo
La aplicación incluye:
- Health checks automáticos
- Logging detallado
- Métricas de predicción

## ⚠️ Consideraciones importantes

1. **Uso responsable**: Esta herramienta es solo para fines educativos y de investigación
2. **Privacidad**: No se almacenan los textos analizados
3. **Rendimiento**: El primer análisis puede ser más lento debido a la carga del modelo
4. **Recursos**: Requiere recursos computacionales significativos

## 🛠️ Desarrollo local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo (opcional)
python training.py

# Ejecutar la aplicación
python app.py
```

## 📝 Logs y debugging

Los logs se almacenan en:
- Contenedor: `/app/logs/`
- Local: `./logs/`

Para ver logs en tiempo real:
```bash
docker logs -f suicide-detection
```

## 🔒 Seguridad

- La aplicación no almacena datos sensibles
- Implementa rate limiting básico
- Usa HTTPS en producción (configurado por EasyPanel)

## 📞 Soporte

Para problemas o dudas:
1. Revisa los logs de la aplicación
2. Verifica el health check endpoint
3. Consulta la documentación de EasyPanel

---

**⚠️ Advertencia**: Esta herramienta es solo para fines educativos y de investigación. Si tú o alguien que conoces está en crisis, busca ayuda profesional inmediatamente.
