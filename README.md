# 🧠 Detección de Suicidio - AI Model

Una aplicación web basada en inteligencia artificial para la detección de contenido con riesgo suicida en textos, desarrollada con BERT y Flask.

## 🚀 Despliegue en EasyPanel

### Prerrequisitos
- Cuenta en EasyPanel
- Acceso a una VPS
- Git instalado

### Pasos para el despliegue:

#### 1. Preparar el repositorio
```bash
# Clonar o subir tu proyecto a GitHub/GitLab
git init
git add .
git commit -m "Initial commit - Suicide Detection App"
git remote add origin <tu-repositorio>
git push -u origin main
```

#### 2. Configurar en EasyPanel

1. **Accede a tu panel de EasyPanel**
2. **Crea una nueva aplicación:**
   - Tipo: **Docker**
   - Fuente: **GitHub/GitLab**
   - Repositorio: Tu repositorio
   - Branch: `main`

3. **Configuración de la aplicación:**
   - **Puerto interno:** `8000`
   - **Puerto externo:** `80` o `443` (para HTTPS)
   - **Variables de entorno:**
     ```
     ENVIRONMENT=production
     PYTHONUNBUFFERED=1
     TOKENIZERS_PARALLELISM=false
     ```

4. **Recursos recomendados:**
   - **CPU:** 2 cores mínimo
   - **RAM:** 4GB mínimo (8GB recomendado)
   - **Almacenamiento:** 10GB mínimo

#### 3. Configuración de dominio
- Asigna un dominio personalizado en EasyPanel
- Habilita SSL automático

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
