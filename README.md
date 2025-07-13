# 🧠 Enhanced Suicide Detection API v2.0

API avanzada basada en inteligencia artificial para detectar contenido con riesgo suicida usando BERT con análisis comprehensivo y evaluación de niveles de riesgo.

## ✨ Nuevas Características Principales

### 🔬 Sistema de Análisis Mejorado

- **🎯 Análisis de sentimientos** usando NLTK VADER para evaluación emocional
- **� Detección de 25+ indicadores específicos** de riesgo suicida (palabras clave como "pain", "death", "tired", etc.)
- **⚠️ Evaluación automática de niveles de riesgo** (Bajo, Medio, Alto)
- **👤 Análisis lingüístico detallado** (pronombres primera persona, longitud del texto)
- **💬 Respuestas contextualizadas** basadas en el nivel de riesgo detectado
- **📊 Logging comprehensivo** para seguimiento profesional

### 📊 Respuesta Mejorada del Endpoint `/predict`

```json
{
  "prediction": "suicide",
  "confidence": 0.8542,
  "suicide_probability": 0.8542,
  "non_suicide_probability": 0.1458,
  "processed_text": "texto procesado...",
  "risk_level": "High",
  "analysis": {
    "indicators": ["pain", "death", "tired"],
    "first_person_count": 3,
    "text_length": 125,
    "word_count": 20,
    "sentiment": -0.7231
  },
  "response_message": "Esta conversación muestra indicadores significativos de riesgo suicida. Se recomienda intervención profesional inmediata. Palabras preocupantes detectadas: pain, death, tired."
}
```

### 🎯 Niveles de Riesgo Definidos

- **🟢 Bajo**: Sin indicadores fuertes de riesgo inmediato, monitoreo regular recomendado
- **🟡 Medio**: Algunos patrones preocupantes detectados, evaluación profesional recomendada
- **🔴 Alto**: Indicadores significativos presentes, intervención inmediata recomendada

### 🔍 Factores de Evaluación

El sistema ahora considera múltiples factores:

1. **Probabilidad del modelo BERT** (factor principal)
2. **Presencia de indicadores específicos** (25+ palabras clave monitoreadas)
3. **Análisis de sentimientos** (escala de -1 a +1)
4. **Patrones lingüísticos** (uso de pronombres en primera persona)
5. **Características del texto** (longitud, número de palabras)

## 🚀 Instalación y Configuración

### Requisitos

```bash
pip install -r requirements.txt
```

### Dependencias Principales

- FastAPI >= 0.104.1
- transformers >= 4.37.1
- torch >= 2.1.0
- nltk >= 3.8.1 (para análisis de sentimientos)
- pydantic >= 2.5.0

### Configuración del Entorno

```bash
# Variables de entorno recomendadas
export ENVIRONMENT=production
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

## 🐳 Despliegue

### Opción 1: Docker Compose (Recomendado)

```bash
docker-compose up -d --build
```

### Opción 2: Ejecución Local

```bash
python fastapi_app.py
```

### Opción 3: EasyPanel

1. **Tipo:** Docker
2. **Puerto:** 8000
3. **Variables de entorno:** Ver configuración arriba
4. **Recursos mínimos:** 2 CPU cores, 4GB RAM

## 🧪 Scripts de Prueba Incluidos

### 1. **`test_enhanced_api.py`** - Tests Automatizados

```bash
python test_enhanced_api.py
```

Ejecuta pruebas comprehensivas de todos los endpoints con casos de diferentes niveles de riesgo.

### 2. **`demo_enhanced_system.py`** - Demostración Interactiva

```bash
python demo_enhanced_system.py
```

Muestra ejemplos detallados de cómo funciona el análisis mejorado.

### 3. **`comparison_demo.py`** - Comparación de Sistemas

```bash
python comparison_demo.py
```

Demuestra las diferencias entre el sistema básico y el mejorado.

## 🌐 Endpoints de la API

### Endpoints Principales

| Endpoint    | Método | Descripción                           |
| ----------- | ------ | ------------------------------------- |
| `/`         | GET    | Información básica de la API          |
| `/health`   | GET    | Estado detallado del sistema          |
| `/predict`  | POST   | Predicción comprehensiva con análisis |
| `/api/info` | GET    | Información completa de capacidades   |
| `/docs`     | GET    | Documentación interactiva Swagger     |
| `/redoc`    | GET    | Documentación alternativa             |

### Ejemplo de Uso

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Me siento muy triste y sin esperanza"}'
```

## 📊 Casos de Prueba Recomendados

### 🟢 Riesgo Bajo

- "Tuve un día difícil pero mañana será mejor"
- "Estoy cansado después del trabajo pero emocionado por el fin de semana"

### 🟡 Riesgo Medio

- "Me siento perdido y sin esperanza últimamente"
- "Todo me abruma y no veo mejoras en mi situación"

### 🔴 Riesgo Alto

- "No puedo soportar más este dolor, quiero que todo termine"
- "He estado pensando en hacerme daño, nadie me extrañaría"

## 📈 Mejoras Implementadas

### Comparación con Sistema Anterior

| Característica              | Sistema Anterior | Sistema Mejorado  |
| --------------------------- | ---------------- | ----------------- |
| Predicción básica           | ✅               | ✅                |
| Análisis de sentimientos    | ❌               | ✅                |
| Detección de indicadores    | ❌               | ✅ (25+ palabras) |
| Niveles de riesgo           | ❌               | ✅ (3 niveles)    |
| Respuestas contextualizadas | ❌               | ✅                |
| Análisis lingüístico        | ❌               | ✅                |
| Logging detallado           | Básico           | Comprehensivo     |

### Indicadores Monitoreados

```python
suicide_indicators = [
    'kill', 'die', 'suicide', 'end', 'pain', 'life', 'anymore', 'want', 'hope',
    'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad',
    'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless'
]
```

## ⚠️ Consideraciones Importantes

### 🏥 Uso Profesional

- **Esta herramienta es de apoyo para profesionales capacitados**
- No debe usarse como único método de evaluación de riesgo suicida
- Siempre consulte con profesionales de salud mental para evaluación e intervención

### 🎯 Interpretación de Resultados

- **Alto (🔴)**: Intervención inmediata recomendada
- **Medio (🟡)**: Evaluación profesional recomendada
- **Bajo (🟢)**: Monitoreo continuo recomendado

### 🔒 Privacidad

- Los textos se procesan temporalmente para análisis
- No se almacenan datos personales
- Implementar medidas adicionales de privacidad en producción

## 📞 Recursos de Ayuda

En caso de emergencia:

- **España**: 717 003 717 (Teléfono de la Esperanza)
- **México**: 55 5259 8121 (SAPTEL)
- **Argentina**: 135 (Centro de Asistencia al Suicida)
- **Internacional**: Befrienders Worldwide

## 🚀 Próximas Mejoras

- [ ] Soporte multiidioma
- [ ] Integración con sistemas de alerta
- [ ] Análisis de patrones temporales
- [ ] Dashboard de monitoreo en tiempo real
- [ ] API de notificaciones automáticas

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork del repositorio
2. Crear una rama feature
3. Realizar cambios con tests
4. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

---

**⚡ Desarrollado para proporcionar herramientas avanzadas de apoyo en la prevención del suicidio con tecnología de IA responsable y análisis comprehensivo.**
-v $(pwd)/logs:/app/logs \
 suicide-detection

```

## 📊 Estructura del proyecto

```

suicide-detection/
├── app.py # Aplicación web Flask
├── training.py # Script de entrenamiento del modelo
├── requirements.txt # Dependencias Python
├── Dockerfile # Configuración Docker
├── docker-compose.yml # Configuración Docker Compose
├── start.sh # Script de inicio
├── .dockerignore # Archivos a ignorar en Docker
├── input/
│ └── Suicide_Detection.csv # Dataset
└── models/ # Modelos entrenados (generado)

````

## 🌐 Endpoints de la API

- **`GET /`** - Interfaz web principal
- **`GET /health`** - Health check
- **`POST /predict`** - Predicción de texto
  ```json
  {
    "text": "Texto a analizar"
  }
````

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
