# Cambios Realizados en la API

## Resumen de Mejoras Implementadas

### 1. **Corrección en la Tokenización**

- **Antes**: Usaba `tokenizer()` con parámetros básicos
- **Ahora**: Usa `tokenizer.encode_plus()` con los mismos parámetros del entrenamiento:
  - `add_special_tokens=True`
  - `return_token_type_ids=False`
  - `padding='max_length'`
  - `return_attention_mask=True`
  - `return_tensors='pt'`

### 2. **Mejores Nombres de Clases**

- **Antes**: "suicide" / "non-suicide"
- **Ahora**: "suicidal" / "non-suicidal" (coincide con el script de entrenamiento)

### 3. **Predicción Mejorada**

- Separación correcta de `input_ids` y `attention_mask`
- Uso explícito de `model.eval()` antes de la predicción
- Cálculo de probabilidades usando `softmax` de la misma manera que en el entrenamiento

### 4. **Niveles de Riesgo**

- **Nuevo**: Agregado campo `risk_level` con valores:
  - "low": probabilidad suicida ≤ 0.5
  - "medium": probabilidad suicida > 0.5 y ≤ 0.8
  - "high": probabilidad suicida > 0.8

### 5. **Análisis Detallado**

- **Nuevo endpoint**: `/predict/detailed`
- Incluye análisis de indicadores de riesgo:
  - Palabras clave relacionadas con suicidio
  - Conteo de pronombres en primera persona
  - Estadísticas del texto (longitud, conteo de palabras)

### 6. **Indicadores de Riesgo**

Agregadas las mismas palabras clave del script de referencia:

```python
['kill', 'die', 'suicide', 'end', 'pain', 'life', 'anymore', 'want', 'hope',
 'help', 'death', 'dead', 'hate', 'tired', 'pills', 'hurt', 'alone', 'sad',
 'depression', 'anxiety', 'lost', 'cut', 'empty', 'worthless']
```

## Nuevos Endpoints

### `/predict` (mejorado)

```json
{
  "prediction": "suicidal",
  "confidence": 0.8234,
  "suicidal_probability": 0.8234,
  "non_suicidal_probability": 0.1766,
  "processed_text": "processed text here",
  "risk_level": "high"
}
```

### `/predict/detailed` (nuevo)

```json
{
  "prediction": "suicidal",
  "confidence": 0.8234,
  "suicidal_probability": 0.8234,
  "non_suicidal_probability": 0.1766,
  "processed_text": "processed text here",
  "risk_level": "high",
  "analysis": {
    "indicators_found": ["sad", "hopeless", "end"],
    "indicator_count": 3,
    "first_person_count": 5,
    "text_length": 45,
    "word_count": 8
  }
}
```

## Archivos Modificados

1. **`fastapi_app.py`**: Función de predicción mejorada y nuevo endpoint
2. **`test_api.py`**: Actualización de tests para nuevos campos
3. **`validate_model.py`**: Nuevo script de validación

## Cómo Probar

1. **Ejecutar el servidor**:

   ```bash
   python fastapi_app.py
   ```

2. **Validar modelo**:

   ```bash
   python validate_model.py
   ```

3. **Probar API**:

   ```bash
   python test_api.py
   ```

4. **Ver documentación**:
   - http://localhost:8000/docs
   - http://localhost:8000/redoc

## Beneficios de los Cambios

- ✅ **Compatibilidad**: Tokenización idéntica al entrenamiento
- ✅ **Precisión**: Mejor manejo de `attention_mask`
- ✅ **Análisis**: Información detallada sobre indicadores de riesgo
- ✅ **Consistencia**: Nombres coherentes con el modelo entrenado
- ✅ **Validación**: Script para verificar funcionamiento del modelo
