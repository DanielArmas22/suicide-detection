# EasyPanel Configuration for Suicide Detection App

## Application Settings
- **Name:** suicide-detection
- **Type:** Docker
- **Source:** GitHub/GitLab Repository

## Build Configuration
- **Dockerfile:** `Dockerfile` (in root directory)
- **Build Context:** Repository root
- **Build Args:** None required

## Runtime Configuration
- **Port:** 8000 (internal)
- **Protocol:** HTTP
- **Health Check:** `/health`

## Environment Variables
```
ENVIRONMENT=production
PYTHONUNBUFFERED=1
TOKENIZERS_PARALLELISM=false
PORT=8000
```

## Resource Requirements
### Minimum:
- **CPU:** 1 core
- **RAM:** 2GB
- **Storage:** 5GB

### Recommended:
- **CPU:** 2 cores
- **RAM:** 4GB
- **Storage:** 10GB

### Optimal (for production):
- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 20GB

## Volumes (Optional)
- **Models:** `/app/models` - Para persistir modelos entrenados
- **Logs:** `/app/logs` - Para almacenar logs de la aplicación

## Domain Configuration
- **Custom Domain:** Configure tu dominio personalizado
- **SSL:** Enable automatic SSL certificate
- **Force HTTPS:** Recommended for production

## Monitoring
- **Health Check URL:** `https://tu-dominio.com/health`
- **Logs Location:** Available in EasyPanel dashboard
- **Metrics:** Monitor CPU, RAM, and disk usage

## Security Notes
- La aplicación no almacena datos sensibles
- Todos los análisis son procesados en memoria
- No se requieren configuraciones especiales de firewall
- Habilitar HTTPS es altamente recomendado

## Deployment Steps for EasyPanel

1. **Push to Repository:**
   ```bash
   git add .
   git commit -m "Deploy to EasyPanel"
   git push origin main
   ```

2. **Create App in EasyPanel:**
   - Go to your EasyPanel dashboard
   - Click "Create Application"
   - Select "Docker" as application type
   - Connect your GitHub/GitLab repository

3. **Configure Application:**
   - Set internal port to `8000`
   - Add environment variables listed above
   - Configure resource allocation based on your needs

4. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete (first build may take 10-15 minutes)
   - Monitor deployment logs for any issues

5. **Configure Domain:**
   - Add custom domain in EasyPanel
   - Enable SSL certificate
   - Test the application at your domain

## Troubleshooting

### Build Issues:
- Check that all files are committed and pushed
- Verify Dockerfile syntax
- Monitor build logs in EasyPanel

### Runtime Issues:
- Check application logs in EasyPanel dashboard
- Verify environment variables are set correctly
- Test health check endpoint: `/health`

### Performance Issues:
- Increase CPU/RAM allocation
- Monitor resource usage
- Consider optimizing model loading

## Support Commands

### View Logs:
```bash
# If using docker-compose locally
docker-compose logs -f

# If using docker directly
docker logs -f suicide-detection
```

### Health Check:
```bash
curl https://tu-dominio.com/health
```

### API Test:
```bash
curl -X POST https://tu-dominio.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test message"}'
```
