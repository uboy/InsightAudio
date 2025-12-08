# Настройка GPU для InsightAudio

## Проблема
Если в логах видно:
```
torch.cuda.is_available(): False
nvidia-smi не найден в PATH
```

Это означает, что Docker контейнер не имеет доступа к GPU.

## Решение

### 1. Установка NVIDIA Container Toolkit (на хосте)

**Ubuntu/Debian:**
```bash
# Добавляем репозиторий NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Устанавливаем
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Перезапускаем Docker daemon
sudo systemctl restart docker
```

**Windows:**
- Установите [NVIDIA Container Toolkit для Windows](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker-desktop)
- Или используйте WSL2 с установленным NVIDIA драйвером

### 2. Проверка установки

```bash
# Проверяем что nvidia-smi работает на хосте
nvidia-smi

# Проверяем что Docker видит GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. Настройка docker-compose.yml

В `docker-compose.yml` секция `deploy` для GPU уже добавлена для обоих сервисов (`insightaudio` и `worker`).

Если GPU недоступен, убедитесь что:
1. Секция `deploy` не закомментирована
2. Docker Compose версии 1.28+ (поддерживает `deploy.resources`)
3. NVIDIA Container Toolkit установлен

### 4. Запуск с GPU

```bash
# Остановите текущие контейнеры
docker-compose down

# Пересоберите образы (если нужно)
docker-compose build

# Запустите с GPU
docker-compose up -d

# Проверьте логи
docker-compose logs insightaudio | grep -i cuda
docker-compose logs worker | grep -i cuda
```

### 5. Альтернатива: использование Dockerfile.gpu

Если стандартный Dockerfile не работает с GPU:

```bash
# Соберите образ с GPU поддержкой
docker build -f Dockerfile.gpu -t insightaudio:gpu .

# Обновите docker-compose.yml чтобы использовать этот образ
# Измените `build: .` на `image: insightaudio:gpu`
```

### 6. Проверка работы GPU

После запуска в логах должно быть:
```
torch.cuda.is_available(): True
CUDA доступна, используется GPU
```

Если все еще `False`:
1. Проверьте что `nvidia-smi` работает на хосте
2. Проверьте версию Docker Compose: `docker-compose version` (должна быть >= 1.28)
3. Проверьте что NVIDIA Container Toolkit установлен: `dpkg -l | grep nvidia-container-toolkit`
4. Перезапустите Docker daemon: `sudo systemctl restart docker`

### 7. Отключение GPU (если не нужен)

Если GPU не нужен, можно отключить в конфиге:

В `config/default_settings.json`:
```json
{
  "default_config": {
    "USE_CUDA": false
  }
}
```

Или закомментируйте секцию `deploy` в `docker-compose.yml`.

