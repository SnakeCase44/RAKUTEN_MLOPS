#!/bin/bash

echo "🔧 Démarrage du build Docker..."

# Vérifie si un GPU est disponible
if command -v nvidia-smi &> /dev/null && nvidia-smi -L | grep -q "GPU"; then
  echo "✅ GPU détecté. Build avec docker-compose.gpu.yml"
  docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build
else
  echo "⚠️ Aucun GPU détecté. Build standard."
  docker-compose -f docker-compose.yml build
fi
