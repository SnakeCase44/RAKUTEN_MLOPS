#!/bin/bash

if command -v nvidia-smi &> /dev/null && nvidia-smi -L | grep -q "GPU"; then
  echo "✅ GPU détecté. Lancement avec support GPU..."
  docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
else
  echo "⚠️  Aucun GPU détecté. Lancement standard..."
  docker-compose up
fi
