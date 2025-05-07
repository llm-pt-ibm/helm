#!/bin/bash

# Vai para o diretório do frontend
cd helm-frontend || exit
echo "Executando build_sync.sh..."
bash build_sync.sh

# Volta para o diretório helm
cd .. || exit
echo "Instalando pacote com pip..."
pip install .

# Executa o servidor
echo "Compilando..."
helm-summarize --suite my-suite

# Executa o servidor
echo "Iniciando helm-server com suite 'my-suite'..."
helm-server --suite my-suite
