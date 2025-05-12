#!/bin/bash
yarn build
cp -r dist/* ../src/helm/benchmark/static_build/
echo "Frontend rebuilt and copied to backend"