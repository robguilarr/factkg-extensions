#!/bin/bash
set -e

REPO_NAME="factkg-extensions"
PKG_NAME="factkg_ext"

mkdir -p $REPO_NAME
cd $REPO_NAME

mkdir -p scripts
mkdir -p src/$PKG_NAME/original
mkdir -p src/$PKG_NAME/gap1_llm_rag
mkdir -p src/$PKG_NAME/gap2_kg_update
mkdir -p src/$PKG_NAME/gap3_dense_retrieval
mkdir -p src/$PKG_NAME/common
mkdir -p src/$PKG_NAME/constants
mkdir -p src/$PKG_NAME/utilities
mkdir -p conf
mkdir -p tests
mkdir -p data
mkdir -p docs

touch pyproject.toml
touch README.md
touch src/$PKG_NAME/__init__.py
touch src/$PKG_NAME/original/__init__.py
touch src/$PKG_NAME/gap1_llm_rag/__init__.py
touch src/$PKG_NAME/gap2_kg_update/__init__.py
touch src/$PKG_NAME/gap3_dense_retrieval/__init__.py
touch src/$PKG_NAME/common/__init__.py
touch src/$PKG_NAME/constants/__init__.py
touch src/$PKG_NAME/utilities/__init__.py
touch conf/default.yaml
touch conf/gap1.yaml
touch conf/gap2.yaml
touch conf/gap3.yaml
touch data/.gitkeep
touch data/README.md
touch docs/00_original_paper.md
touch docs/01_gap_one.md
touch docs/02_gap_two.md
touch docs/03_gap_three.md

echo "Structure created successfully."
