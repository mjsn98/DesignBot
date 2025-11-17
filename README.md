# 🪑 DesignBot – Asistente de Diseño de Muebles con Memoria (Streamlit + LangChain)

DesignBot es un chatbot conversacional diseñado para asistir a usuarios en la creación de **muebles personalizados**.  
Utiliza **memoria contextual**, una **máquina de estados** y herramientas de **Procesamiento del Lenguaje Natural (PLN)** para mantener un flujo conversacional coherente durante todo el diseño.

El proyecto incluye:
- Chat conversacional guiado
- Registro y almacenamiento de pedidos
- Visualización de datos y métricas
- Interfaz completa desarrollada en **Streamlit**

---

## 🚀 Requisitos

### 🔧 Versión de Python
**Python 3.10 obligatorio**  
(El proyecto NO funciona correctamente en versiones superiores debido a dependencias específicas de LangChain y Streamlit).

### 📦 Dependencias
Se recomienda crear un entorno virtual.

---

## 📥 Instalación

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/mjsn98/DesignBot.git
cd DesignBot
```
### 2️⃣ Crear entorno virtual (recomendado)
```bash
pip install -r requirements.txt
```
### 3️⃣ Instalar dependencias
```bash
python3.10 -m venv entorno
source entorno/bin/activate       # Linux / Mac
entorno\Scripts\activate          # Windows
```
### ▶️ Ejecución de la aplicación
```bash
streamlit run app.py
```
### ▶️ Abrir en navegador
```bash
http://localhost:8501
```
