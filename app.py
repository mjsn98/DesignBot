import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Importar las clases de tu notebook
from langchain.llms.base import BaseLanguageModel
from langchain.schema import BaseMessage, AIMessage, LLMResult, Generation, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Any, Optional, List
from enum import Enum
import asyncio
from langchain_core.prompt_values import PromptValue

# 1. Estados del diseño
class EstadoDiseno(Enum):
    INICIO = 0
    ESPERANDO_TIPO_MUEBLE = 1
    ESPERANDO_MATERIAL = 2
    ESPERANDO_COLOR = 3
    ESPERANDO_DIMENSIONES = 4
    CONFIRMACION = 5
    ESPERANDO_INFO_CONTACTO = 6
    PEDIDO_GUARDADO = 7

# 2. Gestor del Diseño
class DisenoManager:
    def __init__(self):
        self.reiniciar_diseno()

    def reiniciar_diseno(self):
        self.estado = EstadoDiseno.INICIO
        self.tipo_mueble = None
        self.material = None
        self.color = None
        self.dimensiones = None
        self.info_contacto = None
        self.pedido_listo_para_guardar = False

    def establecer_tipo_mueble(self, tipo):
        self.tipo_mueble = tipo
        self.estado = EstadoDiseno.ESPERANDO_MATERIAL
        return True

    def establecer_material(self, material):
        if not self.tipo_mueble:
            return False
        self.material = material
        self.estado = EstadoDiseno.ESPERANDO_COLOR
        return True

    def establecer_color(self, color):
        if not self.material:
            return False
        self.color = color
        self.estado = EstadoDiseno.ESPERANDO_DIMENSIONES
        return True

    def establecer_dimensiones(self, dim):
        if not self.color:
            return False
        self.dimensiones = dim
        self.estado = EstadoDiseno.CONFIRMACION
        return True

    def establecer_info_contacto(self, contacto):
        self.info_contacto = contacto
        self.pedido_listo_para_guardar = True
        self.estado = EstadoDiseno.PEDIDO_GUARDADO

    def obtener_resumen_actual(self):
        return f"""
🪑 Tipo de mueble: {self.tipo_mueble}
🔩 Material: {self.material}
🎨 Color: {self.color}
📏 Dimensiones: {self.dimensiones}
"""

    def obtener_datos_pedido(self, nombre_cliente):
        return {
            'nombre': nombre_cliente or "Cliente",
            'tipo_mueble': self.tipo_mueble,
            'material': self.material,
            'color': self.color,
            'dimensiones': self.dimensiones,
            'contacto': self.info_contacto,
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'estado': 'Confirmado'
        }

# 3. LLM Personalizado - VERSIÓN SIMPLIFICADA
class DesignBotLLM(BaseLanguageModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diseno_manager = DisenoManager()
        self.nombre_cliente = None

    @property
    def _llm_type(self) -> str:
        return "designbot-compatible"

    def _call(self, prompt: str, stop=None, **kwargs):
        # Estrategia múltiple para extraer el input del usuario
        user_input = ""
        
        # Dividir en líneas y buscar el último input del usuario
        lines = prompt.strip().split('\n')
        
        # Buscar desde el final hacia el principio
        for i in range(len(lines)-1, -1, -1):
            line = lines[i].strip()
            
            # Estrategia 1: Buscar "Usuario:"
            if line.startswith('Usuario:'):
                user_input = line[8:].strip()
                break
            
            # Estrategia 2: Buscar "Human:"
            elif line.startswith('Human:'):
                user_input = line[6:].strip()
                break
            
            # Estrategia 3: Buscar "input:"
            elif line.startswith('input:'):
                user_input = line[6:].strip()
                break
            
            # Estrategia 4: Si estamos en la última línea y no tiene prefijo conocido
            elif i == len(lines)-1 and not any(line.startswith(x) for x in ['AI:', 'DesignBot:', 'System:']):
                user_input = line
                break
        
        # Si no encontramos nada, usar la última línea
        if not user_input:
            user_input = lines[-1].strip()
        
        # Limpiar posibles restos de prefijos
        user_input = user_input.replace('Human:', '').replace('Usuario:', '').replace('input:', '').strip()
        
        user_input_lower = user_input.lower()
        
        # --- LÓGICA PRINCIPAL MEJORADA ---
        
        # SALUDO INICIAL
        if self.diseno_manager.estado == EstadoDiseno.INICIO:
            if any(s in user_input_lower for s in ["hola", "hello", "buenos días", "buenas"]):
                if any(indicador in user_input_lower for indicador in ["me llamo", "soy", "nombre es"]):
                    # Extraer nombre
                    if "me llamo" in user_input_lower:
                        nombre = user_input_lower.split("me llamo")[1].strip()
                    elif "soy" in user_input_lower:
                        nombre = user_input_lower.split("soy")[1].strip()
                    elif "nombre es" in user_input_lower:
                        nombre = user_input_lower.split("nombre es")[1].strip()
                    else:
                        nombre = user_input_lower
                    
                    # Tomar solo la primera palabra como nombre
                    self.nombre_cliente = nombre.split()[0].title()
                    self.diseno_manager.estado = EstadoDiseno.ESPERANDO_TIPO_MUEBLE
                    return f"¡Hola {self.nombre_cliente}! ¿Qué tipo de mueble te gustaría diseñar? Tenemos: silla, mesa, sofá, estantería."
                else:
                    return "¡Hola! Soy DesignBot, tu asistente para diseño de muebles. ¿Cómo te llamas?"
            
            # Si es solo un nombre
            elif len(user_input.split()) == 1 and user_input.replace('.', '').replace(',', '').isalpha():
                self.nombre_cliente = user_input.title()
                self.diseno_manager.estado = EstadoDiseno.ESPERANDO_TIPO_MUEBLE
                return f"¡Hola {self.nombre_cliente}! ¿Qué tipo de mueble te gustaría diseñar? Tenemos: silla, mesa, sofá, estantería."

        # TIPO DE MUEBLE
        if self.diseno_manager.estado == EstadoDiseno.ESPERANDO_TIPO_MUEBLE:
            tipos = {
                "silla": "SILLA",
                "mesa": "MESA", 
                "sofá": "SOFÁ",
                "sofa": "SOFÁ",
                "estantería": "ESTANTERÍA",
                "estanteria": "ESTANTERÍA",
                "armario": "ARMARIO",
                "cama": "CAMA"
            }
            for k, v in tipos.items():
                if k in user_input_lower:
                    self.diseno_manager.establecer_tipo_mueble(v)
                    return "¡Excelente elección! ¿Qué material prefieres: madera noble, MDF, metal o vidrio?"
            
            # Si no reconoce el tipo de mueble
            return "No reconozco ese tipo de mueble. ¿Podrías elegir entre: silla, mesa, sofá o estantería?"

        # MATERIAL
        if self.diseno_manager.estado == EstadoDiseno.ESPERANDO_MATERIAL:
            materiales = {
                "madera noble": "MADERA_NOBLE",
                "mdf": "MADERA_MDF", 
                "metal": "METAL",
                "vidrio": "VIDRIO",
                "plástico": "PLÁSTICO",
                "plastico": "PLÁSTICO"
            }
            for k, v in materiales.items():
                if k in user_input_lower:
                    self.diseno_manager.establecer_material(v)
                    return "Perfecto. Ahora elige el color: natural, blanco, negro o madera oscura."
            
            return "No reconozco ese material. ¿Podrías elegir entre: madera noble, MDF, metal o vidrio?"

        # COLOR
        if self.diseno_manager.estado == EstadoDiseno.ESPERANDO_COLOR:
            colores = {
                "natural": "NATURAL",
                "blanco": "BLANCO", 
                "negro": "NEGRO",
                "madera oscura": "MADERA_OSCURA",
                "rojo": "ROJO",
                "azul": "AZUL",
                "verde": "VERDE"
            }
            for k, v in colores.items():
                if k in user_input_lower:
                    self.diseno_manager.establecer_color(v)
                    return "Muy bien. Por último, selecciona las dimensiones: pequeño, estándar o grande."
            
            return "No reconozco ese color. ¿Podrías elegir entre: natural, blanco, negro o madera oscura?"

        # DIMENSIONES
        if self.diseno_manager.estado == EstadoDiseno.ESPERANDO_DIMENSIONES:
            dims = {
                "pequeño": "PEQUEÑO",
                "pequeno": "PEQUEÑO",
                "estándar": "ESTÁNDAR", 
                "estandar": "ESTÁNDAR",
                "grande": "GRANDE",
                "xl": "EXTRA_GRANDE",
                "extra grande": "EXTRA_GRANDE"
            }
            for k, v in dims.items():
                if k in user_input_lower:
                    self.diseno_manager.establecer_dimensiones(v)
                    resumen = self.diseno_manager.obtener_resumen_actual()
                    return f"""
📝 **RESUMEN DE TU DISEÑO**
{resumen}
¿Confirmás este diseño? (responde 'sí' o 'no')
"""
            
            return "No reconozco esas dimensiones. ¿Podrías elegir entre: pequeño, estándar o grande?"

        # CONFIRMACIÓN
        if self.diseno_manager.estado == EstadoDiseno.CONFIRMACION:
            if any(s in user_input_lower for s in ["sí", "si", "confirmo", "ok", "vale", "de acuerdo", "perfecto"]):
                self.diseno_manager.estado = EstadoDiseno.ESPERANDO_INFO_CONTACTO
                return "¡Perfecto! Para finalizar, ¿cuál es tu email o teléfono para contactarte?"
            elif any(s in user_input_lower for s in ["no", "cancelar", "reiniciar", "empezar de nuevo"]):
                self.diseno_manager.reiniciar_diseno()
                return "Entendido. Vamos a empezar de nuevo. ¿Qué mueble te gustaría diseñar?"
            else:
                return "¿Confirmás este diseño? Responde 'sí' para confirmar o 'no' para empezar de nuevo."

        # INFORMACIÓN DE CONTACTO
        if self.diseno_manager.estado == EstadoDiseno.ESPERANDO_INFO_CONTACTO:
            if user_input.strip():  # Solo si hay contenido
                self.diseno_manager.establecer_info_contacto(user_input)
                resumen = self.diseno_manager.obtener_resumen_actual()
                nombre_cliente_temp = self.nombre_cliente or "Cliente"
                
                # Marcar que hay un pedido listo para guardar
                self.diseno_manager.pedido_listo_para_guardar = True

                return f"""
🎉 ¡DISEÑO CONFIRMADO! 🎉

{resumen}
📧 Contactaremos a {nombre_cliente_temp} en: {user_input}

¿Te gustaría diseñar otro mueble? ¡Solo dime qué tipo te gustaría!
"""
            else:
                return "Por favor, proporciona tu email o teléfono para contactarte."

        # RESPUESTA POR DEFECTO - más específica según el estado
        estado_actual = self.diseno_manager.estado
        if estado_actual == EstadoDiseno.INICIO:
            return "¡Hola! Soy DesignBot. Para comenzar, ¿cómo te llamas?"
        elif estado_actual == EstadoDiseno.ESPERANDO_TIPO_MUEBLE:
            return "Por favor, elige un tipo de mueble: silla, mesa, sofá o estantería."
        else:
            return "No entendí tu respuesta. ¿Podrías repetirlo o ser más específico?"

    # Métodos de LangChain (mantener igual)
    def _generate(self, prompts, stop=None, **kwargs):
        generations = []
        for prompt in prompts:
            resp = self._call(prompt)
            generations.append([Generation(text=resp)])
        return LLMResult(generations=generations)

    async def _acall(self, prompt: str, stop=None, **kwargs):
        await asyncio.sleep(0.01)
        return self._call(prompt)

    def invoke(self, input: str, stop=None, **kwargs):
        return self._call(input)

    def predict(self, text: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return self.invoke(text, stop=stop, **kwargs)

    async def apredict(self, text: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return await self._acall(text, stop=stop, **kwargs)

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = await self._acall(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def generate_prompt(self, prompts: List[PromptValue], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return self._generate(prompt_strings, stop=stop, **kwargs)

    async def agenerate_prompt(self, prompts: List[PromptValue], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return await self._agenerate(prompt_strings, stop=stop, **kwargs)

    def predict_messages(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> BaseMessage:
        full_prompt = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                full_prompt += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                full_prompt += f"AI: {msg.content}\n"
            elif isinstance(msg, SystemMessage):
                full_prompt += f"System: {msg.content}\n"
            else:
                full_prompt += f"{msg.content}\n"

        response_text = self.predict(full_prompt, stop=stop, **kwargs)
        return AIMessage(content=response_text)

    async def apredict_messages(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> BaseMessage:
        full_prompt = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                full_prompt += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                full_prompt += f"AI: {msg.content}\n"
            elif isinstance(msg, SystemMessage):
                full_prompt += f"System: {msg.content}\n"
            else:
                full_prompt += f"{msg.content}\n"

        response_text = await self.apredict(full_prompt, stop=stop, **kwargs)
        return AIMessage(content=response_text)

# 4. Cadena Final MODIFICADA para guardar pedidos
class DesignBotChain:
    def __init__(self, llm, prompt, memory):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory

    def predict(self, user_input):
        history = self.memory.load_memory_variables({})["chat_history"]
        prompt_final = self.prompt.format(chat_history=history, input=user_input)
        respuesta = self.llm.invoke(prompt_final)
        
        # VERIFICAR Y GUARDAR PEDIDO después de obtener la respuesta
        if (hasattr(self.llm.diseno_manager, 'pedido_listo_para_guardar') and 
            self.llm.diseno_manager.pedido_listo_para_guardar):
            
            pedido = self.llm.diseno_manager.obtener_datos_pedido(self.llm.nombre_cliente)
            
            # Guardar en session_state
            if 'pedidos' not in st.session_state:
                st.session_state.pedidos = []
            st.session_state.pedidos.append(pedido)
            
            # Reiniciar el flag
            self.llm.diseno_manager.pedido_listo_para_guardar = False
            self.llm.diseno_manager.reiniciar_diseno()
        
        self.memory.save_context({"input": user_input}, {"output": respuesta})
        return respuesta

# 5. Funciones para mostrar pedidos
def mostrar_pedidos():
    """Muestra la lista de pedidos realizados"""
    st.header("📋 Pedidos Realizados")
    
    if 'pedidos' not in st.session_state or len(st.session_state.pedidos) == 0:
        st.info("📝 Aún no hay pedidos realizados. ¡Comienza a diseñar muebles en el chat!")
        return
    
    pedidos = st.session_state.pedidos
    
    # Mostrar métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pedidos", len(pedidos))
    with col2:
        pedidos_hoy = len([p for p in pedidos if p['fecha'].startswith(datetime.now().strftime("%Y-%m-%d"))])
        st.metric("Pedidos Hoy", pedidos_hoy)
    with col3:
        tipos_count = {}
        for p in pedidos:
            tipos_count[p['tipo_mueble']] = tipos_count.get(p['tipo_mueble'], 0) + 1
        tipo_popular = max(tipos_count, key=tipos_count.get) if tipos_count else "N/A"
        st.metric("Mueble Más Popular", tipo_popular)
    with col4:
        st.metric("Clientes Únicos", len(set(p['nombre'] for p in pedidos)))
    
    st.markdown("---")
    
    # Crear DataFrame para mostrar los pedidos
    df_data = []
    for i, pedido in enumerate(pedidos):
        df_data.append({
            'ID': i + 1,
            'Cliente': pedido['nombre'],
            'Mueble': pedido['tipo_mueble'],
            'Material': pedido['material'],
            'Color': pedido['color'],
            'Dimensiones': pedido['dimensiones'],
            'Contacto': pedido['contacto'],
            'Fecha': pedido['fecha'],
            'Estado': pedido['estado']
        })
    
    df = pd.DataFrame(df_data)
    
    # Opciones de filtrado
    col1, col2, col3 = st.columns(3)
    with col1:
        filtro_mueble = st.selectbox(
            "Filtrar por tipo de mueble",
            ["Todos"] + list(df['Mueble'].unique())
        )
    with col2:
        filtro_estado = st.selectbox(
            "Filtrar por estado",
            ["Todos"] + list(df['Estado'].unique())
        )
    with col3:
        filtro_cliente = st.selectbox(
            "Filtrar por cliente",
            ["Todos"] + list(df['Cliente'].unique())
        )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    if filtro_mueble != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Mueble'] == filtro_mueble]
    if filtro_estado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Estado'] == filtro_estado]
    if filtro_cliente != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Cliente'] == filtro_cliente]
    
    # Mostrar tabla de pedidos
    st.subheader(f"📦 Lista de Pedidos ({len(df_filtrado)} pedidos)")
    
    # Estilizar la tabla
    st.dataframe(
        df_filtrado,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Cliente": st.column_config.TextColumn("Cliente", width="medium"),
            "Mueble": st.column_config.TextColumn("Mueble", width="medium"),
            "Material": st.column_config.TextColumn("Material", width="medium"),
            "Color": st.column_config.TextColumn("Color", width="medium"),
            "Dimensiones": st.column_config.TextColumn("Dimensiones", width="medium"),
            "Contacto": st.column_config.TextColumn("Contacto", width="large"),
            "Fecha": st.column_config.DatetimeColumn("Fecha", width="medium"),
            "Estado": st.column_config.SelectboxColumn(
                "Estado",
                width="medium",
                options=["Confirmado", "En producción", "Completado", "Cancelado"]
            )
        }
    )
    
    # Opción para exportar datos
    if st.button("📤 Exportar Pedidos a CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"pedidos_designbot_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Resumen visual simple
    st.markdown("---")
    st.subheader("📊 Resumen Visual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de tipos de muebles
        tipo_counts = df['Mueble'].value_counts()
        fig_tipos = px.bar(
            x=tipo_counts.index,
            y=tipo_counts.values,
            title="Pedidos por Tipo de Mueble",
            labels={'x': 'Tipo de Mueble', 'y': 'Cantidad'},
            color=tipo_counts.values,
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_tipos, use_container_width=True)
    
    with col2:
        # Gráfico de materiales
        material_counts = df['Material'].value_counts()
        fig_materiales = px.pie(
            values=material_counts.values,
            names=material_counts.index,
            title="Uso de Materiales",
            hole=0.4
        )
        st.plotly_chart(fig_materiales, use_container_width=True)

# 6. Configuración de Streamlit
def main():
    st.set_page_config(
        page_title="DesignBot - Asistente de Diseño",
        page_icon="🪑",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar pedidos en session_state si no existen
    if 'pedidos' not in st.session_state:
        st.session_state.pedidos = []
    
    # Sidebar
    with st.sidebar:
        st.title("🪑 DesignBot")
        st.markdown("---")
        
        st.subheader("Configuración")
        if st.button("🧹 Reiniciar Conversación"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            if "chatbot" in st.session_state:
                st.session_state.chatbot.llm.diseno_manager.reiniciar_diseno()
                st.session_state.chatbot.llm.nombre_cliente = None
            st.rerun()
        
        # Botón para limpiar pedidos (útil para testing)
        if st.button("🗑️ Limpiar Pedidos"):
            st.session_state.pedidos = []
            st.rerun()
        
        st.markdown("---")
        st.subheader("Navegación")
        if st.button("📋 Ver Pedidos"):
            st.session_state.mostrar_pedidos = True
        if st.button("💬 Volver al Chat"):
            st.session_state.mostrar_pedidos = False
        
        st.markdown("---")
        st.subheader("Estadísticas Rápidas")
        total_pedidos = len(st.session_state.pedidos)
        pedidos_hoy = len([p for p in st.session_state.pedidos 
                          if p['fecha'].startswith(datetime.now().strftime("%Y-%m-%d"))])
        
        st.metric("Total Pedidos", total_pedidos)
        st.metric("Pedidos Hoy", pedidos_hoy)
        
        st.markdown("---")
        st.markdown("### Acerca de")
        st.info("""
        DesignBot te ayuda a diseñar muebles personalizados paso a paso.
        
        **Características:**
        • Diseño guiado de muebles
        • Historial de pedidos
        • Gestión de clientes
        """)
    
    # Inicializar session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=False, memory_key='chat_history')
    
    if "chatbot" not in st.session_state:
        # Prompt del sistema SIMPLIFICADO
        system_prompt = "Eres DesignBot, un asistente especializado en diseño de muebles personalizados."

        template = """Historial de conversación:
{chat_history}

Usuario: {input}
DesignBot:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=template
        )
        
        llm = DesignBotLLM()
        st.session_state.chatbot = DesignBotChain(llm=llm, prompt=prompt, memory=st.session_state.memory)
    
    if "mostrar_pedidos" not in st.session_state:
        st.session_state.mostrar_pedidos = False
    
    # Título principal
    st.title("🤖 DesignBot - Asistente de Diseño de Muebles")
    
    # Mostrar pedidos o chat
    if st.session_state.mostrar_pedidos:
        mostrar_pedidos()
    else:
        # Área del chat
        st.subheader("💬 Conversación con DesignBot")
        
        # Mostrar historial de mensajes
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input del usuario
        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            # Agregar mensaje del usuario
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Obtener respuesta del bot
            with st.chat_message("assistant"):
                with st.spinner("DesignBot está pensando..."):
                    response = st.session_state.chatbot.predict(prompt)
                st.markdown(response)
            
            # Agregar respuesta del bot al historial
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()