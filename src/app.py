import streamlit as st
from neurona import GenericNeuron

# Configuración de la página
st.set_page_config(page_title="Simulador de Neurona Genérica")

# Estilo personalizado
style = """
<style>
   .appview-container .main .block-container{
       max-width: 90%;
   }
</style>
"""

st.markdown(style, unsafe_allow_html=True)

# Imagen y título
st.image("img/neuron.jpg", use_column_width="always")
st.title('Simulador de Neurona Genérica')

# Selector de número de entradas/pesos
num_entries = st.slider("Selecciona el número de entradas/pesos para la neurona", 1, 10)

# Pesos
st.subheader("Pesos")
weights = []
columns_weights = st.columns(num_entries)

for i in range(num_entries):
    with columns_weights[i]:
        st.markdown(f"Peso w<sub>{i}</sub>", unsafe_allow_html=True)
        weights.append(st.number_input(f"w{i}", key=f"w{i}", label_visibility="collapsed"))

st.write(f"w = {weights}")

# Entradas
st.subheader("Entradas")
inputs = []
columns_inputs = st.columns(num_entries)

for i in range(num_entries):
    with columns_inputs[i]:
        st.markdown(f"Entrada x<sub>{i}</sub>", unsafe_allow_html=True)
        inputs.append(st.number_input(f"x{i}", key=f"x{i}", label_visibility="collapsed"))

st.write(f"x = {inputs}")

# Sesgo y función de activación
colBias, colFunc = st.columns(2)

with colBias:
    st.subheader("Sesgo")
    bias = st.number_input("Introduce el valor del sesgo")

with colFunc:
    st.subheader("Función de Activación")
    function = st.selectbox(
        'Elige la función de activación',
        ('Sigmoide', 'ReLU', 'Tanh', 'Escalón Binario')
    )

    if function == 'Sigmoide':
        function = 'sigmoid'
    elif function == 'Escalón Binario':
        function = 'binary_step'
    else:
        function = function.lower()

# Botón para calcular la salida
if st.button("Calcular salida"):
    # Instancia de la neurona y cálculo de la salida
    neuron = GenericNeuron(weights, bias, function)
    output = neuron.predict(inputs)
    st.write(f"La salida de la neurona es {output}")

st.write("")
