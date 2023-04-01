# %% [markdown]
# # Práctica 4
#
# **IDS330L, Feb-Abr 2023**
#
# Incorporar Redes Neuronales Convolucionales a la clasificación de Imágenes con Tensorflow
#
# En esta ocasión veremos un poco sobre cómo funcionan, cómo se implementan y cómo podemos usar redes neuronales que integran capas convolucionales para un caso de Clasificación. EL orden que seguiremos es:
#
# 1. Ver cómo funcionan las capas convolucionales
# 2. Crear un modelo de red con Keras y Tensorflow
# 3. Entrenar modelos
# 4. Probar la red
# 5. Jugar con la red
#
# Las preguntas del ejercicio están distribuidas a lo largo del cuaderno. Están enumeradas en **negrita** del 1 al 8.
#
# Para garantizar la validez de su trabajo, guarde sus resultados en un PDF y cárguelo al aula virtual **además** del cuaderno.
#

# %% [markdown]
# # Preparación

# %%
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten
import numpy as np
np.random.seed(20)

# %% [markdown]
# En una capa convolucional, lo que antes hemos definido como una neurona que funciona como un producto punto ahora se implementa de forma similar a una convolución bi-dimensional (un filtro) que consume (o procesa) una porción de la imagen a la vez. La neurona aplica el filtrado a diferentes porciones de la imagen, generando así una interpretación que guarda información por separado sobre las diferentes porciones de la imagen por las que pasa. Si la imagen tiene profundidad (capas), como es el caso de las tres capas de una imagen RGB, los filtros incorporan estas capas en sus cálculos.
#
# Para cada "porción" de la imagen, iniciando en una posición $(i,j)$, se lleva a cabo una combinación lineal de los datos con la que se alimenta una función de activación, funcionando de forma similar al producto punto que vimos en en perceptrón de la red neuronal conectada. En este caso, si se tienen capas (profundidad), la información de las capas también incluye en el procesamiento. Específicamente, para un porción de una imagen, en una posición $(i,j)$, cada filtro computa
#
# $$ z_{i,j} = \sigma \left( b + \large\sum_{l=0}^{k_H-1} \large\sum_{m=0}^{k_W-1} \large\sum_{n=0}^{D-1} w_{l,m,n} \cdot x_{i+l,j+m,n} \right)$$
#
# Donde $\sigma$ es la función de activación, $(k_H \times k_W \times D)$ son las dimensiones del filtro (ancho, alto, profundidad/capas), $w$ las ganancias, y $b$ el sesgo de cada filtro.
#
# Una de las dimensiones del $w$ surge de la cantidad de capas, $D$, en los datos, así que el tamaño del filtro se define realmente por $k_W$ y $k_H$, y la profundidad dependerá de las capas. Generalmente, aunque no es regla, se definen los filtros de forma cuadrada $k \times k$, para facilitar algunos detalles. La matriz de dimensiones $k \times k$ (que tiene $k^2$ coeficientes) representa el **núcleo** (_kernel_, en inglés) del filtro, que tendrá $Dk^2$ coeficientes y un ($1$) parámetro de sesgo (inglés, _bias_).
#
#
# En una capa convolucional de una red neuronal, se aplican $N$ filtros, cada uno con $Dk^2 + 1$ parámetros, para procesar la entrada (la imagen). Haciendo esto que tengamos que trabajar con $N \times (Dk^2 +1 )$ parámetros por cada capa convolucional. En un sistema con capacidad de aprendizaje, esto permite reducir grandemente la cantidad de parámetros que entrenar.
#
# Abajo tenemos un método que genera una capa convolutiva de 2 dimensiones para procesar una entrada con `nb_filters` filtros de núcleo $3 \times 3$. Nótese que se especifica nó usar un sesgo (`use_bias=False`), reduciendo aún más los parámetros. Similarmente, se puede notar que usa una función de activación tipor $ReLU$. Con este método podremos facilitar la creación de capas en nuestro modelo.

# %%


def conv3x3(input_x, nb_filters):
    """
    Wrapper around convolution layer
    Inputs:
    input_x: input layer / tensor
    nb_filter: Number of filters for convolution
    """
    return Conv2D(nb_filters, kernel_size=(3, 3), use_bias=False,
                  activation='relu', padding="same")(input_x)

# %% [markdown]
# En un sistema inteligente, su capacidad de aprendizaje surge de poder ajustar los parámetros en los filtros de acuerdo a la experiencia. Esto hace que cada filtro pueda ajustarse a "reconocer" algo en particular de la entrada. Por ejemplo, si $N=32$, se pueden reconocer 32 características distinas en una imagen, como líneas en una orientación específica, una variación de color, etc. Vale notar que estas son características de bajo nivel (al nivel de pixeles) en la imagen. Si existieran más capas convolucionales encima de esta, se pudieran reconocer rasgos más concretos como formas de cara, o el contorno de un objeto, en base a combinaciones de las 32 características de la primera capa. Esto se ilustra en la Figura 3-1 del libro (pg. 76), que muestra cómo cada capa convolucional resume **toda** la entrada, en términos de la cantidad de neurnas que tenga.
#
# Cada filtro se aplica a una porción $k \times k$ de la imagen a la vez, empezando en el punto $(i,j)$ y produciendo un valor $z_{i,j}$, y avanzando hasta completar la imagen. Según las necesidades, se puede definir qué tan "rápido" avanza el filtro por la imagen, tanto a lo largo como lo ancho. Por ejemplo, cada 1 pixel, o cada 2, etc.; que generalmente se aplica igual para ambas direcciones (horizonal o vertical). Esto implica que mientras más grande el "paso" (inglés, _stride_), menos datos se producen como resultado. Por tanto, el tamaño del núcleo y los pasos que toma influyen en la cantidad de datos finales. Esto se ilustra en la Figura 3-2 del libro (pg. 78), donde se observa también que la imagen se **rellena** con ceros (inglés, _padding_) para permitir que el filtro pueda dar todos los pasos que necesita.
#
# Si la cantidad de pasos que puede dar un filtro por la imagen en cada dirección se representa por $H_o \times W_o $, tomando que los $N$ filtros se aplican a toda la imagen, podemos conlucir que cada capa convolucional produce $H_o \times W_o \times N$ datos, que resumen la interpretación de cada filtro (o neurona) de la entrada.

# %% [markdown]
# # Aplicación
#
# Para crear una red neuronal convolucional (inglés, CNN), se debe aplicar cierto criterio sobre cómo se transforma la información a medida que pasa por sus diferentes capas. Existen diferentes arquitecturas para diferentes fines. En nuestro caso, vamos a probar una arquitectura que permita realizar clasificación de objetos en imágenes. Para esto pudiera no funcionarnos la Red Neuronal Conectada (densas) que usamos antes, ya que necesitamos utilizar características espaciales relacionadas con la forma de los elementos.
#
# Para formar una CNN con imágenes, muchas veces se combinan capas convolucionales para extraer características de las imágenes, con capas densas que ayudan a clasificar las imágenes (clasificando igual que en los ejercicios anteriores). Sin embargo, a veces es necesario manipular la cantidad de información entre capas.
#
# Uno de los métodos para manipular la cantidad de información entre capas es insertar entre ellas una capa de _pooling_, que usa filtros **no entrenables** que se usan para resumir los valores en su entrada tomando lo que "ven" por una ventana y extrayendo algún resumen de los datos. Por ejemplo, una capa de _pooling_ que extraiga el valor máximo de una ventana $2\times 2$ ($k=2$), que se avance cada 2 valores en cada dirección ($s=2$), terminaría reduciendo el ancho y largo de los datos a la mitad (diviendo entre 2). Lo importante de esto es que se reducen la cantidad de computaciones en las próximas capas. En el caso de ejemplo, se reduce por $2\times 2=4$ la cantidad de datos con los que debe trabajar la próxima capa.
#
# Como las capas convolucionales y de _pooling_ alteran la cantidad de información, se debe tomar en cuenta cómo evoluciona la cantidad de información entre cada capa. Es por esto que se estudian las arquitecturas de las CNN al tratar de implementarlas para una aplicación específica.

# %% [markdown]
# # Ejercicio
#
# Estaremos aplicando una CNN para clasificar las imágenes del dataset [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist), que tienen propiedades similares al dataset de números de MNIST que usamos para los ejercicios anteriores (por ej.: las imágenes son $28\times 28$ pixeles). Estas, sin embargo, contienen formas de prendas de vestir, también clasificadas en 10 categorías, por lo que aprovecharemos la capacidad de las CNN para procesar información espacial para poder identificar diferentes rasgos en una misma imágen para tener un mejor resultado de estimación.


# %%
# setup parameters
num_classes = 10  # target number of classes
img_h, img_w = 28, 28  # input dimensions

# %% [markdown]
# Tal como habíamos hecho antes, prepararemos el conjunto de datos para tener la forma requerida por nuestra red, en este caso $(28 \times 28 \times 1)$, y a normalizar los datos entre 0 y 1.

# %%


def get_dataset():
    """
    Return processed and reshaped dataset for training
    In this cases Fashion-mnist dataset.
    """
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # test and train datasets
    print("Nb Train:", x_train.shape[0], "Nb test:", x_test.shape[0])
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    in_shape = (img_h, img_w, 1)
    # normalize inputs
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    # convert to one hot vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_dataset()

# %% [markdown]
#
# # La red
#
# Abajo podemos ver cómo se crea el modelo de nuestra CNN, que combina capas convolucionales y de _pooling_ para procesar las imágenes, y usa dos capas densas al final para clasificar las imágenes en una cantidad de categorías específicas.

# %%


def create_model(img_h=28, img_w=28, num_classes=10):
    """
    Creates a CNN model for training.
    Inputs:
    img_h: input image height
    img_w: input image width
    Returns:
    Model structure
    """
    inputs = Input(shape=(img_h, img_w, 1))
    x = conv3x3(inputs, 32)
    x = conv3x3(x, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv3x3(x, 64)
    x = conv3x3(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv3x3(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=preds)
    print(model.summary())
    return model


# %% [markdown]
# En este caso, el método `create_model` hace el trabajo por nosotros, pero otra forma de crear una estructura similar es usando la clase Sequential de Keras, en algo como :
#
# ```python
# model = Sequential() # `Sequential` inherits from tf.keras.Model
# # 1st block:
# model.add(Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(img_h, img_w, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # 2nd block:
# model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # Dense layers:
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dense(84, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
# ```

# %% [markdown]
# Al crear el modelo, notemos la cantidad de parámetros a entrenar en cada capa (columna "Param #") y la forma de la salida.

# %%
# Creamos el modelo de la red
model = create_model(img_h, img_w, num_classes)

# %% [markdown]
# Para crear la red, necesitamos compilar el modelo indicandole los métodos de cálculo de perdida y de optimización que usará para el entrenamiento.

# %%
# setup optimizer, loss function and metrics for model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# %% [markdown]
# ## Entrenamiento
#
# El tamaño del batch puede verse influido por la función de pérdida que hayamos elegido. Por el momento, usaremos 128.
#
# Notemos el reducido número de _epochs_ que estamos usando.

# %%
batch_size = 128  # batch size
num_epochs = 10  # training epochs

# %% [markdown]
# Keras permite aplicar métodos de _callback_ con los que podemos monitorear el progreso del entrenamiento El siguiente método de callback permite ir guardando el modelo al final de cada epoch, en caso de que queramos interrumpir el entrenamiento.

# %%
# This is optional if we would like to save our model after every epoch

# To save model after each epoch of training
callback = ModelCheckpoint('mnist_cnn.h5')

# %%
# start training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[callback])

# %% [markdown]
# El modelo permite evaluar los resultados del entrenamiento con el conjunto de datos de prueba, aunque parte del entrenamiento implica validación

# %%
# Evaluate and print accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# %%
model2 = create_model()

# %% [markdown]
# En este ejemplo usamos un método distinto de optimización, el de _stochastic gradient descent_ (descenso gradiente estocástico), que fue el que implementamos en el ejercicio anterior para entrenar el la Red Neuronal Conectada (pg. 40 del libro). En el modelo arriba, usamos el "Adam" (_adaptive moment estimation_), presentando en la página 102 del libro.
#
# El callback en este caso es uno que detiene el proces si la precisión de validación no mejora.

# %%
model2.compile(optimizer='sgd', loss=keras.losses.categorical_crossentropy,
               metrics=['accuracy'])
# We also instantiate some Keras callbacks, that is, utility functions
# automatically called at some points during training to monitor it:
callbacks = [
    # To interrupt the training if `val_loss` stops improving for over 3epochs:
    keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')]


# %%
# Finally, we launch the training:

model2.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=num_epochs,
           verbose=1,
           validation_data=(x_test, y_test),
           callbacks=callbacks)

# %%
# Evaluate and print accuracy
score2 = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])


# %%
model.save('model1.h5')
model2.save('model2.h5')

# %%
np.random.seed(20)

model = keras.models.load_model('model1.h5')
model2 = keras.models.load_model('model2.h5')

# setup parameters
num_classes = 10  # target number of classes
img_h, img_w = 28, 28  # input dimensions


def get_dataset():
    """
    Return processed and reshaped dataset for training
    In this cases Fashion-mnist dataset.
    """
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # test and train datasets
    print("Nb Train:", x_train.shape[0], "Nb test:", x_test.shape[0])
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    in_shape = (img_h, img_w, 1)
    # normalize inputs
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    # convert to one hot vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_dataset()

# %% [markdown]
# **1.** ¿Cómo se comparan la precisión de validación (_accuracy_ reportada) de los dos métodos?
#

# %% [markdown]
# Tenemos los mejores resultados en el primer modelo ya que tienen mayor porcentaje de acierto y el modelo generalizó mejor para el dataset o sea que tiene menor test loss.

# %% [markdown]
# |               | Model 2 | Model 1 |
# |---------------|---------|---------|
# | Test loss     | 0.4182  | 0.2423  |
# | Test accuracy | 0.8446  | 0.9248  |

# %% [markdown]
# ## Probemos nuestra suerte
#
# Esta vez, probemos nuestra suerte usando el modelo para intentar clasificar una imagen fuera del conjunto de entrenamiento.
#
# Para esto usaremos OpenCV para tomar una imagen de un objeto similar a los del conjunto de entrenamiento y ver si nos funciona.
#
# Primero veamos cómo usar la red para obtener una predicción:

# %%


# %% [markdown]
# Veamos cómo se ve una de las imágenes del conjunto de prueba:

# %% [markdown]
# **2.** ¿Cuál es la categoría de esta imagen? Modifique el siguiente bloque para comprobar

# %%
index = 18
plt.subplot(2, 2, 1)
plt.title("Ejemplo de MNIST")
plt.imshow(x_test[index])

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

category_number = np.argmax(np.array(y_test[index]))

cat = class_names[category_number]
print("Categoría:", cat)

# %%
index = 10
plt.subplot(2, 2, 1)
plt.title("Ejemplo de MNIST")
plt.imshow(x_test[index])

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

cat = class_names[np.argmax(np.array(y_test[index]))]
print("Categoría:", cat)

# %% [markdown]
# **3.** ¿Cuáles son las categorías del conjunto de datos Fashion MNIST? ¿A cuál pertenece la imagen?

# %% [markdown]
# class_names = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot"
# ]

# %% [markdown]
# Veamos si el modelo de la CNN predice lo mismo. Primero debemos usar `np.reshape` para preparar la imagen a tener el mismo formato que el que pide la red en su entrada. Luego usamos `predict` para ver lo que dice el modelo:

# %%
a = x_test[18].reshape(1, 28, 28, 1)
res = model.predict(a)
cat_a = res.argmax(axis=1)

name = class_names[np.argmax(np.array(res))]

print("Categoría predicha: {}".format(cat_a))
print("Nombre de categoría predicha: {}".format(name))


# %% [markdown]
# **4.**
#
# ¿Por qué es necesario usar `argmax` para obtener la categoría, en vez de solo `res`?

# %%
print(res)

# %% [markdown]
# **argmax** sirve para indicarnos el indice de máximo valor en nuestra capa de salida. Nuestra capa de salida será las diez clases o categorías que se indicaron en su configuración por la cual el índice de mayor índice indicará el valor más probable de ser el correcto indicado por nuestro modelo.

# %% [markdown]
# Entonces, ¿Qué significa lo que devuelve `predict`? Pista: revise la forma de la salida de la última capa del modelo.

# %% [markdown]
# **res** solo te devuelve los 10 valores indicando por sus índices la probabilidad o certeza que tiene de que sea cada uno de los índices/clases/categorías de nuestro modelo.

# %% [markdown]
# ### Probemos ahora una imagen fuera del conjunto.
#
# Empezamos con la imagen "bolso.jpg" disponible junto con este archivo, pasándola por un proceso en el que usamos OpenCV para volverla similar al conjunto de entrenamiento. Esto es, volver la imagen un cuadro 28x28 en escala de grises, normalizado a valores entre 0 y 1. El siguiente método toma una imagen y devuelve la conversión.

# %%


def preparar_imagen(img, img_h=28, img_w=28):
    '''Convierte una imagen '''

    res = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convertir a Hue, Sat, Val
    res[:, :, 2] = 255 - res[:, :, 2]  # Invertir Value
    # Convertir definición del color
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    # Convertir a Grises (1-dimensional)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(res, (img_w, img_h), cv2.INTER_AREA)  # Cambiar de tamaño
    res = res.reshape(img_h, img_w, 1)
    res = res.astype('float32')
    res /= 255.0
    print(np.shape(res))
    return res

# %% [markdown]
# Para obtener la imagen y procesarla:


# %%
bolso1 = cv2.imread(r"bolso.jpg")

b1 = preparar_imagen(bolso1)

# %%
plt.subplot(2, 2, 1)
plt.title("Original")
# OpenCV importa los colores en formato BGR en vez de RGB, esto intercambia las capas
plt.imshow(cv2.cvtColor(bolso1, cv2.COLOR_RGB2BGR))
plt.subplot(2, 2, 2)
plt.title("Preparada")
plt.imshow(b1)

# %% [markdown]
# **5.** Utilice las dos redes que entrenamos para predecir la clase de objeto que está en la imagen.

# %%
pred1 = model.predict(b1)
pred2 = model2.predict(b1)

clase1 = pred1.argmax()
clase2 = pred2.argmax()

# %% [markdown]
# **6.** Intente con las otras imágenes incluidas con este archivo. ¿Qué puede decir de los resultados?

# %% [markdown]
# **7.** Intente con dos imágenes  extraídas de Amazon de artículos de las categorías que hemos visto. Las imágenes deberían ser **solo la prenda** y con el fondo blanco.

# %% [markdown]
# **8.** Finalmente, ¿hubo algún caso donde no concordaron los dos modelos? ¿Cuál considera estuvo más cerca? ¿Considera que alguna característica del artículo influyó?
