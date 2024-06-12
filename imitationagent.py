from gamestate import gamestate
from copy import deepcopy
import keras
from keras.layers import Dense
import pandas as pd
import random



class imitationagent:
    """
	implementacion de un agente que realiza imitation learning para hex.
	"""

    def __init__(self, state=gamestate(8)):

        self.model = keras.Sequential()
        self.size=0
        self.set_gamestate(state)

        "se crea la red de neuronas que se va a utilizar con tantas neuronas en la capa de entrada como grande sea el tablero"
        self.model.add(keras.Input(shape=(self.dim,)))
        self.model.add(Dense(self.dim, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        "queremos que la salida sean dos(fila y columna) y con funcion de activacion relu para sacar numeros enteros enteros positivos"
        self.model.add(Dense(2, activation='relu'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        "cargamos pesos anteriores"
        #self.model.load_weights('model'+str(self.dim))
        print(self.model.summary())

    def best_move(self):
        """
		predice un movimiento usando el modelo y lo devuelve.
		"""
        if (self.rootstate.winner() != gamestate.PLAYERS["none"]):
            "cuando termina el juego guarda los pesos de las neuronas"
            self.model.save_weights('model')
            return gamestate.GAMEOVER


        board = [self.anterior.board.flatten().astype(int).tolist()]
        "selecciona el movimiento"
        bestmove = self.model.predict(board)
        "nos aseguramos que los resultados esten en los valores aceptables"
        resul = (bestmove[0][0].astype(int)%self.size, bestmove[0][1].astype(int)%self.size)
        if(resul in self.rootstate.moves()):
            return resul
        else:
            """para evitar que seleccione una casilla elegida cada vez que lo haga le 
            penalizamos seleccionando una casilla al azar. 
            este movimiento al azar tambien formara parte de su entrenamiento 
            para que aprenda a evitar casillas incorrectas """
            print("casilla ya seleccionada")
            print("seleccionando casilla aleatoria de las disponibles")
            resul = random.choice(self.rootstate.moves())
            return resul


    def move(self, move):
        """
		mover ficha elegida y entrenamiento
		"""

        trainX = self.anterior.board.flatten().astype(int).tolist()
        trainy = move
        self.databaseX = pd.DataFrame([trainX])
        self.databaseY = pd.DataFrame([trainy])

        "se realiza data augmentation"
        for i in range(100):
            X_train = pd.concat([self.databaseX, pd.DataFrame([trainX])], ignore_index=True)
            y_train = pd.concat([self.databaseY, pd.DataFrame([trainy])], ignore_index=True)


        "se entrena con cada movimiento"
        self.model.fit(X_train, y_train, epochs=100, verbose=0)

        self.rootstate.play(move)

    def set_gamestate(self, state):
        """
		Actualiza el estado del juego
		"""
        self.rootstate = deepcopy(state)
        self.anterior = state
        self.size = state.size
        self.dim = state.size * state.size

