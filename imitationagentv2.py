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

        self.rootstate = deepcopy(state)
        self.aciertos = 0
        self.aciertosx = 0
        self.aciertosy = 0
        self.ocupado = 0
        self.total = 0
        "se crea la red de neuronas que se va a utilizar con tantas neuronas en la capa de entrada como grande sea el tablero"
        self.modelx = keras.Sequential()
        self.modelx.add(keras.Input(shape=(64,)))
        self.modelx.add(Dense(64, activation='relu'))
        self.modelx.add(Dense(64, activation='relu'))
        self.modelx.add(Dense(32, activation='relu'))
        self.modelx.add(Dense(16, activation='relu'))
        self.modelx.add(Dense(8, activation='relu'))
        "queremos que la salida sean dos(fila y columna) y con funcion de activacion relu para sacar numeros enteros enteros positivos"
        self.modelx.add(Dense(1, activation='relu'))
        self.modelx.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print(self.modelx.summary())

        #self.modelx.load_weights('modelx')
        self.modely = keras.Sequential()
        self.modely.add(keras.Input(shape=(64,)))
        self.modely.add(Dense(64, activation='relu'))
        self.modely.add(Dense(64, activation='relu'))
        self.modely.add(Dense(32, activation='relu'))
        self.modely.add(Dense(16, activation='relu'))
        self.modely.add(Dense(8, activation='relu'))
        "queremos que la salida sean dos(fila y columna) y con funcion de activacion relu para sacar numeros enteros enteros positivos"
        self.modely.add(Dense(1, activation='relu'))
        self.modely.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # self.modely.load_weights('modely')

    def best_move(self):
        """
		predice un movimiento usando el modelo y lo devuelve.
		"""
        if (self.rootstate.winner() != gamestate.PLAYERS["none"]):
            "cuando termina el juego guarda los pesos de las neuronas"

            return gamestate.GAMEOVER

        board = [self.anterior.board.flatten().astype(int).tolist()]
        "selecciona el movimiento"
        bestmove = self.model.predict(board)
        "nos aseguramos que los resultados esten en los valores aceptables"
        resul = (bestmove[0][0].astype(int) % self.size, bestmove[0][1].astype(int) % self.size)
        self.modelx.save_weights('modelx')
        self.modely.save_weights('modely')
        if (resul in self.rootstate.moves()):
            return resul
        else:
            """para evitar que seleccione una casilla elegida cada vez que lo haga le 
            penalizamos seleccionando una casilla al azar. 
            este movimiento al azar tambien formara parte de su entrenamiento 
            para que aprenda a evitar casillas incorrectas """
            print("casilla ya seleccionada")
            print("seleccionando casilla aleatoria de las disponibles")

            def get_adjacent_indices(i, j, m, n):
                adjacent_indices = []
                if i > 0:
                    adjacent_indices.append((i - 1, j))
                if i + 1 < m:
                    adjacent_indices.append((i + 1, j))
                if j > 0:
                    adjacent_indices.append((i, j - 1))
                if j + 1 < n:
                    adjacent_indices.append((i, j + 1))
                return adjacent_indices

            adjacents = get_adjacent_indices(resul[0], resul[1], 8, 8)
            for adj in adjacents:
                if adj in self.rootstate.moves():
                    resul = adj
                    igual = False
            if igual:
                resul = random.choice(self.rootstate.moves())

            return resul

    def move(self, move):
        """
		mover ficha elegida y entrenamiento
		"""

        trainX = self.anterior.board.flatten().astype(int).tolist()
        trainy = move
        trainyx = trainy[0] + 1
        trainyy = trainy[1] + 1
        self.databaseX = pd.DataFrame([trainX])
        self.databaseYx = pd.DataFrame([trainyx])
        self.databaseYy = pd.DataFrame([trainyy])

        "se realiza data augmentation"
        for i in range(5):
            self.databaseX = pd.concat([self.databaseX, pd.DataFrame([trainX])], ignore_index=True)
            self.databaseYx = pd.concat([self.databaseYx, pd.DataFrame([trainyx])], ignore_index=True)
            self.databaseYy = pd.concat([self.databaseYy, pd.DataFrame([trainyy])], ignore_index=True)

        "se entrena con cada movimiento"
        self.modelx.fit(self.databaseX, self.databaseYx, epochs=100, verbose=0)
        self.modelx.save_weights('modelx')
        self.modely.fit(self.databaseX, self.databaseYy, epochs=100, verbose=0)
        self.modely.save_weights('modely')

        self.rootstate.play(move)

    def set_gamestate(self, state):
        """
		Actualiza el estado del juego
		"""
        self.rootstate = deepcopy(state)
        self.anterior = state
        self.size = state.size
        self.dim = state.size * state.size
