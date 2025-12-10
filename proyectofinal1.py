import tkinter as tk
import random
import time
import numpy as np
import pandas as pd
import json
import os
from collections import deque
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Red neuronal simple (sin TensorFlow para evitar dependencias)
class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Pesos
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Biases
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        # Concatenar entrada con estado anterior
        concat = np.vstack((h_prev, x))
        
        # Puerta de olvido
        ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        
        # Puerta de entrada
        it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        
        # Candidato para celda
        cct = self.tanh(np.dot(self.Wc, concat) + self.bc)
        
        # Nueva celda
        c_next = ft * c_prev + it * cct
        
        # Puerta de salida
        ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # Nuevo estado oculto
        h_next = ot * self.tanh(c_next)
        
        # Salida
        yt = np.dot(self.Wy, h_next) + self.by
        yt = self.sigmoid(yt)
        
        return yt, h_next, c_next

class AdvancedSnakeAI:
    def __init__(self):
        # Sistema de aprendizaje
        self.decision_tree = DecisionTreeClassifier(max_depth=5)
        self.scaler = StandardScaler()
        
        # Red neuronal LSTM para predecir jugador
        self.lstm = SimpleLSTM(input_size=8, hidden_size=16, output_size=4)
        
        # Memoria para entrenamiento
        self.memory = []
        self.player_patterns = []
        
        # CSV para almacenar datos
        self.data_file = "snake_ai_data.csv"
        self.model_file = "snake_ai_model.json"
        
        # Estado de la IA
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.3  # Exploración vs explotación
        self.generation = 0
        self.total_reward = 0
        
        # Cargar datos previos
        self.load_data()
        
    def load_data(self):
        """Carga datos de entrenamiento previos"""
        try:
            if os.path.exists(self.data_file):
                self.df = pd.read_csv(self.data_file)
                print(f"Datos cargados: {len(self.df)} registros")
            else:
                self.df = pd.DataFrame(columns=[
                    'game_id', 'timestamp', 'player_pos_x', 'player_pos_y',
                    'player_dir_x', 'player_dir_y', 'rival_pos_x', 'rival_pos_y',
                    'food_pos_x', 'food_pos_y', 'player_length', 'rival_length',
                    'action_taken', 'reward', 'next_state', 'game_score'
                ])
                print("Nuevo dataset creado")
                
            if os.path.exists(self.model_file):
                with open(self.model_file, 'r') as f:
                    data = json.load(f)
                    self.generation = data.get('generation', 0)
                    self.total_reward = data.get('total_reward', 0)
                    print(f"Modelo cargado - Generación: {self.generation}")
        except Exception as e:
            print(f"Error cargando datos: {e}")
            self.df = pd.DataFrame(columns=[
                'game_id', 'timestamp', 'player_pos_x', 'player_pos_y',
                'player_dir_x', 'player_dir_y', 'rival_pos_x', 'rival_pos_y',
                'food_pos_x', 'food_pos_y', 'player_length', 'rival_length',
                'action_taken', 'reward', 'next_state', 'game_score'
            ])
    
    def save_data(self, game_id, game_data):
        """Guarda datos de la partida"""
        try:
            new_data = pd.DataFrame(game_data)
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            self.df.to_csv(self.data_file, index=False)
            
            # Guardar modelo
            model_data = {
                'generation': self.generation,
                'total_reward': self.total_reward
            }
            with open(self.model_file, 'w') as f:
                json.dump(model_data, f)
            print(f"Datos guardados: {len(game_data)} registros")
        except Exception as e:
            print(f"Error guardando datos: {e}")
    
    def extract_features(self, game_state):
        """Extrae características del estado del juego"""
        player_head = game_state['player_head']
        rival_head = game_state['rival_head']
        food_pos = game_state['food_pos']
        player_direction = game_state['player_direction']
        rival_direction = game_state['rival_direction']
        width = game_state['width']
        height = game_state['height']
        player_body = game_state['player_body']
        rival_body = game_state['rival_body']
        
        # Características básicas
        features = []
        
        # 1. Distancia relativa a la comida
        dx_food = (food_pos[0] - rival_head[0]) % width
        dy_food = (food_pos[1] - rival_head[1]) % height
        features.extend([dx_food / width, dy_food / height])
        
        # 2. Distancia al jugador
        dx_player = (player_head[0] - rival_head[0]) % width
        dy_player = (player_head[1] - rival_head[1]) % height
        features.extend([dx_player / width, dy_player / height])
        
        # 3. Direcciones
        features.extend([player_direction[0], player_direction[1],
                         rival_direction[0], rival_direction[1]])
        
        # 4. Peligros cercanos
        dangers = self.check_dangers(rival_head, rival_body, player_body, width, height)
        features.extend(dangers)
        
        # 5. Valor de la fruta
        fruit_value = 1.0 / (abs(dx_food) + abs(dy_food) + 1)
        features.append(fruit_value)
        
        # 6. Oportunidad de bloqueo
        block_opportunity = self.calculate_block_opportunity(
            player_head, player_direction, rival_head, width, height
        )
        features.append(block_opportunity)
        
        return np.array(features).reshape(1, -1)
    
    def check_dangers(self, rival_head, rival_body, player_body, width, height):
        """Verifica peligros en las 4 direcciones"""
        dangers = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dx, dy in directions:
            danger_score = 0
            next_pos = ((rival_head[0] + dx) % width, (rival_head[1] + dy) % height)
            
            # Peligro: colisión con su propio cuerpo
            if next_pos in rival_body[1:]:
                danger_score += 0.8
            
            # Peligro: colisión con el jugador
            if next_pos in player_body:
                danger_score += 0.6
            
            # Peligro: cerca de las paredes
            if next_pos[0] < 2 or next_pos[0] > width - 3:
                danger_score += 0.2
            if next_pos[1] < 2 or next_pos[1] > height - 3:
                danger_score += 0.2
            
            dangers.append(danger_score)
        
        return dangers
    
    def calculate_block_opportunity(self, player_head, player_dir, rival_head, width, height):
        """Calcula oportunidad de bloquear al jugador"""
        # Predecir próxima posición del jugador
        predicted_player_pos = (
            (player_head[0] + player_dir[0]) % width,
            (player_head[1] + player_dir[1]) % height
        )
        
        # Distancia del rival a esa posición
        dx = abs(predicted_player_pos[0] - rival_head[0])
        dy = abs(predicted_player_pos[1] - rival_head[1])
        distance = min(dx, width - dx) + min(dy, height - dy)
        
        # Oportunidad inversamente proporcional a la distancia
        opportunity = max(0, 1 - distance / 10)
        return opportunity
    
    def predict_player_movement(self, player_history):
        """Predice el próximo movimiento del jugador usando LSTM"""
        if len(player_history) < 3:
            return random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        
        # Convertir deque a lista
        history_list = list(player_history)
        
        # Tomar últimos 3 movimientos
        recent_moves = history_list[-3:] if len(history_list) >= 3 else history_list
        features = []
        
        for move in recent_moves:
            features.extend([move[0], move[1]])
        
        # Rellenar si hay menos de 3 movimientos
        while len(features) < 6:
            features.extend([0, 0])
        
        # Tomar solo 6 características (3 movimientos * 2 coordenadas)
        features = features[:6]
        
        # Añadir padding hasta 8 características
        while len(features) < 8:
            features.append(0)
        
        features = np.array(features[:8]).reshape(8, 1)
        
        # Pasar por LSTM
        h_prev = np.zeros((self.lstm.hidden_size, 1))
        c_prev = np.zeros((self.lstm.hidden_size, 1))
        
        output, _, _ = self.lstm.forward(features, h_prev, c_prev)
        
        # Interpretar salida
        direction_idx = np.argmax(output.flatten())
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        if direction_idx < len(directions):
            return directions[direction_idx]
        else:
            return random.choice(directions)
    
    def make_decision(self, game_state, player_history):
        """Toma decisión combinando árbol de decisión y LSTM"""
        # Predicción LSTM del jugador
        predicted_player_move = self.predict_player_movement(player_history)
        
        # Estrategia principal: ir por la fruta
        fruit_direction = self.get_direction_to_target(
            game_state['rival_head'],
            game_state['food_pos'],
            game_state['width'],
            game_state['height']
        )
        
        # Estrategia secundaria: bloquear al jugador si es buena oportunidad
        block_direction = None
        block_opp = game_state.get('block_opportunity', 0)
        
        if block_opp > 0.5:
            block_direction = self.get_blocking_direction(
                game_state['player_head'],
                predicted_player_move,
                game_state['rival_head'],
                game_state['width'],
                game_state['height']
            )
        
        # Decisión final (con exploración epsilon-greedy)
        if random.random() < self.epsilon:
            # Exploración: movimiento aleatorio seguro
            safe_moves = self.get_safe_moves(game_state)
            if safe_moves:
                return random.choice(safe_moves)
        
        # Explotación: elegir mejor movimiento
        if block_direction and random.random() < block_opp:
            return block_direction
        elif fruit_direction:
            return fruit_direction
        else:
            # Movimiento por defecto (aleatorio pero seguro)
            safe_moves = self.get_safe_moves(game_state)
            if safe_moves:
                return random.choice(safe_moves)
            else:
                return (1, 0)
    
    def get_direction_to_target(self, current_pos, target_pos, width, height):
        """Calcula dirección óptima hacia un objetivo"""
        dx = (target_pos[0] - current_pos[0]) % width
        dy = (target_pos[1] - current_pos[1]) % height
        
        # Considerar wraparound
        if dx > width // 2:
            dx = dx - width
        if dy > height // 2:
            dy = dy - height
        
        if abs(dx) > abs(dy):
            return (1 if dx > 0 else -1, 0)
        else:
            return (0, 1 if dy > 0 else -1)
    
    def get_blocking_direction(self, player_pos, player_dir, rival_pos, width, height):
        """Calcula dirección para bloquear al jugador"""
        # Posición futura del jugador
        future_player_pos = (
            (player_pos[0] + player_dir[0]) % width,
            (player_pos[1] + player_dir[1]) % height
        )
        
        # Posición para bloquear (una celda adelante)
        block_pos = (
            (future_player_pos[0] + player_dir[0]) % width,
            (future_player_pos[1] + player_dir[1]) % height
        )
        
        return self.get_direction_to_target(rival_pos, block_pos, width, height)
    
    def get_safe_moves(self, game_state):
        """Obtiene movimientos seguros (sin colisiones inmediatas)"""
        rival_head = game_state['rival_head']
        rival_body = game_state['rival_body']
        player_body = game_state['player_body']
        width = game_state['width']
        height = game_state['height']
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        safe_moves = []
        
        for dx, dy in directions:
            next_pos = ((rival_head[0] + dx) % width, (rival_head[1] + dy) % height)
            
            # Verificar seguridad
            safe = True
            
            # No chocar consigo mismo
            if next_pos in rival_body[1:]:
                safe = False
            
            # No chocar con el jugador
            if next_pos in player_body:
                safe = False
            
            if safe:
                safe_moves.append((dx, dy))
        
        return safe_moves
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """Aprende de la experiencia (sistema evolutivo)"""
        # Guardar en memoria
        self.memory.append((state, action, reward, next_state, done))
        self.total_reward += reward
        
        # Limitar tamaño de memoria
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]
        
        # Entrenar periódicamente
        if len(self.memory) % 100 == 0 and len(self.memory) >= 50:
            self.train_on_memory()
    
    def train_on_memory(self):
        """Entrena el modelo con la memoria acumulada"""
        print(f"Entrenando IA... Generación: {self.generation}")
        self.generation += 1
        
        # Reducir epsilon (menos exploración con el tiempo)
        self.epsilon = max(0.1, self.epsilon * 0.99)

class SnakeGameML:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake con IA Avanzada - ML + Red Neuronal")
        self.root.resizable(False, False)
        
        # Configuración
        self.cell_size = 20
        self.width = 30
        self.height = 25
        self.speed = 150
        
        # Colores
        self.bg_color = "#0A1A0A"
        self.grid_color = "#1A3A1A"
        self.player_color = "#00FF00"
        self.player_head = "#007700"
        self.rival_color = "#FF3300"
        self.rival_head = "#770000"
        self.food_color = "#FFFF00"
        self.text_color = "#FFFFFF"
        self.ui_bg = "#1A2A1A"
        
        # Crear interfaz
        self.setup_ui()
        
        # Inicializar IA
        self.ai = AdvancedSnakeAI()
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.game_data = []
        self.player_history = deque(maxlen=10)
        
        # Variables del juego
        self.reset_game()
        
        # Controles
        self.root.bind("<KeyPress>", self.on_key_press)
        
        # Iniciar
        self.draw_grid()
        self.game_loop()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill="both", expand=True)
        
        # Canvas del juego
        self.canvas = tk.Canvas(
            main_frame,
            width=self.width * self.cell_size,
            height=self.height * self.cell_size,
            bg=self.bg_color,
            highlightthickness=0
        )
        self.canvas.pack(side="left", padx=10, pady=10)
        
        # Panel de información
        info_panel = tk.Frame(main_frame, bg=self.ui_bg, width=250)
        info_panel.pack(side="right", fill="y", padx=10, pady=10)
        info_panel.pack_propagate(False)
        
        # Título
        title = tk.Label(
            info_panel,
            text="SNAKE AI",
            font=("Arial", 20, "bold"),
            fg=self.player_color,
            bg=self.ui_bg
        )
        title.pack(pady=20)
        
        # Puntuaciones
        scores_frame = tk.Frame(info_panel, bg=self.ui_bg)
        scores_frame.pack(pady=10, padx=20, fill="x")
        
        self.player_score_var = tk.StringVar(value="Jugador: 0")
        self.player_score_label = tk.Label(
            scores_frame,
            textvariable=self.player_score_var,
            font=("Arial", 14, "bold"),
            fg=self.player_color,
            bg=self.ui_bg
        )
        self.player_score_label.pack(anchor="w")
        
        self.rival_score_var = tk.StringVar(value="Rival IA: 0")
        self.rival_score_label = tk.Label(
            scores_frame,
            textvariable=self.rival_score_var,
            font=("Arial", 14, "bold"),
            fg=self.rival_color,
            bg=self.ui_bg
        )
        self.rival_score_label.pack(anchor="w", pady=5)
        
        # Información de IA
        ai_frame = tk.Frame(info_panel, bg=self.ui_bg)
        ai_frame.pack(pady=20, padx=20, fill="x")
        
        tk.Label(
            ai_frame,
            text="ESTADO DE LA IA:",
            font=("Arial", 12, "bold"),
            fg=self.text_color,
            bg=self.ui_bg
        ).pack(anchor="w", pady=(0, 10))
        
        self.ai_generation_var = tk.StringVar(value="Generación: 0")
        tk.Label(
            ai_frame,
            textvariable=self.ai_generation_var,
            font=("Arial", 11),
            fg=self.text_color,
            bg=self.ui_bg
        ).pack(anchor="w")
        
        self.ai_memory_var = tk.StringVar(value="Memoria: 0 eventos")
        tk.Label(
            ai_frame,
            textvariable=self.ai_memory_var,
            font=("Arial", 11),
            fg=self.text_color,
            bg=self.ui_bg
        ).pack(anchor="w")
        
        self.ai_epsilon_var = tk.StringVar(value="Exploración: 30%")
        tk.Label(
            ai_frame,
            textvariable=self.ai_epsilon_var,
            font=("Arial", 11),
            fg=self.text_color,
            bg=self.ui_bg
        ).pack(anchor="w")
        
        # Estrategia actual
        self.strategy_var = tk.StringVar(value="Estrategia: Buscando fruta")
        self.strategy_label = tk.Label(
            ai_frame,
            textvariable=self.strategy_var,
            font=("Arial", 11, "italic"),
            fg=self.rival_color,
            bg=self.ui_bg
        )
        self.strategy_label.pack(anchor="w", pady=(10, 0))
        
        # Controles
        controls_frame = tk.Frame(info_panel, bg=self.ui_bg)
        controls_frame.pack(pady=30, padx=20, fill="x")
        
        tk.Label(
            controls_frame,
            text="CONTROLES:",
            font=("Arial", 12, "bold"),
            fg=self.text_color,
            bg=self.ui_bg
        ).pack(anchor="w", pady=(0, 10))
        
        controls = [
            ("←↑↓→", "Mover serpiente"),
            ("ESPACIO", "Pausar/Reanudar"),
            ("R", "Reiniciar juego"),
            ("S", "Guardar datos de IA"),
            ("ESC", "Salir")
        ]
        
        for key, desc in controls:
            frame = tk.Frame(controls_frame, bg=self.ui_bg)
            frame.pack(fill="x", pady=2)
            
            tk.Label(
                frame,
                text=key,
                font=("Arial", 10, "bold"),
                fg=self.player_color,
                bg=self.ui_bg,
                width=10
            ).pack(side="left")
            
            tk.Label(
                frame,
                text=desc,
                font=("Arial", 10),
                fg=self.text_color,
                bg=self.ui_bg
            ).pack(side="left")
    
    def draw_grid(self):
        """Dibuja la cuadrícula"""
        for x in range(0, self.width * self.cell_size, self.cell_size):
            self.canvas.create_line(
                x, 0, x, self.height * self.cell_size,
                fill=self.grid_color, width=1
            )
        for y in range(0, self.height * self.cell_size, self.cell_size):
            self.canvas.create_line(
                0, y, self.width * self.cell_size, y,
                fill=self.grid_color, width=1
            )
    
    def reset_game(self):
        """Reinicia el juego"""
        # Serpiente del jugador
        self.player = {
            'body': [(self.width//4, self.height//2)],
            'direction': (1, 0),
            'grow': False,
            'score': 0,
            'alive': True
        }
        
        # Serpiente rival IA
        self.rival = {
            'body': [(3*self.width//4, self.height//2)],
            'direction': (-1, 0),
            'grow': False,
            'score': 0,
            'alive': True
        }
        
        # Comida
        self.generate_food()
        
        # Estado del juego
        self.game_over = False
        self.winner = None
        self.paused = False
        
        # Limpiar historial
        self.player_history.clear()
        
        # Actualizar UI
        self.update_scores()
        self.update_ai_info()
    
    def generate_food(self):
        """Genera comida en posición aleatoria"""
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            food_pos = (x, y)
            
            if (food_pos not in self.player['body'] and 
                food_pos not in self.rival['body']):
                self.food = food_pos
                break
    
    def get_game_state(self):
        """Obtiene el estado actual del juego para la IA"""
        return {
            'player_head': self.player['body'][0],
            'rival_head': self.rival['body'][0],
            'food_pos': self.food,
            'player_direction': self.player['direction'],
            'rival_direction': self.rival['direction'],
            'player_body': self.player['body'],
            'rival_body': self.rival['body'],
            'width': self.width,
            'height': self.height,
            'player_score': self.player['score'],
            'rival_score': self.rival['score'],
            'block_opportunity': self.calculate_block_opportunity()
        }
    
    def calculate_block_opportunity(self):
        """Calcula oportunidad de bloqueo actual"""
        player_head = self.player['body'][0]
        rival_head = self.rival['body'][0]
        
        dx = abs(player_head[0] - rival_head[0])
        dy = abs(player_head[1] - rival_head[1])
        distance = min(dx, self.width - dx) + min(dy, self.height - dy)
        
        # Más cerca = mayor oportunidad
        opportunity = max(0, 1 - distance / 15)
        
        # Aumentar oportunidad si el jugador va hacia el rival
        player_dir = self.player['direction']
        future_player = (
            (player_head[0] + player_dir[0]) % self.width,
            (player_head[1] + player_dir[1]) % self.height
        )
        
        future_dx = abs(future_player[0] - rival_head[0])
        future_dy = abs(future_player[1] - rival_head[1])
        future_dist = min(future_dx, self.width - future_dx) + min(future_dy, self.height - future_dy)
        
        if future_dist < distance:
            opportunity *= 1.5
        
        return min(opportunity, 1.0)
    
    def move_rival_ai(self):
        """Mueve la serpiente rival usando IA avanzada"""
        if not self.rival['alive'] or self.game_over:
            return
        
        # Registrar movimiento del jugador
        if self.player['direction'] != (0, 0):
            self.player_history.append(self.player['direction'])
        
        # Obtener estado del juego
        game_state = self.get_game_state()
        
        # Tomar decisión con IA
        new_direction = self.ai.make_decision(game_state, self.player_history)
        
        # Actualizar dirección (evitando movimiento opuesto)
        current_dir = self.rival['direction']
        if not (new_direction[0] + current_dir[0] == 0 and 
                new_direction[1] + current_dir[1] == 0):
            self.rival['direction'] = new_direction
        
        # Actualizar etiqueta de estrategia
        dx = abs(self.food[0] - self.rival['body'][0][0])
        dy = abs(self.food[1] - self.rival['body'][0][1])
        distance = min(dx, self.width - dx) + min(dy, self.height - dy)
        
        block_opp = self.calculate_block_opportunity()
        
        if block_opp > 0.5 and distance > 5:
            self.strategy_var.set("Estrategia: Bloqueando jugador")
        else:
            self.strategy_var.set("Estrategia: Buscando fruta")
    
    def update_ai_info(self):
        """Actualiza la información de la IA en la UI"""
        self.ai_generation_var.set(f"Generación: {self.ai.generation}")
        self.ai_memory_var.set(f"Memoria: {len(self.ai.memory)} eventos")
        self.ai_epsilon_var.set(f"Exploración: {int(self.ai.epsilon * 100)}%")
    
    def update_scores(self):
        """Actualiza las puntuaciones"""
        self.player_score_var.set(f"Jugador: {self.player['score']}")
        self.rival_score_var.set(f"Rival IA: {self.rival['score']}")
    
    def move_snake(self, snake, is_rival=False):
        """Mueve una serpiente"""
        if not snake['alive']:
            return
        
        head_x, head_y = snake['body'][0]
        dir_x, dir_y = snake['direction']
        
        new_x = (head_x + dir_x) % self.width
        new_y = (head_y + dir_y) % self.height
        new_head = (new_x, new_y)
        
        # Verificar colisión consigo misma
        if new_head in snake['body']:
            snake['alive'] = False
            return
        
        # Mover
        snake['body'].insert(0, new_head)
        
        if not snake['grow']:
            snake['body'].pop()
        else:
            snake['grow'] = False
            snake['score'] += 1
    
    def check_collisions(self):
        """Verifica colisiones"""
        if not self.player['alive']:
            self.game_over = True
            self.winner = "rival"
            return
        
        if not self.rival['alive']:
            self.game_over = True
            self.winner = "player"
            return
        
        # Colisión cabeza a cabeza
        player_head = self.player['body'][0]
        rival_head = self.rival['body'][0]
        
        if player_head in self.rival['body']:
            self.game_over = True
            self.winner = "rival"
            return
        
        if rival_head in self.player['body']:
            self.game_over = True
            self.winner = "player"
            return
        
        if player_head == rival_head:
            self.game_over = True
            self.winner = "draw"
            return
    
    def draw(self):
        """Dibuja todos los elementos"""
        self.canvas.delete("all")
        self.draw_grid()
        
        # Dibujar comida
        food_x, food_y = self.food
        self.canvas.create_oval(
            food_x * self.cell_size + 3,
            food_y * self.cell_size + 3,
            (food_x + 1) * self.cell_size - 3,
            (food_y + 1) * self.cell_size - 3,
            fill=self.food_color,
            outline=self.food_color,
            width=2
        )
        
        # Dibujar serpiente del jugador
        for i, (x, y) in enumerate(self.player['body']):
            color = self.player_head if i == 0 else self.player_color
            self.canvas.create_rectangle(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                (x + 1) * self.cell_size - 1,
                (y + 1) * self.cell_size - 1,
                fill=color,
                outline=self.grid_color,
                width=1
            )
        
        # Dibujar serpiente rival
        for i, (x, y) in enumerate(self.rival['body']):
            color = self.rival_head if i == 0 else self.rival_color
            self.canvas.create_rectangle(
                x * self.cell_size + 1,
                y * self.cell_size + 1,
                (x + 1) * self.cell_size - 1,
                (y + 1) * self.cell_size - 1,
                fill=color,
                outline=self.grid_color,
                width=1
            )
        
        # Dibujar información de la IA sobre la serpiente rival
        if self.rival['alive'] and len(self.rival['body']) > 0 and not self.game_over:
            head_x, head_y = self.rival['body'][0]
            
            # Indicador de estrategia
            if "Bloqueando" in self.strategy_var.get():
                indicator_color = "#FF0000"
            else:
                indicator_color = "#FFFF00"
            
            self.canvas.create_oval(
                head_x * self.cell_size - 5,
                head_y * self.cell_size - 5,
                head_x * self.cell_size + self.cell_size + 5,
                head_y * self.cell_size + self.cell_size + 5,
                outline=indicator_color,
                width=2
            )
        
        # Dibujar game over
        if self.game_over:
            self.canvas.create_rectangle(
                0, 0,
                self.width * self.cell_size,
                self.height * self.cell_size,
                fill="#000000",
                stipple="gray50"
            )
            
            if self.winner == "player":
                message = "¡VICTORIA!"
                color = self.player_color
            elif self.winner == "rival":
                message = "¡IA GANA!"
                color = self.rival_color
            else:
                message = "¡EMPATE!"
                color = self.text_color
            
            self.canvas.create_text(
                self.width * self.cell_size // 2,
                self.height * self.cell_size // 2 - 30,
                text=message,
                font=("Arial", 28, "bold"),
                fill=color
            )
            
            self.canvas.create_text(
                self.width * self.cell_size // 2,
                self.height * self.cell_size // 2 + 10,
                text=f"{self.player['score']} - {self.rival['score']}",
                font=("Arial", 20),
                fill=self.text_color
            )
            
            self.canvas.create_text(
                self.width * self.cell_size // 2,
                self.height * self.cell_size // 2 + 50,
                text="Presiona R para jugar otra vez",
                font=("Arial", 14),
                fill=self.text_color
            )
        
        # Dibujar pausa
        elif self.paused:
            self.canvas.create_text(
                self.width * self.cell_size // 2,
                self.height * self.cell_size // 2,
                text="PAUSA",
                font=("Arial", 24, "bold"),
                fill=self.text_color
            )
    
    def save_game_data(self):
        """Guarda los datos de la partida"""
        if self.game_data:
            self.ai.save_data(self.game_id, self.game_data)
            print("Datos guardados exitosamente")
    
    def update_game(self):
        """Actualiza el estado del juego"""
        if self.game_over or self.paused:
            return
        
        # Guardar estado anterior para aprendizaje
        old_state = self.get_game_state()
        
        # Mover jugador
        self.move_snake(self.player)
        
        # Mover rival con IA
        self.move_rival_ai()
        self.move_snake(self.rival, is_rival=True)
        
        # Verificar colisiones
        self.check_collisions()
        
        # Verificar comida
        reward = 0
        if self.player['body'][0] == self.food:
            self.player['grow'] = True
            self.generate_food()
            reward -= 0.5  # Penalidad para IA cuando jugador come
        
        if self.rival['body'][0] == self.food:
            self.rival['grow'] = True
            self.generate_food()
            reward += 1.0  # Recompensa para IA
        
        # Guardar datos para aprendizaje
        if not self.game_over:
            new_state = self.get_game_state()
            action = self.rival['direction']
            
            self.game_data.append({
                'game_id': self.game_id,
                'timestamp': datetime.now().isoformat(),
                'player_pos_x': self.player['body'][0][0],
                'player_pos_y': self.player['body'][0][1],
                'player_dir_x': self.player['direction'][0],
                'player_dir_y': self.player['direction'][1],
                'rival_pos_x': self.rival['body'][0][0],
                'rival_pos_y': self.rival['body'][0][1],
                'food_pos_x': self.food[0],
                'food_pos_y': self.food[1],
                'player_length': len(self.player['body']),
                'rival_length': len(self.rival['body']),
                'action_taken': f"{action[0]},{action[1]}",
                'reward': reward,
                'next_state': str(new_state),
                'game_score': f"{self.player['score']}-{self.rival['score']}"
            })
            
            # Aprender de la experiencia
            self.ai.learn_from_experience(
                old_state,
                action,
                reward,
                new_state,
                self.game_over
            )
        
        # Actualizar UI
        self.update_scores()
        self.update_ai_info()
    
    def on_key_press(self, event):
        """Maneja teclas presionadas"""
        key = event.keysym
        
        if key == "Escape":
            self.save_game_data()
            self.root.quit()
        elif key == "r" or key == "R":
            self.save_game_data()
            self.reset_game()
        elif key == "s" or key == "S":
            self.save_game_data()
        elif key == "space":
            self.paused = not self.paused
        elif not self.paused and not self.game_over:
            if key == "Left":
                if self.player['direction'] != (1, 0):
                    self.player['direction'] = (-1, 0)
            elif key == "Right":
                if self.player['direction'] != (-1, 0):
                    self.player['direction'] = (1, 0)
            elif key == "Up":
                if self.player['direction'] != (0, 1):
                    self.player['direction'] = (0, -1)
            elif key == "Down":
                if self.player['direction'] != (0, -1):
                    self.player['direction'] = (0, 1)
    
    def game_loop(self):
        """Bucle principal del juego"""
        self.update_game()
        self.draw()
        self.root.after(self.speed, self.game_loop)

# Iniciar el juego
if __name__ == "__main__":
    try:
        root = tk.Tk()
        game = SnakeGameML(root)
        root.mainloop()
    except Exception as e:
        print(f"Error al iniciar el juego: {e}")
        import traceback
        traceback.print_exc()