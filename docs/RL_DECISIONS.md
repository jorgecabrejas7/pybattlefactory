# Decisiones de diseño de RL

Registro de lo decidido en la conversación de diseño (2026-09-24). Cada decisión lleva su motivo.
El código vive en `rl/`; las observaciones del juego, en `pybattle/view.py` (ver `docs/GUIDE.md` §7).

---

## 1. Agentes y cómo se entrenan

| Decisión | Valor | Motivo |
|---|---|---|
| Agentes | **Táctico** (alquiler + intercambios) y **combatiente** (movimientos y cambios) | La estructura del juego |
| Algoritmo | **PPO** (actor-crítico, on-policy) | Sencillo con máscaras y con muchos entornos en paralelo; el simulador es rápido, así que la eficiencia en muestras de Rainbow importa menos |
| Entrenamiento | **Los dos a la vez** | Si no aprende, plan B: **alternancia** (entrenar uno con el otro congelado, y turnarse) |
| Codificadores | **Compartidos** entre táctico y combatiente (mismas tablas de embeddings y mismos codificadores de Pokémon) | Lo que uno aprende de "qué es un buen Pokémon" sirve al otro. Revisable si hay interferencia |

## 2. Episodios, terminales y truncamiento

| | Combatiente | Táctico |
|---|---|---|
| Paso de tiempo | Cada decisión en `BATTLE` / `FORCED_SWITCH` | Cada alquiler y cada intercambio |
| Episodio | **Un combate** | **Toda la racha, hasta perder**. No termina al completar un reto de 7 combates |
| Terminal | Fin del combate (victoria o derrota) | Perder un combate |
| γ | **0,99** | **1** |

- Motivo del episodio del combatiente: entre combates el equipo se cura (`HealPlayerParty`), así que nada de lo que
  hace en un combate afecta al siguiente salvo ganar. Hacer el episodio más largo solo añade varianza.
- Motivo de γ = 0,99: con 0,9 el horizonte efectivo era de unas 10 decisiones. Un combate dura una mediana de
  26 decisiones, así que las primeras decisiones apenas recibían señal y se premiaba ganar rápido frente a ganar seguro.
- Motivo de γ = 1 en el táctico: todas las victorias importan igual.
- **Truncamiento neutro.** Gen 3 no garantiza que un combate termine: cambiar de Pokémon no gasta PP, y la curación
  puede superar al retroceso de Forcejeo. Tras `max_turns` decisiones se corta el combate como **truncamiento**, no
  como derrota. El objetivo se calcula con bootstrapping, r + γ·V(s), y no con r solo. Como el simulador no puede
  continuar la racha tras cortar, el episodio del táctico también se trunca, haciendo bootstrapping con su último valor.

## 3. Recompensas

**Combatiente**: +1 al ganar, 0 al perder. No hay −1 al perder: con γ < 1 crearía un incentivo a alargar los combates perdidos.
Además lleva **shaping basado en potencial** (Ng, Harada y Russell, 1999), que no cambia la política óptima:

    F = β · (γ·Φ(s') − Φ(s)),   β = 0,5,   Φ(terminal) = 0

    Φ(s) = 0,4 · (PS̄_propios − PS̄_rival)
         + 0,4 · (vivos_propios − vivos_rival) / 3
         + 0,2 · E(s)

    E(s) = media sobre las 7 etapas (Atk, Def, Vel, AtEsp, DefEsp, Prec, Eva) de (etapa_propia − etapa_rival) / 12

- PS̄ es la media de las fracciones de PS de los 3 Pokémon. Para el rival se usan los **PS reales**. La recompensa solo
  se usa al entrenar, así que no filtra nada a la política. Un rival que aún no ha salido cuenta como PS completos.
- Las 7 etapas entran por separado y con el mismo peso. Φ es una fórmula fija; qué stat importa más lo aprende el
  crítico, que ve cada etapa por separado en la observación.
- Aceptado: al cambiar de Pokémon se pierden las etapas, así que ese paso da shaping negativo. Al principio del
  entrenamiento el agente cambiará menos de lo debido.

**Táctico**: +1 por victoria (variante A, la del primer entrenamiento). La variante B, r = racha, queda para comparar.
Cuidado con B: sus retornos crecen con el cuadrado de la racha y habrá que escalarlos.

## 4. Acciones

**Combatiente**: 4 movimientos + cambiar a la posición 0, 1 o 2 del equipo. Las acciones ilegales se enmascaran
(logit = −∞) según `usable_moves` y `switch_targets`. **Sin acción de rendirse**: con recompensa 0 al perder, rendirse
nunca mejora el retorno. La puntuación de cada acción sale del vector de su entidad: el movimiento *i* o el Pokémon *j*.

**Táctico, alquiler**: **red puntero autorregresiva**.
1. Elige el Pokémon que sale primero entre los 6 candidatos.
2. Elige la pareja que lo acompaña entre las 10 combinaciones de los 5 restantes, condicionada al primero.

Solo importa el orden de la posición 0, porque en combate los cambios los decide el combatiente. Por eso hay
6·C(5,2) = 60 alquileres distintos. log π = log π(primero) + log π(pareja | primero). Se enmascaran las especies repetidas.

**Táctico, intercambio**: 10 acciones: quedarse, o (posición propia 0–2, posición rival 0–2). Se enmascaran las
especies ya presentes en el equipo. El Pokémon nuevo ocupa la posición del que sale, y la posición 0 es la que sale
primero en el siguiente combate.

## 5. Observaciones

Regla: **solo lo que un jugador podría saber**. Es decir, la pantalla, los mensajes, la memoria perfecta del combate y el
conocimiento del juego. Nunca PS exactos del rival, sets sin revelar, PP del rival, duraciones aleatorias restantes ni el RNG.

**Combatiente**: `BattleView` ampliada. La lista completa de campos está en `docs/GUIDE.md` §7:
- rivales que aún no han salido, sin ningún dato;
- base stats y tipos de los rivales vistos;
- habilidad del rival si se ha anunciado;
- último movimiento de cada bando, si fue crítico, quién actuó primero y el daño hecho y recibido;
- contadores legales: turnos dormido transcurridos, Canto Mortal, pantallas, etc.

Se quitan **IVs, EVs y naturaleza**, porque ya están incluidos en los stats.

**Táctico**: **sin pistas** (`hint_type`/`hint_style`). Recibe:
- contexto de la racha: racha, número de reto, combate dentro del reto y contador de alquileres del juego
  (`factoryRentsCount`, que con más alquileres mejora los Pokémon de alquiler iniciales de los retos siguientes,
  según `battle_factory.c:551`);
- en el alquiler, los 6 candidatos completos;
- en el intercambio, su equipo completo, con un indicador de la posición 0, y el equipo derrotado: especie más lo
  revelado en el combate (movimientos, objeto, habilidad), con el token *desconocido* en lo demás.

**No** se le dan los sets posibles de cada especie. Se descarta `frontier_ids` porque es redundante con el set completo.

## 6. Codificación y red

- **Tablas de embeddings por tipo de entidad**: especie, movimiento, efecto de movimiento, tipo, objeto y habilidad.
  Cada tabla es **compartida** en todos los sitios donde aparece esa entidad. Cada tabla reserva un índice para
  **desconocido**, distinto de **ninguno**.
- Los números van con **escalas fijas sacadas de los límites del juego**:

  | Dato | Escala |
  |---|---|
  | PS propios | hp/max_hp |
  | PS del rival | píxeles/48 |
  | max_hp y stats | /500 |
  | Base stats | /255 |
  | PP | pp/pp_max |
  | Cambios de stats | log del multiplicador real |
  | Contadores | /máximo |
  | Turno | recortado a 50 y /50 |

  Tipos, estado alterado y clima van en one-hot o multi-hot.
- **Arquitectura**:
  1. Codificador de movimiento: embedding + números.
  2. Codificador de Pokémon: especie, tipos, objeto, habilidad, números y movimientos.
  3. **Atención** (transformer sin codificación posicional) sobre los Pokémon + un token de campo/contexto.
  4. Un indicador marca el Pokémon activo.

  El combatiente es simétrico ante permutaciones del banquillo, y la atención lo garantiza por construcción.
- Tamaño: embeddings de 64, capa oculta de 128, 2 capas de atención × 4 cabezas.

## 7. Hiperparámetros de PPO

| | Combatiente | Táctico |
|---|---|---|
| Entornos en paralelo | 64 (compartidos) | ← |
| Transiciones por actualización | 8.192 | 1.024 |
| γ / λ (GAE) | 0,99 / 0,95 | 1 / 0,95 |
| ε (clip) | 0,2 | 0,2 |
| Épocas × minibatches | 4 × 8 | 4 × 4 |
| Tasa de aprendizaje (Adam) | 3·10⁻⁴ | 3·10⁻⁴ |
| Coeficiente de entropía / de valor | 0,01 / 0,5 | 0,01 / 0,5 |

## 8. Evaluación

- **Líneas base**:
  - combatiente **aleatorio**: acciones legales uniformes;
  - combatiente de **máxima potencia × eficacia de tipo**: potencia · STAB · eficacia contra el activo rival; nunca cambia
    por voluntad propia y, tras un debilitamiento, saca al mejor atacante contra el rival.

  El táctico de referencia elige al azar y no intercambia.
- **Métricas**: racha media hasta perder, porcentaje de victorias por combate y porcentaje de retos completados.
  Se miden sobre semillas de evaluación fijas, con la política determinista (argmax) y con la estocástica.
- Para la publicación: varias semillas de entrenamiento, muchas ejecuciones de las líneas base e intervalos de confianza.

## 9. Monitorización

TensorBoard (`runs/`) registra:
- el progreso de la racha;
- las pérdidas de PPO y sus diagnósticos (KL, fracción recortada, varianza explicada, entropía);
- la distribución de acciones;
- la velocidad;
- evaluaciones periódicas contra las líneas base.

---

## 10. Implementación

| Pieza | Archivo |
|---|---|
| Vistas → tensores y máscaras | `rl/encode.py` |
| Red compartida (embeddings, codificadores, atención, cabezas) | `rl/model.py` |
| Entornos en paralelo, recompensas, Φ y truncamiento | `rl/envs.py` |
| PPO (buffers por agente y entorno, GAE, actualización) | `rl/ppo.py` |
| Bucle de entrenamiento conjunto | `rl/train.py` |
| Líneas base | `rl/baselines.py` |
| Evaluación periódica con semillas fijas | `rl/evaluate.py` |

Detalles fijados al implementar:
- **Truncamiento**: tras 300 decisiones del combatiente en un combate (`--max-decisions`). Cuentan también los turnos de
  Forcejeo que el juego juega solo cuando no queda nada elegible.
- **Decisiones sin elección**: si todos los movimientos están a 0 PP y no hay cambio posible, el juego usa Forcejeo.
  El entorno la juega sin consultar al agente, porque no hay nada que decidir.
- **Contador de alquileres**: se lee de `SaveBlock2 + 0xDF4`. El comentario de `global.h` dice 0xDF6: está desplazado 2 bytes.
  Suma 1 por el alquiler de cada reto y 1 por cada intercambio, y se actualiza en la victoria siguiente.
- **Máscara de intercambio**: replica `Swap_AlreadyHasSameSpecies` (`host_factory_screen.c`).
- **Rachas**: todas empiezan en 0 (`win_streak=0`). Empezar más adelante está pendiente de decidir.
- **Comprobado**:
  - el retorno del táctico = racha;
  - la suma del shaping de un combate = victoria − β·Φ(s₀) con γ = 1.
  - memoria: 0,85 M parámetros; pico de GPU de 281 MiB en la actualización del combatiente; 40 MiB de RAM por buffer.

## 11. Cómo lanzarlo y verlo

```bash
tmux new -s factory
python -m rl.train --name ppo_joint_v1                       # ventana 1: entrenamiento
python -m rl.evaluate --run runs/ppo_joint_v1 --watch        # ventana 2: evaluación de cada checkpoint
tensorboard --logdir runs --port 6006                        # ventana 3: http://localhost:6006
```

Qué mirar en TensorBoard:
- `eval_greedy/streak_mean` y `eval_sampled/streak_mean` frente a `eval_baseline_*/streak_mean`. Es la métrica
  principal, con intervalo de confianza del 95 % por bootstrap (`streak_ci95_low/high`).
- `train/streak_mean`, `train/battle_win_rate` y `train/challenge_completed_rate`: el progreso durante el entrenamiento, con exploración.
- `battler/*` y `tactician/*`:
  - `explained_variance`: cerca de 1 si el crítico predice bien;
  - `approx_kl` y `clip_frac`: si crecen mucho, las actualizaciones son demasiado grandes;
  - `entropy`: si cae a 0 muy pronto, el agente deja de explorar.
- `actions/*`: la fracción de cambios del combatiente y de intercambios del táctico.
- `perf/*`: decisiones por segundo, uso de GPU y proporción de tiempo actualizando.

---

## 12. Rainbow DQN (implementado en `rl/rainbow.py` y `rl/train_rainbow.py`; se lanzará cuando termine PPO)

Se entrena igual que PPO para que la comparación aísle el algoritmo: los dos agentes a la vez, con la misma red base,
las mismas observaciones, el mismo Φ y el mismo truncamiento.

- **Las seis piezas**: Double DQN, replay priorizado (α = 0,5; β de 0,4 a 1 a lo largo de 50 M decisiones), red dueling
  (media de A solo sobre las acciones legales), retornos a n pasos (5 para el combatiente, 3 para el táctico),
  distribucional C51 (51 átomos) y noisy nets (σ₀ = 0,5, en las cabezas).
- **Acciones**: una Q por acción completa.
  - Combate: 7.
  - Intercambio: 10.
  - Alquiler: 90 casillas (primero × pareja). Se desenmascaran las 60 válidas, y la cabeza puntero combina los tres Pokémon elegidos.
- **Soportes de C51**:
  - combatiente, [−1, 2];
  - táctico, [0, 60] con γ = 1, recortado. Es la opción C2: mismo objetivo que PPO, con los átomos separados 1,2 victorias.
- **Replay**: 1 M transiciones del combatiente (4,3 GiB en float16) y 100 k del táctico.
- **Reutilización**: cada transición se muestrea 8 veces de media, como en Rainbow de Atari (lote 32 cada 4 pasos).
  Con lote 256, es un paso de gradiente cada 32 transiciones nuevas.
- **Truncamiento**:
  - la cola a n pasos del combatiente hace bootstrapping con la observación de corte;
  - la del táctico se descarta, porque no hay observación suya en ese momento (le ocurre a ~0,1 % de los combates).
- **Resto de parámetros**: tasa de aprendizaje 1·10⁻⁴ (Adam, ε = 1,5·10⁻⁴), red objetivo copiada cada 8.000 pasos de gradiente,
  calentamiento de 20 k transiciones (combatiente) y 2 k (táctico).
- **Comprobado**:
  - los retornos a n pasos coinciden con el cálculo a mano;
  - la proyección de C51 tiene los índices acotados (en GPU, el error de coma flotante los sacaba del soporte);
  - la prueba corta aprende.
- **No correr a la vez que PPO**: decisión del usuario.

## 13. PPO v2: rachas que empiezan en rondas altas (`ppo_joint_v2`, desde cero)

v1 empezaba siempre en la racha 0, así que casi no veía las rondas 3–6. En la evaluación de v1 a 42 M decisiones:
- ≥ 1 ronda: 61 %;
- ≥ 2 rondas: 35 %;
- ≥ 3 rondas: 8,8 %;
- ≥ 6 rondas (racha 42, el reto completo): 0,2 %.

- **Reparto de inicios**: el 50 % de las rachas empieza en la ronda 1 (racha 0). El otro 50 % empieza al inicio de una
  ronda de la 2 a la 6, elegida uniformemente: racha 7k con k de 1 a 5. Tiene que ser un múltiplo de 7, porque el juego
  reinicia el número de combate dentro del reto.
- **No hace falta un guardado**: entre retos solo se conservan la racha y el contador de alquileres, y el equipo se
  alquila de nuevo. Empezar con `reset(win_streak=7k, rents_count=…)` equivale a un guardado en ese punto.
- **Contador de alquileres inicial**: aleatorio y uniforme en **[k, 7k]**, el rango factible.
  - Cada ronda suma 1 por el alquiler, porque el script llama a `factory_setswapped` antes de `factory_rentmons`
    (`BattleFrontier_BattleFactoryPreBattleRoom/scripts.inc:40`).
  - Suma además de 0 a 6 por los intercambios.
  - Comprobado en el simulador: sin intercambiar nunca, el contador sube 1 por ronda.
- **La evaluación principal sigue empezando en la racha 0**, con las mismas semillas que v1.
  Nuevas métricas: `eval_*/reach_round_k` y `eval_*/rounds_completed_mean`.
- **Los checkpoints guardan también el optimizador** desde v2 (`--init-from` permite continuar).

---

## Pendiente (ideas apuntadas, sin decidir)

- **La ronda en one-hot** (propuesta del usuario, 2026-09-24). Hoy entra como el escalar `min(reto, 10)/10` en
  `rl/encode.py::_context`. En one-hot (rondas 1–6, más una casilla para la 7 en adelante), cada ronda tendría su propia
  dirección en la entrada y a la red le llegaría una señal más clara para adaptar la estrategia a la ronda.
  El **combate dentro de la ronda ya va en one-hot de 7**, así que solo falta la ronda.
  Cambiar la codificación modifica el tamaño de la entrada: exige entrenar de nuevo (o al menos reiniciar `ctx_enc`)
  y no es compatible con los checkpoints actuales.
- **¿Redes separadas o compartidas para táctico y combatiente?** (2026-09-24). Hoy comparten embeddings, codificadores y
  transformer; solo las cabezas son distintas. Hay un indicio de interferencia:
  - la pérdida de valor del táctico es ~300× la del combatiente (8,6 frente a 0,023 en v1), porque sus retornos son
    rachas (~7) y no probabilidades (~0,7);
  - su norma de gradiente es ~10× mayor (5,9 frente a 0,6);
  - con el recorte a 0,5, las actualizaciones del táctico mueven el tronco común casi solo para ajustar su crítico.

  Experimento propuesto, con varias semillas cada uno:
  - (a) compartido, como ahora;
  - (b) compartido, con los retornos del táctico normalizados;
  - (c) solo los embeddings compartidos;
  - (d) todo separado.

  Métricas: evaluación desde racha 0, rondas alcanzadas y varianza explicada de cada crítico.

---

## 14. PPO v3 (`ppo_joint_v3`, desde cero, 2026-09-25)

Motivos: en el informe de 2 h de v2 (§13), v2 no cambiaba nada en la ronda 1 y mejoraba poco en las rondas 4 y 6.
Además, el análisis de las derrotas mostró tres cosas:
- Noland gana el 44 % de sus combates, frente al 15 % de los rivales normales de la ronda 3;
- los combates 1–2 de cada ronda son los más difíciles, porque el agente todavía juega con el alquiler recién elegido;
- la ronda 6 es un muro: 66–70 % de victorias en todas las posiciones, contra legendarios.

v3 mantiene todo lo de v2 y cambia lo siguiente:

- **Fallo del entorno corregido: el símbolo de plata.** Las rachas que empiezan en la racha 21 o más lo reciben
  (`FLAG_SYS_FACTORY_SILVER`, con el nuevo `Gen3Game.set_flag`). Sin él, el juego no programa a Noland en el combate 42
  (`GetFrontierBrainStatus`), y un jugador con esa racha lo tiene necesariamente. v2 nunca vio ese combate, y su
  evaluación de la ronda 6 era más fácil que la real.
- **Recompensa del combatiente: solo ganar** (decisión del usuario: da igual perder Pokémon si al final se gana).
  - El shaping con Φ se **retira gradualmente**: β baja de 0,5 a 0 de forma lineal durante los primeros 20 M decisiones.
    Hace de andamio al principio y luego desaparece. Se retira poco a poco para que el crítico se adapte.
  - **γ del combatiente = 1**, para que no se prefiera ganar rápido. Los combates son cortos y el truncamiento
    garantiza que terminan.
- **Solo se comparten los embeddings.** El táctico tiene sus propios codificadores y su propio transformer (`trunk_t`).
  La red pasa de 0,85 M a 1,29 M parámetros.
- **Normalización del valor** para los dos agentes. El crítico predice el retorno normalizado con una media y una
  desviación móviles (`ValueNorm`). Así las pérdidas de valor tienen escalas comparables (antes, ~0,03 frente a ~8).
- **La ronda en one-hot** (7 casillas: rondas 1–6, y 7 en adelante). Es la codificación versión 3; `--encode-version 2`
  reproduce v1 y v2.
- **Estimaciones de daño y velocidad** para el combatiente (`rl/damage.py`). Aproximadas y **sin usar los sets de la
  Factory**, porque un jugador no se los sabe de memoria:
  - Se aplica la fórmula de daño de la tercera generación: STAB, eficacia, stats y sus etapas, quemadura,
    Reflejo/Pantalla Luz, clima, objetos y habilidades conocidos, movimientos de daño fijo o de varios golpes.
  - Los stats ocultos del rival se acotan con las stats base de la especie, nivel 100 y los IVs fijos de la ronda,
    con cualquier EV entre 0 y 252 y cualquier naturaleza (0,9–1,1). El rango se amplía con la tirada aleatoria (85–100 %).
  - Por cada movimiento propio: daño [mín., máx.] al rival activo en fracción de sus PS, y si puede debilitarlo ya.
  - Por cada movimiento revelado del rival: lo mismo contra nuestro activo.
  - Por cada Pokémon propio: la peor amenaza revelada (fracción y posibilidad de KO) y si es más rápido seguro o quizás.
  - **Validado** contra el daño real del juego: el 95,2 % de 8.876 golpes nuestros y el 92,4 % de 1.988 golpes rivales
    caen dentro del rango. Los que quedan fuera suelen quedar por encima, lo que encaja con críticos que la vista no detecta.
- **Evaluación** (`rl/evaluate.py`): además de la evaluación desde la racha 0, cada checkpoint se evalúa **por ronda**
  (`eval_round_k/complete`, `eval_round_k/battle_win_rate`, 192 rachas por ronda). Esta métrica es la que se puede
  comparar entre versiones entrenadas con repartos de inicio distintos.

Comando:

```bash
python -m rl.train --name ppo_joint_v3 --gamma-b 1.0 --start-p0 0.5 --start-max-round 5 --share embeddings \
    --value-norm 1 --beta 0.5 --beta-anneal-steps 20e6 --encode-version 3
```
