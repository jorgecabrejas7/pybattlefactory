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
  revelado en el combate (movimientos, objeto, habilidad), con el token *desconocido* en lo demás;
- desde la codificación v4, por cada rival derrotado, cómo de difícil fue en el combate (§17).

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
- **Rachas**: v1 empezaba siempre en 0 (`win_streak=0`); desde v2 hay inicios en rondas altas (§13).
- **Comprobado**:
  - el retorno del táctico = racha;
  - la suma del shaping de un combate = victoria − β·Φ(s₀) con γ = 1.
  - memoria: 0,85 M parámetros; pico de GPU de 281 MiB en la actualización del combatiente; 40 MiB de RAM por buffer.

## 11. Cómo lanzarlo y verlo

Las opciones por defecto de `rl.train` son las de v3 (§14). Para reproducir v1:
`--gamma-b 0.99 --start-p0 1 --share all --value-norm 0 --beta-anneal-steps 0 --encode-version 2`.

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

Se entrena igual que PPO v3 (§14) para que la comparación aísle el algoritmo: los dos agentes a la vez, con la misma
red base (el táctico con su propio tronco, `--share embeddings`), las mismas observaciones, el mismo Φ retirado
gradualmente (`--beta-anneal-steps`), γ del combatiente = 1, los mismos inicios de racha (`--start-p0 0.5`) y el
mismo truncamiento. (Hasta 2026-09-25 se entrenaba como v1: γ 0,99, β fijo, inicios en 0 y tronco compartido.)

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
- **Ruido de la red objetivo**: se sortea de nuevo en cada paso de aprendizaje (`target.reset_noise()`), como el de la
  red online al actuar. Antes se quedaba la muestra copiada en la última sincronización durante 8.000 pasos.
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

---

## 15. Búsqueda MCTS para el combatiente (`rl/search.py` + C++)

Motivo: PPO explora con ruido independiente en cada turno, así que casi nunca prueba planes de varios turnos.
La búsqueda los evalúa de forma explícita. Antes se comprobó con una regla forzada (`rl/forced_rules.py`) que
**forzar Doble Equipo hasta +2 empeora** (−4 a −8 puntos de victoria en los combates donde actúa, rondas 2–5) y que
Tóxico forzado no cambia nada. La red acertaba al no usarlos a ciegas.

- **Algoritmo:** MCTS determinizado en conjunto.
  - K árboles (por defecto 8), cada uno con la información oculta sorteada y una semilla de azar fijas.
  - PUCT con pérdida virtual, lotes de 32 hojas y N simulaciones repartidas entre los árboles.
  - Las hojas se evalúan con la red PPO: priors de la política y valor del crítico desnormalizado ≈ P(ganar).
  - Las hojas terminales valen 1 o 0; en 300 decisiones se trunca y se usa el valor de la red.
  - Decisión = argmax de las visitas sumadas en la raíz.
- **Modos:**
  - **legal** (el que cuenta): la información oculta se sortea **solo con lo que sabe un jugador real**
    (`rl/determinize.py`; criterio estricto decidido por el usuario el 2026-09-25: ninguna decisión puede tener en
    cuenta IVs, EVs ni nada que un jugador no sepa). No se usan la lista de sets de la Factory, el grupo de la ronda,
    los IVs fijos del juego ni el estado real:
    - especie vista: la suya; no vista: cualquiera de la lista de especies de la Frontera (sin repetir en el equipo);
    - movimientos: los revelados, y el resto al azar entre los que la especie puede aprender a su nivel (nivel,
      MT/MO, tutor, huevo y preevoluciones: `pybattle/data/learnsets.json`, `scripts/extract_learnsets.py`);
    - objeto: el revelado, o uno al azar entre los que tienen efecto en combate (los específicos solo para su
      especie), sin repetir en el equipo; uno consumido o quitado sigue sin estar;
    - IVs 0–31 por stat, EVs aleatorios hasta 510 (≤ 252 por stat) y naturaleza al azar;
    - habilidad: la anunciada, o 50/50 (un Pokémon visto que no anunció Intimidación no la tiene);
    - PS exactos dentro de la barra, con los PS máximos del sorteo;
    - contadores ocultos sorteados con lo que el jugador contó: sueño y confusión (los dos bandos), Atadura,
      Alboroto, Enfado, Anulación y Otra Vez (turnos restantes compatibles con los transcurridos), PS del Sustituto
      rival, daño pendiente de Premonición / Deseo Oculto (recalculado con los Pokémon actuales) y el bloqueo de
      Cinta Elegida del rival (según el objeto sorteado).
  - **perfect**: el estado real. **Solo es un techo de referencia; está prohibido en el entrenamiento**:
    `mark_training()` en `rl/train*.py`, `SearchBattler.mode` de solo lectura y comprobado al preparar raíces, y el
    `Searcher` C++ rechaza raíces que no sean determinizaciones completas cuando `PYB_TRAINING` está activo.
  - **En los dos modos**, cada raíz vuelve a sortear el azar del turno (`Gen3Game.redraw_turn`): semilla del RNG,
    la tirada de Garra Rápida del turno (`gRandomTurnNumber`) y **la elección del rival para este turno**, que su IA
    ya ha hecho mientras el jugador decide: se deshace y la IA vuelve a elegir desde el estado de la raíz. En un
    cambio forzado con los dos debilitados, se vuelve a elegir también su sustituto.
- **Implementación:**
  - C++ (`src/gen3/`):
    - `sim_step`, `determinize`, `set_rng`;
    - `MctsTree`;
    - un observador y codificador v3 **idénticos bit a bit** a los de Python (`ObsMemory`: 19.279 comparaciones, 0 diferencias);
    - el bucle `Searcher`, que llama a la red en Python una vez por lote.
  - Servidor de inferencia en GPU opcional (`rl/inference.py`): agrupa las hojas de todos los procesos.
  - Las implementaciones Python y C++ dan las **mismas visitas y la misma acción** con las mismas semillas.
- **Velocidad** (máquina libre, 18 procesos, ronda 3, 256 simulaciones, K = 8; las tres versiones dan resultados idénticos):

  | Implementación | ms por decisión | vs Python |
  |---|---|---|
  | Python, red en CPU | 176,8 | 1× |
  | C++, red en CPU | 47,8 | 3,7× |
  | C++, servidor de red en GPU | 28,1 (25,2 con lotes de 128) | 6,3–7× |
  | C++ + GPU con 64 simulaciones | 14,3 | — |

  Lo que queda es la preparación de las raíces en Python (sorteo de K determinizaciones, `from_python`, raíz;
  ~9 ms) y la latencia de ida y vuelta al servidor.
- **Resultados** (checkpoint final de v3, evaluación por ronda con las mismas semillas y un entorno nuevo por racha;
  P(completar la ronda) en %; 608 rachas por ronda salvo 64 y 1.024, con 320):

  | | R1 | R2 | R3 | R4 | R5 | R6 | Producto ≈ P(6 rondas) |
  |---|---|---|---|---|---|---|---|
  | Red sola | 68,8 | 70,2 | 38,0 | 57,6 | 21,9 | 18,6 | 0,43 % |
  | MCTS legal 64 | 73,8 | 69,7 | 39,7 | 58,8 | 19,4 | 20,0 | 0,46 % |
  | MCTS legal 256 | 71,4 | 74,3 | 43,6 | 64,3 | 26,5 | 20,7 | 0,82 % |
  | MCTS legal 1.024 | 72,2 | 74,4 | 48,8 | 67,8 | 24,4 | 27,2 | 1,18 % |
  | Información perfecta 256 (techo) | 72,9 | 72,7 | 39,8 | 65,8 | 26,6 | 23,2 | 0,86 % |

  **⚠ Resultados invalidados (revisión de código, 2026-09-25).** Cuando el jugador decide, la IA rival ya ha elegido
  su acción del turno (`gChosenActionByBattler[1]`, `gChosenMoveByBattler[1]`…). El clon + `determinize` + `set_rng`
  no la borraba, así que **la búsqueda, también en modo legal, conocía de antemano la acción del rival**:
  - la copia cambia de Pokémon el 94 % de las veces si el rival real iba a cambiar, y el 0 % si no;
  - el movimiento coincide el 96 % de las veces.

  Por eso legal ≈ perfect. Hay que repetir la tabla tras corregirlo. El entrenamiento PPO no se ve afectado,
  porque no usa búsqueda. **Corregido** el 2026-09-25 (`redraw_turn`, ver Modos); además la determinización legal ya
  no usa la lista de sets de la Factory, así que la tabla nueva no es comparable con esta.

  Lectura original (pendiente de confirmar):
  - la búsqueda mejora todas las rondas con 256 simulaciones (+2 a +7 puntos) y más aún con 1.024 (hasta +11);
  - con 64 simulaciones apenas aporta;
  - conocer la información oculta no mejora sobre el modo legal: el límite está en la profundidad de la búsqueda y en
    la calidad de la red, no en la información oculta.
- **Limitaciones conocidas** (lo que queda sin sortear o simplificado; también en `src/gen3/search_host.c`):
  - no se usa la evidencia negativa (p. ej., no ver Restos tras recibir daño no descarta Restos);
  - el sueño de Descanso (3 turnos fijos) no se distingue del aleatorio;
  - el daño acumulado de Venganza (`gBideDmg`) del rival y los registros de daño del turno de Contraataque / Manto
    Espejo no se sortean;
  - la amistad es 0 (como la construye la Frontera; solo afecta a Retribución / Frustración);
  - 26 de los 3.528 movimientos de los sets de la Factory solo se obtienen por intercambio con otros juegos y no
    están en los learnsets (Smeargle puede tener cualquiera);
  - el recuento de turnos transcurridos del observador coincide con el contador real en el 99,7 % de los casos
    medidos; en el resto el sorteo se queda con el valor más bajo posible.
- **Reproducibilidad**: el simulador es determinista (los locales C se inicializan a cero,
  `-ftrivial-auto-var-init=zero`; antes dos clones del mismo estado podían divergir en ~18 % de los pasos según lo que
  el proceso hubiera ejecutado antes). `FactoryEnv.reset()` no arrastra estado del juego; lo que hacía irreproducible
  la evaluación por ronda era el consumo doble de semillas tras una derrota, ya corregido en `eval_round`.

Pendiente (se suman a la lista de abajo):
- ~~**Codificación v4**~~ (decisiones del usuario, 2026-09-25): **hecho**, ver §17. v3 queda igual, bit a bit
  (Python y C++), para que los checkpoints v3 sigan funcionando.
- ~~**Entrenar con búsqueda** (expert iteration, solo en modo legal).~~ Implementado: alphazero_v1 (§18).
- **KL del táctico** (revisión 2026-09-25): sus lotes mezclan transiciones de políticas anteriores (quedan pendientes
  hasta su siguiente decisión, y las actualizaciones del combatiente mueven los embeddings compartidos), de ahí los
  picos de approx_kl. No es un fallo de cálculo. `ppo_update` registra ahora `stale_kl` / `stale_clip_frac` (antes de
  ningún paso) y `rl.train` acepta `--target-kl-t` (parada temprana) y `--refresh-logp-t 1` (recalcular los log-prob
  viejos al empezar la actualización). Por defecto nada cambia: **decidir** si se activan.

---

## 16. Prueba de capacidad de la red (`rl/capacity_test.py`, 2026-09-25)

Datos: 1,49 M decisiones del combatiente y 117 k del táctico, jugadas con v3. Se entrenó **solo el crítico, desde
cero**, sobre esos datos fijos, con la red actual (1,3 M parámetros) y una ancha (d = 256, 3 capas, 5,9 M).
La validación se hace con **combates y rachas completos que no se ven en el entrenamiento**.

La primera versión separaba por decisión. Daba 0,84 frente a 0,91 de varianza explicada, pero era **memorización**:
las decisiones de un mismo combate comparten el mismo resultado, así que la validación contenía combates ya vistos.

Varianza explicada en entrenamiento / en validación (por combate o racha):

| | Época 1 | Época 4 | Época 12 |
|---|---|---|---|
| Combatiente, red actual | 0,48 / **0,21** | 0,73 / 0,02 | 0,91 / −0,12 |
| Combatiente, red ancha | 0,48 / 0,18 | 0,78 / 0,00 | 0,96 / −0,14 |
| Táctico, red actual | 0,11 / 0,10 | 0,14 / **0,13** | 0,22 / 0,10 |
| Táctico, red ancha | 0,11 / 0,11 | 0,14 / 0,13 | 0,19 / 0,11 |

Conclusión:
- **La red actual no se queda corta.** La ancha no generaliza mejor, y ambas memorizan en cuanto se entrena más.
- El límite es el **ruido del objetivo**: un solo resultado por combate o racha. Lo generalizable de él cabe en la red
  actual.

**Decisión** (regla del usuario: agrandar solo si la prueba sale positiva): **alphazero_v1 mantiene el tamaño actual**
(d = 128, 2 capas), con tasa de aprendizaje *cosine*.

**Implicación para AlphaZero:** el objetivo de valor *z* (resultado de un combate) es muy ruidoso. Conviene mezclarlo
con el valor de la raíz de la búsqueda, que es menos ruidoso, y vigilar el sobreajuste a los datos del buffer:
reutilizar pocas veces cada muestra y comprobar el crítico con combates de validación.

---

## 15b. Comparación de la búsqueda, repetida tras las correcciones (2026-09-25)

Checkpoint final de v3. Evaluación por ronda con las mismas semillas, un entorno nuevo por racha y la búsqueda ya
corregida:
- redibuja el turno del rival (sin fuga de su acción);
- usa la determinización estricta de jugador.

Hojas evaluadas con el servidor de GPU. P(completar la ronda) en %.

| | R1 | R2 | R3 | R4 | R5 | R6 | ≈ P(6 rondas) | ms por decisión |
|---|---|---|---|---|---|---|---|---|
| Red sola (608 rachas/ronda) | 68,8 | 70,2 | 38,8 | 57,6 | 21,9 | 19,4 | 0,46 % | 1,4 |
| MCTS legal 256 (608) | 71,2 | 72,0 | 39,3 | 59,4 | 23,5 | 23,2 | 0,65 % | 24,9 |
| MCTS legal 1.024 (320) | 72,2 | 71,9 | 42,8 | 59,4 | 23,4 | 21,6 | 0,67 % | 78,5 |
| Información perfecta 256 (608) | 70,6 | 72,5 | 42,1 | 60,2 | 24,3 | 24,5 | 0,77 % | 25,5 |

Lectura:
- **La búsqueda legal sigue mejorando en las 6 rondas**, pero **mucho menos** que en la tabla contaminada: +0,5 a +4
  puntos por ronda. Ninguna ronda es significativa por separado (z < 2), pero el signo es positivo en las 6
  (p ≈ 0,016 en una prueba de signos; z combinado ≈ 1,9).
- Buena parte de la mejora anterior venía de conocer la acción del rival.
- **1.024 simulaciones no mejoran sobre 256.** Profundizar más no ayuda; el límite está en la **calidad del valor y la
  política de la red** que guían la búsqueda. Es justo lo que mejora *expert iteration*.
- La información perfecta ayuda algo (0,77 % frente a 0,65 %; es la variante más clara, z combinado ≈ 3,2). Esa es la
  pérdida por no conocer lo oculto con el sorteo estricto.
- Hubo 13 errores de búsqueda en ~100 k decisiones (se juega la acción de la red): están por investigar.

## 17. Codificación v4 (`--encode-version 4`, 2026-09-25)

Motivo: la regla del usuario (ninguna decisión puede usar IVs, EVs ni nada que un jugador real no sepa) y dos
fallos de las estimaciones de v3. Además, el táctico recibe cómo de difícil fue cada rival derrotado.
**v2 y v3 no cambian** (Python idéntico al de antes bit a bit, comprobado con el código anterior en 1.571 decisiones
de combate, 40 alquileres y 25 intercambios; C++ v3 idéntico a Python v3), así que los checkpoints v3 cargan y
codifican igual.

- **Estimaciones de daño y velocidad sin los IVs del rival** (`rl/damage.py`, `version=4`). Los stats ocultos
  del rival se acotan con **cualquier IV de 0 a 31**, EVs 0–252 y naturaleza 0,9–1,1, al **nivel que se ve** (antes:
  nivel 100 y la tabla de IVs fijos de la ronda, que el jugador no conoce):
  - stat: mín. = ⌊(⌊2·base·nivel/100⌋ + 5)·0,9⌋, máx. = ⌊(⌊(2·base + 31 + 63)·nivel/100⌋ + 5)·1,1⌋;
  - PS: mín. = ⌊2·base·nivel/100⌋ + nivel + 10, máx. = ⌊(2·base + 31 + 63)·nivel/100⌋ + nivel + 10.

  El rango contiene el de cualquier IV fijo (test). Ya no hace falta el contexto de la racha para calcularlas.
- **Objetos de especie corregidos**: Hueso Grueso ×2 al Ataque de **Cubone y Marowak**, Bola Luminosa ×2 al At. Esp.
  de **Pikachu** (en v3, `SP_CUBONE, SP_MAROWAK, SP_PIKACHU` salen en orden de id: Pikachu, Cubone, Marowak; se
  mantiene así en v3).
- **Dificultad de cada rival derrotado** (para el táctico, en el intercambio). El `BattleObserver` la acumula durante
  el combate en un `FoeRecord` por posición del equipo rival, y `SwapView.defeated` la entrega (los dos backends
  construyen la vista con el mismo observador). Cada turno se atribuye al rival que estaba en el campo (al que entra,
  si cambió voluntariamente). Solo lo que ve el jugador: lo nuestro exacto (PS, estados, debilitados); lo del rival,
  solo por la pantalla (barra de PS, cambios de stats anunciados).

  | Número (token rival del intercambio) | Escala |
  |---|---|
  | PS que perdió nuestro equipo en sus turnos (por cualquier causa: golpes, estado, clima, retroceso) | / PS máx. totales del equipo, recortado a 1 |
  | Pokémon nuestros debilitados en sus turnos | /3 |
  | Turnos en el campo | recortado a 20, /20 |
  | Turnos en que un ataque nuestro le bajó la barra de PS («golpes necesarios») | recortado a 10, /10 |
  | Máxima suma de sus etapas de stats positivas | recortado a 12, /12 |
  | Si alguno de los nuestros recibió un estado alterado en sus turnos (no cuenta nuestro Descanso) | 0/1 |

  - Se añaden 6 números al final de `mon_num` de **todos** los tokens (ceros en los tokens propios, en el alquiler y
    en el combate; también en un `SwapView` sin registros).
  - El último turno del combate no pasa por ninguna decisión: `BattleObserver.finish(ram)` lo cuenta en el momento en
    que el juego decide el combate (`gBattleOutcome`). `SimBackend` ejecuta ahora el combate con `Gen3Game.run` hasta
    ese momento, llama a `finish` y después `factory_run_battle` cierra el combate como antes (los mismos frames).
    `EmuBackend` llama a `finish` cuando `advance_factory` detecta el final (hasta 8 frames después, mientras sale el
    mensaje de victoria; el test comprueba que da lo mismo).
  - Los golpes de varios turnos seguidos sin decisión (Enfado, etc.) cuentan como uno.
- **Tamaños v4**: `MON_IDS` = 19, **`MON_NUM` = 112** (v3: 106, v2: 102), `MOVE_NUM` = 17, `CTX_IDS` = 2,
  `CTX_NUM` = 99 (`FIELD_NUM` = 76 + `CONTEXT_NUM` = 23). Solo cambia el codificador de Pokémon (`mon_enc`): hay que
  entrenar desde cero (o reiniciar `mon_enc`).
- **C++** (`ObsMemory`, `Searcher`): codifica v3 y v4 con un interruptor global en tiempo de ejecución
  (`pybattle_native.set_encode_version`; `rl.encode.set_version` lo llama). `EncodedObs` reserva sitio para la
  disposición más grande y guarda los anchos de cada observación (`mon_w`, `move_w`, `ctx_w`), así que el `Searcher`
  con el observador Python acepta cualquier versión. El intercambio (y sus registros) solo existe en Python. No hace
  falta regenerar tablas (los ids de especie ya estaban en `observer_tables.h`).
- **Validación**:
  - C++ v4 = Python v4 **bit a bit**: 14.289 codificaciones en rachas completas (rondas 1–8, Noland, cambios
    forzados, `from_python`) y 5.598 en caminos de búsqueda (determinización, rebase, `sim_step`); lo mismo para v3;
  - los registros suman lo que pasó (138 intercambios): PS perdidos = PS perdidos por el equipo (exacto en los 102 sin
    curación visible; ≥ con curación), debilitados = nuestros Pokémon a 0 PS, turnos ≥ decisiones del combate,
    mejoras ≥ las vistas;
  - emulador = simulador: los registros coinciden en 12 combates grabados en la ROM (también leyendo el final 16 frames
    más tarde).

Comando (igual que v3, con la codificación nueva):

```bash
python -m rl.train --name ppo_joint_v4 --gamma-b 1.0 --start-p0 0.5 --start-max-round 5 --share embeddings \
    --value-norm 1 --beta 0.5 --beta-anneal-steps 20e6 --encode-version 4
```

---

## 18. alphazero_v1: expert iteration desde cero para los dos agentes (`rl/alphazero.py`, `rl/tactician_search.py`)

Decisiones del usuario: **los dos agentes desde cero**, búsqueda del combatiente **solo en modo legal**, tamaño de red
de §16 (d = 128, 2 capas, `share="embeddings"`), tasa de aprendizaje *cosine*.

**Iteración** (se repite `--iterations` veces):
1. **Autojuego** con una copia congelada de la red: 18 procesos (`--workers`) juegan rachas. La red del combatiente
   corre en el servidor de inferencia de la GPU (`rl/inference.py`), que ahora admite `reload()`: carga los pesos
   nuevos sin reiniciarse, copiándolos en su sitio para que los grafos CUDA sigan siendo válidos. El táctico usa la
   copia en CPU de cada proceso. Las rachas siguen de una iteración a la siguiente: una muestra se completa cuando
   termina su combate (combatiente) o su racha (táctico).
2. **Entrenamiento** con un buffer de repetición de las últimas `--window` = 4 iteraciones.
3. **Evaluación por ronda** y **checkpoint**.

**Combatiente**:
- Decide el MCTS en C++ de §15, siempre en modo legal (`AZBattler` no tiene parámetro de modo): K = 8
  determinizaciones, `redraw_turn` en cada raíz y 128 simulaciones (`--sims`).
- **Ruido de Dirichlet** en los priors de la raíz (α = 0,3, peso 0,25), añadido en Python antes de la búsqueda.
- La acción se muestrea de las visitas con temperatura 1 (`--temperature`).
- **Objetivos**:
  - política: la distribución de visitas de la raíz;
  - valor: ½ z + ½ q, donde z es el resultado del combate (1/0) y q el valor de la raíz (media de Q ponderada por
    visitas). La mezcla se cambia con `--value-mix-b`.
- Si hay una sola acción legal no se busca: la política es one-hot y el valor, solo z.

**Táctico: búsqueda por simulación** (`rl/tactician_search.py`):
- Cada opción (alquileres legales (primero, pareja), hasta 60; intercambios, ≤ 10) se valora con combates simulados
  en juegos clonados:
  1. se aplica la opción (`factory_rent` / `factory_swap`, que ya rellenan `gEnemyParty` con el rival real);
  2. **antes del primer frame del combate se sustituye el equipo rival entero** por uno sorteado con el criterio
     estricto de §15, con los 3 Pokémon sin ver: especie uniforme entre las de la Frontera, movimientos aprendibles,
     objeto con efecto, IVs/EVs/naturaleza/habilidad al azar y PS completos. Como plantilla se usa una copia de
     nuestro equipo (nivel e id de entrenador), y el RNG se vuelve a sembrar con el de la búsqueda.
     **Nada del rival real** (RAM, pista del encargado, posición del RNG) llega a la simulación: hay un test que cambia
     el rival real (`gFrontierTempParty`) y comprueba que la búsqueda da exactamente lo mismo;
  3. el combate lo juega la red congelada del combatiente, de forma voraz y sin búsqueda. Todos los combates
     simulados avanzan a la vez, con una llamada por lotes a la red en cada paso. Los lotes pequeños (≤ 8) van a la
     copia en CPU, porque una ida y vuelta al servidor cuesta más;
  4. valor de una simulación, en las unidades del retorno del táctico (victorias hasta el final de la racha, γ = 1):
     0 si pierde; si gana, 1 + V_t(siguiente decisión del táctico) según la red de valor del táctico
     (`--t-bootstrap 1`; la pantalla de intercambio simulada solo muestra la especie del equipo derrotado), o solo 1
     (`--t-bootstrap 0`). Se corta a las 100 decisiones (`--t-max-decisions`), y entonces vale lo que estime la red.
- **Reparto**: Gumbel top-m + *sequential halving* (Danihelka et al., 2022). Se simulan las m = 16 opciones con mayor
  log π + ruido de Gumbel. Con 256 combates por decisión (`--t-budget`) quedan 4 rondas: 4 combates por opción, la
  mejor mitad pasa con 8, luego 16 y luego 32. La elegida es la que sobrevive al final.
- **Objetivos**:
  - política: la distribución de visitas del *sequential halving* (`--t-target visits`), o softmax(Q/T)
    (`--t-target softmax`, T = 0,5 victorias);
  - valor: ½ z + ½ q. Aquí z son las victorias desde la decisión hasta el final de la racha y q, el valor de la
    opción elegida (`--value-mix-t`).
- El alquiler se entrena con el **objetivo conjunto**: −Σ π(l, p) [log q(l) + log q(p | l)], es decir, el marginal del
  primero más el condicional de la pareja. Para ello se añadió `FactoryNet.rental_joint`, que da las 6 filas de parejas
  a la vez.

**Pérdidas y optimización**:
- Entropía cruzada con la política de la búsqueda, más el MSE del valor normalizado con `ValueNorm` (una
  actualización por iteración con todos los objetivos del buffer, ritmo 0,3), más *weight decay* (AdamW, 1·10⁻⁴).
- Tasa de aprendizaje 3·10⁻⁴ con coseno sobre las iteraciones planeadas (suelo del 5 %). Recorte del gradiente a 1.
- Cada muestra nueva se usa `--reuse` = 4 veces de media (con el buffer lleno):
  - lote de 512 decisiones del combatiente por paso;
  - alquileres e intercambios comparten un lote de al menos 16, repartido según cuántos llegan.
- El **5 %** de los combates y de las rachas (`--holdout`) **no se entrena**. Sirve para medir la varianza explicada
  de cada crítico en datos no vistos (`*/explained_variance_heldout`, y `_z` contra el resultado solo): es lo que
  recomendaba §16.

**Currículo**: el 30 % de las rachas empieza en la ronda 1. El resto, en las rondas 2–6, con peso proporcional a
1 − P(completar la ronda k), según la última evaluación por ronda (uniforme al principio). `SimBackend.reset` pone los
símbolos de plata y oro según la racha de inicio.

**Evaluación y registros**:
- En cada iteración, evaluación por ronda con la red sola y voraz (192 rachas por ronda, las mismas semillas que
  `eval_round`): `eval_round_k/*`. Con `--eval-search-sims 64` se evalúa también con búsqueda:
  `eval_search_round_k/*`.
- TensorBoard (eje x: decisiones del combatiente de autojuego):
  - pérdidas, entropía, KL respecto a la política de la búsqueda y varianza explicada (entrenamiento y validación):
    `battler/*` y `tactician_rental|swap/*`;
  - velocidad de autojuego: `perf/*`;
  - búsqueda: `search/*` (simulaciones, profundidad media y máxima, Q de la raíz, entropía de las visitas, cambios
    de la acción respecto al prior);
  - búsqueda del táctico: `tsearch/*`;
  - currículo: `curriculum/*`.
- Checkpoints en cada iteración (`ckpt_<decisiones>.pt` y `latest.pt`, escritura atómica). Guardan la red, el
  optimizador, `value_norm` y `args` con `algo="alphazero"`. `rl.policy`, `rl.evaluate`, `rl.eval_rounds` y
  `rl.full_report` los leen como los de PPO; `full_report` tiene `--grid` para la rejilla de pasos.
- Profundidad de la búsqueda: `Searcher.search` devuelve ahora `max_depth` y `mean_depth`.

Comando:

```bash
python -m rl.alphazero --name alphazero_v1        # valores por defecto = los de arriba
python -m rl.az_bench --workers 18 --decisions 200  # velocidad de autojuego
```

**Pendiente de decidir** (elegido provisionalmente al implementar):
- El rival de las simulaciones del táctico sale del sorteo estricto (sets aleatorios), así que es **mucho más débil**
  que un rival real de la Factory. Al inicio de la ronda 3 (256 combates simulados), la red v3 gana el 98 % de esos
  combates y una red sin entrenar, el 79 %; contra los rivales reales de esa ronda, v3 completa solo el 38 % de las
  rondas. Por eso los Q del táctico son optimistas y distinguen poco entre opciones.
  Alternativas, si el usuario las autoriza: sortear los sets de la lista de la Factory (conocimiento de un jugador
  experto, excluido por la regla estricta), o un rival sorteado más fuerte (p. ej., los mejores movimientos de la
  especie).
- El valor de una simulación del táctico usa el *bootstrap* con V_t, para que Q y z estén en las mismas unidades.
  Sin él (`--t-bootstrap 0`), Q sería P(ganar el siguiente combate) y no se podría mezclar con z.
- Objetivo de política del táctico: visitas del *halving* frente a softmax(Q/T).
- Tamaño de la iteración: 40.000 decisiones del combatiente (`--decisions-per-iter`) y 200 iteraciones planeadas.
- La evaluación por ronda en cada iteración (192 × 6 rachas) para el autojuego mientras dura.
