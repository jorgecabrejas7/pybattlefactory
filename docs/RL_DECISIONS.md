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
  determinizaciones, `redraw_turn` en cada raíz y 128 simulaciones (`--sims`). El rival de cada determinización se
  sortea con `--opponent-prior` (por defecto `factory_sets`, ver «Rival de las búsquedas de entrenamiento» abajo).
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
  2. **antes del primer frame del combate se sustituye el equipo rival entero** por uno sorteado con
     `--opponent-prior`: por defecto de la lista de sets de la Factory (`factory_sets`, abajo); con `strict`, con el
     criterio estricto de §15 y los 3 Pokémon sin ver (especie uniforme entre las de la Frontera, movimientos
     aprendibles, objeto con efecto, IVs/EVs/naturaleza/habilidad al azar y PS completos). Como plantilla se usa una copia de
     nuestro equipo (nivel e id de entrenador), y el RNG se vuelve a sembrar con el de la búsqueda.
     **Nada del rival real** (RAM, pista del encargado, posición del RNG) llega a la simulación: hay un test que cambia
     el rival real (`gFrontierTempParty`) y comprueba que la búsqueda da exactamente lo mismo;
  3. el combate lo juega la red congelada del combatiente, de forma voraz y sin búsqueda. Todos los combates
     simulados avanzan a la vez, con una llamada por lotes a la red en cada paso. Los lotes pequeños (≤ 8) van a la
     copia en CPU, porque una ida y vuelta al servidor cuesta más;
  4. valor de una simulación, en las unidades del retorno del táctico (victorias hasta el final de la racha, γ = 1):
     0 si pierde; si gana, 1 + V_t(siguiente decisión del táctico) según la red de valor del táctico
     (`--t-bootstrap 1`; la pantalla de intercambio simulada muestra el equipo derrotado tal como lo vio el jugador
     en el combate simulado, ver «Pantalla de intercambio simulada» abajo), o solo 1
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

### Rival de las búsquedas de entrenamiento: la lista de sets de la Factory (decidido por el usuario, 2026-09-25)

Definiciones, de abajo arriba:
- Un **set** es una ficha fija de la Battle Factory: especie, 4 movimientos, objeto equipado, naturaleza y reparto de
  EVs. El juego tiene 882 (`gBattleFrontierMons`). Cuando genera un rival no inventa nada: elige sets de esa lista.
- El **pool** de una ronda es el rango de sets del que el juego sortea los rivales de esa ronda
  (`sInitialRentalMonRanges`: 96 sets en las rondas 1–4 a nivel abierto, 510 desde la 5; a nivel 50 sin los de nivel
  alto; nunca Unown). Reglas visibles al generar un equipo: nada de especies repetidas, nada de objetos repetidos, y
  ninguna especie de las que había en pantalla cuando se generó (los 6 alquileres, o nuestro equipo más el equipo
  derrotado en la pantalla de intercambio). Noland (el jefe) sortea del mismo pool pero excluye *sets*, no especies
  (los de nuestro equipo tras el intercambio y los del equipo derrotado), y lo hace al empezar el combate.
- Una **determinización** (§15) es un mundo completo sorteado para lo que el jugador no ve del rival (quién es cada
  Pokémon no visto, sus movimientos, objeto, IVs, EVs, naturaleza, habilidad, PS exactos dentro de la barra). El
  **muestreador** (*prior* del rival) es la regla con que se sortea.
- El **muestreador estricto** (§15) usa solo lo que sabe un jugador que no conoce la lista: especie cualquiera de la
  Frontera, movimientos aprendibles al azar, cualquier objeto útil. Produce rivales absurdos y débiles (tabla abajo).

**Decisión**:
- **Durante el entrenamiento**, las dos búsquedas (las determinizaciones del combatiente y los rivales simulados del
  táctico) sortean los rivales de **la lista de sets de la Factory** (`--opponent-prior factory_sets`, el valor por
  defecto de `rl.alphazero`): el pool de la ronda, con las reglas de arriba, tomando del set la especie, los
  movimientos y el objeto; la habilidad es una moneda al aire entre las dos de la especie, como en el juego.
  **Corrección del usuario (2026-09-25):** el rival simulado es un rival **real** de la Factory, construido como lo
  construye el juego: también el reparto de EVs y la naturaleza del set, y el IV que el juego da a ese combate
  (`fixed_iv`: 3, o 6 en el último combate de la ronda por el fallo de la Battle Tower; la fila propia de Noland).
  La restricción de información es para **los modelos**, no para el simulador: IVs, EVs y naturaleza nunca entran en
  la entrada de la red; solo forman parte del mundo que la búsqueda simula. (La primera versión los sorteaba al azar;
  el test comprueba ahora que el IV, los EVs y la naturaleza de los sets sorteados son los del juego.)
- Se mantiene todo lo que el jugador vio: especie, movimientos, objeto y habilidad revelados, la barra de PS, el
  estado, los rivales ya vistos, y las reglas estrictas de §15. Un Pokémon visto solo puede recibir sets compatibles
  con lo revelado (sus movimientos revelados incluidos en el set, su objeto revelado igual al del set); si ningún set
  lo explica (≈ 0,2 % de los Pokémon vistos, p. ej. un movimiento copiado), ese Pokémon se sortea con la regla
  estricta.
- Justificación: un jugador aprende los sets jugando (son pocos por especie y se repiten); la entrada de la red
  **nunca** los contiene, solo la búsqueda que genera los objetivos de entrenamiento.
- **En inferencia y evaluación** (`rl/eval_rounds.py`, `rl/evaluate.py`, agentes de `watch`, el valor por defecto de
  `SearchBattler`) se sigue usando el muestreador estricto.
- **Candado**: `factory_sets` solo existe en un proceso marcado como entrenamiento (`mark_training()` /
  `_TRAINING_MAINS`); fuera de él, pedirlo lanza `PermissionError` (`rl/determinize.py check_prior`, comprobado al
  construir `SearchBattler` / `TacticianSearch` y en cada búsqueda). El modo perfecto sigue prohibido en entrenamiento.
- **Lo que sabe el jugador de la generación** (`pybattle.backend.OpponentKnowledge`): la ronda, si es Noland, y las
  exclusiones, anotadas por `SimBackend` en cada pantalla de alquiler / intercambio (`opponent_knowledge_for`) a
  partir de lo que muestra la pantalla. **El equipo real nunca se lee**: hay tests que cambian el rival real
  (`gFrontierTempParty`) y comprueban que las dos búsquedas dan exactamente lo mismo, y un test que comprueba que el
  equipo real siempre está dentro del pool que describe `OpponentKnowledge`.
- Se mantienen: el valor de una opción del táctico con *bootstrap* 1 + V_t(s′) (V_t: la red de valor del táctico; s′:
  la siguiente decisión del táctico tras ganar el combate simulado), para que Q y z estén en las mismas unidades; el
  objetivo de política del táctico = visitas del *sequential halving*; y las evaluaciones del táctico en la CPU de
  cada proceso.

### Pantalla de intercambio simulada (corregido 2026-09-25)

El *bootstrap* necesita la observación de la pantalla de intercambio que sigue a un combate simulado ganado. Antes,
el equipo derrotado salía solo con su especie y los registros v4 del rival derrotado (§17) iban a cero, porque el
observador C++ de las simulaciones no guardaba lo revelado ni los registros. Ahora `ObsMemory` lleva los registros
(`records`, `team_max_hp`, `finish`, portados de `BattleObserver`) y expone lo revelado; el simulador llama a
`finish` en el momento en que se decide el combate (como `SimBackend`) y la pantalla simulada se construye con esa
memoria. Test: la observación simulada es idéntica, bit a bit, a la de `SimBackend` tras jugar el mismo combate
(observador C++ y Python).

### Diagnóstico: ¿cómo de fuertes son los rivales simulados?

Al inicio de la ronda k (pantalla de alquiler, racha 7(k − 1)), con el alquiler voraz de la propia red, el primer
combate lo juega la red del combatiente (voraz, sin búsqueda, como en las simulaciones) contra un rival sorteado con
cada muestreador o contra el rival real (el que generó el juego, con el RNG resembrado en cada combate). 512 combates
por casilla (128 comienzos × 4). A la derecha, los resultados reales de la red en toda la ronda (`eval_round`, 192
rachas, voraz, solo red). `scripts/diag_sim_opponents.py`.

| Red | Ronda | Rival estricto | Rival `factory_sets` | Rival real | Real: victorias por combate (ronda entera) | Real: rondas completadas |
|---|---|---|---|---|---|---|
| v3 | 1 | 100,0 % | 90,6 % | 94,1 % | 94,6 % | 68,2 % |
| v3 | 3 | 100,0 % | 89,5 % | 91,8 % | 88,5 % | 38,5 % |
| v3 | 5 | 99,6 % | 76,6 % | 73,8 % | 78,1 % | 18,8 % |
| sin entrenar | 1 | 57,4 % | 16,8 % | 25,4 % | 22,9 % | 0,0 % |
| sin entrenar | 3 | 51,2 % | 17,6 % | 15,6 % | 15,4 % | 0,0 % |
| sin entrenar | 5 | 50,0 % | 7,4 % | 7,0 % | 10,3 % | 0,0 % |

Porcentaje de victorias en el primer combate de la ronda (salvo las dos últimas columnas), 256 combates por casilla
(error típico ≈ ±2–3 puntos), con los sets completos (EVs, naturaleza e IV del juego). El muestreador estricto da
rivales tan débiles que v3 gana casi siempre y una red sin entrenar la mitad; con `factory_sets` la tasa de victoria
queda dentro de unos pocos puntos de la del rival real en todas las rondas (con EVs y naturaleza al azar, la primera
versión daba en la ronda 5 un 80,3 % frente a 73,0 %: rivales algo más fáciles de lo real).

**Pendiente de decidir** (elegido provisionalmente al implementar):
- Tamaño de la iteración: 40.000 decisiones del combatiente (`--decisions-per-iter`) y 200 iteraciones planeadas.
- La evaluación por ronda en cada iteración (192 × 6 rachas) para el autojuego mientras dura.

## 18b. Resultado de alphazero_v1 y diagnóstico (2026-09-26)

**Resultado.** Parado en la iteración 169/200 (6,8 M decisiones del combatiente), en meseta desde la ~70. P(completar)
por ronda (red sola, voraz, 512 rachas, mismas semillas para todas las filas; `runs/diag_az1/`):

| | R1 | R2 | R3 | R4 | R5 | R6 | P(6 rondas) |
|---|---|---|---|---|---|---|---|
| PPO v3, red sola | 69,3 | 71,3 | 39,1 | 56,6 | 21,3 | 19,7 | 0,46 % |
| PPO v3 + búsqueda legal 128 | 70,5 | 73,2 | 39,3 | 55,7 | 20,5 | 19,9 | 0,46 % |
| AZ v1, red sola | 66,0 | 66,8 | 41,8 | 56,6 | 21,1 | 19,5 | 0,43 % |
| AZ v1 + búsqueda legal 128 | 71,1 | 73,0 | 43,8 | 60,2 | 27,5 | 27,0 | 1,02 % |
| AZ v1 + búsqueda legal 512 | 72,1 | 74,4 | 52,5 | 68,8 | 30,1 | 29,1 | 1,70 % |
| AZ v1 + búsqueda **perfecta** 512 (solo diagnóstico) | 76,4 | 78,5 | 48,0 | 71,9 | 32,6 | 35,9 | 2,4 % |

**Diagnóstico.**
1. *Combatiente*: la búsqueda mejora mucho a la red de AZ (a v3 no), así que el profesor sabe más que el alumno; lo que
   falla es la transferencia: (a) el objetivo de política son las visitas de 128 simulaciones en 8 árboles con ruido de
   Dirichlet, casi planas (entropía de visitas 1,51 ≈ la de la red); (b) en el autojuego la acción se sortea de las
   visitas con temperatura 1 y el agente gana solo el 55 % de los combates (94 % en voraz).
2. *Crítico del táctico*: ordena como el de v3 (y v3 ya está en el techo: EV ≈ 0,15–0,19 frente a z, casi todo por la
   posición en la racha), pero con la escala mal (predice ~2 victorias donde el juego voraz da 5–8), porque z venía del
   autojuego débil del punto 1b.
3. *Búsqueda del táctico*: su objetivo es casi ruido (dos búsquedas del mismo estado coinciden el 12 % en alquileres) y,
   jugada en el juego real, no elige mejor que la red (alquiler −0,06, cambio −0,24 ± 0,09 victorias). Una variante con
   números aleatorios comunes y pocas candidatas (opción B) es consistente pero tampoco mejora a la red (todas las
   configuraciones dentro de ±2 SE de 0). Causa: el simulador no predice las diferencias entre opciones del táctico.
4. *Techo*: incluso con información perfecta el combatiente gana solo el 85–87 % de los combates en las rondas 5–6:
   buena parte de lo que falta está en el equipo (táctico) y en la calidad de la red, no en la información oculta.

## 19. alphazero_v2 (decidido 2026-09-26, desde cero)

### 19.1 Combatiente: Gumbel AlphaZero
- Raíz con *sequential halving* y ruido de Gumbel (Danihelka et al., 2022) en vez de Dirichlet; **256 simulaciones**.
- En el autojuego se **juega la acción ganadora del halving** (no se sortea de las visitas): el autojuego juega a su nivel
  real, y la exploración la da el ruido de Gumbel.
- Objetivo de política: softmax(logits de la red + σ(Q̂)), donde Q̂ es el valor de cada acción según la búsqueda (el de
  la búsqueda si se visitó; el valor de la red si no, "Q completados") y σ una transformación monótona que lo escala.

### 19.2 Red evaluadora de equipos (nueva)
Definiciones:
- **Ronda**: bloque de 7 combates; empieza con un alquiler y el equipo se devuelve al terminarla (un equipo solo vive
  en su ronda).
- **k**: número del combate dentro de la ronda (1..7) en el que se usará el equipo.
- **Equipo**: los 3 Pokémon, con lo que el jugador sabe de ellos (especie, movimientos, objeto) y su orden (líder).
- **Opción**: cada elección del táctico (alquiler: trío + líder, ~60; cambio: quedarse o uno de los 3×3 cambios, 10);
  cada opción da un **equipo resultante**.
- **W(equipo, ronda, tipo)**: probabilidad de que nuestro combatiente gane **un** combate con ese equipo contra un rival
  sorteado de esa ronda (sets reales de la Factory, construidos como el juego). Tipos: *normal* (combates 1–6, IV 3),
  *último* (combate 7, IV 6), *Noland* (combate 7 de las rondas 3 y 6). Red entrenada de forma supervisada con millones
  de combates simulados (promediar muchos combates por equipo le quita el ruido). Depende del combatiente: se reentrena
  periódicamente con el combatiente actual.
- **pᵢ**: W con el tipo que corresponde al combate i de la ronda.
- **E_ronda**: victorias esperadas en lo que queda de ronda manteniendo el equipo: p_k + p_k·p_{k+1} + … hasta el 7.
- **P(completar)**: p_k·p_{k+1}·…·p₇.
- **V_siguiente**: victorias esperadas desde el inicio de la ronda siguiente (no depende del equipo actual, que se
  devuelve); tabla por ronda aprendida de resultados reales.
- **Δ**: E_ronda del equipo resultante menos la del equipo actual (cambio) o menos la media de las opciones (alquiler).

Usos:
1. **Entrada del táctico, por opción**: [E_ronda, W(normal), W(último o Noland), Δ] del equipo resultante (como las
   estimaciones de daño por movimiento del combatiente).
2. **Shaping** basado en potencial (Ng et al.): Φ(s) = E_ronda + P(completar) × V_siguiente (unidades de victorias);
   recompensa extra Φ(s′) − Φ(s) en cada decisión del táctico. No cambia la política óptima.
3. **Experto del táctico** (ver 19.3).

Información: entrenar W con los sets reales es conocimiento que un jugador adquiere jugando ("este equipo suele ganar en
la ronda 5"), aceptado por el usuario; nunca información oculta de un rival concreto.

#### 19.2.1 Implementación: `rl/team_eval.py`, `rl/team_eval_data.py` (2026-09-26, `runs/team_eval_v1/`)

**Lo que la red ve de un equipo** (`TeamMon`, construido solo desde la vista del táctico):
- nuestros Pokémon (alquileres y nuestro equipo en la pantalla de intercambio): el jugador los ve enteros en la
  pantalla de resumen: especie, 4 movimientos, objeto, habilidad y estadísticas (PS máximos y las otras 5). Los IVs,
  EVs y la naturaleza no se usan: ya están dentro de las estadísticas;
- el Pokémon que entraría con un intercambio: solo lo que se vio de él en el combate (`SeenMon`): especie, los
  movimientos que usó, el objeto y la habilidad si el juego los nombró; lo demás es DESCONOCIDO (su propio token) y
  sus estadísticas, desconocidas;
- el orden: el primero es el líder (una marca en su token). Nunca se lee nada del rival (hay un test que cambia en la
  RAM los movimientos, objetos, IVs y naturalezas del equipo derrotado y el siguiente rival real, y comprueba que las
  características salen idénticas).

**Rondas a partir de la 7**: W(equipo, r) = W(equipo, 6) exactamente, como hace el juego (nivel abierto, el único que
entrenamos): la fila del pool de rivales es min(desafío, 7) y las filas 4–7 son el mismo rango; los rivales normales y
del último combate tienen siempre IV 3 / 6 (el fallo de la Battle Tower: `GetFactoryMonFixedIV` recibe la racha de la
Tower), y el IV de Noland es 31 desde la ronda 6. Por eso la ronda se recorta a 1..6 (la 5 y la 6 comparten incluso
los rivales normales; solo cambian los alquileres que se ofrecen). Noland vuelve en las rondas 9, 12, …
(`battle_types`, con `factory_brain_status`).

**Datos** (`python -m rl.team_eval_data`). Un *caso* es un equipo en una ronda y un tipo de combate, jugado en 12
combates simulados independientes (rival nuevo cada vez):
1. ronda r uniforme en 1..6; tipo normal (combate b uniforme en 1..6) o último (70 % / 30 %); en las rondas 3 y 6 el
   séptimo combate es de Noland, así que allí los tipos son normal / Noland (60 % / 40 %);
2. `SimBackend.reset` en la racha 7(r − 1) + b − 1 con un contador de alquileres al azar dentro de lo posible
   [r − 1, 7(r − 1)] (como `FactoryEnv`): el juego genera los 6 alquileres de la ronda; se alquila un trío legal y un
   líder al azar; se escribe el número de combate en la partida guardada para que el contexto que ve el combatiente
   sea el de ese combate;
3. equipos tras intercambios: con probabilidad 0,5 uno de nuestros Pokémon (dos, en el 30 % de esos casos) se cambia
   por un set del pool de rivales de la ronda, construido como el juego construyó al rival (EVs y naturaleza del set,
   IV 3), de especie distinta a las del equipo; se marca como "intercambiado";
4. cada combate: una copia del juego, el equipo rival sustituido por uno de `sample_factory_sets` (el pool de la
   ronda, las exclusiones de su generación, EVs / naturaleza del set, el IV del juego para ese combate) y el RNG
   resembrado; lo juega la red de alphazero_v1 (`--battler-ckpt`), voraz y sin búsqueda, en el servidor de inferencia
   de la GPU; todos los combates de un proceso avanzan a la vez; corte a las 300 decisiones (marcado como truncado y
   excluido del entrenamiento: 0,12 %).

**Tamaño y velocidad**: 45 minutos con 18 procesos: **1.992.338 equipos, 23,9 millones de combates** (≈ 8.850
combates/s, 126.000 decisiones del combatiente/s).

**Cómo se ve un Pokémon intercambiado en la pantalla de intercambio** (medido en 1.500 Pokémon derrotados con
alphazero_v1 voraz): movimientos revelados 0/1/2/3/4 en el 18 / 44 / 24 / 10 / 4 %, objeto nombrado 19 %, habilidad
anunciada 4 % (especies con dos habilidades). En el entrenamiento, cada Pokémon intercambiado se muestra así (lo no
revelado, DESCONOCIDO; estadísticas desconocidas) con probabilidad 0,5, y entero con probabilidad 0,5 (en la pantalla
de intercambio siguiente ya es nuestro y se ve entero).

**Red** (`TeamNet`, 429.000 parámetros): cada Pokémon es un token (embeddings de especie, tipos, objeto, habilidad y
habilidades posibles; la suma de sus 4 movimientos codificados con embeddings de movimiento, efecto y tipo más
potencia / precisión / prioridad / PP; estadísticas base; estadísticas propias si se conocen; marca de líder); se suma
un embedding de la ronda a cada token; un transformer de 2 capas sin posiciones los mezcla (invariante al orden salvo
por la marca de líder); media + token del líder + ronda → MLP → 3 logits (normal, último, Noland), sigmoide.
Pérdida: la log-verosimilitud binomial de (ganados de jugados) de cada equipo en su cabeza, que es la suma de las BCE
por combate. AdamW, lr 2·10⁻³ con *one-cycle*, lotes de 2.048 equipos, 8 épocas (97 s en la GPU). Validación: el 5 %
de los equipos (99.617, 1,19 M combates), separados por equipo.

Medidas, definidas antes de usarlas:
- **Calibración**: si la red dice 0,8, ¿se gana el 80 %? Se ordenan los combates por la W predicha, se parten en 10
  grupos iguales (deciles) y se compara la media predicha con la proporción ganada de cada grupo.
- **Brier**: media de (W predicha − resultado)², con resultado 1 / 0; más bajo es mejor. Se compara con el de
  predecir siempre la misma media ("constante").
- **AUC**: probabilidad de que un combate ganado tenga una W predicha mayor que uno perdido (0,5 = azar, 1 = perfecto).

**Calibración** (validación, por combate; cada columna es un decil de la probabilidad predicha):

| Decil | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicho | 0,295 | 0,455 | 0,561 | 0,649 | 0,721 | 0,777 | 0,821 | 0,857 | 0,890 | 0,931 |
| Observado | 0,294 | 0,456 | 0,559 | 0,649 | 0,720 | 0,774 | 0,820 | 0,857 | 0,890 | 0,929 |

Brier 0,1732 (0,2121 con la media global); AUC 0,758. Por ronda y tipo (AUC por combate; Brier frente al de predecir
la media de esa casilla; "explicada" = fracción de la varianza real entre equipos de esa casilla que explica la red,
quitando el ruido binomial de 12 combates):

| Ronda | Tipo | Victorias | Predicho | AUC | Brier (constante) | Varianza explicada |
|---|---|---|---|---|---|---|
| 1 | normal | 0,835 | 0,836 | 0,684 | 0,128 (0,138) | 0,97 |
| 1 | último | 0,820 | 0,819 | 0,685 | 0,137 (0,148) | 0,97 |
| 2 | normal | 0,810 | 0,812 | 0,708 | 0,139 (0,154) | 0,99 |
| 2 | último | 0,795 | 0,794 | 0,700 | 0,148 (0,163) | 0,99 |
| 3 | normal | 0,783 | 0,785 | 0,672 | 0,158 (0,170) | 0,96 |
| 3 | Noland | 0,419 | 0,419 | 0,653 | 0,226 (0,244) | 0,93 |
| 4 | normal | 0,822 | 0,824 | 0,671 | 0,138 (0,147) | 0,98 |
| 4 | último | 0,801 | 0,802 | 0,671 | 0,149 (0,160) | 0,98 |
| 5 | normal | 0,572 | 0,572 | 0,669 | 0,223 (0,245) | 0,94 |
| 5 | último | 0,539 | 0,539 | 0,668 | 0,227 (0,248) | 0,98 |
| 6 | normal | 0,617 | 0,619 | 0,674 | 0,215 (0,236) | 0,90 |
| 6 | Noland | 0,373 | 0,372 | 0,662 | 0,216 (0,234) | 0,91 |

Lectura: el AUC por combate es moderado (0,65–0,71) porque un combate suelto depende mucho del rival y del azar, pero
casi toda la diferencia *entre equipos* que existe en estos datos la recoge la red (90–99 %; la desviación típica real
del porcentaje de victorias entre equipos es 0,10–0,16 dentro de cada casilla). Los equipos con intercambios se
explican casi igual que los de alquiler (0,88–1,00 frente a 0,93–1,02; `runs/team_eval_v1/within_round.json`).

**Validación del experto en el juego real** (la condición de 19.3; `python -m rl.team_eval_validate`, misma prueba
que 18b.3). Definiciones:
- **π_eval**: la opción de mayor Q_eval = Φ del estado tras la opción (E_ronda + P(completar) × V_siguiente).
  **π_red**: la de mayor probabilidad según la red del táctico de alphazero_v1.
- **V_siguiente** para esta prueba: victorias medias desde el inicio de la ronda r + 1 con alphazero_v1 voraz (los dos
  agentes), de `runs/diag_az1/q1_az_allgreedy.npz`: 11,02 / 7,97 / 7,45 / 3,79 / 4,19 / 5,67 para r = 1..6.
- **Estados**: los que alcanza alphazero_v1 jugando (combatiente voraz, táctico muestreado): 126 alquileres (primera
  decisión de rachas desde la ronda 1) y 126 cambios de las rondas 3–5 (sin las pantallas antes de Noland).
- **Resultado real de una opción**: se juega en el juego real con 64 semillas emparejadas (las mismas para todas las
  opciones) hasta que acaba la ronda, con el combatiente y el táctico de alphazero_v1 voraces. Aquí se jugaron
  **todas** las opciones (60 alquileres, 10 cambios) de cada estado. "Victorias" = combates ganados hasta el final de
  la ronda; "Φ real" = victorias + V_siguiente si se completó la ronda (las unidades de Q_eval).
- **Ganancia**: resultado real de π_eval menos el de π_red, media sobre estados (± error típico); "donde difieren" solo
  cuenta los estados en que eligen distinto. **P(mejor)**: fracción de esos estados en que π_eval sale mejor.
  **Correlación**: entre la ganancia que el evaluador promete (diferencia de Q_eval) y la real.
- **Mejor opción en el top-3**: la opción con más victorias reales medias está entre las 3 primeras del criterio
  (con 64 semillas la "mejor" es ruidosa: su error típico es ≈ 0,25 victorias). **Spearman**: correlación de rangos,
  en cada estado, entre la puntuación del criterio y el resultado real de todas las opciones.

| | Difieren | Ganancia donde difieren | Ganancia en todos | P(mejor) | Corr. prometida/real | Mejor en top-3: eval / red | Spearman: eval / red |
|---|---|---|---|---|---|---|---|
| Alquiler, victorias | 105/126 | +0,085 ± 0,072 | +0,071 ± 0,060 | 0,51 | +0,15 | 0,17 / 0,21 | 0,57 / 0,53 |
| Alquiler, Φ real | 105/126 | +0,16 ± 0,18 | +0,13 ± 0,15 | 0,50 | +0,19 | 0,18 / 0,23 | 0,57 / 0,53 |
| Cambio, victorias | 58/126 | −0,030 ± 0,076 | −0,014 ± 0,035 | 0,48 | +0,36 | 0,68 / 0,66 | 0,52 / 0,49 |
| Cambio, Φ real | 58/126 | −0,04 ± 0,14 | −0,02 ± 0,07 | 0,52 | +0,34 | 0,68 / 0,62 | 0,54 / 0,50 |

(Con solo E_ronda como criterio, sin V_siguiente, sale lo mismo: +0,088 ± 0,073 y −0,037 ± 0,078.) Referencia: la
búsqueda por simulación de alphazero_v1 daba −0,06 (alquiler) y −0,24 ± 0,09 (cambio) en la misma prueba (18b.3).

**Veredicto**: el evaluador **no elige mejor que la red** por más de 2 errores típicos (alquiler +1,2 SE, cambio
−0,4 SE). Ordena todas las opciones algo mejor que la red (Spearman 0,57 frente a 0,53 y 0,52 frente a 0,49) y, a
diferencia de la búsqueda de §18, no elige peor, pero su argmax no gana más. Según la condición de 19.3, el
coeficiente de imitación debe quedar en 0 con este evaluador (sí pueden usarse sus características como entrada y el
shaping, que no cambia la política óptima). Nota: las primeras 54 rachas de alquiler daban +0,37 ± 0,12; con las 126
bajó a +0,09 ± 0,07: con esta prueba hacen falta todos los estados para no sacar conclusiones de la varianza.

**Interfaz** (la que usa `rl/alphazero_v2.py` vía `rl/tactician_features.py`):
- `TeamEvaluator.load(path, device="cpu")` (método de clase); `ev.w(equipos, ronda)` → [n, 3].
- `team_of_option(view, kind, opción)`: opciones como las da `options_of` (alquiler (líder, índice de pareja); cambio
  0..9, también `None` / `NO_ACTION` / (hueco propio, hueco rival)).
- `option_features(ev, view, kind, ctx, opciones, v_next)` → [n, 4] = [E_ronda, W(normal), W(último o Noland, el que
  sea el combate 7 de esta ronda), Δ].
- `phi(ev, view, kind, ctx, v_next)`: Φ del estado actual. En un cambio, el equipo actual. En un alquiler todavía no
  hay equipo: se define como **la media de Φ sobre los alquileres legales** (la misma referencia que Δ). También
  acepta un `Team` o (kind, opción): el estado tras la elección (`phi_options` da todas).
- `ctx` es `run_ctx`: ronda = `challenge` + 1, k = `battle` + 1. La Q_eval que calcula el entrenador,
  E_ronda + W_n^(7−k)·W_l·v_next, es exactamente Φ del estado tras la opción, también con Noland (W_l es la
  columna del combate 7 que corresponda); hay un test que lo comprueba.
- Coste: ≈ 3 ms en CPU por decisión de alquiler (60 opciones). En procesos hijos (los de `FactoryEnv` de
  `eval_rounds`), torch se pone a 1 hilo: construir la red en un hijo `fork` de un padre que ya usó varios hilos se
  quedaba colgado.

### 19.3 Táctico híbrido
- **π_red**: política del táctico. **Q_eval(opción)**: Φ del estado resultante según el evaluador.
  **π_eval** = softmax(Q_eval / T), el "experto".
- **Ventaja A**: cuánto mejor salió la opción de lo que esperaba el crítico, con resultados reales + shaping (PPO/GAE).
- Pérdida = coeficiente × CE(π_eval, π_red) [imitación, AlphaZero] + término de PPO con A [experiencia real] + valor.
- La búsqueda por simulación del táctico (§18) se elimina.
- **Condición**: antes de dar peso a la imitación se comprueba en el juego real que la opción de π_eval gana más que la
  de π_red (misma prueba que en 18b.3); si no, el coeficiente de imitación es 0.

### 19.4 Evaluación
Red sola cada iteración; red + búsqueda legal de 512 simulaciones cada ~5 iteraciones.

### 19.5 Implementación (`rl/alphazero_v2.py`, 2026-09-26)

Punto de entrada **nuevo**, `python -m rl.alphazero_v2`, que reutiliza piezas de `rl/alphazero.py` (currículo,
procesos de autojuego, buffer del combatiente, paso de entrenamiento del combatiente, evaluación). Motivo: v2 cambia
el táctico de raíz (sin búsqueda, PPO en vez de imitar una búsqueda), así que meterlo tras *flags* en `rl.alphazero`
llenaría v1 de ramas; con un archivo aparte, `python -m rl.alphazero` sigue siendo exactamente alphazero_v1. Los
cambios en archivos compartidos no alteran v1 (tests de v1 y de la búsqueda C++ intactos).

**Combatiente: raíz Gumbel en C++** (`Searcher.search_gumbel`, `src/gen3/search.cpp`; Python en `rl/gumbel.py`).
Términos, de abajo arriba:
- **logits** de la raíz: log de la política de la red (la *prior*), normalizada sobre las acciones legales.
- **Ruido de Gumbel** g(a): un número aleatorio por acción (distribución de Gumbel). Sumar g a los logits y quedarse
  con el máximo equivale a sortear una acción de la prior; así se explora sin temperatura ni Dirichlet.
- **Top-m**: se consideran las m = min(legales, 16) acciones con mayor g + logits. En el combate hay como mucho 7
  acciones, así que se consideran todas las legales; el ruido sigue importando en el desempate y en el ganador.
- **Q completado**: para cada acción, su Q de la búsqueda si se visitó; si no, **v_mix**, una mezcla del valor de la
  red en ese nodo con la media, ponderada por la prior, de los Q de las acciones visitadas (ec. 33 del artículo).
- **σ(q)** = (c_visit + máx. de visitas) · c_scale · q, con c_visit = 50 y c_scale = 0,1, y q reescalado a [0, 1]
  con mínimo y máximo sobre las acciones legales del nodo (el valor por defecto de mctx). Cuantas más visitas, más
  pesa la búsqueda frente a la prior.
- **Sequential halving** (reparto por mitades): ⌈log₂ m⌉ fases; cada fase gasta 256 / fases simulaciones repartidas
  a partes iguales entre las acciones que quedan (la última fase se lleva lo que sobre); tras cada fase se ordenan por
  g + logits + σ(Q̂) y pasa la mejor mitad. Con 6 acciones legales: 14 simulaciones cada una, luego 28 a las 3
  mejores, luego 44 a las 2 mejores (84 + 84 + 88 = 256). La **ganadora** es la última que queda.
- **Con K = 8 determinizaciones**: las simulaciones de cada acción se reparten por turnos entre los 8 árboles; Q̂ de
  una acción = suma de valores / suma de visitas de su hijo en los 8 árboles (media ponderada por visitas). Las
  decisiones del halving usan ese Q̂ agregado. `redraw_turn`, el muestreador `factory_sets` en entrenamiento y el
  estricto en evaluación, y el candado de información perfecta se mantienen igual (`_roots` de `SearchBattler`).
- **Debajo de la raíz** (`--non-root gumbel`, por defecto): la regla determinista del artículo, elegir
  argmax π′(a) − N(a) / (1 + Σ N), con π′ = softmax(logits + σ(Q completado)) de ese nodo; las selecciones en vuelo
  cuentan como visitas de valor 0 (la misma pérdida virtual que PUCT), para poder evaluar hojas en lotes.
  `--non-root puct` deja PUCT debajo de la raíz. Se eligió la regla del artículo porque con pocas simulaciones por
  árbol (~30 por acción) es la que el artículo recomienda y en C++ cuesta lo mismo.
- **Autojuego**: se juega la ganadora (sin temperatura, sin Dirichlet). **Objetivo de política**:
  softmax(logits + σ(Q completado)) sobre las legales. **Objetivo de valor**: ½ z + ½ q_raíz (sin cambios; z sale
  ahora de partidas jugadas a su nivel real).
- Una sola acción legal: no se busca (política one-hot, valor z), como en v1.
- **Velocidad** (18 procesos + servidor GPU, checkpoint de v1, máquina compartida con carga ~8, 150 decisiones por
  proceso; `python -m rl.az_bench --v2`):

  | | ms por decisión del combatiente | decisiones del combatiente / s |
  |---|---|---|
  | v1 (PUCT 128 + búsqueda del táctico) | 24,6 | 211 (≈290 con la máquina libre) |
  | v1 con PUCT 256 | 42,3 | 171 |
  | **v2 (Gumbel 256, regla del artículo)** | 42,8 | **388** |
  | v2 con PUCT debajo de la raíz | 41,9 | 396 |

  La raíz Gumbel cuesta lo mismo que PUCT con el mismo número de simulaciones; el autojuego va más rápido que v1
  porque el táctico ya no simula combates (antes era el 53 % del tiempo de los procesos).

**Táctico híbrido.**
- **Red**: `FactoryNet(..., opt_feat=4)` recibe `opt_feat` por opción: [E_ronda, W(normal), W(último o Noland), Δ]
  del equipo resultante (§19.2). Un MLP pequeño f de esas 4 cifras se suma al logit de cada opción; su media y su
  máximo sobre las opciones legales entran al crítico (capa lineal iniciada a cero). Sin `opt_feat` en la entrada, o
  con la red de v1, todo es como antes; un `state_dict` de v1 carga en la red nueva.
- **Alquiler (cabeza factorizada)**: una opción es (primero, pareja) y sus rasgos van a la celda [primero, pareja].
  Para que la política conjunta sea exactamente π(l, p) ∝ π₀(l) · π₀(p | l) · e^{f(l, p)} se suma f(l, p) a los
  logits de pareja de la fila l y log Σ_p π₀(p | l) e^{f(l, p)} al logit del primero l. Así `rental()` (sortear el
  primero y luego la pareja) y `rental_joint()` dan la misma distribución (test).
- **Rasgos en el entorno** (`rl/tactician_features.py`): en cada alquiler / intercambio se enumeran las opciones como
  el espacio de acciones (`options_of`) y se llama a `option_features(ev, view, kind, ctx, options, v_next)` del
  evaluador; sin `--team-eval`, ceros. `FactoryEnv(obs_hook=...)` los añade también en las evaluaciones.
- **Q_eval y π_eval**: Q_eval(opción) = E_ronda + W_n^(7−k) · W_l · V_siguiente (Φ del estado resultante, a partir
  de los rasgos; k = siguiente combate de la ronda); π_eval = softmax(Q_eval / T), T = `--t-eval-temp` (0,5 victorias,
  provisional).
- **Shaping**: r = victorias desde la decisión anterior + Φ(s′) − Φ(s), γ = 1, Φ(fin de la racha) = 0, con
  Φ = `phi()` del evaluador. La suma de las recompensas de una racha es victorias − Φ(s₀) (test). Un combate cortado
  por las 300 decisiones cierra la racha con el valor estimado de lo que faltaba.
- **V_siguiente**: tabla por ronda de victorias medias desde el inicio de la ronda r hasta el final de la racha, con
  los resultados del autojuego; media móvil por iteración (ritmo `--v-next-rate` 0,3; la primera observación se toma
  tal cual). Se guarda en el checkpoint y se registra en `v_next/round_r`.
- **PPO** (definido en §7): solo con las muestras de la iteración en curso, `--t-ppo-epochs` 4 épocas, recorte 0,2,
  ventajas GAE (λ 0,95, γ 1) sobre las recompensas con shaping, valor con `ValueNorm`, bonus de entropía 0,01, más
  `--t-imitation` × CE(π_eval, π_red) (0 por defecto, hasta que la validación del evaluador diga que π_eval gana más
  que la red). La decisión que queda abierta al acabar la iteración solo sirve de *bootstrap* a las anteriores y no
  se entrena nunca (su log-prob es de la copia anterior de la red). El combatiente mantiene su buffer de 4
  iteraciones. Las dos partes comparten el optimizador (AdamW) y la tasa de aprendizaje coseno.

**Evaluación**: red sola cada iteración (`eval_round_k/*`, con los rasgos del táctico); red + búsqueda legal PUCT de
512 simulaciones con el muestreador estricto cada 5 iteraciones (`eval_search_round_k/*`, 64 rachas por ronda,
comparable con la tabla de §18b). Productos: `eval/product_complete`, `eval_search/product_complete`.

**Registros nuevos**: `gumbel/*` (entropía del objetivo, KL(objetivo ‖ prior), fracción en que la ganadora ≠ argmax
de la prior o del objetivo, acciones consideradas); `tactician/*` (pérdidas, *clip frac*, KL aproximada, entropía,
desviación de las ventajas, media y desviación de la recompensa con shaping, Φ medio, CE de imitación, fracción en
que la red elige el argmax de Q_eval, varianza explicada en entrenamiento y validación); `v_next/*`. Se mantienen los
de v1 que siguen teniendo sentido (`battler/*`, `search/*`, `perf/*`, `train/*`, `curriculum/*`); `tsearch/*` ya no.

**Interfaz esperada del evaluador** (`rl/team_eval.py`, en paralelo; `rl/team_eval_stub.py` la imita para los
tests): `TeamEvaluator.load(path, device)` (de clase o de instancia), `option_features` con las opciones en el
formato de `options_of` y `ctx` = `run_ctx` (`battle` = combates ganados en la ronda, `challenge` = rondas
completadas), y `phi(ev, view, kind, ctx, v_next)`.

Comando:

```bash
python -m rl.alphazero_v2 --name alphazero_v2 --team-eval runs/team_eval/latest.pt   # valores por defecto = los de arriba
python -m rl.az_bench --v2 --workers 18 --decisions 200                             # velocidad de autojuego
```

**Pendiente de decidir**: T de π_eval (0,5) y el coeficiente de imitación (0 hasta la validación); el ritmo de
V_siguiente (0,3); lote de PPO del táctico (256) y coeficiente de valor (1).

### 19.6 Refresco periódico de W durante alphazero_v2, y plan para después (`rl/team_eval_refresh.py`, 2026-09-26)

Definiciones, de abajo arriba:
- **W** (§19.2) es la probabilidad de que **nuestro** combatiente gane un combate con un equipo. Se entrenó con
  combates jugados por el combatiente de v1. El combatiente de v2 cambia al entrenar, así que la W fija se va quedando
  desfasada: describe a un combatiente que ya no es el nuestro.
- **Refrescar W**: volver a ajustarla con combates jugados por el combatiente actual.
- **Ajuste fino** (*fine-tuning*): seguir entrenando una red desde sus pesos actuales, con pocas épocas y una tasa
  de aprendizaje más baja, en vez de entrenarla desde cero.
- **Mezcla de parámetros**: la red nueva es α · (red ajustada) + (1 − α) · (red anterior), peso a peso.
- **Conjunto de sondeo** (*probe*): 1.024 equipos fijos (siempre los mismos, semilla fija) en los que se mide cuánto
  se mueve W en cada refresco: la media de |ΔW|.

**Cuándo** (`RefreshSchedule`): W empieza fija, la de `--team-eval` (entrenada con v1). Los refrescos empiezan
cuando el combatiente de v2 juega al menos como el de v1: la media de las 3 últimas evaluaciones «red sola» de la
media sobre las rondas 1–6 de P(completar la ronda) alcanza la media de la referencia (`--w-refresh-after`, por
defecto la fila «AZ v1, red sola» de §18b: 66,0 / 66,8 / 41,8 / 56,6 / 21,1 / 19,5 %, media 45,3 %). Desde ese
momento se refresca en la misma iteración y luego cada `--w-refresh-every` = 20 iteraciones.

**Cómo** (al final de la iteración, con el autojuego parado):
1. **Datos**: `--w-refresh-battles` = 4 M combates simulados nuevos (≈ 10 min con 18 procesos) jugados por la copia
   actual del combatiente, con el mismo generador y la misma distribución de equipos y rivales que los datos
   originales (`rl/team_eval_data.Generator`, 16 combates por equipo; los procesos de autojuego lo ejecutan con su
   cliente del servidor GPU). Además, los combates **reales** del autojuego de las últimas 5 iteraciones (equipo tal
   como lo conoce el jugador, ronda, tipo de combate y resultado; `RealBattleLog`, cuesta un `decode_party` por
   combate). Son ~0,3 % de los datos. Ojo: esos combates se juegan con la búsqueda Gumbel y con los equipos que elige
   el táctico, no con la red sola y equipos al azar como los simulados; con ese peso apenas influyen.
2. **Ajuste fino** de una copia de W desde sus pesos actuales: 2 épocas, tasa 2·10⁻⁴ con coseno, la misma pérdida
   binomial que el entrenamiento original. El 5 % de los equipos nuevos se aparta para medir.
3. **Mezcla de parámetros** con α = 0,5 (`--w-refresh-alpha`). Se eligió mezclar parámetros y no salidas porque las
   dos redes parten de los mismos pesos y el ajuste es corto, así que su media es una red intermedia (lo comprueban
   las métricas tras mezclar y el |ΔW| del sondeo); se guarda una sola red (el coste del evaluador no crece), y los
   refrescos sucesivos son una media móvil exponencial de los pesos. Mezclar salidas obligaría a guardar y evaluar
   todas las redes anteriores.
4. Se guarda `runs/<nombre>/team_eval_<iteración>.pt` (escritura atómica), y los procesos de autojuego y la
   evaluación la cargan desde la siguiente decisión del táctico.

**Registros** (`w_refresh/*`): combates simulados y reales, segundos (generación y total), BCE, Brier y AUC por
combate en los equipos apartados con la W anterior (`*_before`), la ajustada (`*_finetuned`) y la mezclada
(`*_after`); |ΔW| medio en el sondeo; `trigger_mean_complete`, `armed` y `count`.

**Checkpoints y reanudación**: cada checkpoint guarda la W en uso (`team_eval`) y el estado del calendario
(`w_refresh`: historial, iteración de arranque, último refresco). `ckpt_<pasos>.pt` se escribe antes del refresco
de su iteración (la W que usó su autojuego) y `latest.pt` después. Con `--init-from` se recupera la W y el calendario.

**Φ cambia en cada refresco.** La garantía de Ng et al. (el shaping F = Φ(s′) − Φ(s) no cambia la política óptima)
es exacta solo si Φ es fija. Si Φ cambia, cada trozo de trayectoria entre dos refrescos sigue cumpliendo la suma
telescópica con su propia Φ, pero el objetivo que optimiza PPO cambia un poco en cada refresco. La mezcla con α = 0,5
y los refrescos espaciados (20 iteraciones) hacen que el cambio sea pequeño (se vigila con |ΔW| del sondeo). Además,
las muestras de PPO solo se usan en la iteración en que se generan, así que nunca se mezclan recompensas de dos Φ
distintas en una misma actualización.

**Plan acordado con el usuario para después de v2:**
- **Etapa 2**: congelar el combatiente de v2; reentrenar W desde cero con él (W_final); entrenar **solo el táctico**
  con W_final fija. Ahora el problema es estacionario (ni el combatiente ni Φ cambian), así que la garantía de Ng es
  exacta.
- **Etapa 3 (opcional)**: congelar el táctico y ajustar el combatiente con los equipos que elige; y alternar.

Comando (igual que en §19.5; los valores por defecto activan el refresco):

```bash
python -m rl.alphazero_v2 --name alphazero_v2 --team-eval runs/team_eval_v1/model.pt
```

### 19.7 Corrección del PPO del táctico (2026-09-26, tras 22 iteraciones del primer lanzamiento)
- **Síntoma**: KL aproximada del táctico 0,11–0,15 y fracción recortada 0,52 en cada iteración (lo normal en PPO es
  ~0,01–0,02 y ~0,1–0,2), entropía cayendo deprisa (3,18 → 1,82) y EV del crítico negativa; la red sola iba por
  detrás de v1 a la misma altura, sobre todo en las rondas 5–6.
- **Causa**: en cada iteración se entrenaba primero el combatiente y después el táctico. Comparten los embeddings,
  así que al empezar el PPO del táctico su política ya no era la que generó sus muestras: el cociente
  π_nuevo/π_viejo empezaba lejos de 1 (datos fuera de política), y el recorte actuaba desde el primer paso.
- **Corrección**: el PPO del táctico se hace **antes** del entrenamiento del combatiente (sus log-probabilidades
  guardadas son exactamente las de la red actual) y con 2 épocas en vez de 4 (`--t-ppo-epochs 2`). Se relanza
  alphazero_v2 desde cero.

## 19.8 Resultado final de alphazero_v2 (2026-09-28)
Cortado en la iteración 192/200 por un apagado de la máquina (tasa de aprendizaje ya ≈ 0; checkpoint de la it. 192).
Evaluación final, 512 rachas por ronda, mismas semillas que §18b:

| | R1 | R2 | R3 | R4 | R5 | R6 | P(6 rondas) |
|---|---|---|---|---|---|---|---|
| AZ v2, red sola | 77,7 | 70,5 | 46,5 | 62,7 | 27,7 | 25,0 | 1,11 % |
| AZ v2 + búsqueda legal 512 | 80,1 | 75,2 | 47,1 | 68,6 | 29,9 | 26,6 | 1,55 % |

La red sola es 2,4× PPO v3 / AZ v1; la búsqueda ya solo añade ~0,4 puntos (a v1 le añadía ~1,3): la red absorbió buena
parte de lo que sabía la búsqueda. Seguía mejorando cuando la tasa de aprendizaje se agotó.

## 20. alphazero_v3 (decidido 2026-09-28, desde cero)
1. **Calendario "constante y caída final"** (warmup-stable-decay): tasa constante la mayor parte del entrenamiento y
   caída solo en el último ~15 %; ~600 iteraciones. Permite alargar sin haber gastado la tasa.
2. **Planificador del táctico con W** (expectimax): el valor de un equipo en el combate k es p_k·(1 + media, sobre
   rivales sorteados del pool de la ronda coherentes con la pista, del máximo sobre las opciones de cambio del valor
   del equipo resultante en k+1); profundidad 1–2 y E_ronda después. Así una elección tiene en cuenta los cambios
   futuros ("planear hacia delante"). Entrada por opción del táctico y, si pasa la prueba en el juego real, experto
   a imitar.
3. **Pistas del juego** (tipo más común del próximo rival, 19 valores; estilo, 8 valores) como dos one-hot separados
   en la entrada del táctico; W condicionada a la pista; rivales sorteados coherentes con la pista.
4. Pruebas: capacidad del combatiente con objetivos de Gumbel; revalidar W_final de v2 en el juego real.

### 20.1 Implementación del entrenador (`rl/alphazero_v3.py`, 2026-09-28)

Punto de entrada **nuevo**, `python -m rl.alphazero_v3`, construido sobre `rl/alphazero_v2.py` (que a su vez usa
`rl/alphazero.py`). Los cambios en archivos compartidos no alteran v1 ni v2: se comprobó que v2 da, con la misma
semilla, exactamente las mismas muestras de autojuego, las mismas métricas del PPO del táctico y los mismos pesos
tras la actualización que el código anterior (bit a bit). Todo lo que no se nombra aquí es igual que en v2
(§19.5–19.7): combatiente Gumbel con 256 simulaciones, PPO del táctico **antes** que el entrenamiento del
combatiente, shaping con Φ, tabla V_siguiente, refresco de W cada 20 iteraciones desde que se arma, evaluación de la
red sola cada iteración y con búsqueda de 512 simulaciones cada 5.

**1. Tasa de aprendizaje "calentamiento, constante y caída" (WSD, *warmup-stable-decay*).** Definiciones:
- **Tasa de aprendizaje** (lr): cuánto se mueven los pesos en cada paso del optimizador.
- **Calentamiento** (*warmup*): las primeras `--lr-warmup` = 3 iteraciones la tasa sube en línea recta:
  lr·(it + 1)/4, es decir, 25 %, 50 % y 75 % de `--lr` (3·10⁻⁴). Evita pasos grandes con la red recién iniciada.
- **Tramo constante**: después, `--lr` fija.
- **Caída**: en el último `--decay-frac` = 15 % de `--iterations` (600 por defecto: iteraciones 510–599) la tasa
  baja hasta `--lr` × `--lr-min-frac` (5 %), con forma de coseno (`--lr-decay cosine`, por defecto) o recta
  (`linear`). En la última iteración se llega exactamente al mínimo; más allá de `--iterations`, el mínimo.

**Alargar el entrenamiento.** El plan solo depende de `--iterations` mientras la caída no ha empezado: al reanudar
(`--init-from`) con más iteraciones antes de la caída, la caída se desplaza al nuevo final (p. ej. de 600 a 800: la
caída pasa a 680–799). Si la caída ya empezó (el checkpoint ya entrenó su primera iteración de caída), se mantiene
el plan del checkpoint y se avisa por pantalla: para alargar una ejecución terminada, se reanuda desde un
`ckpt_<pasos>.pt` anterior a su caída (se guardan todos). Si se reanuda con un plan más corto cuya caída ya debería
haber empezado, la caída empieza en la iteración reanudada. El plan se guarda en el checkpoint (`lr_schedule`).
El combatiente y el táctico comparten el optimizador (AdamW, como en v2) y por tanto el calendario: no se vio motivo
para separarlos (el táctico ya controla su paso con el recorte de PPO y las épocas adaptativas de abajo).

**2. Pistas en la entrada del táctico: codificación versión 5** (`--encode-version 5`, obligatoria en v3).
- **One-hot**: un vector de ceros con un 1 en la casilla del valor.
- La pista del encargado sobre el **próximo** rival: tipo más común (19 casillas, la 19.ª = "sin pista", valor 18) y
  estilo (8 casillas), de `RentalView` / `SwapView` (`hint_type`, `hint_style`). Antes de Noland no hay pista: el
  tipo va a la casilla "sin pista" y el estilo a la 0 (así lo da el juego).
- Van en el token de contexto del táctico (alquiler y cambio), **en las mismas posiciones que en el combatiente** (las
  últimas 27 cifras del bloque de campo, que en el táctico eran ceros hasta la v4). Los tamaños no cambian y nada más
  cambia: la observación del combate de la v5 es idéntica a la de la v4 (test).
- **C++**: el codificador C++ solo codifica combates, así que con la v5 funciona como v4 (`encode.native_version`);
  el observador C++ de la búsqueda se sigue usando (`encode.CPP_VERSIONS` = 3, 4, 5).
- **Checkpoints anteriores**: los de v2 (codificación 4) siguen cargando y jugando sin pistas (test con
  `runs/alphazero_v2/latest.pt`); su red también puede iniciar una red v3 (las capas por opción de otra anchura se
  quedan con su inicialización).

**3. Rasgos por opción del evaluador v3.** La interfaz la implementa el otro lado (evaluador, planificador, W
condicionada a la pista); aquí se usa así:
- `option_features_v3(ev, vista, tipo, ctx, opciones, tabla_v_next, pista, planner_cfg)` → [n_opciones,
  N_OPT_FEATURES], al menos [E_ronda, W(normal | pista), W(último o Noland), Δ, Q_plan, Q_plan − media]. La anchura
  de la entrada por opción de la red (`opt_feat`) es **N_OPT_FEATURES**, leída del evaluador cargado (su checkpoint)
  o, si no la tiene, del módulo; no está fijada en el código. El MLP pequeño por opción, la inclinación de la política
  conjunta del alquiler y la entrada al crítico (media y máximo) son los de v2 con esa anchura.
- **Φ** (shaping) = `phi_v3(...)` con la pista. **Q_eval** = `q_eval_v3(rasgos)` (la Q_plan del planificador), y
  **π_eval** = softmax(Q_eval / T), T = `--t-eval-temp` (0,5 victorias), detrás de `--t-imitation` (0 por defecto:
  se fijará con la validación en el juego real del otro lado).
- `tabla_v_next` es la tabla completa por ronda (la misma media móvil de v2); `planner_cfg` es `--planner-cfg`
  (JSON, `{}` = los valores por defecto del módulo).
- **Módulo**: `--team-eval-module`; por defecto, el primero de `rl.team_eval_v3` / `rl.team_eval` que tenga la
  interfaz. Con `--team-eval stub` o `stub:varied`, `rl/az3_stub.py` (6 rasgos; sin planificador: Q_plan = Φ del
  estado tras la opción, y la pista solo mueve un poco W, para las pruebas).
- El refresco de W (§19.6) reutiliza el de v2; si el módulo trae sus propias `generate_battles` / `probe_teams` (por
  ejemplo, para generar combates con pistas), se usan esas.

**4. Señal del táctico.** Definiciones:
- **Época** (§7): una pasada por todas las muestras de la iteración.
- **KL aproximada**: cuánto se ha alejado la política actual de la que jugó las muestras; media de (r − 1) − log r
  con r = π_nuevo(a)/π_viejo(a). Ahora se mide sobre todas las muestras de la iteración, sin gradiente.
- `--t-ppo-epochs` = 2 épocas (como v2) y después **épocas extra mientras la KL aproximada siga por debajo de
  `--t-kl-target` = 0,01**, hasta `--t-max-epochs` = 4 en total. Así se saca más señal de las muestras cuando la
  política apenas se ha movido, y si la KL ya pasa del objetivo tras 2 épocas no se hace ninguna extra (no se vuelve
  a la situación de §19.7, KL 0,11–0,15). `--t-kl-target 0` = exactamente v2. Registros: `tactician/epochs`,
  `tactician/kl_between_epochs`, `tactician/kl_final`.
- **Peso hacia las rondas difíciles** (`--t-round-weight` α, 0 por defecto = v2): dificultad de la ronda r =
  1 − P(completar r) de la última evaluación de la red sola (la misma que usa el currículo de rondas de salida; la
  ronda 7 en adelante cuenta como la 6; sin evaluación, todas iguales). Cada muestra pesa
  (1 − α) + α · dificultad / media de las dificultades de las muestras de la iteración (peso medio 1, así que la
  tasa efectiva no cambia). Pesa los términos de política y de imitación; no el valor ni la entropía. El currículo de
  rondas de salida (`--start-p0`) sigue igual que en v2. Se deja en 0 hasta ver cómo aprende el táctico con las
  pistas y los rasgos nuevos.

**Evaluación fuera del entrenamiento.** `rl.policy.Policy` carga los checkpoints de v3 (anchura y codificación de
sus `args`), y `rl.eval_rounds` añade por defecto los rasgos del táctico de un checkpoint v3 (`--features auto`; la W,
la tabla V_siguiente y `planner_cfg` del checkpoint; `--team-eval` cambia la W). Los demás checkpoints se evalúan
como antes. `python -m rl.az_bench --v3` mide la velocidad del autojuego.

**Velocidad** (`python -m rl.az_bench`, 18 procesos + servidor GPU, pesos de `runs/alphazero_v2/latest.pt`, evaluador
de prueba `stub:varied`, 100 decisiones por proceso; máquina muy cargada por otros trabajos, carga 43–50 con 20
núcleos, así que solo vale la comparación): v3 76,6 decisiones del combatiente/s (221 ms por decisión), v2 en las
mismas condiciones 83,0 (203 ms); diferencia dentro del ruido de la carga (el táctico es el 1 % del tiempo de los
procesos, 22 ms por decisión con el evaluador de prueba). Con la máquina libre v2 daba 388/s (§19.5). El coste real
del táctico dependerá del planificador del evaluador v3; conviene repetir la medida con `runs/team_eval_v3/model.pt`.

**Registros nuevos**: `tactician/hint_none_frac` (fracción de decisiones sin pista), las tres de épocas de arriba,
`tactician/weight_min` / `weight_max` (con α > 0) y `train/lr` con el calendario nuevo.

Comando:

```bash
python -m rl.alphazero_v3 --name alphazero_v3 --team-eval runs/team_eval_v3/model.pt \
    [--team-eval-module rl.team_eval_v3] [--planner-cfg '{"depth": 1}'] [--t-imitation 0]   # resto por defecto
python -m rl.az_bench --v3 --workers 18 --decisions 200 --team-eval runs/team_eval_v3/model.pt
```

### 20.2 Pruebas previas a alphazero_v3 (2026-09-28, `runs/diag_v3/`)
- **T1, capacidad del combatiente con objetivos de Gumbel.** 324 k decisiones del autojuego de v2; redes desde cero de
  1,3 M, 5,5 M y 14,2 M parámetros, validación agrupada por combate. Mejor KL(objetivo‖red) en validación: 0,747 /
  0,748 / 0,738 (la red de v2, con ~7,7 M decisiones, 0,677). Las tres sobreajustan desde la época 8; lo que baja la
  KL es la cantidad de datos (25/50/100 %: 0,82/0,77/0,73), no el tamaño. **Suelo de ruido**: dos búsquedas del mismo
  estado eligen la misma acción solo el 61 % de las veces; KL(objetivo ‖ objetivo medio de 8 búsquedas) ≈ 0,38–0,43,
  que ninguna red puede quitar. Veredicto: **no se agranda la red**; la KL ~0,68 de v2 es sobre todo ruido del objetivo.
- **T2, W_final de v2 (`team_eval_0188`) frente al táctico de v2 en el juego real** (protocolo §19.2.1): alquiler
  +0,166 ± 0,084 victorias donde difieren (1,98 errores típicos; P(mejor) 0,51, correlación prometida/real ≈ 0);
  cambio +0,062 ± 0,077 (no significativo). W ordena todas las opciones mejor que la red (Spearman 0,51–0,54 frente a
  0,45–0,46), pero su favorita no gana claramente más. Veredicto: no justifica imitación (> 0) por la regla de §19.3.

### 20.3 Implementación del evaluador: pistas, W condicionada a la pista y planificador (2026-09-28)

Archivos: `rl/determinize.py` (pista y muestreadores), `rl/team_eval.py` (W_v3, planificador, interfaz v3),
`rl/team_eval_data.py`, `rl/team_eval_refresh.py`, `rl/team_eval_validate.py`, `rl/tactician_features.py`;
pruebas en `tests/python/test_team_eval_v3.py`.

#### 20.2.1 La pista del asistente

Definiciones:
- **Pista**: antes de cada combate normal, el asistente de la Factory dice dos cosas del próximo rival: su **tipo más
  común** (`hint_type`, 0..17 el tipo; 18 = "ninguno") y su **estilo** (`hint_style`, 0..8). La ven `RentalView`
  (del primer rival de la ronda), `SwapView` (del siguiente) y `BattleView` (del rival actual).
- **Estilo**: el juego tiene 7 listas de movimientos (preparación, lento y seguro, aguante, alto riesgo, debilitar,
  impredecible, clima). **Puntos de un estilo** = cuántos de los 12 movimientos del rival están en su lista (un
  movimiento en dos listas, Snatch, cuenta solo para la primera). Un estilo **se cumple** si llega a su umbral
  (3 para los tres primeros, 2 para los demás).

Cómo la calcula el juego (`battle_factory.c`, `GetOpponentMostCommonMonType` / `GetOpponentBattleStyle`), y cómo la
replica `hint_of_team` al pie de la letra:
- tipo: se cuentan los tipos de las 3 especies (un tipo doble cuenta sus dos tipos; uno simple, una vez); se
  recorren los tipos en orden guardando el primero con más cuenta y el último que empató con él; si el primero
  supera estrictamente al empatado, ese tipo; si no, 18;
- estilo: ningún estilo cumplido → 0; uno o dos → el último de ellos en el orden de las listas; más de dos → 8
  ("sin estilo único"; el `encode` del táctico solo tiene 8 casillas, así que ahí el 8 no marca ninguna).
- Antes del combate de Noland no hay pista (`NO_HINT` = (18, 0), que también es una pista posible de un rival
  normal: por eso "desconocida" no puede ser (18, 0)).

Prueba contra el juego: en 600 pantallas de alquiler y 40+ pantallas de cambio reales (`SimBackend`), la pista que
calcula el juego coincide siempre con `hint_of_team` sobre los sets reales del rival (leídos de `gFrontierTempParty`
solo en la prueba). Frecuencias (1.500 rivales del primer combate de ronda): tipo "ninguno" 65 %, agua 8 %, normal
5 %; estilo 0 el 42 %, 4 (alto riesgo) el 39 %, estilo 8 una vez.

#### 20.2.2 Muestreadores coherentes con la pista

- **Muestreo por rechazo**: se sortea un equipo rival entero con el muestreador de siempre, se calcula su pista como
  el juego y solo se acepta si es la pista oída; si no, se vuelve a sortear. Así el equipo aceptado sigue exactamente
  la distribución del muestreador **condicionada** a la pista (prueba: frecuencias de especies iguales a las de los
  sorteos libres que por casualidad tienen esa pista).
- `sample_factory_sets(..., hint=)` (entrenamiento) y `sample_determinization(..., hint=)` (el estricto, inferencia):
  cuentan todos los Pokémon del rival, también los debilitados ya vistos (sus movimientos no revelados cuentan para el
  estilo, así que se sortean para la comprobación y luego se dejan como están). Sorteos vectorizados por lotes (64,
  256, luego 1.024) hasta 4.096 equipos; si ninguno cumple, se usa el último y se cuenta en
  `hint_stats["fallback"]`. Con Noland, o `hint=None`, todo es como antes (mismo código).
- Coste: pista común ((18, 4)): 0,9 ms (sets de la Factory) / 2,6 ms (estricto); pistas raras 3–9 ms / 13–46 ms,
  con algún *fallback* en pistas casi imposibles para ese pool (p. ej. (6, 6): 45 de 100).
- `battle_hint(view, noland)` da la pista a pasar (None antes de Noland). `determinization(..., hint=)` la reenvía.

#### 20.2.3 W condicionada a la pista (W_v3)

- **W(equipo, ronda, tipo, pista)**: como W (§19.2) pero sabiendo la pista del rival de ese combate. **Pista
  desconocida**: un símbolo propio (ni (18, 0) ni ninguna pista real) para los combates cuya pista no se ha oído
  todavía (los futuros), para Noland y para W "sin rival concreto" (E_ronda de los combates siguientes).
- **Abandono de pista** (*hint dropout*): en el entrenamiento, el 30 % de los combates se presentan con la pista
  desconocida; así W(· | desconocida) aprende la media sobre las pistas (lo que se espera sin haberla oído).
- **Red**: la de §19.2.1 (`TeamNet(hint=True)`); la parte del equipo no depende de la pista, así que cada equipo se
  codifica una vez y la cabeza se evalúa por combate: entrada de la cabeza = [media, líder, ronda, embedding de la
  pista (tipo 19 + desconocido, estilo 9 + desconocido, sumados), lectura por atención de los 3 Pokémon con una
  consulta hecha con la pista] → MLP → 3 logits. Pérdida: BCE por combate (con una sola pista por equipo es la
  binomial de v1). Una W de v1 (sin pista) sigue funcionando en todas partes: la pista se ignora.
- **Datos** (`rl.team_eval_data`, con el combatiente de alphazero_v2): cada combate guarda la pista de su rival
  (`hint_of_team` de los sets sorteados; NO_HINT para Noland) y el combatiente la ve en su observación, como en un
  combate real.

**Datos** (`runs/team_eval_v3/data`, 45 minutos con 18 procesos, combatiente `runs/alphazero_v2/latest.pt` voraz y
sin búsqueda): **1.354.324 equipos, 21,6 millones de combates** (8.000 combates/s; 16 por equipo), 0,13 % cortados.
**Entrenamiento**: 8 épocas, lotes de 2.048 equipos (todos sus combates), lr 2·10⁻³ *one-cycle*, abandono de pista
0,3 (143 s en la GPU; 630.000 parámetros). Validación: 5 % de los equipos (67.716; 1,08 M combates), separados por
equipo.

Medidas (definidas en §19.2.1: calibración, Brier, AUC). "Con pista" = cada combate con la pista de su rival;
"desconocida" = la misma red con el símbolo desconocido (lo que se espera sin haber oído la pista):

| | Brier con pista | Brier desconocida | Brier constante | AUC con pista | AUC desconocida |
|---|---|---|---|---|---|
| Todos los combates | 0,1427 | 0,1446 | 0,1742 | 0,775 | 0,766 |
| Combates normales (sin Noland, que no tiene pista) | 0,1288 | 0,1310 | 0,1483 | 0,747 | 0,735 |

Calibración (deciles de W predicha, con pista): predicho 0,389 / 0,561 / 0,670 / 0,752 / 0,813 / 0,855 / 0,888 /
0,915 / 0,939 / 0,967, observado 0,392 / 0,563 / 0,670 / 0,754 / 0,814 / 0,856 / 0,887 / 0,913 / 0,938 / 0,966. A
nivel de equipo (pista desconocida) explica el 98 % de la varianza real entre equipos (correlación 0,87).

Por pista (combates normales; Brier con pista / desconocida; AUC con pista / desconocida):

| Pista | Combates | Victorias | Brier | AUC |
|---|---|---|---|---|
| tipo "ninguno" (18) | 610.234 | 0,822 | 0,1289 / 0,1294 | 0,739 / 0,736 |
| agua | 78.152 | 0,796 | 0,1394 / 0,1450 | 0,746 / 0,724 |
| psíquico | 44.898 | 0,786 | 0,1403 / 0,1449 | 0,768 / 0,749 |
| normal | 39.217 | 0,830 | 0,1259 / 0,1280 | 0,732 / 0,715 |
| tierra | 30.415 | 0,835 | 0,1178 / 0,1247 | 0,762 / 0,720 |
| planta | 17.927 | 0,872 | 0,0996 / 0,1083 | 0,748 / 0,684 |
| fuego | 17.912 | 0,849 | 0,1064 / 0,1133 | 0,785 / 0,757 |
| dragón | 4.794 | 0,650 | 0,1799 / 0,2013 | 0,774 / 0,749 |
| estilo 0 (ninguno) | 377.538 | 0,806 | 0,1341 / 0,1364 | 0,754 / 0,743 |
| estilo 4 (alto riesgo) | 356.656 | 0,820 | 0,1305 / 0,1322 | 0,733 / 0,722 |
| estilo 2 (lento y seguro) | 102.420 | 0,837 | 0,1209 / 0,1231 | 0,741 / 0,727 |

(el resto de tipos y estilos, en `runs/team_eval_v3/model_heldout.json`; todos mejoran con la pista.)

**¿Mejora la pista?** Sí, pero poco en media: en los combates normales el Brier baja un 1,7 % (0,1310 → 0,1288) y
el AUC sube 0,012; es el 13 % de lo que W ya mejora sobre la constante. Casi toda la ganancia está en el 35 % de
combates con un tipo concreto (p. ej. dragón: Brier 0,201 → 0,180, y W pasa de 0,75 a 0,65, la tasa real; planta:
AUC 0,68 → 0,75): ahí la pista dice mucho del rival. Con tipo "ninguno" (65 %) apenas cambia nada. Por ronda, el AUC
con pista sube 0,014–0,03 en todas las casillas normales / último.

#### 20.2.4 El planificador (Q_plan)

Definiciones, de abajo arriba:
- **Hoja**: V₀(T, k) = Φ = E_ronda + P(completar) · V_siguiente desde el combate k, con p_k = W(· | pista de k) y
  los siguientes con la pista desconocida.
- **Opciones de cambio** tras ganar al rival del combate k: quedarse, o uno de los 3×3 cambios con los Pokémon de ese
  rival, vistos como se ven en la pantalla de cambio (el enmascarado del generador de datos: movimientos revelados,
  objeto y habilidad con las frecuencias medidas; estadísticas desconocidas); el nuevo ocupa el hueco del que sale.
- **Valor con planificación** (expectimax de profundidad D):
  V_D(T, k) = p_k(T | h_k) · (1 + media sobre rivales r de [ máx sobre opciones o de V_{D−1}(T ⊕ o, k+1) ])
  para k < 7; en k = 7, o con D = 0, la hoja. Los rivales r del combate k se sortean del pool de la ronda como los
  sortea el juego (sin especies repetidas ni objetos repetidos, sin las especies que hay en pantalla) y coherentes
  con la pista oída h_k; más adelante las pistas son desconocidas y los rivales del combate k+1 excluyen además las
  especies del rival r.
- **Números aleatorios comunes**: los mismos rivales (y el mismo enmascarado) para todas las opciones de una
  decisión, para que las diferencias entre opciones no sean ruido del sorteo.
- **Haz (beam) en profundidad 2**: para cada (opción, rival) solo se expanden las 3 mejores opciones del nivel 1
  según V₀. Como V₁ ≥ V₀ (quedarse siempre es posible), V₂ ≥ V₁ ≥ V₀ con los mismos sorteos (probado).
- **Q_plan(opción)** = V_D(equipo tras la opción, k). Con D = 0 es Φ con pista (Q_eval de §19.3).
- Por defecto: D = 1, 8 rivales, (D = 2: 4 rivales del nivel 2 por rival, haz 3). Todas las W de una decisión se
  calculan de una vez sobre los equipos distintos (se deduplican).

Prueba de la definición: para varias opciones de alquiler y cambio, un cálculo a mano con bucles (objetos `Team`,
`ev.w`) da exactamente el mismo Q_plan que el cálculo vectorizado.

**Coste** (ms por decisión, W_v3 = `runs/team_eval_v3/model.pt`, 1 hilo, máquina libre; 10 decisiones de cada
clase; "equipos" = equipos distintos evaluados por la red):

| | alquiler CPU | cambio CPU | alquiler GPU | cambio GPU | equipos (alquiler / cambio) |
|---|---|---|---|---|---|
| Profundidad 0 (Φ) | 5 | 2 | — | — | 60 / 10 |
| **Profundidad 1** (8 rivales) | **63** | **19** | 8 | 4 | 1.560 / 510 |
| Profundidad 2 (8 × 4 rivales, haz 3) | 522 | 167 | 32 | 13 | 12.600 / 4.540 |

Casi todo el coste es la red (el transformer de W, ~40 µs por equipo en un hilo). Con 18 procesos a la vez (la
validación de abajo) la profundidad 1 costó 106 / 36 ms y la 2, 939 / 232 ms. **Recomendación: `{}` = profundidad 1,
8 rivales** (en CPU, ~60 ms por alquiler y ~20 ms por cambio, pequeño frente a los combates); la profundidad 2 no
eligió mejor en la validación y cuesta 8–9 veces más (solo tendría sentido con W en la GPU).

#### 20.2.5 Interfaz para el entrenador (`rl.team_eval`, también `TeamFeaturesV3` en `rl.tactician_features`)

```
N_OPT_FEATURES = 7;  WIN_SCALE = 10
option_features_v3(ev, view, kind, ctx, options, v_next_table, hint, planner_cfg) -> [n_opt, 7] (float32)
    # [E_ronda | pista, W(normal | pista), W(último o Noland), Δ, Q_plan, Q_plan − media(Q_plan), Φ con pista]
    # las columnas en victorias (0, 3, 4, 5, 6) van divididas por WIN_SCALE: entradas de orden 1 para la red
phi_v3(ev, view, kind, ctx, v_next_table, hint, planner_cfg) -> float          # en victorias
    # cambio: Q_plan del equipo actual (quedarse); alquiler: media de Q_plan sobre los alquileres legales
q_eval_v3(features) -> [n_opt]   # Q_plan en victorias (columna 4 × WIN_SCALE)
generate_battles(...), probe_teams(...)   # para el refresco de W (combates con pista)
```
`v_next_table[r]` = victorias esperadas desde el inicio de la ronda r (se usa `table[challenge + 2]`, la ronda
siguiente); `hint` = (hint_type, hint_style) de la vista, o None; `planner_cfg` = `PlannerConfig` o dict
(`depth`, `n_opp`, `n_opp2`, `beam`, `seed`). `phi_v3` reutiliza el plan de `option_features_v3` de la misma decisión.
`TeamFeaturesV3(path, planner_cfg=..., v_next=...)` hace lo mismo desde un backend (toma la pista de la vista y la
anula antes de Noland) y sirve de `obs_hook`. El refresco (`team_eval_refresh`) funciona igual con W_v3: los
combates generados y los reales guardan la pista y el ajuste fino usa la misma pérdida por combate con abandono 0,3.

#### 20.2.6 Validación en el juego real (condición de §19.3), agentes de alphazero_v2

`python -m rl.team_eval_validate run --version 3` (`runs/team_eval_v3/validate.pkl`, 942 s con 18 procesos). La
prueba de §19.2.1 con los agentes de v2:
- **π_red**: el argmax de la red del táctico de alphazero_v2 (sus rasgos por opción, de la W con la que se entrenó,
  `team_eval_0188.pt`). **Criterios** (argmax, con la pista del próximo combate y W_v3): Q_plan de profundidad 1
  (`q_plan1`) y 2 (`q_plan2`), Φ con pista (`phi_h`, profundidad 0) y E_ronda con pista (`e_round`, sin V_siguiente).
- **Estados**: 126 alquileres (primera decisión de rachas desde la ronda 1) y 126 cambios de las rondas 3–5 (sin los
  de antes de Noland), de partidas de v2 (combatiente voraz, táctico muestreado). **Resultado real**: 64 semillas
  emparejadas por opción, jugando hasta el final de la ronda con el combatiente y el táctico de v2 voraces (con sus
  rasgos). Se juegan todos los cambios y, en el alquiler, la unión de las 6 mejores opciones de la red y de cada
  criterio (10,8 de media). V_siguiente = la tabla del checkpoint de v2 (se usa también en "Φ real").
- Columnas como en §19.2.1: ganancia de π_eval sobre π_red donde eligen distinto (± error típico entre estados), en
  todos los estados, P(mejor) donde difieren y correlación entre la ganancia prometida (diferencia de la puntuación)
  y la real.

| | Criterio | Difieren | Ganancia donde difieren | En todos | P(mejor) | Corr. prometida/real |
|---|---|---|---|---|---|---|
| Alquiler, victorias | q_plan1 | 111/126 | **+0,185 ± 0,077** (2,4 SE) | +0,163 ± 0,068 | 0,60 | −0,05 |
| | q_plan2 | 109/126 | **+0,193 ± 0,080** (2,4 SE) | +0,167 ± 0,069 | 0,58 | −0,02 |
| | phi_h | 110/126 | +0,151 ± 0,079 (1,9 SE) | +0,131 ± 0,069 | 0,54 | +0,03 |
| | e_round | 110/126 | +0,148 ± 0,079 (1,9 SE) | +0,129 ± 0,069 | 0,55 | +0,03 |
| Alquiler, Φ real | q_plan1 | 111/126 | **+0,54 ± 0,24** (2,3 SE) | +0,48 ± 0,21 | 0,59 | −0,08 |
| | q_plan2 | 109/126 | **+0,56 ± 0,24** (2,3 SE) | +0,48 ± 0,21 | 0,57 | −0,04 |
| | phi_h | 110/126 | +0,43 ± 0,25 (1,8 SE) | +0,38 ± 0,21 | 0,58 | −0,00 |
| | e_round | 110/126 | +0,42 ± 0,24 (1,7 SE) | +0,37 ± 0,21 | 0,57 | −0,01 |
| Cambio, victorias | q_plan1 | 81/126 | +0,109 ± 0,064 (1,7 SE) | +0,070 ± 0,041 | 0,57 | +0,23 |
| | q_plan2 | 80/126 | +0,075 ± 0,054 (1,4 SE) | +0,048 ± 0,035 | 0,57 | +0,32 |
| | phi_h | 75/126 | +0,058 ± 0,051 (1,1 SE) | +0,035 ± 0,030 | 0,55 | +0,12 |
| | e_round | 78/126 | +0,068 ± 0,049 (1,4 SE) | +0,042 ± 0,030 | 0,56 | +0,11 |
| Cambio, Φ real | q_plan1 | 81/126 | **+0,34 ± 0,16** (2,1 SE) | +0,22 ± 0,10 | 0,57 | +0,32 |
| | q_plan2 | 80/126 | +0,25 ± 0,15 (1,7 SE) | +0,16 ± 0,09 | 0,56 | +0,40 |
| | phi_h | 75/126 | +0,18 ± 0,14 (1,3 SE) | +0,11 ± 0,08 | 0,53 | +0,20 |
| | e_round | 78/126 | +0,21 ± 0,14 (1,5 SE) | +0,13 ± 0,08 | 0,54 | +0,12 |

Victorias medias hasta el final de la ronda: alquiler: red 5,89, q_plan1 6,05, q_plan2 6,06, phi_h 6,02, e_round
6,02 (la mejor opción jugada, 6,46); cambio: red 3,74, q_plan1 3,81, q_plan2 3,79, phi_h 3,78, e_round 3,79 (4,10).
Comparaciones directas entre criterios (misma prueba, todos los estados): q_plan1 − phi_h = +0,031 ± 0,024
(alquiler) y +0,035 ± 0,024 (cambio) victorias; q_plan2 − q_plan1 = +0,004 ± 0,023 y −0,022 ± 0,022.

**Veredicto** (criterio de §19.3: > 2 errores típicos mejor que la red):
- **q_plan1 (profundidad 1): sí en el alquiler** (2,4 SE en victorias, 2,3 SE en Φ real); **en el cambio, solo en Φ
  real** (2,1 SE; 1,7 SE en victorias). Es el mejor criterio en el cambio y casi el mejor en el alquiler
  (q_plan2, +0,01).
- **q_plan2**: sí en el alquiler (2,4 / 2,3 SE), no en el cambio (1,4 / 1,7 SE); no mejora a la profundidad 1.
- **phi_h y e_round** (W_v3 sin planificar): no (1,1–1,9 SE), aunque ahora siempre por delante de la red (con v1 y
  la W de v1 daban +1,2 y −0,4 SE, §19.2.1).
- Planificar añade poco sobre Φ con pista (+0,03 victorias, 1,3–1,5 SE) y la diferencia prometida apenas predice la
  real en el alquiler (correlación ≈ 0; en el cambio 0,2–0,4). Con 16 comparaciones, un 2,1–2,4 SE aislado podría
  ser suerte; lo robusto es que todos los criterios con W_v3 salen por delante de la red de v2 y que Q_plan
  (profundidad 1 o 2) va por delante de phi_h y e_round en las cuatro tablas.
- Para el entrenador: q_plan1 cumple la condición de §19.3 en el alquiler (y en el cambio en unidades de Φ), así
  que puede darse un coeficiente de imitación pequeño con **π_eval = softmax(Q_plan de profundidad 1 / T)**; la
  profundidad 2 no aporta y cuesta 8 veces más. Nota: la prueba compara contra la red de v2 al final de su
  entrenamiento; la red de v3 empieza desde cero, así que al principio el experto le sacará más ventaja.

### 20.4 Lanzamiento de alphazero_v3 (2026-09-28)
- W_v3 (`runs/team_eval_v3/model.pt`, combatiente de v2, pistas) y planificador de profundidad 1 (8 rivales), la
  configuración por defecto: la profundidad 2 no añade nada en la validación y cuesta ~8× más.
- **Imitación**: el planificador de profundidad 1 elige mejor que el táctico de v2 en el juego real en los alquileres
  (+0,185 ± 0,077 victorias, 2,4 errores típicos) y en los cambios en unidades de Φ (2,1; 1,7 en victorias), y todos
  los criterios de W_v3 van por delante de la red. Cumple la condición de §19.3 en los alquileres: se activa la
  imitación con un coeficiente pequeño, `--t-imitation 0.1` (π_eval = softmax(Q_plan / T), T = 0,5 victorias), para
  que la experiencia real (PPO) siga pudiendo corregir al experto.
- Estilo de pista 8 ("más de dos estilos"): queda fuera de las 8 casillas del one-hot y se codifica como todo ceros,
  que es distinguible de "sin pista" (estilo 0): no se pierde información.

### 20.5 Objetivo y orden de trabajo tras alphazero_v3 (decidido 2026-09-29)
- **Objetivo**: completar el reto **de forma consistente** (no basta con completarlo una vez). Exige ganar ~98,4 % de
  los combates; v3 está en ~95 % (rondas 1–2) y ~83 % (rondas 5–6).
- **Prioridad**: mejorar el agente ahora; el enfoque de investigación (varias semillas, ablaciones, intervalos de
  confianza) se hará después.
- **Orden**: (1) evaluación final de v3 (512 rachas/ronda, red sola y búsqueda 512); (2) **autopsia de derrotas** de
  las rondas 5–6: clasificar cada derrota en azar / equipo / error del combatiente, y buscar patrones por rival y por
  evento; (3) diseñar la v4 atacando la causa dominante.

## 21. Autopsia de derrotas (`rl/autopsy.py`, 2026-09-29)

Objetivo (§20.5): antes de diseñar la v4, saber **por qué** pierde alphazero_v3 los combates de las rondas 5–6. La
herramienta toma un checkpoint de v3, junta N derrotas (500 por defecto), clasifica cada una en *azar*, *equipo* o
*error del combatiente*, y busca patrones por rival, pista y evento. Es **solo diagnóstico**: usa información
perfecta, que nunca entra en el entrenamiento.

### 21.1 Definiciones (de abajo arriba)

- **Semilla del RNG del combate**: el juego saca todo lo aleatorio (críticos, precisión, daño, efectos secundarios,
  duración del sueño...) de un único número, `gRngValue`, que cambia en cada uso. Fijar ese número antes del primer
  fotograma de un combate fija toda su suerte; cambiarlo da "el mismo combate con otra suerte".
- **Receta de repetición**: el simulador es determinista, así que una derrota se guarda como la semilla del entorno
  de la racha más **todas** las acciones que se dieron al juego desde que empezó (alquiler, cambios, cada decisión
  del combate). Volver a aplicarlas reconstruye cada estado exactamente. Se comprueba con un *hash* del estado al
  final del combate (equipos, Pokémon en combate, RNG, resultados, número de fotogramas; no los bytes crudos del
  estado, que contienen punteros distintos en cada proceso).
- **Decisión previa**: la decisión del táctico justo antes del combate perdido: el cambio (combates 2–7 de la ronda)
  o el alquiler (combate 1). Sus **opciones** son las de §19.2 (cambio: quedarse o uno de los 3×3 cambios, hasta 10;
  alquiler: hasta 60).
- **Inicio del combate**: el estado tras aplicar una opción en la decisión previa, justo antes del primer fotograma
  del combate. Para comparar opciones se **reescribe el equipo rival verdadero** (el que generó el juego con nuestra
  opción real; con Noland depende de nuestro equipo, porque excluye nuestros sets) y se fija la semilla del RNG.
  Prueba: con la opción real, el rival verdadero y el RNG propio del juego, las acciones registradas reproducen el
  combate real bit a bit (mismo *hash* final).
- **Combatiente fuerte**: la búsqueda MCTS de §15 con muchas simulaciones (`--strong-sims`, 1.024 por defecto,
  8 determinizaciones) en **modo perfecto** (ve el estado verdadero: el equipo rival real y sus PS exactos; solo
  el azar futuro le es desconocido). Es un techo: juega mejor que cualquier jugador con información legal.
  `rl.autopsy` no es un punto de entrada de entrenamiento: no llama a `mark_training()`, ningún módulo de
  entrenamiento lo importa, y se niega a arrancar dentro de un proceso marcado como entrenamiento (tests).
- **P(ganar | estado)**: la fracción de victorias al jugar desde ese estado hasta el final con el combatiente fuerte,
  repitiendo con varias semillas del RNG. Con M semillas, su error típico es ≤ 0,5/√M (0,09 con M = 32).
- **Números aleatorios comunes**: comparar dos opciones con **las mismas** semillas, para que la diferencia no sea
  ruido de la suerte (las dos empiezan con los mismos números aleatorios). Se usan en (b) y (c).
- **Rebobinar**: volver al estado de una de nuestras decisiones del combate real (reconstruido con la receta) y
  jugar desde allí otra cosa. Al rebobinar se sortea de nuevo la suerte del turno (`Gen3Game.redraw_turn`: semilla
  nueva, y la elección del rival de ese turno, que su IA ya había hecho, se deshace), como en las raíces de §15.

### 21.2 Qué hace

**1. Recogida** (`collect.jsonl`). Rachas que empiezan al principio de la ronda k (k ∈ `--rounds`, 5 y 6 por
defecto, alternando; 3 opcional), con el protocolo de `rl.eval_rounds` (la racha se para al completar la ronda k o
al perder). Juegan los agentes de v3 **como se evalúan**: táctico = su red, voraz, con sus rasgos por opción (W y
planificador, `features_hook` del checkpoint); combatiente = su red, voraz (`--battler net`), o la búsqueda legal de
la evaluación (`--battler search`: 512 simulaciones, 8 determinizaciones). De **cada** combate (también de los
ganados, para tener la tasa base) se guarda: nuestro equipo y el rival **verdadero** (especie, movimientos, objeto y
sets de la Factory; solo para el análisis), la pista, si es Noland, y los **eventos**. De las derrotas, además, la
receta de repetición y el registro de turnos (en cada decisión nuestra: turno, acción, Pokémon en combate de los dos
lados con sus PS verdaderos, y los eventos hasta la decisión siguiente).

**Eventos** (definidos por el mensaje que imprime el juego). Se añadió al núcleo C (`src/gen3/host.c`) un gancho
que ve cada mensaje de combate (`CONTROLLER_PRINTSTRING`) con los combatientes implicados; `Gen3Game.log_messages()`
lo registra solo para esa partida (los clones no registran: la búsqueda no se ve afectada; sin registrar, el coste es
una comprobación por mensaje) y `take_messages()` lo devuelve. Eventos, cada uno con el lado al que le pasa (`own_`
nosotros, `opp_` el rival): `crit` ("¡Un golpe crítico!", de quien ataca), `miss` (falla la precisión), `ohko_hit`
("¡KO de un golpe!"), `confusion_hit` (se hiere a sí mismo por la confusión), `flinch` (retrocede), `freeze` /
`paralysis` / `sleep` / `confused` (le causan el estado), `frozen_turn` / `full_para` / `asleep_turn` / `love_turn`
(pierde el turno por congelación, parálisis total, sueño o enamoramiento), `wake` (despierta), `focus_band` (aguanta
con Cinta Focus), `explosion` (usa Explosión / Autodestrucción), `ohko_move` (usa Fisura, Perforador, Guillotina o
Frío Polar), `perish_song` (Canto Mortal) y `crit_ko` (un crítico tras el que cae el objetivo de ese mismo golpe:
"decidido por un crítico"). **Balance de suerte** de un combate = eventos que nos perjudican (crítico del rival,
fallo nuestro, parálisis total nuestra...) menos los que nos benefician (los simétricos).

**2. Análisis de cada derrota** (`analysis.jsonl`; en paralelo; reanudable):
- **(a) Azar**: **P0** = P(ganar) desde el inicio del combate con la opción real y el rival verdadero, con el
  combatiente fuerte y `--luck-seeds` = 32 semillas. Si P0 es alta, el combate era ganable y se perdió por suerte o
  por el combatiente; si es baja, el emparejamiento era malo. También P0 con la red sola (`p0_net`).
- **(b) Error del combatiente**: desde nuestra **última** decisión hacia atrás, la búsqueda fuerte (en modo legal y
  en modo perfecto) dice qué jugaría en el estado real. Donde difiere de nuestra jugada: P_d = P(ganar) jugando fuerte
  desde ahí (`--mistake-seeds` = 8 semillas; se corta en cuanto ya no puede llegar a 0,5) y, si P_d ≥ 0,5, P_nuestra
  = P(ganar) jugando nuestra jugada y después fuerte, con las mismas semillas. **Error decisivo** = la decisión más
  tardía con P_d ≥ 0,5 y Δ = P_d − P_nuestra ≥ `--mistake-delta` (0,25). La definición sin la condición de Δ
  ("difiere y la posición seguía ganable") se informa aparte (`p_found`); la condición evita llamar error a una
  jugada distinta pero igual de buena (dos búsquedas del mismo estado coinciden solo el ~61 %, §20.2). Se revisan
  como mucho `--max-rewinds` = 4 decisiones distintas por modo. Se informa la fracción de derrotas con error
  decisivo, el turno típico, cuántas decisiones antes del final, y qué jugamos frente a qué habría jugado la
  búsqueda (movimiento concreto o cambio).
- **(c) Equipo**: las opciones de la decisión previa contra el **mismo** rival verdadero y con las mismas semillas.
  Etapa 1 (criba): todas las opciones con la red sola (`--team-screen-seeds` = 16). Etapa 2: las `--team-top` = 2
  mejores alternativas de la criba con el combatiente fuerte, en las primeras `--team-seeds` = 16 semillas de (a)
  (nuestra opción: su resultado de (a) en esas mismas semillas). **Ganancia de equipo** = mejor alternativa − la
  nuestra. Para cada opción se guardan también W (la de ese combate: normal, o último / Noland) y Q_plan del
  planificador: ¿veían la alternativa mejor? (`planner_saw` = Q_plan la prefería a la nuestra; `w_saw`, igual con W).

**Etiqueta de cada derrota**, con esta **precedencia** (la primera que se cumple):
1. `battler_error`: hay error decisivo (modo legal por defecto, `--label-mistake-mode`). Va primero porque es lo
   que muestra la partida real: una posición ganable se tiró.
2. `team`: una alternativa gana ≥ `--team-gain` = 0,3 más.
3. `luck`: P0 ≥ `--luck-p` = 0,8: el combatiente fuerte gana casi siempre ese inicio y no hubo error claro.
4. `mixed`: 0,5 ≤ P0 < 0,8, sin error ni equipo mejor.
5. `bad_matchup`: P0 < `--bad-p` = 0,5 y ninguna alternativa claramente mejor: el rival era malo para cualquiera de
   nuestras opciones.

Los indicadores no son exclusivos: el informe da también sus combinaciones (p. ej. `mistake+p0_high`), y
`--report-only` vuelve a etiquetar con otros umbrales sin repetir el análisis.

**3. Informe** (`runs/autopsy_<nombre>/`): `summary.json`, `losses.csv` (una fila por derrota) y `report.md`
(corto, en español). Fracciones de cada etiqueta con **intervalo bootstrap al 95 %** (se remuestrean las derrotas
2.000 veces con reemplazo y se toman los percentiles 2,5 y 97,5 de la fracción), en total, en Noland, sin Noland y
por ronda. **Patrones**: para cada rasgo (especie o set del rival, estilo y tipo de la pista, evento, especie
nuestra, ronda, combate de la ronda, Noland, balance de suerte), **lift** = P(rasgo | derrota) / P(rasgo |
victoria) (lift 3 = el rasgo aparece 3 veces más a menudo en las derrotas), con un IC al 95 % aproximado (del
logaritmo del cociente). Como las derrotas duran más (en una muestra de 60: 21,7 decisiones nuestras frente a 12,5
en las victorias), parte del lift de un evento es solo más tiempo para que ocurra; por eso se da también la tasa
por cada 10 decisiones nuestras.

### 21.3 Cómo lanzarlo

```bash
python -m rl.autopsy --ckpt runs/alphazero_v3/latest.pt --name v3        # 500 derrotas, rondas 5-6, 18 procesos + GPU
python -m rl.autopsy --ckpt ... --name v3_search --battler search        # recogida con la búsqueda legal 512
python -m rl.autopsy --ckpt ... --name v3_r3 --rounds 3,5,6
python -m rl.autopsy --name v3 --report-only [--team-gain 0.2 ...]       # rehace el informe (y las etiquetas)
python -m rl.autopsy --estimate [presupuestos]                           # coste estimado, sin ejecutar nada
```
Si se interrumpe, al relanzar con el mismo `--name` reutiliza `collect.jsonl` y salta las derrotas ya analizadas.

### 21.4 Coste

Medido en 14 derrotas de las rondas 5–6 (presupuestos de semillas por defecto, pocas simulaciones, en CPU) para
contar decisiones, y con la velocidad de la búsqueda de 2.048 simulaciones en la máquina libre (`runs/diag_az1`: ~215
ms por decisión y proceso, con 18 procesos y el servidor GPU; proporcional a las simulaciones, §15b). Una derrota
cuesta, en decisiones del combatiente fuerte: (a) ~640 (32 semillas × ~20 decisiones: los combates perdidos son
largos), (b) ~24 comparaciones + ~380 de rebobinados, (c) ~640 (2 alternativas × 16 semillas × ~20); más ~5.000
decisiones de la red (criba y P0 de la red), que son baratas. La recogida de 500 derrotas son unos minutos (la red
gana el 83 % de los combates de las rondas 5–6: ~3.000 combates).

| Presupuesto | Horas (500 derrotas, 18 procesos) |
|---|---|
| Plan inicial: 2.048 simulaciones, 32 semillas en (a) y en las **10** opciones de (c) | ~11 |
| 2.048, 32 semillas, (c) con 3 alternativas × 32 semillas | ~5 |
| 2.048, resto por defecto | ~2,9 |
| **1.024 simulaciones, resto por defecto (propuesto, valores por defecto)** | **~1,5** |
| 2.048, 16 semillas en (a) y (c), 2 rebobinados | ~2,1 |

La estimación tiene ±30 % (la calibración es pequeña y la búsqueda fuerte de verdad juega combates de otra
duración). `--estimate` la recalcula para cualquier combinación de presupuestos.

### 21.5 Pruebas (`tests/python/test_autopsy.py`)
- Los identificadores de mensajes coinciden con los del decomp; el análisis de mensajes asigna bien el lado de cada
  evento y detecta `crit_ko`; el registro de mensajes funciona en una partida real y los clones no registran.
- Extremo a extremo con 2 derrotas y presupuestos mínimos (recogida, (a), (b), (c), informe y `--report-only`).
- Determinismo: la receta reproduce el combate en **otro proceso** (mismo *hash*); el inicio de combate con la opción
  real, el rival verdadero y el RNG propio reproduce el combate real; la misma semilla da el mismo resultado (red,
  búsqueda perfecta y legal, y los rebobinados); todas las opciones de (c) se enfrentan al rival verdadero.
- Modo perfecto solo aquí: `rl.autopsy` no está entre los puntos de entrada de entrenamiento, nadie del entrenamiento
  lo importa, y en un proceso marcado como entrenamiento se niega a arrancar y a construir la búsqueda perfecta.
- La lógica del error decisivo (recorrido hacia atrás; las posiciones ya perdidas y las jugadas igual de buenas se
  saltan) y la precedencia de las etiquetas.

### 21.6 Pendiente de decidir
- **Presupuesto**: 1.024 simulaciones (propuesto, ~1,5 h) o 2.048 (~2,9 h, o ~2,1 h con 16 semillas).
- **Umbral del error decisivo**: Δ ≥ 0,25 (con 8 semillas, 2 de 8) es un añadido a la definición pedida; con
  `--mistake-delta 0` se recupera la definición sin Δ (las dos se informan siempre). ¿Modo legal (lo que un jugador
  podía saber; por defecto) o perfecto para la etiqueta?
- **(c) en dos etapas**: la criba con la red decide qué alternativas juega el combatiente fuerte; una alternativa que
  la red juega mal pero el fuerte jugaría bien puede quedarse fuera (sesgo hacia "no es el equipo").
- **Noland**: su equipo verdadero depende del nuestro (excluye nuestros sets); al probar otra opción se le mantiene el
  equipo real, aunque con esa opción el juego habría sorteado otro (y alguno de sus sets podría coincidir con uno
  nuestro).
- La recogida no es reproducible bit a bit entre ejecuciones (el planificador del táctico sortea rivales sin semilla,
  como en la evaluación); el análisis sí lo es, porque parte de las acciones registradas.

### 20.6 Resultado final de alphazero_v3 (2026-09-30)
600 iteraciones (24 M decisiones del combatiente, ~24 h). Evaluación final, 512 rachas por ronda, semilla 777 (las
mismas que §18b y §19.8):

| | R1 | R2 | R3 | R4 | R5 | R6 | P(6 rondas) |
|---|---|---|---|---|---|---|---|
| PPO v3, red sola | 69,3 | 71,3 | 39,1 | 56,6 | 21,3 | 19,7 | 0,46 % |
| AZ v2 + búsqueda 512 | 80,1 | 75,2 | 47,1 | 68,6 | 29,9 | 26,6 | 1,55 % |
| **AZ v3, red sola** | 87,7 | 85,5 | 61,5 | 79,5 | 42,0 | 43,2 | **6,65 %** |
| **AZ v3 + búsqueda legal 512** | 87,3 | 86,5 | 63,5 | 79,3 | 45,5 | 44,9 | **7,77 %** |

Victorias por combate (red sola): 98,1 / 97,8 / 93,6 / 96,8 / 88,2 / 89,2 %. La caída de la tasa (WSD, it. 510–600)
subió P(6) de ~3,4 % a ~6,7 %. La búsqueda de 512 apenas añade (la red la ha destilado). Siguiente: autopsia de
derrotas (§21) y la rama "v3-larga" (retomar desde la it. 505 con más iteraciones en fase constante).

## 22. alphazero_v4: modelo del rival aprendido de la propia experiencia (`rl/opponent_model.py`, 2026-09-30)

Decidido por el usuario: v4 = v3 + un **modelo del rival aprendido de lo que el propio agente ha visto en sus
partidas**, usado por la búsqueda del combatiente en evaluación e inferencia en lugar del muestreador estricto. Todo
lo demás es v3 (§20). Motivo (§21 y el diagnóstico de techo): con el rival sorteado de la lista de sets
(`factory_sets`, solo diagnóstico fuera del entrenamiento) la búsqueda legal de 512 de v3 sube de 7,77 % a 9,52 % en
P(6 rondas); un jugador puede aprender esos sets jugando, pero no puede leer la lista del juego.

### 22.1 Qué observa el jugador (comprobado en el decomp)

Definiciones: **resumen** (*Summary*) es la pantalla que muestra un Pokémon entero (especie, 4 movimientos, objeto,
habilidad, naturaleza, estadísticas). **Configuración** de un Pokémon = (especie, conjunto de sus movimientos,
objeto); es lo que el modelo aprende (un set de la Factory sin los EVs).

- **En combate** (`BattleObserver`, `SeenMon`): de cada Pokémon rival que sale, su especie y nivel; los movimientos
  que usa; su objeto si el juego lo nombra (Restos, una baya consumida, Desarme / Ladrón / Truco); su habilidad si
  se anuncia; la barra de PS (48 píxeles), el estado, y en cada turno quién actuó primero y cuánto bajó la barra.
- **Pantalla de intercambio** (tras ganar; `src/battle_factory_screen.c`): se ve el equipo derrotado **solo por su
  especie** (dibujo y nombre, `Swap_PrintMonSpecies`). La opción «Resumen» solo existe para **nuestro** equipo
  (`Swap_Task_OpenSummaryScreen` llama a `ShowPokemonSummaryScreen(..., gPlayerParty, ...)`); al elegir un Pokémon
  rival el juego pasa directamente a «¿Aceptar este Pokémon?» (`Swap_ActionMon` → `Swap_AskAcceptMon`). Así que la
  pantalla de intercambio no añade nada a lo visto en el combate (para ganar, los tres rivales han salido). **Pero**
  el Pokémon que nos quedamos pasa a ser nuestro y su resumen lo muestra entero: esa es una configuración real de
  un rival, vista completa.
- **Pantalla de alquiler**: los 6 alquileres, enteros (`ShowPokemonSummaryScreen(SUMMARY_MODE_LOCK_MOVES, ...)`).
  El juego los sortea con el mismo generador y casi del mismo rango de sets que los rivales de la ronda
  (`GetFactoryMonId`; algunos del rango de la ronda siguiente).
- **Nunca**: los movimientos no usados, el objeto no nombrado, los IVs / EVs del rival, ni la lista de sets.

`SwapView` / `RentalView` / `BattleView` ya exponían exactamente esto (`swap_candidates` construye el equipo
derrotado con la especie y lo revelado en el combate).

### 22.2 Qué se registra (`ExperienceCollector`)

Un **registro** es una observación: (fuente, ronda, especie, movimientos, objeto, naturaleza, completo). El
recolector vigila un `FactoryEnv` (envuelve `step` / `reset`) y solo lee la fase, las vistas, `run_info()` y la
memoria del combate del jugador (`backend._observer`):
- en cada pantalla de alquiler: los 6 alquileres, **completos** (fuente `RENTAL`, con su naturaleza);
- al acabar cada combate (ganado, perdido o cortado): cada rival que salió, **parcial** (especie, movimientos
  usados, objeto nombrado o «no revelado»); si en la pantalla de intercambio siguiente nos quedamos con uno de
  ellos, ese registro pasa a ser **completo** (su resumen). Cada rival que sale es exactamente un registro;
- la ronda es el `challenge_num` en que se vio (cubos 0..6 y 7+ juntos).

### 22.3 El modelo (`ExperienceModel`)

Definiciones, de abajo arriba:
- **Suavizado** (Dirichlet): a una frecuencia observada se le suman pseudo-cuentas de una distribución de partida,
  para que lo nunca visto tenga probabilidad pequeña pero no cero.
- **Especie**: P_r(s) = (n_r(s) + β P(s)) / (N_r + β), con P(s) = (n(s) + a0/|F|) / (N + a0) sobre las especies de
  la Frontera F (lo que ya sabe el muestreador estricto). n_r(s) = veces vista en la ronda r (rivales 1, alquileres
  `rental_w` = 1). β = 5 lleva cada ronda hacia la media de todas; a0 = 1.
- **Configuraciones conocidas** de una especie: las vistas completas alguna vez. Además una opción **«nueva»**: una
  configuración nunca vista, sorteada con las reglas estrictas (movimientos aprendibles, objeto útil).
  P_r(c | s) = (n_r(s, c) + β P(c | s)) / (N_r(s) + β); P(c | s) = n(s, c) / (N(s) + a_cfg) y
  P(nueva | s) = (n(s, nueva) + a_cfg) / (N(s) + a_cfg), a_cfg = 1.
- **EM** (esperanza-maximización) para los registros parciales: un rival visto en combate con movimientos R y
  objeto I es compatible con varias configuraciones. Cada avistamiento reparte su cuenta entre las compatibles en
  proporción a su probabilidad actual (y a «nueva» con la de las reglas estrictas, ver abajo); se repite 3 veces.
  Así los combates enseñan con qué frecuencia aparece cada configuración como **rival**, no solo como alquiler.
- **Posterior en una decisión**: para un rival visto (especie s, R, I):
  P(c) ∝ P_r(c | s) · [R ⊆ movimientos(c)] · [objeto(c) = I si se reveló; si no, que no lo lleve un compañero];
  P(nueva) ∝ P_r(nueva | s) · L(R, I | s), con L la probabilidad de que el sorteo estricto contenga R
  (C(L−r, 4−r)/C(L, 4) sobre sus L movimientos aprendibles) y ese objeto. **Nada visto es imposible**: si ninguna
  configuración conocida lo explica, sale una nueva con las reglas estrictas. Un hueco no visto: especie de P_r(s)
  sin repetir la de un compañero, y configuración de P_r(c | s) sin repetir objeto.
- **Lo demás, como el estricto**: IVs al azar, EVs al azar, habilidad a cara o cruz salvo anunciada, PS dentro de
  la barra. La naturaleza, de las observadas para esa configuración (+1 pseudo-cuenta repartida entre las 25).
- **Pista** (`use_hint`): se sortean equipos enteros (vectorizado, lotes de 64/256/1024 hasta 512) y se acepta el
  primero cuya pista (`hint_of_arrays`, especie + movimientos, como el juego) sea la oída; si no hay ninguno, el
  último (`hint_stats["fallback"]`). En este camino las exclusiones (especie u objeto repetidos) se aplican por
  rechazo sobre el par (especie, configuración). Coste: 0,15 ms por determinización sin pista, 0,5 ms con pista
  (el estricto, 0,08 / 4,5 ms).
- No implementado (opcional en la propuesta): repesar por la barra de PS / el daño y por el orden de velocidad.

**Muestreador «experience»** (`rl/determinize.py` `PRIORS`): permitido en cualquier proceso (es conocimiento del
jugador), a diferencia de `factory_sets` (solo entrenamiento). `SearchBattler(opponent_prior="experience",
opponent_model=modelo o ruta)`, `GumbelBattler` igual; `rl.eval_rounds --opponent-prior experience
--opponent-model RUTA`, y `--opponent-prior auto` (por defecto) usa «experience» con el modelo del propio checkpoint
si lo lleva (los de v4) y el estricto si no. `strict` sigue disponible; `factory_sets` sigue bloqueado fuera del
entrenamiento (salvo el diagnóstico explícito).

### 22.4 Por qué respeta la regla de información

- El modelo solo recibe registros construidos desde vistas y la memoria del combate; en la decisión, solo la vista
  (`view.enemy_party`), la ronda y la pista. Nunca `gEnemyParty`, `gFrontierTempParty`, la lista de sets
  (`FRONTIER_MONS`, `SET_*`), los pools ni los IVs fijos.
- Lo que el modelo «sabe» de la lista de sets lo ha visto: alquileres y rivales de sus propias partidas. Es lo que
  aprende un jugador humano que juega mucho la Factory.
- Pruebas (`tests/python/test_opponent_model.py`): (a) el código de `rl/opponent_model.py` y `rl/alphazero_v4.py`
  (sin docstrings ni comentarios) no contiene ninguno de esos nombres; (b) el recolector, ejecutado sobre un backend
  «solo jugador» que falla si se toca otra cosa que la fase, las vistas, `run_info` y la memoria del combate, da los
  mismos registros que sobre el backend real; y los registros son fieles (el Pokémon tomado es el del rival
  entero); (c) se sobrescribe el rival real (ids de sets y los huecos no vistos) y el muestreador y la búsqueda dan
  exactamente lo mismo. Además: guardar / cargar, determinismo de la actualización en línea (mismo resultado en
  cualquier reparto y orden de los registros), exp(log P) = frecuencia del muestreador, rechazo por pista, búsqueda,
  `eval_rounds` y una ejecución diminuta de v4 de extremo a extremo (con reanudación).

### 22.5 Aprendizaje en línea y arranque en caliente

- **En línea** (`rl/alphazero_v4.py`): cada proceso de autojuego lleva un recolector en su entorno; `play()` devuelve
  los registros nuevos (`opp_obs`) junto a las muestras; el proceso principal los suma **en el orden de los
  procesos** (determinista) al modelo en cada iteración. Las simulaciones de las búsquedas y los combates simulados
  del refresco de W no se registran (no son partidas vistas). Las cuentas se guardan en cada checkpoint
  (`opponent_model`) y en `runs/<nombre>/opponent_model.pt`; al reanudar se recuperan. Registros en TensorBoard:
  `opp_model/*`.
- **Arranque en caliente** (`python -m rl.opponent_model_data collect`): registros de 30.006 combates jugados por
  alphazero_v3 (red sola, voraz, los dos agentes con sus rasgos; rachas que empiezan en las rondas 1–6), con el
  mismo recolector: 120.925 registros (89.005 avistamientos de rivales, 12.681 de ellos completos por intercambio;
  31.920 alquileres), 514 configuraciones y 109 especies. 157 s con 18 procesos. `runs/opp_model_v3/model.pt`.
- Cada iteración de v4 añade ~3.000 combates: a las pocas iteraciones los datos en línea superan al arranque.

### 22.6 Validación

**(1) Probabilidad del rival verdadero.** Combates apartados: 3.006 combates de v3 con otras semillas
(`opponent_model_data heldout`, 34.228 decisiones); de cada combate 3 decisiones al azar; en cada una, cada hueco
rival vivo. La etiqueta (la configuración verdadera, leída de `gEnemyParty` al empezar el combate) solo se usa para
puntuar. Medidas:
- **log P media**: logaritmo natural de la probabilidad que el muestreador da a la configuración verdadera de ese
  hueco (más alto es mejor; +1 = e ≈ 2,7 veces más probabilidad). Suelo de log 1e-12 para lo imposible (0,06 %:
  movimientos copiados, imposibles también para el estricto).
- **top-k**: fracción en que la configuración verdadera está entre las k más probables del modelo (huecos no vistos:
  sobre especie × configuración).
- **Referencia factory_sets**: la misma medida con el sorteo uniforme de los sets del pool de la ronda compatibles
  con lo visto (lee la lista del juego: solo como techo).

| | Huecos | log P modelo | log P estricto | log P ref. factory_sets | top-1 | top-5 | top-10 |
|---|---|---|---|---|---|---|---|
| Todos | 18.026 | **−2,71** | −17,78 | −2,68 | 0,436 | 0,519 | 0,545 |
| No vistos | 8.949 | −5,14 | −22,47 | −5,10 | 0,009 | 0,042 | 0,085 |
| Vistos (vivos) | 9.077 | −0,31 | −13,16 | −0,29 | 0,856 | 0,990 | 0,999 |

Con los 3.000 primeros combates el modelo ya da −2,84 (casi lo mismo). Por ronda, log P −2,3 en las rondas 1–4
(pool de 96 sets) y −3,5 en las 5–7 (510 sets); el estricto, −17,5 a −18,1 en todas. Variantes (misma prueba):
sin alquileres (`rental_w` 0) −4,60 y el 7,6 % de configuraciones imposibles (muchas configuraciones solo se ven
completas en el alquiler); β = 1 −2,70, β = 20 −2,74, una sola vuelta de EM −2,71, a_cfg 0,1 o 5 sin cambios.
**Con la pista** (Monte Carlo, 200 equipos por decisión, 2.845 huecos no vistos): log P de la especie −4,79 → −3,87
y de la configuración −5,75 → −4,82 (0,9 nats mejor), con un 1,5 % de sorteos sin ningún equipo compatible.

**(2) Juego: alphazero_v3 + búsqueda legal de 512** (8 determinizaciones, 512 rachas por ronda, semilla 777, las
mismas que §20.6), cambiando solo el muestreador del rival:

| Muestreador del rival | R1 | R2 | R3 | R4 | R5 | R6 | P(6 rondas) |
|---|---|---|---|---|---|---|---|
| estricto (§20.6) | 87,3 | 86,5 | 63,5 | 79,3 | 45,5 | 44,9 | 7,77 % |
| **experience** (arranque en caliente, sin pista) | 87,3 | 86,9 | 64,5 | 77,0 | 50,2 | 47,7 | **9,02 %** |
| experience con pista | 87,7 | 83,6 | 64,3 | 79,7 | 48,2 | 48,8 | 8,84 % |
| factory_sets (techo de diagnóstico, lee la lista) | 88,3 | 86,1 | 67,2 | 79,5 | 50,2 | 46,7 | 9,52 % |

Victorias por combate en las rondas 5 / 6: estricto 89,1 / 89,7 %; experience 90,5 / 90,5 %; con pista 90,0 /
90,7 %; factory_sets 90,5 / 90,1 %. Error típico de P(6) ≈ ±0,7 puntos por fila (las filas comparten semillas, así
que sus diferencias tienen algo menos de ruido). Coste: 44–48 ms por decisión, igual que el estricto.

Lectura: el modelo aprendido recupera ~70 % de la distancia entre el estricto y el techo `factory_sets`
(+1,25 puntos de P(6), +1,4 / +0,8 puntos de victorias por combate en las rondas 5 / 6), sin leer nada que el
jugador no haya visto. La pista mejora la probabilidad del rival (22.6 (1)) pero en el juego no se nota (8,84 %,
dentro del ruido): se deja **desactivada** por defecto (`use_hint` false; activable con `--opponent-model-cfg
'{"use_hint": true}'`).

### 22.7 Lanzamiento de alphazero_v4

- Desde cero, como v3: `python -m rl.alphazero_v4` con los valores por defecto de v4 = los del lanzamiento de v3
  (`--team-eval runs/team_eval_v3/model.pt --t-imitation 0.1 --eval-every 3 --eval-search-every 15`), 1.500
  iteraciones con WSD (caída en el último 15 %: iteraciones 1275–1499), Gumbel 256 y `factory_sets` en la búsqueda
  de entrenamiento (como v3), refresco de W, planificador y pistas sin cambios.
- Modelo del rival: arranca de `runs/opp_model_v3/model.pt` (22.5) y aprende en línea; la evaluación con búsqueda
  (cada 15 iteraciones, 512 simulaciones) usa el muestreador «experience» con el modelo del momento.
- `rl.eval_rounds --battler search` sobre un checkpoint de v4 usa por defecto su propio modelo.

```bash
python -m rl.alphazero_v4 --name alphazero_v4 --opponent-model-init runs/opp_model_v3/model.pt \
    --team-eval runs/team_eval_v3/model.pt --t-imitation 0.1 --eval-every 3 --eval-search-every 15 --iterations 1500
python -m rl.opponent_model_data collect --ckpt runs/alphazero_v3/latest.pt --battles 30000 --out runs/opp_model_v3/model.pt
python -m rl.opponent_model_data validate --model runs/opp_model_v3/model.pt --heldout runs/opp_model_v3/heldout.pkl --per-battle 3
```

### 22.x Techo con información perfecta de alphazero_v3 y lanzamiento de v4 (2026-09-30)
Búsqueda **perfecta** (conoce al rival real; solo diagnóstico) con la red final de v3, semilla 777:

| | R1 | R2 | R3 | R4 | R5 | R6 | P(6 rondas) | victorias por combate R5 / R6 |
|---|---|---|---|---|---|---|---|---|
| perfecta 2048, 512 rachas | 89,3 | 88,1 | 71,3 | 79,9 | 67,0 | 62,9 | 18,9 % | 94,3 / 93,7 % |
| perfecta 8192 (R5–6, 128 rachas) | – | – | – | – | 62,5 | 62,5 | – | 93,3 / 93,7 % |

Con información perfecta y la red de v3, el combatiente gana ~94 % de los combates de las rondas 5–6, y pasar de 2048 a
8192 simulaciones no mejora: la búsqueda ya está saturada con esta red. El 98,4 % por combate que exige completar el
reto de forma consistente **no es alcanzable con la red actual ni conociendo al rival**; hace falta una red (valor y
política) mucho mejor, que suba el techo, o equipos mejores. alphazero_v4 (modelo del rival aprendido de la experiencia,
1500 iteraciones, desde cero) se lanzó igualmente, como decidió el usuario.

## 24. alphazero_v3 en el juego real: velocidad, agentes, registros, repetición exacta y vídeo (2026-10-01)

Objetivo (decidido por el usuario): jugar el reto en el juego **real** (el emulador con la ROM, no el simulador) lo
más rápido posible con el mejor checkpoint de v3, registrar todo lo que el agente sabe en cada decisión y, cuando una
racha complete el reto (42 victorias, la última contra Noland con el símbolo de oro), **repetirla exactamente** con la
interfaz completa y grabarla en vídeo. Código: `pybattle/emu/options.py`, `pybattle/emu/mirror.py`,
`rl/real_agents.py`, `rl/real_play.py`, `scripts/play_real.py`, `scripts/real_play.sh`; pruebas en
`tests/python/test_real_play.py`.

### 24.1 Definiciones (de abajo arriba)
- **Fotograma** (*frame*): la consola dibuja 60 imágenes por segundo y el juego avanza su lógica una vez por
  fotograma. "Velocidad ×2" = 120 fotogramas por segundo de reloj.
- **Sin tope** (*uncapped*): el emulador ejecuta fotogramas tan rápido como puede la CPU, sin esperar al reloj.
- **Renderizar**: convertir la memoria de vídeo de la consola en una imagen. Es solo salida: no cambia la lógica del
  juego. **Frameskip n**: dibujar una imagen de cada n + 1 fotogramas; con n enorme no se dibuja nunca (sin ventana no
  hace falta).
- **Opciones del juego**: el menú OPCIONES guarda en `SaveBlock2` (offset 0x14) la velocidad del texto (3 bits:
  lenta / media / rápida) y "Escena de combate" (1 bit, `optionsBattleSceneOff`: con 1 no se muestran las animaciones
  de los movimientos). El juego las lee en el momento, así que escribirlas en RAM tras cargar la partida equivale a
  elegirlas en el menú. El .sav del disco nunca se toca (el emulador guarda en una copia en memoria).
- **Generador aleatorio** (RNG, `gRngValue`): un número de 32 bits que cada llamada a `Random()` transforma con una
  fórmula fija (congruencial lineal). Mismo valor y mismas llamadas ⇒ mismos números.
- **Savestate**: una foto completa del estado del emulador (CPU, RAM, vídeo, temporizadores). Cargarla y dar las
  mismas pulsaciones reproduce exactamente la misma partida. **No incluye** la memoria de guardado del cartucho
  (*flash*, donde el juego guarda la partida), así que se guarda aparte.
- **Reloj del cartucho** (RTC): Esmeralda lleva un reloj de tiempo real. Por defecto mGBA le da la hora del ordenador,
  que cambia entre una partida y su repetición. Ahora el reloj **empieza en una fecha fija y avanza con los fotogramas
  emulados** (`set_rtc_fake_epoch`): el contador de fotogramas va en el savestate, así que la hora que ve el juego en
  una repetición es la misma.
- **Determinismo**: mismo estado inicial + mismas entradas ⇒ mismos fotogramas, mismo RNG, mismo resultado. El
  emulador lo es, y el conductor (`pybattle/emu/driver.py`) también, porque decide qué teclas pulsa leyendo la RAM.
- **Espejo** (*mirror*): una copia del combate actual en el simulador que avanza a la par que el emulador (24.4).

### 24.2 Velocidad
Cambios: el host del emulador (`emu/host/emulator.cpp`) gana `set_frameskip`, `set_rtc_fake_epoch` y
`save_data` / `load_save_data` (la flash); `EmuBackend(options=, frameskip=, rtc_epoch_ms=)` los aplica (por
defecto, todo como antes). Hay que **recompilar** `pybattle_emu` en la copia principal
(`cmake --build build --target pybattle_emu`).

Medidas (`play_real.py bench`, 40 combates por fila desde la racha 0, 1 hilo de torch, 2 procesos a la vez, máquina
cargada por el entrenamiento de v4; "fotogramas/combate" incluye lo que pasa entre combates: sala, encargado, cambio):

| Combatiente | Escena | Dibuja | Combates/hora/proceso | Fotogramas/combate | ms por decisión del combatiente |
|---|---|---|---|---|---|
| red sola | off | no | **807** | 11.519 | 3,5 |
| red sola | off | sí | 731 | 11.519 | 3,1 |
| red sola | on | no | 601 | 16.065 | 3,7 |
| búsqueda 512 | off | no | **897** | 8.828 | 66 |
| búsqueda 512 | on | no | 666 | 13.592 | 66 |

(Cada fila juega combates distintos, así que la duración de los combates varía por azar; el agente se lleva el 14 %
del tiempo con búsqueda y el 2 % con la red sola: casi todo es emular.) Lectura: el emulador corre a ~2.300–2.700
fotogramas/s (~40× la velocidad real); no dibujar ahorra ~10 %; quitar las animaciones, ~25–30 % de los fotogramas; el
texto ya estaba en "rápido" en la partida del usuario. La búsqueda de 512 simulaciones cuesta ~66 ms por decisión en
CPU (el árbol en C++): **apenas reduce el ritmo**.

### 24.3 ¿Cambian las opciones el juego? (experimento)
- **Dentro de un combate, no.** En combates de la Frontera el juego no avanza el RNG en cada fotograma
  (`VBlankCB_Battle` lo salta con `BATTLE_TYPE_FRONTIER`) y las animaciones no llaman a `Random()`. Prueba: el espejo
  en el simulador (que no tiene animaciones) coincide byte a byte con el emulador en 100 decisiones con la escena
  apagada y en 82 con la escena encendida, también en un combate contra Noland.
- **Fuera de los combates, sí.** Una racha jugada con la escena apagada y repetida con la escena **encendida** coincide
  en el RNG en las 31 primeras decisiones (3 combates) y luego diverge **en el mundo exterior** tras el 3.er combate
  (`CB2_Overworld`): allí el número de llamadas a `Random()` depende del número absoluto de fotogramas (que las
  animaciones cambian) y los rivales siguientes salen distintos. Por tanto **una repetición tiene que usar las mismas
  opciones que la partida** (consistencia del estado): el registro guarda las opciones y la repetición las reutiliza
  (`--scene on/off` existe para experimentar y avisa de cada divergencia).
- **Compromiso elegido**: la escena apagada da ~25–30 % más combates por hora, pero el primer reto completo llega en
  pocos cientos de combates (P(6 rondas) ≈ 9 % con búsqueda, §22.6), es decir en minutos u horas, así que la velocidad
  importa poco y un vídeo con animaciones se entiende mucho mejor. Lanzamiento: **2 procesos con búsqueda y la escena
  encendida** (su reto completo se graba con animaciones) + **1 con la red sola y la escena apagada** (la configuración
  más rápida). `--scene off` vale para todos si se prefiere velocidad.

### 24.4 Los agentes (`rl/real_agents.py`, `V3Player`)
- **Táctico**: exactamente el de la evaluación (`rl.eval_rounds` con `--features auto`): la red de
  `runs/alphazero_v3/latest.pt`, voraz, con sus rasgos por opción de `rl.alphazero_v3.features_hook` (la W con la que
  entrenó, `team_eval_0580.pt`, su tabla V_siguiente y el planificador de profundidad 1) y la pista del encargado
  (ninguna antes de Noland). El planificador sortea sus 8 rivales con una **semilla por decisión** que se registra (en
  la evaluación se sorteaban sin semilla: misma distribución, pero así cada decisión es reproducible).
- **Combatiente, modo red**: el argmax de la política de la red.
- **Combatiente, modo búsqueda**: `SearchBattler` legal, 512 simulaciones, 8 determinizaciones, c_puct 1,5, rivales
  del modelo de la experiencia (`runs/opp_model_v3/model.pt`, §22).
- **La búsqueda en el emulador**: la búsqueda clona y avanza un `Gen3Game` nativo, que el emulador no tiene. Pero el
  simulador *es* el código de combate del juego y, desde el punto de partida del combate (los dos equipos, las
  banderas del combate, los datos de la Frontera y el RNG en `BeginBattleIntro`) y con las mismas acciones, reproduce
  la RAM de combate y el RNG del emulador byte a byte (`pybattle.diff`). Así que `WatchedMachine` mira en cada
  fotograma si empieza un combate y guarda ese punto de partida; `SimMirror` arranca con él un combate en el
  simulador, lo lleva a cada decisión, **lo compara** con el emulador (RNG, regiones de combate, los dos equipos) y le
  da la misma acción que al emulador. `MirroredBackend` presenta a la búsqueda las vistas del emulador con el juego del
  simulador. Si alguna comparación fallase, ese combate sigue con la red (`search_fallback`); en las pruebas no ha
  fallado nunca. (La información oculta del rival se sigue sorteando: la búsqueda legal sobrescribe lo no visto.)
- **Paridad**: en 400 decisiones de un `FactoryEnv` (desde la ronda 3), `V3Player` elige exactamente la misma acción
  que los agentes de `eval_rounds` (prueba `test_agent_parity_with_eval`).
- **"Victorias esperadas" del táctico**: su crítico aprendió el retorno con *shaping*, victorias − Φ(s) (§19); por eso
  se muestra crítico + Φ, con Φ = la media de Q_plan de los alquileres, o la Q_plan de quedarse en un cambio.

### 24.5 Registros (`rl/real_play.py`)
Una **racha** (*run*) va desde el vestíbulo hasta perder (después de 42 victorias sigue jugando). Por proceso, en
`runs/real_play/w<k>/`:
- `runs/run_NNNNN/`: `start.state.z` (savestate en el vestíbulo justo antes de empezar), `save.z` (la flash),
  `run.json` (argumentos del inicio, opciones, reloj, banderas del conductor, agente y semillas), `actions.jsonl`
  (cada decisión: fase, acción, RNG y fotograma en ese momento, racha, combate y los detalles del agente; y cada fin
  de combate) y `result.json`. Una racha con 42 victorias también tiene `FULL_WIN.json` y una línea en
  `runs/real_play/full_wins.jsonl`. Se borran los savestates de rachas viejas y cortas (se guardan los de las 40
  últimas, los de ≥ 14 victorias y los de los retos completos).
- `playlog.jsonl`: `rl.playlog.PlayLogger` (lo escribe otro agente; mientras no esté, `rl/playlog_stub.py` con la
  misma interfaz): en cada decisión la vista, la política (probabilidades y valor) y los extras: rasgos por opción
  del táctico (W normal / último, E_ronda, Δ, Q_plan, Φ), la pista, visitas y Q de la búsqueda, y las creencias del
  modelo del rival sobre cada Pokémon rival (los 3 sets más probables y la probabilidad de un set nunca visto).
- `stats.json`: rachas, combates, combates/hora, la mejor racha, retos completos, desincronizaciones del espejo.

### 24.6 Repetición exacta y vídeo
- `replay_run`: un emulador nuevo recibe el savestate, la flash, el reloj y (van en la RAM del savestate) las mismas
  opciones; se empieza la racha con los mismos argumentos y se le dan **las acciones registradas, sin recalcular
  nada**. En cada decisión se comprueban la fase, el RNG y el número de fotograma, y en cada combate el resultado.
  Prueba: una racha corta con búsqueda se repite idéntica (`test_replay_is_deterministic`), dibujando o sin dibujar.
- `record_run`: la misma repetición con la ventana completa (SDL sin pantalla) a **×2**: un fotograma de vídeo cada 4
  fotogramas emulados (30 imágenes/s), así que el vídeo va exactamente a ×2 aunque se grabe más deprisa; en cada
  decisión la imagen se mantiene 0,5 s para que se lean los paneles. 1920×1080, H.264 con NVENC (un codificador de la
  GPU aparte de los núcleos que usa el entrenamiento; si no, libx264). Junto al .mp4 se copian `actions.jsonl` y
  `run.json`. Se para tras la 42.ª victoria. Prueba: 1 combate = 1,9 min de vídeo grabados en 0,6 min.
- Automático: el supervisor (`play_real.py play`) vigila `full_wins.jsonl`; con el primer reto completo pausa un
  proceso (el de la red sola) para no pasar de 3 emuladores, graba en `runs/real_play/videos/` y lo reanuda.

### 24.7 La ventana (`pybattle/emu/viewer.py`, `ui.py`)
- Cabecera: fase, ronda, combate n/7 (con NOLAND), racha, victorias de la racha, modo (red / búsqueda 512 / REPLAY).
- Combatiente: P(ganar este combate) según la red; por acción, la probabilidad de la red, las visitas de la búsqueda
  (en %) y su Q (resultado medio de sus simulaciones, 1 = victoria), y el daño estimado (o el riesgo en los cambios).
- Táctico: cada opción con π, E_ronda, W normal / último, Q_plan y Δ (en el alquiler, las 8 más probables + la
  elegida + la de mejor Q_plan, marcada ★), las victorias esperadas y la pista del encargado.
- Modelo del rival (panel bajo la pantalla del juego): por cada Pokémon rival, lo revelado y sus 3 sets más probables
  (objeto y movimientos) con su probabilidad; de los no vistos, las especies más probables.

### 24.8 Uso
```bash
scripts/real_play.sh                                   # 2 búsqueda (escena on) + 1 red (escena off), en segundo plano
python scripts/play_real.py status --out runs/real_play
touch runs/real_play/STOP                              # parar; reanudar: scripts/real_play.sh --resume
python scripts/play_real.py verify runs/real_play/w0/runs/run_00012
python scripts/play_real.py replay runs/real_play/w0/runs/run_00012 --speed 2       # en la ventana
python scripts/play_real.py record RUN --video win.mp4 --until-wins 42
python scripts/watch.py --rom ROM --save SAV --v3 search --speed 2                  # mirar en directo
```

## 23. Estudio de cómo juega el mejor agente de v3 (`rl/playlog.py`, `rl/play_agent.py`, `rl/play_sim.py`, `rl/playstats.py`, 2026-10-01)

Objetivo (decidido por el usuario): describir **cómo** juega `runs/alphazero_v3/latest.pt` (táctico con sus rasgos W /
planificador; combatiente = la red sola **y** la búsqueda legal de 512 con el modelo del rival «experience»), por ronda
(1–6, Noland aparte) y a lo largo de la racha. Hay dos fuentes de partidas con el **mismo** registro: el juego real
(emulador; el *runner* lo escribe otro agente y llama a este registrador) y una réplica en el simulador (muchos más
combates). Es solo diagnóstico: nada de esto entra en el entrenamiento.

### 23.1 El registrador (`rl/playlog.py`)

Definiciones:
- **Registro** (*log*): un archivo de texto con un objeto JSON por línea (JSONL); comprimido con gzip si el nombre acaba
  en `.gz`. Cada línea es un **registro** de uno de cuatro tipos: `meta` (la primera: configuración), `decision`,
  `battle_end` y `run_end`.
- **Vista del jugador**: lo que el juego enseña (`RentalView`, `SwapView`, `BattleView` de `pybattle/view.py`). **Verdad**:
  lo que el jugador no ve (equipo verdadero del rival, sus PS exactos...).

Interfaz (fija; el *runner* del emulador programa contra ella):

```
class PlayLogger:
    def __init__(self, path, meta: dict)
    def decision(self, kind: str, view, run_info, policy: dict, action, extras: dict) -> None
        # kind: rental | swap | battle | forced_switch; view = la vista del jugador (RentalView/SwapView/BattleView);
        # policy: {"probs": [...], "value": float, "legal": [...]} (+ claves extra); extras: cualquier cosa
    def battle_end(self, won: bool, info: dict) -> None
    def run_end(self, streak: int, info: dict) -> None
    def close(self) -> None
read_log(path) -> registros (dicts);  iter_runs(paths) -> rachas agrupadas (combates -> decisiones)
```

- Se llama a `decision` **antes** de dar la acción al juego. La acción se guarda como lista JSON: alquiler
  `[líder, j, k]`, cambio `null` (quedarse) o `[hueco propio, hueco rival]`, combate `["move", i]` / `["switch", i]`.
- **Serialización**: dataclasses → objetos, numpy → listas, conjuntos → listas ordenadas, tuplas → listas, NaN / ∞ →
  `null`; lo que no se puede convertir queda como `{"__repr__": ...}`.
- **Separación de la verdad**: la clave `"truth"` de `extras` (decisión) o de `info` (fin de combate / de racha) se
  saca a un campo propio `"truth"` del registro. Todo lo demás es vista del jugador. La línea `meta` lo declara
  (`player_view_fields`, `truth_fields`); `strip_truth(registro)` lo quita. Los análisis que describen al jugador no
  leen `truth` (solo la sección marcada del informe).
- Contadores propios: `run` (racha dentro del archivo; sube tras cada `run_end`) y `battle` (combate de la racha; sube
  tras cada `battle_end`). Una decisión del táctico lleva el número del combate que precede.
- Se vacía el búfer cada 200 registros: un proceso cortado conserva lo escrito; `read_log` lee hasta el corte y
  `iter_runs` descarta la racha incompleta del final.

### 23.2 El agente con sondas (`rl/play_agent.py`)

`PlayAgent(ckpt, battler="net" | "search")`: `tactician(backend, kind, obs, ctx)` y `battler(backend, obs, ctx,
decisiones)` devuelven `(acción, policy, extras)` listos para `decision`. Juegan **exactamente** como en la evaluación
(`rl.eval_rounds`):
- **Táctico**: su red, voraz (alquiler: el líder de mayor probabilidad y luego la pareja de mayor probabilidad dado ese
  líder), con los 7 rasgos por opción de W y del planificador de profundidad 1 (`features_hook` del checkpoint: su W
  `team_eval_0580.pt`, su tabla V_siguiente y `planner_cfg`).
- **Semilla del planificador por decisión** (`planner_seed`): el planificador sortea rivales; con una semilla por
  decisión las sondas de abajo ven los **mismos** sorteos (números aleatorios comunes) y su diferencia no es ruido del
  sorteo. El juego del agente no cambia en distribución (los sorteos siguen siendo nuevos en cada decisión).
- **Combatiente**: la red sola (máximo de su política) o la búsqueda legal (`SearchBattler`, 512 simulaciones, 8
  determinizaciones, muestreador «experience» con `runs/opp_model_v3/model.pt`, §22).

Qué se registra en cada decisión (además de la vista):
- `policy`: probabilidades de la red sobre las opciones legales (`legal`, en el orden de `probs`), valor (del crítico;
  en el combatiente, probabilidad de ganar), `choice` (índice elegido). Alquiler: además `lead_probs` (6) y
  `trio_probs` (20 tríos).
- Táctico, `extras`: `options`, `features` (los 7 rasgos por opción: E_ronda, W(normal | pista), W(último o Noland), Δ,
  Q_plan, Q_plan − media, Φ; los de victorias divididos por 10), `q_plan` (en victorias), `hint`, `noland`,
  `planner_seed` y `probes`.
- **Sondas contrafactuales** (`probes`; solo la red, no se juegan): la elección y las probabilidades de la red con la
  entrada cambiada:
  - `hint_none_input`: la pista de la entrada se cambia por «ninguna» (tipo 18, estilo 0), rasgos sin tocar;
  - `hint_none_full`: además los rasgos se recalculan sin pista (W con el símbolo «desconocida», planificador sin
    rechazo por pista), con la misma semilla del planificador;
  - `hint_<t>_input` / `hint_<t>_full`: el tipo de la pista se cambia por t, para 3 tipos t al azar entre los otros 16
    reales (no el verdadero, no el tipo «misterio»); el estilo se mantiene;
  - `feat_zero`: los rasgos por opción a cero.
  Antes de Noland no hay pista: solo `feat_zero`. Con pista (18, 0) no hay `hint_none_*`.
- Combatiente, `extras`: `estimates` (de `rl/damage.py`, lo que calcula un jugador: por movimiento del activo
  [daño mínimo, máximo, ¿puede dejar KO?] en fracción de los PS máximos del rival; `max_damage_move`; amenazas de los
  movimientos vistos del rival; quién es más rápido), `search` (visitas y Q por acción, si hay búsqueda) y `beliefs`
  (las 3 configuraciones más probables del rival activo según el modelo «experience», con P(nueva)).

### 23.3 La réplica en el simulador (`rl/play_sim.py`)

- **Racha** i: semilla `stable_seed("play_sim", semilla, ronda de salida, i)`; las mismas rachas para los dos modos
  (comparación emparejada). Empieza en el alquiler de la ronda 1, como el juego real, y termina con la primera derrota
  (o un combate que llega a 300 decisiones: derrota marcada `truncated`; o `--max-battles` = 200 combates: `capped`).
- **Rachas tardías** (`--late-runs`): empiezan en la ronda 5 o 6 (contador de alquileres al azar, como `FactoryEnv`),
  marcadas `late_start`; solo añaden muestras de las rondas altas.
- Además de lo del agente se registra: en cada decisión de combate los **eventos** desde nuestra decisión anterior
  (`events_prev`: el análisis de mensajes del juego de `rl/autopsy.py`: críticos, fallos, estados, retrocesos...); en
  cada fin de combate turnos, decisiones, PS que nos quedan (`own_hp`, nuestros: el jugador los ve), recuento de
  eventos y balance de suerte; y como **verdad** el equipo verdadero del rival (fin de combate) y los PS / set
  verdaderos de su activo (cada decisión).
- Procesos: cada uno escribe su archivo `<out>/<modo>/<modo>_p<j>[sufijo].jsonl.gz`; las rachas se reparten en turno
  rotatorio. `--run-offset` permite un segundo lanzamiento en la misma carpeta con otras rachas.

### 23.4 Las métricas (`rl/playknow.py`, `rl/playstats.py`)

**Convenciones.**
- **Ronda** de un combate = `challenge_num + 1`; columnas R1..R6, **R7+** (rondas 7 y siguientes) y **Noland** (los
  combates contra el Jefe, aparte). Una decisión del táctico pertenece al combate que precede. **Posición** = 1..7 dentro
  de la ronda.
- **Intervalo de confianza (IC) al 95 %**: *bootstrap* por rachas: 1.000 réplicas en las que cada racha recibe un peso
  aleatorio Poisson(1) (todas sus decisiones con el mismo peso, porque dentro de una racha no son independientes); se
  toman los percentiles 2,5 y 97,5 de la media ponderada.
- **Diferencia controlada** por un estrato s (normalmente la ronda): Σ_s w_s (media del grupo A en s − media del grupo B
  en s), w_s = observaciones de s / las de todos los estratos con los dos grupos. Es una asociación, no un efecto
  causal (el agente elige cuándo hace cada cosa).
- **Categoría** de un movimiento (3.ª generación): la da su tipo: normal, lucha, volador, veneno, tierra, roca, bicho,
  fantasma y acero son **físicos**; fuego, agua, planta, eléctrico, psíquico, hielo, dragón y siniestro **especiales**.
- **Ataque con tipo**: movimiento con potencia > 0 cuyo daño sigue la tabla de tipos (no cuentan Sísmico, Tinieblas,
  Furia Dragón, Bomba Sónica, Psicoonda, Superdiente, Esfuerzo, Contraataque, Manto Espejo, golpes de KO directo,
  Venganza). Poder Oculto toma su tipo de los IV (los nuestros se ven); Meteorobola, normal.
- **Eficacia** de un tipo sobre un Pokémon: producto de la tabla de tipos por sus tipos, con la habilidad si el
  jugador la conoce (anunciada, o la única posible de la especie): Levitación (tierra ×0), Absorbe Agua, Absorbe Elec.,
  Absorbe Fuego, Superguarda (no súper eficaz ×0), Sebo (fuego / hielo ×0,5). **Resiste** = ×0,5 o menos (inmunidad
  incluida); **débil** = ×2 o más.
- Un **mon** del análisis: nuestro (o un alquiler) con el set completo; un rival de la pantalla de cambio, solo con los
  movimientos que usó en combate (lo que ve el jugador).

**A. Alquiler.**
- **Opción** = (líder, pareja): 60. **Trío** = equipo sin orden: 20. P(trío) = Σ de P(opción) de sus 3 líderes.
  **P(en el equipo)** de un Pokémon ofrecido = Σ P(trío) de los 10 tríos que lo contienen.
- **Tasa de elección** de una especie = elegida / ofrecida; **probabilidad media** = media de P(en el equipo) al
  ofrecerse; **seguridad** = media de |2·P(en el equipo) − 1| (1 = siempre clara; 0 = duda).
- **Líder**: tasa de líder si elegido; especies elegidas ≥ 20 veces que nunca son líder. **Líder frente a la pista**
  (alquileres con tipo en la pista): fracción de líderes que resisten el tipo de la pista frente a la media de los 3
  elegidos; igual con «tiene un ataque con tipo súper eficaz contra ese tipo». Velocidad base del líder − media del
  equipo.
- **Elegidos frente a ofrecidos**: media de los 3 elegidos − media de los 6 ofrecidos de: suma de estadísticas base,
  cada base, Velocidad real, **aguante físico real** = PS máx. × Def / 1000 (especial: × DefEsp), movimientos de estado
  (potencia 0), ataques con tipo. **Objetos**: tasa de elección por objeto.
- **Papel** de un set (el primero que se cumple): **barredor con mejora** = un movimiento que sube Ataque, At.
  Especial o Velocidad (efectos ATTACK_UP(_2), SPEED_UP(_2), SPECIAL_ATTACK_UP(_2), CALM_MIND, DRAGON_DANCE, BULK_UP,
  CURSE, BELLY_DRUM) y ≥ 2 ataques con tipo; **apoyo / estados** = ≥ 2 movimientos de estado; **barredor rápido** =
  Velocidad base ≥ 90; **muro / tanque** = PS + (Def + DefEsp)/2 base ≥ 170 y máx(Atq, AtEsp) base < 100; **atacante
  lento / robusto** = el resto. **Lift** de un papel = fracción entre los elegidos / fracción entre los ofrecidos.

**B. Cambios.**
- Opciones: quedarse + 3×3 cambios (el que entra ocupa el hueco del que sale). **Tasa de cambio** = cambia / decisiones;
  **tasa de salida** de una especie = sale / estaba en el equipo en una pantalla de cambio; **tasa de entrada** = entra /
  se ofrece. **Permanencia** = combates que un set (especie, movimientos, objeto) pasa en el equipo; *censurada* si la
  racha acaba con él. **Tipo de cambio por papel**: (papel del que sale → papel del que entra, éste con su set completo,
  visto en el combate siguiente).
- **Cobertura defensiva** de un equipo, sobre los 17 tipos de ataque: **resistidos** (algún miembro resiste), **nadie
  resiste** (17 − resistidos), **debilidades compartidas** (≥ 2 miembros débiles); también solo físicos (9 tipos) y
  solo especiales (8). **Cobertura ofensiva**: de los 17 tipos simples, cuántos golpea ×2 algún ataque con tipo del
  equipo; también solo con físicos y solo con especiales. **Aguante físico / especial** base = media de PS × Def (o
  DefEsp) base / 1000.
- Para cada decisión (alquiler: los 20 tríos; cambio: las 10 opciones, el rival visto como en la pantalla) se da la
  medida de lo **elegido**, la **media de las opciones**, la **mejor opción** y el **percentil** = fracción de las otras
  opciones a las que supera lo elegido (empates medio punto; 0,5 = al azar). **Antes / después** de cada cambio:
  después − antes, con el que entra con su set completo.
- **Fracción física** = ataques físicos / ataques con tipo del equipo. **Encaje** de un miembro = fracción de sus ataques
  con tipo cuya categoría coincide con su mayor ataque base (Atq ≥ AtEsp → físicos); el del equipo, la media;
  **desajustados** = miembros con encaje < 0,5.
- **Arquetipo** (medias de base): **rápido y frágil** = Velocidad ≥ 85 y PS·(Def + DefEsp)/2 < 7.500; **lento y robusto**
  = Velocidad < 70 y ese producto ≥ 7.500; **mixto**, el resto. **Supera con seguridad** a un rival = Velocidad real >
  la máxima posible del rival (IV 31, EV 252, naturaleza +, nivel 100); **fracción superada** de un miembro = fracción
  de los rivales vistos en esa ronda en los registros (ponderados por frecuencia) a los que supera con seguridad.
  **Correlaciones** de Spearman entre velocidad, aguantes, ataque y PS medios de los equipos de cada combate.
- **No cambia aunque otra opción parezca mejor**: Δq = máx Q_plan de los cambios − Q_plan de quedarse; Δw igual con
  W(normal | pista); Δbase = máx(suma de base del que entra − la del que sale). «Dice cambiar» = Δq > 0,5 victorias /
  Δw > 0,02 / Δbase ≥ 40. Fracción de los «quedarse» en que cada uno decía cambiar; «cambia aunque quedarse es mejor» =
  Q_plan(elegido) < Q_plan(quedarse) − 0,5; acuerdo con el máximo de Q_plan. Resultado cuando el planificador decía
  cambiar: victoria en el combate siguiente quedándose − cambiando, controlado por ronda.
- **Calidad por el resultado**: cambiar − quedarse en la victoria del combate siguiente y en completar la ronda,
  controlado por estratos (ronda × tercio de Q_plan(quedarse) dentro de la ronda = estados comparables).

**C. Sinergias** (booleanas por equipo):
- **lluvia** = Danza Lluvia y (un ataque de agua, Trueno o Nado Rápido); **sol** = Día Soleado y (ataque de fuego,
  Rayo Solar, Meteorobola o Clorofila); **arena** = (Tormenta Arena o Chorro Arena) y ≥ 2 miembros roca / tierra / acero;
  **granizo** = Granizo y ≥ 2 miembros de hielo; **Relevo** = un Pokémon con Relevo y una mejora o Sustituto;
  **pantallas** = Reflejo o Pantalla de Luz; **Púas**; **clérigo** = Cascabel Cura / Aromaterapia; **núcleo
  fuego / agua / planta** = un miembro de cada tipo; **pareja defensiva** = dos miembros que resisten cada uno todas las
  debilidades del otro.
- **Línea base** de una decisión = fracción de sus opciones (20 tríos, 10 opciones de cambio) con la sinergia; **lift**
  = presencia en lo elegido / línea base, sobre las decisiones con alguna opción que la tiene; también elegida − base.
- **Usada** en combate: se eligió el movimiento que la activa (o salió nuestro Chorro Arena); **aprovechada**: lluvia /
  sol = ataque de agua o Trueno (fuego, Rayo Solar, Meteorobola) con ese clima, o Nado Rápido en lluvia; arena /
  granizo = una decisión con ese clima y el rival activo sin inmunidad; Relevo = relevar con mejoras o Sustituto;
  pantallas = alguna decisión con la pantalla puesta; Púas = el rival cambia de Pokémon con Púas puestas; clérigo =
  usado con un miembro con estado.

**D. Influencia de la pista y de los rasgos** (de las sondas): **cambia** = la opción de la sonda ≠ la real; **equipo
cambia** = el equipo resultante es otro. **Δ resistencias**: sonda «ninguna» = resistencias al tipo de la pista del
equipo real − del de la sonda; sonda «tipo t» = resistencias a t del equipo de la sonda − del real (positivo: la pista
hace añadir resistencias al tipo anunciado). **Δ x2**, igual con miembros con un ataque súper eficaz contra el tipo.
**ΔQ_plan** = Q_plan(elección real) − Q_plan(elección de la sonda). Acuerdo con el máximo de Q_plan, con y sin rasgos.

**E. Estados y mejoras** (decisiones de combate):
- Grupos por efecto: Tóxico; envenenar; parálisis; sueño; Bostezo; confusión (Rayo Confuso, Contoneo, Camelo, Danza
  Caos); Fuego Fatuo; Drenadoras. **Disponible** = el activo tiene ese movimiento utilizable; **tasa de uso** = elegido /
  disponible; **cuándo**: PS propios, barra del rival, *amenaza KO* (algún movimiento visto del rival puede dejarnos KO,
  `rl/damage.py`), rival ya con estado, turno, al usarlo frente a cuando estaba disponible.
- **Acierta** = en la decisión siguiente el rival muestra el estado nuevo (o confusión / Drenadoras / Bostezo). **Error**
  = no podía funcionar con lo que se veía: rival tras Sustituto; Velo Sagrado; ya con estado mayor (o confuso, o con
  Drenadoras); inmune por tipo (los movimientos de parálisis comprueban la tabla de tipos: Onda Trueno contra tierra;
  Tóxico / veneno contra veneno o acero; Fuego Fatuo contra fuego; Drenadoras contra planta); habilidad conocida
  (Flexibilidad, Inmunidad, Insomnio / Espíritu Vital, Ritmo Propio, Velo Agua).
- **Mejoras** ofensivas (las del barredor con mejora) y defensivas (Def, DefEsp, evasión, precisión, Masa Cósmica,
  Reducción, Rizo Defensa, Reserva): uso, cuándo, *cae después* (nuestro Pokémon se debilita ese turno) y *tope* (las
  estadísticas que sube ya en +6).
- **Resultado**: victoria del combate con − sin cada rasgo (usar un estado, una mejora, Explosión, cambiar de Pokémon;
  cada sinergia, papel o arquetipo del equipo), controlado por ronda.

**F. Elección de movimiento**: eficacia elegida y la mejor disponible; elige la mejor eficacia; ataque resistido
habiendo uno neutro; elige el de **máximo daño** (estimación media de `rl/damage.py`); **KO seguro** = daño mínimo
estimado ≥ barra de PS del rival: lo toma, y con el más preciso cuando hay varios; **riesgo innecesario** = había otro
ataque más preciso con ≥ 85 % del daño medio del elegido; precisión media; golpes de KO directo y Explosión (uso y
momento: PS, barra rival, vivos, turno, amenaza); **efecto secundario** (probabilidad > 0 o crítico alto) entre ataques
de daño parecido (≥ 85 % del máximo) frente a elegir al azar entre ellos; **STAB** y «sin STAB pero más eficaz que el
mejor STAB»; **prioridad** con el rival a ≤ 25 % (separando si somos seguro más rápidos); **contacto** frente a
habilidades que lo castigan (Elec. Estática, Punto Tóxico, Efecto Espora, Cuerpo Llama, Piel Tosca, Gran Encanto
posibles en la especie) con − sin, controlado por ronda.

**G. Cambios en combate**: **voluntario** = cambiar pudiendo atacar; tasa de cambio dada cada situación (sí / no y
diferencia controlada por ronda): **desventaja** (algún tipo de ataque visto o STAB del rival nos hace ×2), PS < 1/3,
rival con subidas ≥ 2, estado propio, amenaza KO. **Resistencia** del que entra = fracción de esos tipos que resiste;
frente a la media de los otros posibles; elige el que más resiste. Igual en los cambios forzados. **Se queda y cae** =
amenazado de KO, con cambio posible, ataca y se debilita ese turno.

**H. Resultados y confianza**: victorias por combate (por ronda y posición), P(completar la ronda | llegar), racha
media (rachas desde la ronda 1); turnos y decisiones por combate; **margen** = fracción de PS que nos quedan al ganar;
**entropía** −Σ p log p sobre las opciones legales y **prob. máxima** por tipo de decisión; victoria por tercios de
prob. máxima (táctico: combate siguiente; combate: prob. máxima media, controlado por ronda); **calibración** del
valor del combatiente en su primera decisión; acuerdo búsqueda–red; **suerte**: fracción de derrotas y victorias con
algún evento en contra (`BAD_FOR_US` de §21), balance de suerte y eventos por combate.

**Sección marcada (verdad)**: fracción de combates perdidos según las especies del equipo verdadero del rival.

### 23.5 Cómo lanzarlo

```bash
python -m rl.play_sim --mode net --battles 22000 --late-runs 300 --procs 3 --out runs/playstudy_v3
python -m rl.play_sim --mode search --battles 14000 --late-runs 200 --procs 2 --suffix _a --out runs/playstudy_v3
python -m rl.playstats --logs runs/playstudy_v3/net --out runs/playstudy_v3/report_net
python -m rl.playstats --logs runs/playstudy_v3/search --out runs/playstudy_v3/report_search
python -m pytest tests/python/test_playlog.py
```
El *runner* del emulador crea un `PlayLogger` y en cada decisión llama a `PlayAgent.tactician` / `battler` (o a su
propio agente) y a `logger.decision(...)`; `rl.playstats --logs <carpeta>` funciona igual sobre esos registros (las
claves que falten en `extras` dejan vacías las métricas que las necesitan).

### 23.6 Réplica en el simulador: muestra, velocidad y primeros resultados (2026-10-01)

Con alphazero_v4 entrenando en la máquina (18 procesos), con `nice 10` y como mucho 3 procesos:

| modo | rachas (tardías) | combates | decisiones | tiempo | combates / hora |
|---|---|---|---|---|---|
| red sola | 1.128 (300) | 22.057 | 264.493 | 25,6 min, 3 procesos | 51.650 |
| búsqueda legal 512, «experience» | 978 (300) | 21.048 | 247.680 | 2,03 h, 2 + 1 procesos | ~10.350 (3.450 por proceso) |

Registros en `runs/playstudy_v3/{net,search}/` (~3,7 kB por combate comprimido). Informes:
`runs/playstudy_v3/report_net/report.md` y `runs/playstudy_v3/report_search/report.md` (más `stats.json` y
`plots/`). El análisis tarda ~5 min por modo (un proceso).

Lo principal (IC 95 % en los informes; R = ronda):
- **Resultados**: victorias por combate, red / búsqueda: R1 98,2 / 98,6 %, R5 87,5 / 90,3 %, R6 91,0 / 91,7 %, Noland
  79,8 / 81,5 %. Racha media desde la ronda 1: 22,2 / 25,4. En el 86 % de las derrotas hubo algún evento de mala
  suerte (60 % en las victorias). La búsqueda coincide con el máximo de la red en el 97,8 % de las decisiones
  (94,8 % contra Noland).
- **Alquiler**: elige sets más fuertes que lo ofrecido (+11,7 de suma de base, más PS y aguante, menos movimientos de
  estado) y legendarios casi siempre (Suicune, Snorlax, Regice, Metagross > 97 %); descarta Shuckle, Xatu, Misdreavus,
  Raichu. Registeel nunca es líder. El líder no resiste más el tipo de la pista que el resto del equipo, pero sí lo
  golpea x2 más (+6,7 puntos). Busca barredores con mejora (lift 1,35) y evita equipos de apoyo (0,85).
- **Cobertura**: lo elegido supera a ~55–66 % de las opciones (mejor que al azar, lejos de la mejor); lo que más
  busca es aguante (percentil 0,65); al cambiar en las rondas 5–6 sube sobre todo el aguante especial.
- **Cambios**: cambia el 72 % de las veces en R1 y el 34 % en R6; coincide con el máximo de Q_plan en el 67 %; casi
  nunca se queda cuando el planificador dice cambiar (0,1 %) pero sí cuando las estadísticas base lo dicen (45 %).
  Cambiar frente a quedarse en estados comparables: sin diferencia en el combate siguiente (+0,2 puntos [−0,4, 0,8]).
- **Sinergias**: evita el sol (lift 0,47) y busca parejas defensivas (1,38); casi no usa las que tiene: Danza Lluvia
  en el 2 % de los combates en que la tiene, Relevo **nunca** (764 decisiones con Relevo disponible).
- **Pista y rasgos**: la pista, solo en la entrada de la red, cambia la elección en ~2 % de los casos; con W y el
  planificador recalculados, en el 21 % (alquiler) y 13 % (cambio). Con los rasgos a cero cambia el 95 % / 75 %: el
  táctico depende sobre todo de W / planificador.
- **Estados**: los usa ~20 % de las veces que puede, casi siempre con el rival entero; errores por lo visible del
  17–26 % (Tóxico a un rival ya con estado o de tipo veneno / acero, Drenadoras a uno ya plantado o de tipo planta),
  Fuego Fatuo solo 7 %.
- **Movimientos**: mejor eficacia en el 92 %, máximo daño en el 74 %, toma el KO seguro en el 81 %; evita el contacto
  contra habilidades que lo castigan (−17 puntos). **Cambios en combate**: cambia +10 puntos más en desventaja de
  tipos y el que entra resiste mejor que la alternativa en el 87 %; amenazado de KO, se queda y cae el 20 %.

## 25. Sondas por agente: cuánto ha aprendido cada uno, por separado (`rl/agent_probe.py`, 2026-10-03)

Objetivo (decidido por el usuario): saber **por separado** cuánto ha aprendido el combatiente y cuánto el táctico, y si
a cada uno le queda margen, para no confundir una meseta del conjunto (P(6 rondas) que deja de subir) con que los dos
agentes se hayan estancado. Es solo diagnóstico: nada de esto entra en el entrenamiento (el módulo se niega a
arrancar en un proceso marcado como entrenamiento).

### 25.1 Definiciones (de abajo arriba)

- **Estado de decisión del táctico**: una pantalla de alquiler o de cambio de una racha concreta, reconstruida
  exactamente con la **receta de repetición** de §21.1 (semilla del entorno + todas las acciones dadas al juego desde
  el principio de la racha). Al reconstruirlo se comprueba el *hash* del estado del juego y el de la observación del
  táctico; si no coinciden con los de la generación, se para.
- **Inicio de combate** (como en §21.1): un estado de decisión del táctico + una opción aplicada. El rival es el que
  el juego genera con esa opción (el verdadero; se comprueba con un *hash* de `gEnemyParty`) y la semilla del RNG del
  combate se fija antes del primer fotograma. Con la misma semilla, el mismo combatiente juega exactamente el mismo
  combate.
- **Banco fijo** (`runs/probe/battles_v1/`): el conjunto de estados guardado **una vez** y reutilizado para todos los
  checkpoints. Así, dos checkpoints juegan los mismos estados con las mismas semillas.
- **Comparación emparejada**: con los mismos estados y semillas, la diferencia entre dos agentes se mide estado a
  estado (d_s = media de A en s − media de B en s) y su IC al 95 % es media(d) ± 1,96·desv(d)/√n. El azar común se
  cancela en buena parte, así que el IC de la diferencia es mucho más estrecho que el de cada tasa por separado.
- **IC al 95 %** de una media: ± 1,96·desv/√n sobre las unidades independientes (los estados; las semillas de un
  mismo estado se promedian antes, porque no son independientes entre sí).
- **Q_plan** (§20): el valor que el planificador del táctico (W + expectimax) da a cada opción, en victorias
  esperadas. **Planificador de referencia**: el de v3 final (su W `team_eval_0580.pt`), con la misma semilla del
  planificador en cada estado para todos los checkpoints.
- **Regret** de una elección = valor de la mejor opción − valor de la opción elegida (0 = eligió la mejor). Se mide
  de dos maneras:
  - **Regret_Q**: con el Q_plan de referencia (barato, pero es la opinión del planificador, no el resultado).
  - **Regret por resultado real**: con las victorias que se consiguen de verdad jugando cada opción.
- **Sesgo del máximo**: el máximo de varias medias con ruido es, en promedio, mayor que el máximo de los valores
  verdaderos; por eso "mejor − elegida" con las mismas semillas sobrestima el regret. **Regret validado**: se elige la
  mejor opción con la mitad de las semillas y se mide con la otra mitad (y al revés, promediando): no tiene ese sesgo
  (tiende a quedarse corto, porque la mitad de las semillas a veces elige mal la mejor). La verdad queda entre los
  dos.
- **Planificador como táctico**: el táctico sustituido por "elegir el máximo de Q_plan" con el mismo W / planificador
  del checkpoint. Mide si la red del táctico añade algo sobre sus propios rasgos.
- **Juego cruzado**: táctico de un checkpoint X con combatiente de otro checkpoint Y (los dos con la codificación v5 y
  el táctico con los rasgos de **su** checkpoint).

### 25.2 El banco (`generate`, una vez)

Rachas de alphazero_v3 final (`runs/alphazero_v3/latest.pt`; táctico = su red voraz con sus rasgos W / planificador,
la semilla del planificador fija por estado; combatiente = su red voraz) desde el principio de las rondas 5 / 6 / 3
(40 / 40 / 20 %), con el protocolo de `rl.eval_rounds` (la racha se para al completar la ronda o al perder). Cada
decisión del táctico es un estado. De ellos, muestreo estratificado (fijo):
- **B**: inicios de combate con la opción que eligió v3 (nuestro equipo y el rival verdadero fijos): cuotas
  R5 650, R6 650, R3 300, Noland 250 (las que haya); **K = 4 semillas** por estado.
- **C1**: 300 alquileres + 600 cambios. **C2**: subconjunto de C1, 50 alquileres + 100 cambios estratificados por
  ronda, con **32 semillas** por estado.

### 25.3 Qué mide cada sonda

**B. Solo el combatiente.** Cada combatiente juega exactamente los inicios de combate de B con sus K semillas →
victorias por combate (total y por estrato: R3, R5, R6, Noland) con IC, y la diferencia emparejada con v3 final.
Combatientes: la red de cualquier checkpoint (voraz, por lotes en la GPU); opcional, sobre un subconjunto fijo de B, la
búsqueda legal de 512 con el modelo del rival «experience» (el del checkpoint si lo lleva, si no
`runs/opp_model_v3/model.pt`) y la búsqueda **perfecta** de 2048 (información perfecta: un techo de referencia, solo
diagnóstico).

**C. Solo el táctico, con un combatiente fijo** (la red de v3 final, voraz).
- **C1**: en cada estado de C1, la elección de la red del táctico de cada checkpoint (con sus propios rasgos) y la del
  planificador como táctico; Regret_Q con el planificador de referencia y acuerdo con su máximo.
- **C2**: cada opción de un estado de C2 (cambio: las ≤ 10; alquiler: las 8 mejores según el Q_plan de referencia
  más todas las que haya elegido algún táctico evaluado) se juega con el combatiente fijo desde ese estado con las 32
  semillas emparejadas **hasta el final de la ronda** (entre combates decide el táctico de v3 final, también fijo).
  Resultado de una opción = victorias que quedan en la ronda. La tabla de resultados se calcula **una vez por opción**
  y la comparten todos los checkpoints: un checkpoint nuevo solo añade las opciones que nadie había elegido. Por
  checkpoint: regret bruto y validado (media ± IC sobre estados), por tipo (alquiler / cambio), y el valor medio de lo
  elegido.

**A. Juego cruzado.** Protocolo de `rl.eval_rounds` (rondas 1–6, N rachas por ronda, semilla 777; la racha j de la
ronda k usa el entorno de semilla (777 + 1000k)·100003 + j, y empieza en el alquiler de la ronda k), combatiente de
red sola. Combinaciones: (táctico X, combatiente Y) para X, Y ∈ {v3 final, v4 último} y el planificador como táctico
(del mismo checkpoint) con su combatiente. Las mismas rachas para todas las combinaciones (la semilla del
planificador también es fija por decisión), así que sus diferencias por ronda son emparejadas. P(6 rondas) = producto
de las P(completar) por ronda, con un IC *bootstrap*.

### 25.4 Implementación

- Un único bucle **en paralelo por lotes** (`play_lockstep`): hasta 256 partidas a la vez por proceso; en cada paso
  todas las decisiones de combate pendientes se evalúan juntas en la GPU. Las decisiones sin elección se juegan como
  en `FactoryEnv` (movimiento 0) y cuentan para el límite de 300 decisiones (un combate cortado cuenta como derrota).
- Caché y reanudación: cada resultado se escribe en una línea (`runs/probe/results/...`, un archivo por agente y por
  proceso) y al relanzar se saltan los que ya están. El coste de cada ejecución queda en `runs/probe/timing.jsonl`.
- Conductor de curvas (`curve`): genera el banco si falta, y para cada checkpoint de la lista (más v3 final) B, C1 y
  la ampliación de la tabla de C2; después la rejilla de A y las búsquedas de B en un subconjunto; informe en
  `runs/probe/report/` (`report.md`, `summary.json`, `curves.png`).
- Pruebas (`tests/python/test_agent_probe.py`): todo estado se reconstruye exactamente (hash del juego y de la
  observación, mismo rival); el inicio de combate con el RNG propio del juego y la red de v3 reproduce **el mismo
  resultado** que en la generación; los resultados de B no dependen del tamaño del lote y se reanudan; la elección de
  la red de v3 en C1 es la de la generación; el planificador elige el máximo de Q_plan; la tabla de C2; el juego
  cruzado manda cada decisión a la red que toca; el regret validado no tiene el sesgo del máximo (prueba con opciones
  iguales); se niega a arrancar en un proceso de entrenamiento.

### 25.5 Cómo lanzarlo

```bash
# todo (banco si falta, B / C1 / C2 / A, informe), reanudable; procesos y prioridad bajos mientras entrena v4
nice -n 15 python -m rl.agent_probe curve --procs 2 --a-runs 512 \
    --ckpts runs/alphazero_v4/ckpt_00004041414.pt,...,runs/alphazero_v4/latest_copia.pt \
    --search search:runs/alphazero_v3/latest.pt,perfect:runs/alphazero_v3/latest.pt
# piezas sueltas
python -m rl.agent_probe battler --battler net:CKPT            # B (o search:CKPT[:SIMS], perfect:CKPT[:SIMS] --subset 300 --seeds 2)
python -m rl.agent_probe tactician --tactician net:CKPT        # C1 (red y planificador)
python -m rl.agent_probe outcomes --tactician net:CKPT         # C2: añade a la tabla lo que elige CKPT
python -m rl.agent_probe crossplay --tactician net:X --battler net:Y --a-runs 512
python -m rl.agent_probe report --ckpts ...                     # rehace el informe desde la caché
python -m pytest tests/python/test_agent_probe.py
```
Para un checkpoint nuevo basta volver a lanzar `curve` con él añadido a `--ckpts`: lo ya calculado se reutiliza. El
checkpoint de la iteración i de v4 es `runs/alphazero_v4/ckpt_<40014·(i+1), 11 cifras>.pt`; conviene copiarlo con su
`team_eval_*.pt` (el que nombra el checkpoint) porque `latest.pt` cambia mientras entrena.

### 25.6 Primeros resultados (2026-10-03; v4 en la iteración 899 de 1500, aún sin la caída de la tasa)

Ejecución: `curve` con v3 final y v4 en las iteraciones 50, 100, 200, …, 800 y 899; 2 procesos con `nice 15`
mientras v4 entrenaba. Banco: 500 rachas de v3 (2.559 combates, 100 s); B = 1.779 inicios (R5 650, R6 650, R3 300,
Noland 179) × 4 semillas; C1 = 300 alquileres + 600 cambios; C2 = 50 + 100 decisiones × 32 semillas. Coste: B ~35 s
por checkpoint (~90.000 decisiones); C1 ~22 s; C2 (1.412 opciones, 45.000 partidas hasta el final de la ronda) 39 min
la primera vez (cada checkpoint nuevo solo añade sus opciones nuevas); A ~6,5 min por combinación (512 rachas × 6
rondas); búsqueda 512 sobre 300 × 2: 5–7 min; perfecta 2048: 11–13 min. Informe: `runs/probe/report/report.md`.

**B. Combatiente** (red voraz; victorias por combate, ± IC 95 %; Δ emparejada con v3 final):

| | total | R3 | R5 | R6 | Noland | Δ vs v3 |
|---|---|---|---|---|---|---|
| v3 final | 89,0 ± 1,0 | 96,4 | 87,7 | 90,4 | 76,5 | – |
| v4 it 50 | 83,3 | 93,3 | 81,3 | 83,7 | 72,3 | −5,7 ± 1,0 |
| v4 it 300 | 85,9 | 95,3 | 84,1 | 87,4 | 70,9 | −3,2 ± 0,9 |
| v4 it 600 | 87,1 | 95,5 | 86,0 | 87,9 | 74,4 | −1,9 ± 0,9 |
| v4 it 899 | 87,5 ± 1,1 | 95,8 | 85,9 | 88,5 | 76,0 | −1,5 ± 0,9 |

En un subconjunto de 300 × 2 (mismos estados y semillas que la red): la búsqueda legal de 512 no añade nada a la red
(v3 −0,2 ± 1,6; v4 +0,8 ± 2,0 puntos); la búsqueda **perfecta** de 2048 añade +5,0 ± 2,5 (v3) y +4,5 ± 2,3 (v4).

**C1. Táctico, Regret_Q** (victorias según el Q_plan de v3): la red de v4 baja de 0,31 (alquiler) / 0,13 (cambio) en
la it. 50 a 0,06 / 0,03 en la 899, lo mismo que la red de v3 final (0,06 / 0,03); acuerdo con el máximo de Q_plan
59 % / 82 %. El planificador de v4 (su propio W) se separa del de v3 en 0,02 / 0,006.

**C2. Táctico, resultado real** (combatiente fijo = red de v3): valor de la opción elegida ≈ 3,42 victorias en lo que
queda de ronda para la red de v3 **y** la de v4 it. 899 (Δ emparejada −0,003 ± 0,056); ningún checkpoint ni el
planificador se separan de la red de v3 más allá del ruido (Δ entre −0,07 y 0,00). Regret: bruto 0,34 victorias
(10 puntos de P(completar) por decisión), validado 0,08 ± 0,09 victorias (0,6 ± 2,1 puntos de P(completar)): casi
todo el regret bruto es el sesgo del máximo; el regret real del táctico por decisión está entre esos dos números y
con 32 semillas no se distingue de 0.

**A. Juego cruzado** (P(completar) por ronda, 512 rachas, semilla 777; v3 + v3 reproduce §20.6: 85,4 / 85,9 / 65,2 /
79,9 / 43,0 / 43,8, P(6) 7,2 %):

| táctico | combatiente | R5 | R6 | P(6) |
|---|---|---|---|---|
| red v3 | v3 | 43,0 | 43,8 | 7,19 % [6,0, 8,4] |
| red v4 899 | v4 899 | 38,3 | 33,2 | 4,20 % [3,4, 5,1] |
| red v3 | v4 899 | 35,4 | 35,7 | 3,87 % |
| red v4 899 | v3 | 41,2 | 42,0 | 6,36 % |
| planificador v4 | v4 899 | 38,9 | 37,9 | 5,05 % |
| planificador v3 | v3 | 41,8 | 40,2 | 6,55 % |

Diferencias emparejadas de P(completar), media de R1–R6: combatiente v4 − v3 = −5,4 ± 2,1 (con el táctico de v3) y
−3,4 ± 2,0 (con el de v4); táctico v4 − v3 = +1,1 ± 1,8 / −0,9 ± 1,8 (no significativo); planificador − red v4 como
táctico +1,2 ± 1,7 (no significativo).

**Lectura.**
- **Combatiente**: es el agente que sigue aprendiendo en v4 (+4 puntos por combate de la it. 50 a la 899, todavía
  subiendo despacio: +0,4 de la 600 a la 899) y el que explica todo el retraso de v4 frente a v3 (−1,5 puntos por
  combate en B, −3 a −5 puntos de P(completar) por ronda en A). v3 final terminó **después** de la caída de la tasa
  (WSD), que en v3 dio +3 puntos de P(6); v4 todavía no la ha empezado (it. 1275–1499), así que esta distancia es la
  esperada a estas alturas y no una meseta. **Margen**: la búsqueda perfecta saca +4,5–5 puntos por combate en el
  mismo subconjunto (techo con información perfecta; la búsqueda legal no saca nada), y R5 / Noland (86 / 76 %) son
  donde más falta. Si tras la caída v4 no pasa de ~89 % en B, el combatiente estará en la meseta de esta red.
- **Táctico**: llegó pronto (Regret_Q de v3 hacia la it. 300–900) y por el resultado real no se distingue de v3 ni
  del planificador; su regret real por decisión es pequeño (validado ~0,6 puntos de P(completar), indistinguible de
  0). Margen del táctico **con este combatiente**: pequeño; no es el cuello de botella. El planificador como táctico
  no es mejor que la red (en A, +1,2 ± 1,7 puntos; en C2, −0,02): la red del táctico ha destilado su planificador y
  no le añade casi nada.
- Para saber cuánto falta de verdad al táctico haría falta C2 con más semillas (el IC del regret validado, ±0,09
  victorias con 32 semillas, se estrecha como 1/√semillas) o un combatiente más fuerte (con el techo perfecto los
  equipos podrían importar más).

## 26. Rumbo de alphazero_v5 (decidido con el usuario 2026-10-03, a confirmar con las sondas de §25)
- **Combatiente**: (1) el modelo del rival aprendido como ENTRADA ("rasgos de amenaza": para cada rival, a partir de
  sus sets probables, la probabilidad de que tenga un ataque súper eficaz contra cada uno de los nuestros y el daño
  máximo esperado que puede hacerles); (2) el modelo aprendido también en la búsqueda del entrenamiento (en lugar de
  `factory_sets`), para entrenar en el mismo mundo en que se juega.
- **Táctico**: Q_plan como línea base de PPO (ventaja = retorno − Q_plan, menos ruido); táctico = planificador +
  corrección aprendida (la red solo aprende cuándo desviarse); planificador mejorado (más rivales sorteados, más
  profundidad y planificación explícita para Noland: IVs altos y exclusión de nuestros sets). (La W condicionada a la
  pista ya existe desde v3.)
- Muchas iteraciones con WSD, como v4.

## 27. Experimentos sin conexión del combatiente antes de alphazero_v5 (`rl/offline_data.py`, `rl/offline_train.py`, 2026-10-04)

Pregunta: ¿cambios en la red del combatiente (más capacidad, más cómputo, otro objetivo de valor, objetivos de
búsqueda mejores) la mejoran **mucho**, antes de pagar una ejecución larga de v5? Diagnóstico de partida: la red es
pequeña (tronco ~0,54 M parámetros, 2 capas), entrena 2–3 s por iteración frente a 85 s de autojuego (la GPU está
casi parada), su política queda a KL 0,65 del objetivo de Gumbel y su valor explica ~45 % de la varianza: ¿le falta
capacidad? Todo se mide en el banco B de §25 (red sola, voraz), emparejado con v4 (`runs/alphazero_v4/stopped_it1111.pt`,
it. 1111, **antes** de su caída de la tasa) y con un control.

### 27.1 Definiciones (de abajo arriba)

- **Destilación sin conexión** (*offline*): se juega una vez con v4 + búsqueda y se guardan las decisiones con lo que
  la búsqueda recomendó; después se entrena una red **desde cero** solo con ese archivo, sin volver a jugar. Permite
  comparar redes con exactamente los mismos datos, en minutos en lugar de días.
- **Posición**: una decisión del combatiente en el autojuego. Se guarda su entrada codificada (la de la red), la máscara
  de acciones legales, el **objetivo de política** (la distribución de Gumbel softmax(logits + σ(Q completada)) de
  §19.1, exactamente lo que entrena v4), la Q de la raíz (media de Q ponderada por visitas), el resultado del combate
  **z** (1 victoria / 0 derrota) y ronda, combate, Noland.
- **Clave de repetición**: (run, li) = la racha (su semilla de `SimBackend.reset` y la lista de **todas** las acciones
  dadas al juego) y cuántas de esas acciones van antes de la decisión. Como el simulador es determinista, aplicarlas
  reconstruye el estado exacto y la memoria del jugador. Contrato y API: `runs/v5_offline/CONTRACT.md`.
- **Apartado** (*held-out*): el 5 % de las rachas (todas sus posiciones) nunca se entrena; ahí se miden las métricas.
- **KL(objetivo ‖ red)**: cuánta información pierde la política de la red respecto al objetivo de la búsqueda (0 =
  idénticas). **Top-1**: fracción en que la acción más probable de la red es la del objetivo.
- **Suelo de ruido**: dos búsquedas del mismo estado no dan el mismo objetivo (azar de la búsqueda). La red óptima
  aprende el **objetivo medio**; su KL a un objetivo concreto nunca baja de E[KL(objetivo ‖ objetivo medio)]. Así,
  KL de la red = suelo + KL(objetivo medio ‖ red), y solo el segundo término se puede reducir.
- **EV** (varianza explicada, §7) del valor contra el objetivo de valor con que se entrenó; **EV z**: contra z.
- **Brier**: media de (valor predicho − z)², en combates no cortados. Mide si el valor acierta el resultado real
  (menor es mejor; predecir siempre la media daría ≈ 0,10).
- **Objetivo de valor con mezcla m**: (1 − m) z + m Q_raíz. v4 usa m = 0,5. m = 0: solo el resultado; m = 1: solo la Q
  de la búsqueda.
- **Semilla de entrenamiento**: la inicialización y el orden de los lotes. Dos redes con la misma receta y distinta
  semilla juegan distinto en B; esa variación (unos ±0,5 puntos aquí) es ruido que hay que separar de los efectos.
- **IC bootstrap al 95 %**: se remuestrean los 1.779 estados de B con reemplazo 2.000 veces y se toman los percentiles
  2,5 y 97,5 de la media (de la tasa o de la diferencia emparejada, §25.1). Para un grupo de semillas, cada estado
  usa la media de las redes del grupo.

### 27.2 El conjunto de datos (`python -m rl.offline_data record`)

- **Cómo**: el trabajador de autojuego de v4 (`SelfPlayerV4`) con la red de it. 1111 en el servidor de la GPU, su
  táctico (muestreado, con sus rasgos W / planificador y su tabla v_next), su currículo de rondas de salida, y la
  búsqueda de entrenamiento: Gumbel, 256 simulaciones, 8 determinizaciones, top-16, rival `factory_sets` (permitido
  solo en entrenamiento: el módulo llama a `mark_training()`, así que no puede existir búsqueda perfecta). Nada se
  entrena durante la grabación. Un `SimBackend` que anota cada `reset` / `act` da la clave de repetición.
- **Tamaño**: 3.335.841 posiciones de 20.336 rachas en **3 h** con 12 procesos (310 decisiones/s de media; bajó a
  170–250/s mientras otro trabajo usaba la GPU). Apartadas 172.001 (5 % por racha). Rondas 1–6: 0,61 / 0,51 / 0,50 /
  0,34 / 0,50 / 0,52 M; 7+: 0,36 M; Noland 141.253. 17 GB en `runs/v5_offline/main/` (un `.npy` por campo; los
  fragmentos originales en `main/shards/`, otros 17 GB, se pueden borrar).
- **Repetición exacta**: `python -m rl.offline_data check` (500 posiciones al azar) y las pruebas
  (`tests/python/test_offline_data.py`: cada posición reconstruida da **la misma entrada codificada**, dos repeticiones
  dan el mismo hash de estado, las dos formas de clave coinciden, el backend entregado es una copia, y una posición
  repetida se puede volver a buscar). Las 3 re-búsquedas de abajo (374.224 posiciones) también comprueban cada
  entrada: ninguna diferencia.
- **Variantes de objetivo para las mismas posiciones** (`research`): las posiciones se reconstruyen con su clave y se
  buscan otra vez. Sobre 171.938 posiciones (rachas enteras al azar; el plan era 400.000, pero la de 1.024 simulaciones
  iba a 35/s con la GPU compartida y se cortó): (a) **1.024 simulaciones** (`factory_sets`), 77 min; (b) **256 con el
  modelo del rival aprendido** (`experience`, el del checkpoint), 13 min; (c) sobre 30.348 de ellas, **otra búsqueda
  de 256 igual que la original**, para medir el ruido.

**El ruido del objetivo** (con (c) y (a)): dos búsquedas de 256 del mismo estado eligen la misma mejor acción solo el
**57,8 %** de las veces (§20.2: 61 %); KL(objetivo₁ ‖ media de los dos) = 0,20, así que el suelo de ruido es ≈ 0,4
(con dos muestras se ve la mitad, como en §20.2: 0,38–0,43). La búsqueda de 1.024 coincide con la de 256 en la mejor
acción el 54,6 % y es más afilada (entropía 0,24 frente a 0,43). La red de v4 está a KL **0,719** de los objetivos
apartados: ≈ 0,4 de suelo + ≈ 0,3 que sí se podría reducir.

### 27.3 El entrenador (`python -m rl.offline_train`)

Receta = la pérdida del combatiente de `rl.alphazero.train_iteration`: entropía cruzada al objetivo de política +
0,5·MSE del valor normalizado (ValueNorm con la media y varianza exactas de los objetivos de entrenamiento), AdamW
(3·10⁻⁴, decaimiento 10⁻⁴), recorte 1,0, lotes de 512, calentamiento de 1.000 pasos y caída coseno al 5 % a lo largo
de 10 épocas (pasadas por los 3,16 M de posiciones de entrenamiento). **Control** = la arquitectura de v4 (d 128,
2 capas, 4 cabezas) con esta receta, 3 semillas (`control`, y `thr_ctrl_s1/s2`, idénticas, entrenadas por §27.y).
Se entrena solo el combatiente: las cabezas del táctico quedan sin entrenar, así que los checkpoints de
`runs/v5_offline/models/` son combatientes, no agentes (B los juega con `net:`; llevan el modelo del rival de v4 para
`search:`). ~10–25 min por red en la GPU (más con la GPU compartida).

### 27.4 Resultados

Tabla completa: `runs/v5_offline/report.md` (y `report_e4.md`). B = victorias por combate en B (red sola, voraz,
1.779 inicios × 4 semillas); Δ = diferencia emparejada en puntos con IC bootstrap al 95 %; "fwd" = milisegundos por
pasada de la red con lotes de 256 en la GPU (fp32, medido intercalado, `speed.json`), relativo al control.

| variante | semillas | parámetros | KL apart. | top-1 | Brier | B % | R5 | R6 | R3 | Noland | Δ vs control | Δ vs v4 | fwd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 it 1111 (en línea) | 1 | 0,65 M | 0,719 | 0,592 | 0,092 | 87,9 | 86,6 | 88,2 | 95,8 | 78,4 | – | – | – |
| **control** (d 128, 2 capas, 10 ép., m 0,5) | 3 | 0,65 M | 0,717 | 0,580 | 0,084 | 88,8 | 87,5 | 89,4 | 96,2 | 78,6 | – | +0,9 [+0,1, +1,6] | 1,00× |
| datos 25 % (0,8 M) | 1 | 0,65 M | 0,779 | 0,555 | 0,092 | 86,4 | 83,5 | 88,2 | 94,7 | 76,7 | −2,4 [−3,2, −1,6] | −1,5 [−2,5, −0,5] | 1,00× |
| datos 50 % (1,6 M) | 1 | 0,65 M | 0,744 | 0,571 | 0,087 | 87,9 | 86,6 | 88,4 | 96,0 | 77,0 | −0,9 [−1,7, −0,2] | −0,1 [−1,0, +0,9] | 1,00× |
| 20 épocas | 2 | 0,65 M | 0,717 | 0,582 | 0,087 | 89,5 | 88,3 | 90,2 | 96,5 | 79,7 | +0,7 [+0,2, +1,2] | +1,6 [+0,8, +2,4] | 1,00× |
| 4 capas | 1 | 0,92 M | 0,716 | 0,580 | 0,085 | 88,2 | 86,3 | 89,3 | 95,8 | 78,1 | −0,6 [−1,3, +0,1] | +0,3 [−0,7, +1,1] | 1,49× |
| 6 capas | 1 | 1,18 M | 0,718 | 0,580 | 0,084 | 88,6 | 86,7 | 89,6 | 96,0 | 78,9 | −0,2 [−0,8, +0,5] | +0,6 [−0,3, +1,6] | 1,98× |
| doble anchura (d 256) | 2 | 2,10 M | 0,713 | 0,583 | 0,088 | 89,3 | 87,8 | 90,2 | 96,3 | 80,2 | +0,6 [+0,0, +1,1] | +1,4 [+0,6, +2,2] | 1,62× |
| d 256, 4 capas | 1 | 3,15 M | 0,723 | 0,580 | 0,088 | 89,3 | 88,0 | 90,2 | 96,3 | 79,1 | +0,6 [−0,2, +1,2] | +1,4 [+0,5, +2,3] | **2,54×** |
| valor m = 0 (solo z) | 1 | 0,65 M | 0,727 | 0,577 | 0,110 | 88,3 | 86,7 | 89,6 | 95,8 | 77,2 | −0,4 [−1,1, +0,2] | +0,4 [−0,5, +1,3] | 1,00× |
| **valor m = 1 (solo Q raíz)** | 3 | 0,65 M | **0,708** | 0,585 | **0,080** | 88,8 | 87,7 | 89,6 | 96,6 | 77,0 | +0,0 [−0,4, +0,5] | +0,9 [+0,1, +1,7] | 1,00× |
| **d 256 + m = 1** | 2 | 2,10 M | **0,705** | 0,587 | 0,081 | **89,8** | 88,5 | 90,3 | 96,6 | 80,9 | **+1,0 [+0,5, +1,5]** | **+1,9 [+1,0, +2,7]** | 1,62× |

(KL apart. y Brier: media de las semillas, última época. El Brier de v4 sale de su propio valor en la raíz, grabado.)

**E1, capacidad y cómputo.** Más profundidad no mejora nada (4 y 6 capas: KL 0,716 / 0,718, B −0,6 / −0,2). El doble de
anchura baja la KL 0,004 y da +0,6 [0,0, +1,1] en B, a 1,6× de coste; d 256 con 4 capas ya sobreajusta (KL 0,723) y
cuesta 2,5×. Más épocas: 20 frente a 10 dejan la KL igual (0,717) pero dan +0,7 [+0,2, +1,2] en B (2 semillas). La
KL de **entrenamiento** tampoco baja con más capacidad (≈ 0,69 en todas, salvo d 256: 0,65, que sobreajusta): **la red
no está corta de capacidad**; la KL ~0,7 es ≈ 0,4 de ruido del objetivo + lo que permiten los datos. Dónde deja de
mejorar la KL apartada: ya en el tamaño de v4 (d 128, 2 capas) y a las ~10 épocas.

**Datos.** Es lo único que mueve mucho la aguja: con la misma red y receta, 0,8 / 1,6 / 3,2 M posiciones dan KL
0,779 / 0,744 / 0,717 (−0,03 por cada duplicación, sin aplanarse todavía) y B 86,4 / 87,9 / 88,8 %; con 0,16 M
(E4, abajo) 79,6 %. Una red desde cero con 3,2 M posiciones de un solo checkpoint ya iguala o pasa a v4, que vio
~45 M posiciones pero siempre de una ventana de 4 iteraciones (160.000 posiciones), con reuso 4.

**E3, objetivo de valor.** Con m = 0 (solo z) el valor sobreajusta en seguida (las posiciones de un mismo combate
comparten z: la red memoriza combates): Brier 0,085 en la época 1 → 0,110 en la 10, y arrastra a la política (KL
0,727). Con m = 0,5 (v4) pasa lo mismo, más despacio (0,0775 → 0,084). Con **m = 1** el valor no sobreajusta (Brier
0,080 estable; 0,107 en las rondas 5–7 frente a 0,113 del control y 0,122 de v4; Noland 0,128 / 0,144 / 0,153) y la
política mejora (KL 0,708, la mejor con la red de v4; reproducido en 3 semillas: 0,7083 / 0,7083 / 0,7081). En B con la
red sola no se nota (+0,0), lo esperado: B solo usa la política. Con búsqueda (búsqueda legal de 256 con el modelo
del rival de v4, 600 estados × 2 semillas): v4 87,4 %, control 87,8 %, m = 1 88,6 %; m = 1 − control +0,8 [−1,1,
+2,4], dirección coherente pero sin potencia. Nota: el valor de v4 en línea (Brier 0,092) es peor que cualquiera de
los entrenados sin conexión, incluso el control (0,084).

**E4, objetivos de búsqueda mejores** (mismas 171.938 posiciones; red y receta del control, 6 épocas, 3 semillas por
variante; con tan pocos datos las redes rinden ~80 % en B, así que solo cuentan las diferencias):

| objetivo | KL a su objetivo | KL al objetivo de 256 grabado | Brier | B % | Δ vs 256 `factory_sets` |
|---|---|---|---|---|---|
| 256, `factory_sets` (el grabado) | 0,920 | 0,920 | 0,091 | 79,6 | – |
| 1.024, `factory_sets` | 1,103 | 0,937 | 0,091 | 80,1 | +0,4 [−0,2, +1,1] |
| 256, modelo del rival aprendido | 0,916 | 0,926 | **0,086** | 80,1 | +0,5 [−0,1, +1,1] |

Ni 4× más simulaciones ni el modelo aprendido dan un salto: +0,4 / +0,5 puntos, dentro del ruido. El modelo aprendido
en la búsqueda de entrenamiento (punto 2 del combatiente en §26) **no empeora** el objetivo respecto a `factory_sets`
(que lee la lista real) y su Q de raíz predice algo mejor el resultado (Brier 0,086 frente a 0,091).

### 27.5 Advertencias

- **Un solo paso de mejora.** Todas las redes destilan las búsquedas de **una** red (v4 it. 1111). Mejorar la
  imitación de ese objetivo no garantiza que, en bucle (autojuego → búsqueda con la red nueva → objetivo nuevo), la
  mejora se acumule o se mantenga; en particular el valor con m = 1 se entrena con la Q de una búsqueda que usa ese
  mismo valor en las hojas (en línea podría reforzar sus propios errores; aquí la Q es de la búsqueda de v4, fija).
- **B mide la política**, no el valor; el valor solo se juzga por el Brier apartado y por la prueba con búsqueda (sin
  potencia).
- **Ruido entre semillas** de ±0,5 puntos en B: las variantes de una sola semilla con |Δ| < 1 no son concluyentes.
- **El control sin conexión ya supera a v4** (+0,9 [+0,1, +1,6]): parte puede ser la caída de la tasa (v4 it. 1111
  aún no la había hecho, §25.6) y parte el entrenar con 3,2 M posiciones de una sola política en lugar de una ventana
  pequeña.
- E4 se hizo con 0,17 M posiciones (red débil); con muchos más datos un objetivo más limpio podría importar más o menos.

### 27.6 Recomendación para v5

1. **No hace falta una red más grande ni más profunda.** El cuello de botella no es la capacidad. Si se quiere algo de
   anchura, d 256 (2 capas) da +0,6 / +1,0 puntos en B (con m = 1) a 1,6× de coste por pasada (no pasa del límite de
   2×); 4+ capas no aportan y d 256 × 4 capas cuesta 2,5×: no.
2. **Valor con m = 1 (Q de la raíz)** en lugar de 0,5: mejor valor (Brier 0,080 frente a 0,084–0,092, sobre todo en las
   rondas 5–7 y Noland) y mejor política (KL −0,009), sin coste. Vigilar en línea el riesgo de auto-refuerzo (p. ej.
   m = 0,8–0,9 como término medio, o empezar en 0,5 y subir).
3. **Más datos por actualización, no más red**: lo que mueve la aguja es la cantidad de posiciones distintas. En v5,
   una ventana de repetición mucho mayor (p. ej. 20–40 iteraciones = 0,8–1,6 M posiciones en lugar de 4) y más pasos
   de entrenamiento por iteración (entrenar ocupa ~3 % de cada iteración; 20 épocas frente a 10 dio +0,7 en B). Coste
   de inferencia: ninguno.
4. El **modelo aprendido en la búsqueda de entrenamiento** (§26 punto 2) se puede adoptar: no empeora los objetivos.
   Más simulaciones (1.024) no compensan su coste de 4×.
5. Expectativa honesta: ninguno de estos cambios del combatiente, solo, da una mejora **grande** (los mejores, +1 a
   +2 puntos por combate en B sobre v4). Para ver el conjunto, d 256 + m = 1 + ventana grande es la combinación que
   indican los datos.

### 27.7 Cómo repetirlo

`runs/v5_offline/README.md` (todas las órdenes); `runs/v5_offline/CONTRACT.md` (la clave de repetición y la API
`replay_positions`, que usa §27.y). Las redes y sus métricas por época: `runs/v5_offline/models/<nombre>/`
(`metrics.json`); los registros de entrenamiento en `runs/v5_offline/logs/`. Las primeras pruebas con 0,6 M posiciones
(`_pv_*`, un adelanto de los datos) mostraron lo mismo: con m = 1 el valor no sobreajusta y la red más grande sobreajusta
antes.

```bash
python -m rl.offline_data record --out runs/v5_offline/main --workers 12 --hours 3
python -m rl.offline_data consolidate --out runs/v5_offline/main
python -m rl.offline_train --name control                       # d 128, 2 capas, 10 épocas, m 0,5
python -m rl.offline_train --name w256vq --d 256 --heads 8 --value-mix 1
python -m rl.offline_train --bench control,w256vq               # banco B (rl.agent_probe)
python -m rl.offline_train --report "control=control+thr_ctrl_s1+thr_ctrl_s2,w256vq=w256vq+w256vq_s1"
python -m pytest tests/python/test_offline_data.py
```

### 27.x Planificador mejorado (`rl/planner2.py`, medido como táctico, sin entrenar; 2026-10-04)

Objetivo (§26): las sondas de §25 dicen que la red del táctico ya es su planificador (expectimax de profundidad 1,
8 rivales sorteados, números aleatorios comunes, W como hoja), así que el techo del táctico es la calidad del
planificador. Antes de decidir v5 se construye un planificador mejor, con cada mejora como **opción que se enciende
por separado**, y se mide como táctico con las sondas C1, C2 y A de §25.

#### 27.x.1 Definiciones (de abajo arriba)

- **Planificador antiguo**: `rl.team_eval.plan_values` (§20.2.4), el de v3 / v4 (y del que salen los rasgos Q_plan
  de sus redes).
- **Muestreador de rivales del planificador**: de dónde salen los rivales sorteados del combate k (sus Pokémon son
  los que podríamos coger en el cambio siguiente). **experience**: el modelo aprendido de las propias partidas (§22),
  cubo de la ronda, sin especies ni objetos repetidos, sin las especies en pantalla y coherente con la pista oída
  (muestreo por rechazo). **factory_sets**: la lista de sets de la Factory (información oculta; solo entrenamiento o
  diagnóstico con `PYBATTLE_DIAGNOSTIC_PRIOR=factory_sets`).
- **Ruido de muestreo**: con otra semilla se sortean otros rivales y Q_plan cambia un poco. Se mide con (1) la
  **desviación típica entre semillas del Q_plan centrado** (Q − su media sobre las opciones: es lo que mueve el
  máximo), (2) el **acuerdo entre semillas** (probabilidad de que dos semillas elijan lo mismo) y (3) el
  **regret_ref** = valor de la mejor opción − el de la elegida, ambos según la misma variante con 256 rivales y
  profundidad 1 (el planificador «convergido»): lo que cuesta el ruido según el propio planificador.
- **Pista del combate siguiente** (`h=1`): en la pantalla de cambio el asistente ya ha dicho la pista del próximo
  rival. El planificador antiguo valora el cambio futuro (tras ganar el combate k) con W sin pista: elige la
  respuesta como si no fuera a saber nada más. Con `h=1`, por cada rival sorteado n del combate k se sortea también
  el rival del combate k+1, se calcula su pista h'_n como el juego, y la respuesta se elige con p_{k+1} = W(· | h'_n).
  Es «la media de los máximos» en vez de «el máximo de la media»: el planificador sabe que cuando cambie sabrá algo
  más. Antes de Noland no hay pista (desconocida, como en el juego).
- **Profundidad 2** (`d=2`): dos cambios por delante (ya existía en §20.2.4: haz de 3 respuestas, n2 = 4 rivales del
  combate k+1 por rival del k; ahora, con `h=1`, esos rivales salen con la pista h'_n).
- **Reglas de Noland** (comprobadas en el decomp, `FillFactoryBrainParty` / `GetFactoryMonFixedIV` /
  `GetFrontierBrainStatus`, en `SimBackend.opponent_knowledge_for` y con pruebas contra el juego): aparece en los
  combates 21 y 42 de la racha (combate 7 de las rondas 3 y 6) y después cada 21; sus sets salen del **mismo rango**
  que los rivales normales de la ronda, sin especies ni objetos repetidos; **excluye sets, no especies**: los de los
  6 Pokémon de la pantalla de cambio (nuestro equipo tras el cambio y el equipo derrotado; el que regalamos vuelve a
  ser posible); todos sus IVs = `GetFactoryMonFixedIV(reto + 2)`: **15 en la ronda 3, 31 desde la 6** (los rivales
  normales tienen 3 / 6). No da pista.
- **Noland simulado** (`nol=mc`): solo en el cambio justo antes de Noland, p_7 de cada opción = victorias en 32
  combates simulados (combatiente = la red del checkpoint, voraz) contra equipos de Noland sorteados del modelo de
  experiencia con sus reglas (exclusión de las configuraciones de los 6 Pokémon en pantalla, IVs 15 / 31, EVs de
  Pokémon entrenado: 252 en dos estadísticas, naturaleza del modelo), mezcladas con W: p = (8·W_Noland + victorias) /
  (8 + 32). Mismos equipos y semillas para todas las opciones.
- **Números aleatorios comunes (NAC)**: los mismos rivales sorteados (y semillas) para todas las opciones de una
  decisión; con la misma semilla, el valor de una opción no depende de qué otras opciones se evalúen (probado).

#### 27.x.2 Lo que estaba mal o flojo en el planificador antiguo (punto (d))

1. **Regla de información**: el planificador antiguo sortea los rivales del combate k (y los Pokémon que podríamos
   coger) de la **lista de sets de la Factory** (`DET.pool`, `SET_*`, `seen_set_mon`) **también en evaluación**, sin
   pasar por `check_prior`. Eso afecta al planificador como táctico y a los rasgos Q_plan de las redes de v3 / v4 en
   todas sus evaluaciones (§20.6, §22, §25). `planner2` usa por defecto el modelo de experiencia; con
   `p=fs` reproduce el antiguo **exactamente** (mismas llamadas al generador; probado) y exige la marca de diagnóstico.
   Medido: quitar la fuga no cuesta nada (E8 − antiguo, abajo: +0,1 / −0,5 puntos de P(6), dentro del ruido).
2. **El valor de la pista futura se ignoraba** (arreglado con `h=1`, ver arriba).
3. **Noland**: el planificador antiguo **nunca sortea a Noland**: solo sortea rivales para los combates k < 7 (tras el
   combate 7 la ronda acaba y no hay cambio); Noland entra solo por la cabeza Noland de W, que se entrenó con combates
   contra equipos generados con sus reglas exactas (`team_eval_data`: IVs de su fila, exclusión de sets). Así que la
   premisa «sortea un entrenador normal en lugar de Noland» no se cumple. Comprobación: en 16 cambios previos a
   Noland, la media de W_Noland sobre las opciones es 0,71 y la de 64 combates simulados contra Noland con los sets
   reales (diagnóstico) 0,69: **W_Noland está calibrada**. Con el modelo de experiencia, los Noland simulados son más
   débiles: 0,85 con EVs al azar (como el muestreador de la búsqueda) y 0,75 con EVs de Pokémon entrenado (los EVs y
   naturalezas de los sets no los aprende el modelo).
4. Menores: el sesgo del máximo en el nivel 1 (el máximo de 10 valores de W con ruido) favorece un poco a los equipos
   con más opciones de cambio; `round_of` se fía del contador de combates de la racha (`run_ctx`), que vale 0 en una
   racha que el simulador empieza a mitad de ronda (solo pasa en pruebas: las sondas empiezan en el primer combate).

#### 27.x.3 Coste y ruido (`python -m rl.planner2_noise`, 40 alquileres + 60 cambios de C1, 4 semillas, 1 hilo, CPU con la máquina cargada)

| variante | ms alquiler | ms cambio | desv. Q centrado (alq. / cambio) | acuerdo entre semillas | regret_ref (victorias) |
|---|---|---|---|---|---|
| antiguo (factory_sets, 8) | 68 | 24 | 0,081 / 0,062 | 0,84 / 0,97 | 0,003 / 0,000 |
| E8 (experience, 8) | 68 | 25 | 0,082 / 0,058 | 0,83 / 0,98 | 0,004 / 0,000 |
| n=16 | 131 | 45 | 0,059 / 0,042 | 0,82 / 0,98 | 0,004 / 0,000 |
| n=32 | 269 | 85 | 0,039 / 0,032 | 0,92 / 0,99 | 0,002 / 0,000 |
| n=64 | 576 | 172 | 0,029 / 0,021 | 0,96 / 0,99 | 0,000 / 0,000 |
| n=32, h=1 | 662 | 187 | 0,043 / 0,035 | 0,88 / 0,99 | 0,002 / 0,000 |
| n=64, h=1 | 1.289 | 394 | 0,031 / 0,025 | 0,97 / 0,99 | 0,000 / 0,000 |
| d=2 (8 × 4) | 626 | 156 | 0,067 / 0,046 | 0,90 / 0,99 | (0,004 / 0,000)* |
| n=16, h=1, d=2 | 1.417 | 380 | 0,054 / 0,044 | 0,89 / 0,98 | (0,005 / 0,002)* |

\* respecto a la referencia de profundidad 1. `nol=mc`: +0,4–1,2 s (GPU) solo en el cambio previo a Noland (como
mucho 2 por racha). El ruido baja como 1/√n (0,082 → 0,029 con 64 rivales) y el acuerdo sube de 0,83 a 0,96 en los
alquileres, pero **el ruido casi no cuesta nada**: con 8 rivales el regret_ref es 0,004 victorias por alquiler y
0 por cambio (las opciones entre las que duda valen casi lo mismo). En el juego cruzado: tiempo total del táctico
en 3.072 rachas: antiguo / E8 ~610–650 s, n=32 h=1 2.500–4.500 s, n=16 h=1 d=2 9.600 s.

#### 27.x.4 C1 y C2 (banco de §25; combatiente fijo = red de v3; W, v_next y modelo de v3 final)

C1: acuerdo con el máximo del planificador antiguo (misma semilla) y Regret_Q bajo él. C2: P(completar la ronda)
de la opción elegida (32 semillas emparejadas), diferencia emparejada con la elección del planificador antiguo y con
la de la red de v3, y regret validado (puntos de P(completar), ± IC 95 %):

| variante | acuerdo alq. / cambio | C2 valor | Δ vs antiguo | Δ vs red v3 | regret validado |
|---|---|---|---|---|---|
| antiguo | – | 61,3 | – | −0,0 ± 1,1 | +0,7 ± 2,1 |
| E8 | 0,90 / 0,99 | 61,1 | −0,1 ± 0,7 | −0,2 ± 1,2 | +0,8 ± 2,1 |
| n=32 | 0,90 / 0,99 | 61,5 | +0,2 ± 0,7 | +0,1 ± 1,1 | +0,5 ± 2,1 |
| n=32, h=1 | 0,89 / 0,98 | 61,4 | +0,1 ± 0,7 | +0,1 ± 1,1 | +0,6 ± 2,1 |
| n=64, h=1 | 0,91 / 0,99 | 61,3 | +0,0 ± 0,8 | −0,0 ± 1,1 | +0,7 ± 2,1 |
| n=16, h=1, d=2 | 0,86 / 0,98 | 61,3 | +0,0 ± 0,8 | −0,0 ± 1,2 | +0,7 ± 2,1 |
| n=32, h=1, nol=mc | 0,89 / 0,95 | 61,4 | +0,1 ± 1,2 | +0,1 ± 1,5 | +0,6 ± 2,3 |
| p=fs, n=32, h=1 (diagnóstico) | 0,92 / 0,98 | 61,5 | +0,2 ± 0,7 | +0,1 ± 1,1 | +0,5 ± 2,1 |

En los 25 cambios previos a Noland de C2, `nol=mc` elige distinto en 11 y la diferencia es 0,0 ± 5,9 puntos.
**Todas las variantes eligen casi lo mismo** (86–92 % de los alquileres, 95–99 % de los cambios) y por el resultado
real no se distinguen del antiguo ni de la red.

#### 27.x.5 Juego cruzado A (512 rachas por ronda, semilla 777, combatiente = red del mismo checkpoint)

P(completar) por ronda (%) y P(6) con IC *bootstrap*:

| táctico | combatiente | R1 | R2 | R3 | R4 | R5 | R6 | P(6) |
|---|---|---|---|---|---|---|---|---|
| red v3 | v3 | 85,4 | 85,9 | 65,2 | 79,9 | 43,0 | 43,8 | 7,19 [6,0, 8,4] |
| antiguo v3 | v3 | 88,3 | 86,7 | 63,7 | 79,9 | 41,8 | 40,2 | 6,55 [5,5, 7,7] |
| E8 v3 | v3 | 86,9 | 87,1 | 63,5 | 81,4 | 42,2 | 40,4 | 6,68 |
| n=32 h=1 v3 | v3 | 87,5 | 87,1 | 62,3 | 80,5 | 42,4 | 39,1 | 6,33 |
| n=16 h=1 d=2 v3 | v3 | 87,7 | 88,5 | 63,7 | 81,1 | 43,8 | 40,8 | 7,15 [6,0, 8,4] |
| p=fs n=32 h=1 v3 (diag.) | v3 | 88,5 | 87,1 | 63,7 | 80,7 | 43,6 | 39,5 | 6,80 |
| red v4 (it. 1111) | v4 | 86,5 | 84,0 | 62,3 | 73,4 | 37,7 | 38,3 | 4,80 [4,0, 5,7] |
| antiguo v4 | v4 | 82,8 | 82,6 | 64,6 | 78,7 | 42,0 | 35,2 | 5,14 [4,2, 6,1] |
| E8 v4 | v4 | 82,0 | 82,0 | 62,5 | 78,9 | 40,4 | 35,0 | 4,69 |
| n=32 h=1 v4 | v4 | 84,2 | 82,4 | 64,8 | 78,3 | 42,6 | 34,0 | 5,10 |
| p=fs n=32 h=1 v4 (diag.) | v4 | 84,2 | 81,8 | 63,7 | 78,3 | 42,6 | 36,1 | 5,29 |

Diferencias emparejadas (la racha i de la ronda k empieza igual para los dos tácticos): media de las 6 rondas de
la diferencia de P(completar) (puntos) y diferencia de P(6) (puntos, IC *bootstrap* emparejado). `nol=mc` solo
cambia el cambio previo a Noland, así que se jugó solo en las rondas 3 y 6 (el resto de rachas son idénticas a
n=32 h=1); su «P(6)» es P(R3)·P(R6).

| comparación | combatiente | media por ronda | Δ P(6) |
|---|---|---|---|
| E8 − antiguo (quitar la fuga de información) | v3 | +0,2 ± 0,9 | +0,13 [−0,49, +0,77] |
| E8 − antiguo | v4 | −0,8 ± 0,9 | −0,45 [−0,94, +0,01] |
| n=32 h=1 − antiguo | v3 | −0,3 ± 0,9 | −0,22 [−0,82, +0,36] |
| n=32 h=1 − antiguo | v4 | +0,1 ± 0,9 | −0,04 [−0,54, +0,44] |
| n=32 h=1 − E8 | v3 / v4 | −0,5 ± 0,9 / +0,9 ± 0,9 | −0,35 [−0,97, +0,26] / +0,41 [−0,05, +0,88] |
| n=16 h=1 d=2 − antiguo | v3 | +0,8 ± 1,0 | +0,60 [−0,12, +1,36] |
| n=16 h=1 d=2 − n=32 h=1 | v3 | +1,1 ± 0,9 | +0,82 [+0,18, +1,52] |
| nol=mc − n=32 h=1 (R3, R6) | v3 / v4 | −1,5 ± 1,7 / −0,8 ± 1,8 | P(R3)·P(R6): −1,6 [−3,4, +0,1] / −0,6 [−2,2, +1,2] |
| p=fs − n=32 h=1 (techo con la lista de sets) | v3 / v4 | +0,7 ± 0,7 / +0,1 ± 0,7 | +0,48 [−0,02, +0,98] / +0,19 [−0,26, +0,64] |
| n=32 h=1 − red del táctico | v3 / v4 | −0,7 ± 1,8 / +0,7 ± 1,7 | −0,86 [−2,13, +0,42] / +0,30 [−0,65, +1,25] |

#### 27.x.6 Lectura y recomendación para v5

- **Ninguna mejora del planificador se nota como táctico.** Más rivales reducen el ruido como se espera
  (1/√n), pero ese ruido ya costaba ~0 (regret_ref 0,004 victorias); la pista futura y la profundidad 2 cambian la
  elección en ~10 % de los alquileres y ~1–2 % de los cambios, y ni C2 (32 semillas por opción) ni A (512 rachas)
  distinguen el resultado. La profundidad 2 es la única con un signo consistente en A con v3 (+0,6 puntos de P(6),
  IC [−0,1, +1,4]; +0,8 [+0,2, +1,5] frente a n=32 h=1, pero entre ~30 comparaciones una con IC que no toca el 0 es
  esperable por azar, y en C2 da 0,0 ± 0,8): no hay evidencia suficiente para pagar 20× de coste.
- **Noland simulado empeora o no cambia** (−1,6 / −0,6 puntos de P(R3)·P(R6)): el modelo de experiencia no conoce
  los EVs ni las naturalezas de los sets y su Noland simulado es demasiado débil (0,75 frente a 0,69 de los sets
  reales); W_Noland ya está calibrada. Se descarta.
- **El techo con la lista de sets real** (p=fs) tampoco se separa (+0,5 / +0,2 puntos de P(6)): el muestreador no
  es el cuello de botella.
- Lo que sí hay que hacer: **quitar la fuga de información** (que el planificador y los rasgos Q_plan de la red usen
  el modelo de experiencia, `planner2` E8 o n=16): coste igual y sin pérdida medible.
- Conclusión para v5: con este W como hoja, el planificador ya está en su techo como táctico; mejorarlo (más
  rivales, más profundidad, Noland) no mueve P(6). El táctico solo puede mejorar si mejora **W** (o se sustituye la
  hoja por algo que dependa del rival concreto) o si mejora el **combatiente** (que es lo que fija los valores de
  W). Recomendación: en v5, planificador `planner2` con el modelo de experiencia y 8–16 rivales (h=1 opcional, coste
  ~2,5×), sin profundidad 2 ni Noland simulado, y poner el esfuerzo en el combatiente (§26 punto 1–2).

#### 27.x.7 Uso y pruebas

```bash
python -m rl.agent_probe tactician --tactician "planner2:CKPT:n=32,h=1"          # C1 (opciones: n, d, n2, b, h, p=exp|fs, nol=w|mc, ns, nw)
python -m rl.agent_probe outcomes  --tactician "planner2:CKPT:n=32,h=1"          # C2
python -m rl.agent_probe crossplay --tactician "planner2:CKPT:n=32,h=1" --battler net:CKPT --a-runs 512
python -m rl.planner2_noise                                                       # coste y ruido
python -m rl.planner2_report                                                      # tablas (runs/probe/planner2/report.json)
python -m pytest tests/python/test_planner2.py
```
`p=fs` necesita `PYBATTLE_DIAGNOSTIC_PRIOR=factory_sets`. Scripts de esta medida: `runs/probe/planner2/run_c.sh`,
`run_a.sh`. Pruebas (15): calendario de Noland y sus IVs contra el juego (rondas 3 y 6; se lee el equipo real solo
para comprobar la regla), exclusión de sets de la pantalla de cambio (el regalado vuelve a ser posible), los sorteos
de Noland (sin especies / objetos repetidos, sin las configuraciones excluidas, su especie sí permitida; con la lista
de sets, sin los ids excluidos), `factory_sets` exige la marca de diagnóstico, opciones (lectura / etiqueta), `p=fs`
= planificador antiguo exacto, determinismo con la misma semilla, NAC (el valor de una opción no depende de las
demás), profundidad 0 = Φ, profundidad 2 ≥ profundidad 1, `h=1` comprobado a mano con bucles y sin efecto con una W
sin pista, el Noland simulado (determinista, IVs 15 en el combate reconstruido, mezcla con W) y la especificación
del probe.

### 27.y Rasgos de amenaza (`rl/threat_features.py`, 2026-10-04)

Objetivo (§26, punto 1 del combatiente): dar a la red del combatiente, como **entrada**, lo que el modelo del rival
aprendido de la experiencia (§22) cree sobre los Pokémon rivales, y medir sin conexión (con el conjunto de datos de
§27) si así juega mejor. Motivo: en el banco B (§25) la búsqueda **perfecta** gana +4,5–5 puntos por combate a la red
(conoce los sets del rival; el azar se vuelve a sortear, así que esa distancia es solo conocimiento de los sets),
mientras que la búsqueda legal con el modelo aprendido no añade nada. Si la red recibe directamente "qué puede hacerme
este rival", quizá aproveche parte de esa distancia sin búsqueda.

#### 27.y.1 Definiciones (de abajo arriba)

- **Configuración** de un rival (§22.1): (especie, sus 4 movimientos, objeto).
- **Posterior** de un hueco rival: la probabilidad que el modelo de experiencia da a cada configuración posible de ese
  hueco **dado lo visto** en el combate: especie (si ya salió), movimientos usados, objeto revelado, objetos revelados de
  sus compañeros (no se repiten) y la ronda. Para un hueco **no visto** la especie también es incierta (las especies de
  la ronda menos las ya vistas en el equipo).
- **Átomo**: cada una de las opciones de esa mezcla. Hay dos clases:
  - **conocido**: una configuración que el jugador ha visto entera alguna vez, con su probabilidad;
  - **nuevo** (uno por especie): "una configuración que nunca he visto"; sus movimientos son los revelados más un
    sorteo sin repetición entre los que la especie puede aprender (la regla estricta de §15). Sus rasgos se calculan
    **exactamente** con fórmulas de combinatoria (hipergeométrica: probabilidad de que un sorteo de k movimientos entre
    n contenga alguno de q dados = 1 − C(n−q, k)/C(n, k); esperanza del máximo de un sorteo), sin muestrear.
- **Estadísticas típicas**: las del rival no se ven; se usa IV 15,5 y EV 85 (510/6) en cada estadística, y la
  naturaleza de las observadas para esa configuración (uniforme para un átomo nuevo). Las nuestras son exactas.
- **Fracción de daño**: daño de un movimiento (fórmula de Gen 3 con la tirada media 0,925, STAB, tabla de tipos,
  habilidades y objetos; sin críticos, clima, pantallas ni cambios de estadísticas, que la red ya ve) dividido entre los
  PS máximos del que lo recibe. Habilidades: las nuestras exactas; la del rival, la revelada o la media entre las
  posibles de su especie (Potencia, Levitación…). Objetos: los nuestros; el del átomo (Cinta Elegida, potenciadores de
  tipo, Hueso Grueso, Bola Luminosa); un átomo nuevo, sin objeto.
- **Rasgos de un par** (rival j, nuestro i), 6 números, todos promediados sobre la posterior de j:
  1. `p_se`: P(j tiene un ataque súper eficaz contra i);
  2. `dmg_in`: esperanza del máximo, entre los movimientos de j, de la fracción de daño a i (tope 1,5);
  3. `ko_in`: P(ese máximo ≥ la fracción de PS que le queda a i): "me puede tumbar de un golpe ya";
  4. `faster`: P(j es más rápido que i) (+ la mitad de los empates), con la distribución de su velocidad sobre IVs 0–31
     y EVs 0–252 y su naturaleza; incluye los cambios de velocidad y la parálisis de los dos activos;
  5. `dmg_out`: esperanza del mejor daño de i a j (estadísticas típicas de j, su especie incierta si no se ha visto);
  6. `ko_out`: P(ese daño ≥ la fracción de PS de j que muestra la barra; no visto: lleno).
- **Rasgos de un hueco rival**: P(tiene algún movimiento de estado: dormir, envenenar, paralizar, quemar, confundir…),
  P(de mejora: Danza Espada, Paz Mental…), P(de recuperación), P(ataque con prioridad), la **entropía** de su posterior
  (cuánta incertidumbre queda, en nats/8) y la masa de los átomos nuevos (`p_novel`: cuánto cree el modelo que es algo
  nunca visto).
- **Disposición en la entrada** (18 números por cada token de Pokémon, N_THREAT = 18):
  - nuestro Pokémon i: [par (rival activo, i), máximo elemento a elemento de los pares (otro rival vivo, i), 6 ceros];
  - rival j (también los no vistos): [par (j, nuestro activo), máximo de los pares (j, otro nuestro vivo), sus 6 rasgos
    de hueco];
  - debilitados: ceros.
- **ThreatNet**: la FactoryNet con esos 18 números añadidos a la entrada del codificador de Pokémon del combatiente: su
  primera capa se ensancha con columnas a cero, así que cargada desde una FactoryNet calcula exactamente lo mismo hasta
  que se entrena (test). El táctico no cambia.

#### 27.y.2 Regla de información

Las entradas son solo la `BattleView` (lo que ve el jugador), el número de ronda y el `ExperienceModel` (cuentas de lo
que el jugador vio). Pruebas (`tests/python/test_threat_features.py`, 10 tests, pasan): (a) el código del módulo, sin
comentarios, no nombra nada del estado verdadero ni de la lista de sets; (b) a lo largo de 4 combates aleatorios, en
cada decisión con un hueco rival sin ver se reescribe en una copia lo que el jugador no ve (ids de sets y los Pokémon
no vistos de `gEnemyParty`): las vistas son iguales y los rasgos, calculados por dos instancias con cachés frías,
idénticos bit a bit; (c) la misma vista da lo mismo con caché fría y caliente; además, las fórmulas cerradas contra
enumeración, la tabla de velocidades contra la fórmula, rangos y disposición, un movimiento súper eficaz revelado da
`p_se` = 1, un modelo vacío da `p_novel` = 1, y la red (carga desde FactoryNet, ceros = sin rasgos, tronco compartido).

#### 27.y.3 Coste

Todo lo que no cambia dentro de un combate (posteriores, tablas de daño por par) se guarda en cachés indexadas por lo
que sabe el jugador; cada decisión solo recalcula lo que depende de los PS y la velocidad. En las posiciones del
conjunto de datos (1 hilo, máquina cargada): **2,3 ms de media por decisión** (mediana 1,5; p90 7; p99 11; la primera
decisión de un combate y cada vez que se revela algo nuevo es la cara), **0,33 ms** si ya está todo en caché. Jugando
el banco B con las redes entrenadas: 0,87–0,90 ms por decisión (el combate entero cuesta 2× el de la red sola).
Con un modelo vacío (todo "nuevo") ~3 ms. Para comparar: la búsqueda de 256
simulaciones cuesta ~200 ms por decisión, así que en la **raíz** el coste es despreciable; **en cada hoja** de la
búsqueda (256 por decisión) sumaría 85–600 ms, demasiado: si se usa en la búsqueda, conviene fijar las posteriores con
lo que se sabe en la raíz (es lo que sabe el jugador al decidir) y recalcular en las hojas solo la parte dinámica
(~0,3 ms, o menos vectorizando por lotes). No implementado aún: v5 tendría que decidirlo.

#### 27.y.4 ¿Predicen al rival verdadero? (diagnóstico; el rival verdadero solo se lee para puntuar)

250 inicios de B jugados por la red de v3; en ~30 % de las decisiones, cada rival vivo frente a cada nuestro.
**Brier** = media de (probabilidad − verdad)² (0 = perfecto; "constante" = predecir siempre la frecuencia media):

| | experiencia | modelo vacío (estricto) | constante |
|---|---|---|---|
| `p_se` del rival activo (visto), n 2.146 | **0,036** | 0,117 | 0,222 |
| P(estado), rivales vistos | **0,044** | 0,124 | 0,221 |
| P(mejora), vistos | **0,036** | 0,094 | 0,217 |
| P(recuperación), vistos | **0,027** | 0,120 | 0,213 |
| P(estado / mejora / recuperación), **no vistos** | 0,216 / 0,189 / 0,166 | 0,215 / 0,193 / 0,177 | 0,214 / 0,189 / 0,165 |

Para un rival que ya ha salido, el modelo sabe casi qué lleva (Brier 3–8 veces menor que el estricto); para uno que no
ha salido no sabe más que la frecuencia media (la especie no se conoce).

#### 27.y.5 Medida sin conexión (conjunto de datos de §27, banco B)

Definiciones:
- **Control** y **+amenaza**: la misma receta de `rl.offline_train` (§27: red de v4, 2 capas, d 128, desde cero, 10
  épocas, lr 3·10⁻⁴ con calentamiento y coseno, lote 512, objetivos de la búsqueda Gumbel de v4), cambiando solo la
  entrada: `--threat runs/v5_offline/threat` usa `ThreatNet` con los rasgos de cada fila. 654.339 parámetros frente a
  652.035 (+2.304, la capa ensanchada). El control con semilla 0 es el `control` de §27 (mismos argumentos).
- **Semilla de entrenamiento**: la inicialización y el orden de los lotes. Se entrenan 3 por brazo (0, 1, 2) y en B se
  promedia, por estado, sobre las 3 redes de cada brazo antes de comparar (así la diferencia no depende de una
  inicialización con suerte).
- **Banco B** (§25): 1.779 inicios de combate × 4 semillas del combate, red voraz; mi ejecutor
  (`python -m rl.threat_features bench`) reproduce exactamente los resultados de `rl.agent_probe battler` con la misma
  red (150 de 150 combates idénticos con la red de v3). **Diferencia emparejada** por estado, IC 95 % *bootstrap* sobre
  estados (2.000 remuestreos).

Rasgos del conjunto de datos: 3.335.841 posiciones, calculadas mientras se grababa (`compute-shards`, 3 procesos,
~1.050 posiciones/s en total incluida la repetición), unidas por fila con `assemble` (comprobado contra `run` / `li`).
Modelo de experiencia: el de `runs/alphazero_v4/stopped_it1111.pt` (el del checkpoint que grabó los datos).

**Conjunto completo (3,16 M filas de entrenamiento, 172.001 apartadas)**, final de la época 10, media de 3 semillas:

| | KL apartada | top-1 | EV objetivo | EV z | Brier | KL Noland |
|---|---|---|---|---|---|---|
| control | 0,7172 | 0,580 | 0,446 | 0,244 | 0,0843 | 0,726 |
| +amenaza | **0,7076** | 0,585 | 0,454 | 0,252 | 0,0834 | 0,717 |

(EV = varianza explicada del valor; z = resultado real del combate.) La KL baja 0,010 en las 3 semillas (control
0,7164 / 0,7177 / 0,7176; +amenaza 0,7070 / 0,7068 / 0,709).

Banco B (victorias por combate, %; Δ emparejada +amenaza − control, puntos):

| | total | R5 | R6 | R3 | Noland |
|---|---|---|---|---|---|
| control (3 semillas) | 88,8 [87,9, 89,7] | 87,5 | 89,4 | 96,2 | 78,6 |
| +amenaza (3 semillas) | 89,0 [88,1, 89,8] | 87,9 | 89,1 | 96,6 | 79,7 |
| **Δ** | **+0,21 [−0,29, +0,70]** | +0,40 [−0,40, +1,26] | −0,35 [−1,17, +0,47] | +0,44 [−0,28, +1,22] | +1,12 [−1,07, +3,17] |
| Δ por semilla (0 / 1 / 2) | +0,49 / +0,25 / −0,13 | | | | |

Ruido entre semillas del control: s1 − s0 = +0,46 [−0,39, +1,29], s2 − s0 = +0,32. Referencias en el mismo banco
(§25.6): v3 final 89,0, v4 it. 899 87,5.

**Piloto con pocos datos (las primeras 600 k posiciones: 569 k de entrenamiento, 4 épocas, 3 semillas):**

| variante | KL apartada | B total | Δ vs control | Noland Δ |
|---|---|---|---|---|
| control | 0,834 | 83,3 | – | – |
| +amenaza (los 18) | 0,808 | 85,5 | **+2,17 [+1,47, +2,83]** | +3,2 [+0,6, +5,5] |
| solo pares (sin el bloque de hueco), 2 semillas | 0,812 | 85,0 | +1,59 [+0,82, +2,32] | +2,2 |
| solo bloque de hueco, 2 semillas | 0,835 | 84,0 | +0,52 [−0,05, +1,12] | +1,1 |
| rasgos con un modelo **vacío** (solo la regla estricta: daño y velocidad sin saber nada de los sets), 2 semillas | 0,826 | 84,8 | +1,34 [+0,60, +2,06] | +1,2 |

+amenaza − modelo vacío (2 semillas): +0,94 [+0,28, +1,59]: con pocos datos, la mitad de la ganancia es el cálculo de
daño / velocidad y la otra mitad lo que el modelo de experiencia sabe de los sets.

#### 27.y.6 Lectura y recomendación

- Los rasgos son **correctos y baratos**: respetan la regla de información (probado), predicen bien lo oculto de un
  rival ya visto (Brier 3–8× menor que la regla estricta) y cuestan ~1 ms por decisión en la raíz.
- **Con pocos datos ayudan mucho** (+2,2 puntos por combate con 570 k posiciones, sobre todo Noland); **con el
  conjunto completo casi nada**: +0,2 puntos [−0,3, +0,7] en B, y solo −0,010 de KL y +0,008 de EV. La red aprende
  sola, de 3 M posiciones, casi todo lo que estos rasgos le dan masticado (los objetivos de la búsqueda de v4 ya se
  calcularon con `factory_sets`, así que llevan implícito el conocimiento de los sets).
- No cierran la distancia con la búsqueda perfecta (+4,5–5 puntos): esa distancia es saber el set **concreto** de
  este rival, y para un rival visto la posterior ya es casi segura (Brier 0,04) y para uno no visto no sabe más que
  la media. Lo que queda está en lo que no se puede saber antes de verlo, no en creencias mal usadas.
- **Recomendación**: no hacerlos parte obligatoria de v5 por su efecto en B (no medible con el conjunto completo).
  Sí son un candidato razonable si v5 arranca **desde cero** (las primeras iteraciones tienen pocos datos, que es
  justo donde ayudan, y la red empieza igual que sin ellos gracias a las columnas a cero), con dos condiciones: (1)
  calcularlos en la raíz y, si la búsqueda los necesita en las hojas, reutilizar las posteriores de la raíz y
  recalcular solo la parte de PS / velocidad (~0,3 ms; sin eso serían +85–600 ms por decisión con 256 simulaciones);
  (2) usar el modelo de experiencia del momento (el que también usaría la búsqueda, §26 punto 2). No se probó con una
  red mayor: la de 4 capas de §27 no mejoró al control (KL 0,716).

#### 27.y.7 Uso

```bash
python -m rl.threat_features compute-shards --watch --procs 3     # mientras se graba runs/v5_offline/main
python -m rl.threat_features assemble                             # -> runs/v5_offline/threat/threat.npy (float16 [N, 6, 18])
python -m rl.threat_features compute --data DIR --out OUT --model MODELO   # de una vez, sobre un conjunto ya consolidado
python -m rl.offline_train --name thr_threat_s0 --threat runs/v5_offline/threat --epochs 10
python -m rl.threat_features bench --ckpt runs/v5_offline/models/thr_threat_s0/final.pt --name main_threat_s0
python -m rl.threat_features table --names "main_threat_s0+main_threat_s1+main_threat_s2" \
    --ref "main_ctrl_s0+main_ctrl_s1+main_ctrl_s2"                # "a+b": media por estado de varias redes
python -m pytest tests/python/test_threat_features.py
```
Resultados de B en `runs/v5_offline/threat/bench/`; modelos `runs/v5_offline/models/thr_*` (piloto: `thr_pilot_*`,
datos del piloto en `runs/v5_offline/threat/pilot_data`). `rl.offline_train` solo ganó la opción `--threat` (las
filas llevan `x["threat"]` y la red es `ThreatNet`); sin ella no cambia nada.
