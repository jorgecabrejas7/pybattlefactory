"""The Spanish markdown report of rl/playstats.py (docs/RL_DECISIONS.md §23): every term is defined before it is
used, then the numbers. render(R, title, plots) -> markdown."""

from . import playknow as K
from .playstats import ROUND_ORDER, _ci, _diff, _round_table

ROLE_ES = K.ROLE_NAMES_ES
ARCH_ES = {"fast_frail": "rápido y frágil", "mixed": "mixto", "bulky_slow": "lento y robusto"}
COV_ES = {
    "resisted": "tipos resistidos (de 17)", "unresisted": "tipos que nadie resiste", "shared_weak": "debilidades compartidas",
    "resisted_phys": "resistidos, físicos (de 9)", "resisted_spec": "resistidos, especiales (de 8)",
    "unresisted_phys": "nadie resiste, físicos", "unresisted_spec": "nadie resiste, especiales",
    "shared_weak_phys": "deb. compartidas, físicos", "shared_weak_spec": "deb. compartidas, especiales",
    "se": "tipos golpeados x2 (de 17)", "se_phys": "golpeados x2 con físicos", "se_spec": "golpeados x2 con especiales",
    "phys_bulk": "aguante físico (PS×Def base, miles)", "spec_bulk": "aguante especial (PS×DefEsp base, miles)",
}
BAL_ES = {"phys_share": "fracción física", "fit": "encaje", "misfits": "desajustados", "base_speed": "vel. base media",
          "max_base_speed": "vel. base máx.", "min_base_speed": "vel. base mín.", "base_offense": "ataque base medio",
          "base_hp": "PS base medios", "status_moves": "movs. de estado"}
STATUS_ES = {"toxic": "Tóxico", "poison": "envenenar (Polvo Veneno...)", "paralysis": "parálisis (Onda Trueno...)",
             "sleep": "sueño (Hipnosis, Espora...)", "yawn": "Bostezo", "confusion": "confusión (Rayo Confuso, Contoneo...)",
             "burn": "Fuego Fatuo", "leech_seed": "Drenadoras"}


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def render(R, title, plots):
    L = []
    w = L.append
    c = R["counts"]
    meta = R.get("meta") or {}
    w(f"# {title}\n")
    w("Informe generado por `rl/playstats.py` (definiciones completas en `docs/RL_DECISIONS.md` §23). Cada término se "
      "define antes de usarse.\n")
    w("## 0. Datos y convenciones\n")
    w(f"- **Agente**: `{meta.get('ckpt', '?')}`; táctico = su red, voraz, con los rasgos por opción de W y del "
      f"planificador; combatiente = **{meta.get('battler', '?')}**"
      + (f" (búsqueda legal, {meta.get('sims')} simulaciones, {meta.get('dets')} determinizaciones, muestreador "
         f"«experience» con `{meta.get('opponent_model')}`)" if meta.get("battler") == "search" else " (la red sola, voraz)")
      + ".")
    w(f"- **Muestra**: {c['runs']} rachas ({c['late_runs']} empezadas en las rondas 5–6, marcadas), {c['battles']} "
      f"combates, {c['rentals']} alquileres, {c['swaps']} cambios, {c['battle_decisions']} decisiones de combate.")
    w("- **Racha**: partida que empieza en la pantalla de alquiler y termina con la primera derrota. **Ronda** r: los "
      "combates 7(r−1)+1 … 7r de la racha. **Posición en la ronda**: 1..7. Las columnas R1..R6 son rondas; **R7+** "
      "las rondas 7 y siguientes; **Noland** los combates contra el Jefe de la Fábrica (combate 21, 42, ...), aparte. "
      "Una decisión del táctico (alquiler o cambio) se asigna al combate que precede.")
    w("- **Intervalo de confianza (IC) al 95 %** (entre corchetes): *bootstrap* por rachas: se rehace la muestra 1.000 "
      "veces dando a cada racha un peso aleatorio (Poisson de media 1; todas sus decisiones juntas, porque las "
      "decisiones de una misma racha no son independientes) y se toman los percentiles 2,5 y 97,5 del resultado.")
    w("- **Diferencia controlada por ronda**: Σ_r w_r (media del grupo A en r − media del grupo B en r), con w_r = "
      "fracción de las observaciones en la ronda r (solo rondas con los dos grupos). Compara situaciones de la misma "
      "ronda. Es una **asociación**, no un efecto causal: el agente elige cuándo hace cada cosa.")
    w("- Todo se mide desde la **vista del jugador** (lo que el juego muestra). La única excepción es la sección final, "
      "marcada, que usa el equipo verdadero del rival.\n")

    # ---- H outcomes first
    H = R["H_outcomes"]
    w("## 1. Resultados (referencia para todo lo demás)\n")
    w("- **Victorias por combate**: fracción de combates ganados. **P(completar la ronda r)**: entre las rachas que "
      "empezaron en la ronda 1 y llegaron a la ronda r, fracción que la completa.\n")
    w(_round_table([("victorias por combate", H["battle_win"]), ("turnos por combate", H["turns"]),
                    ("decisiones nuestras por combate", H["decisions"]),
                    ("PS que quedan al ganar (fracción del total)", H["hp_margin_wins"])]))
    w("")
    pc = H["p_complete_round"]
    w("| ronda | " + " | ".join(k for k in pc) + " |\n|---|" + "---|" * len(pc))
    w("| P(completar | llegar) | " + " | ".join(_ci(pc[k], True) for k in pc) + " |")
    w(f"\nRacha media desde la ronda 1: {_ci(H['streak'], nd=2)} victorias. Por posición en la ronda (sin Noland): "
      + ", ".join(f"{p}: {_ci(H['battle_win_by_pos'][p], True)}" for p in H["battle_win_by_pos"]) + ".\n")
    for cap, p in plots[:1]:
        w(f"![{cap}]({p.split('/')[-2]}/{p.split('/')[-1]})\n")

    # ---- A rental
    A = R["A_rental"]
    w("## 2. Alquiler (sección A)\n")
    w("Definiciones:")
    w("- **Opción de alquiler**: (líder, pareja) de los 6 Pokémon de alquiler; 6 × 10 = 60 opciones, 20 **tríos** "
      "(equipos sin orden). **P(trío)** = suma de las probabilidades de la política sobre sus 3 líderes posibles. "
      "**P(en el equipo)** de un Pokémon = suma de P(trío) de los 10 tríos que lo contienen.")
    w("- **Tasa de elección** de una especie: veces elegida / veces ofrecida. **Probabilidad media**: media de "
      "P(en el equipo) cuando se ofrece. **Seguridad** = media de |2·P(en el equipo) − 1|: 1 si la política siempre "
      "lo tiene claro (sí o no), 0 si duda (P = 0,5).")
    w("- Se listan especies ofrecidas al menos `--min-n` veces.\n")
    sp = A["species"]
    w("**Más elegidos** (tasa de elección, IC 95 %; prob. media; seguridad; líder si elegido):\n")
    w("| especie | ofrecido | tasa | prob. media | seguridad | líder si elegido |\n|---|---|---|---|---|---|")
    for x in sp[:15]:
        w(f"| {x['name']} | {x['offered']} | {_ci(x['pick_rate'], True)} | {x['p_in']:.2f} | {x['certainty']:.2f} | "
          f"{_pct(x['lead_given_picked'])} |")
    w("\n**Menos elegidos**:\n")
    w("| especie | ofrecido | tasa | prob. media | seguridad |\n|---|---|---|---|---|")
    for x in sp[-15:][::-1]:
        w(f"| {x['name']} | {x['offered']} | {_ci(x['pick_rate'], True)} | {x['p_in']:.2f} | {x['certainty']:.2f} |")
    sure = sorted(sp, key=lambda x: -x["certainty"])
    w("\n**Más seguros** (la política casi nunca duda): " + ", ".join(f"{x['name']} ({x['certainty']:.2f})"
                                                                        for x in sure[:10]))
    w("\n**Menos seguros**: " + ", ".join(f"{x['name']} ({x['certainty']:.2f})" for x in sure[-10:][::-1]) + "\n")
    w("**Líder.** Especies elegidas ≥ 20 veces que **nunca** salen de líder: "
      + (", ".join(f"{x['name']} ({x['picked']})" for x in A["never_leads"]) or "ninguna") + ".\n")
    lv = A["lead_vs_hint"]
    w("- **Líder frente a la pista del primer rival** (alquileres con un tipo en la pista; n = "
      f"{lv['n']}). *Resiste* = el tipo de la pista le hace ×0,5 o menos; *x2* = tiene un ataque súper eficaz contra "
      "ese tipo. Se compara el líder con la media de los 3 del equipo:")
    w(f"  - líder resiste: {_ci(lv['lead_resists'], True)}; miembro medio: {_ci(lv['team_member_resists'], True)}; "
      f"diferencia {_ci(lv['lead_minus_team_resists'], True)} puntos.")
    w(f"  - líder con x2: {_ci(lv['lead_se'], True)}; miembro medio {_ci(lv['team_member_se'], True)}; diferencia "
      f"{_ci(lv['lead_minus_team_se'], True)} puntos.")
    w("- **Velocidad base del líder − media del equipo**:\n")
    w(_round_table([("líder − equipo (vel. base)", A["lead_speed_minus_team"]),
                    ("P(trío elegido)", A["mean_p_chosen_trio"]), ("entropía (nats)", A["entropy"])]))
    w("\n**Elegidos frente a ofrecidos.** Para cada alquiler: media de los 3 elegidos − media de los 6 ofrecidos "
      "(estadísticas base; *vel. real* = estadística de Velocidad; *aguante real* = PS máx. × Def (o DefEsp) / 1000; "
      "*movs. de estado* = movimientos sin daño):\n")
    cv = A["chosen_vs_offered"]
    names = {"bst": "suma de base", "base_hp": "PS base", "base_atk": "Atq base", "base_def": "Def base",
             "base_spe": "Vel base", "base_spa": "AtEsp base", "base_spd": "DefEsp base", "speed_stat": "vel. real",
             "phys_bulk_stat": "aguante físico real", "spec_bulk_stat": "aguante especial real",
             "status_moves": "movs. de estado", "attack_moves": "movs. de ataque"}
    w(_round_table([(names[k] + " (elegido − ofrecido)", v["diff"]) for k, v in cv.items()], nd=2))
    w("\n**Objetos** con mayor y menor tasa de elección: "
      + ", ".join(f"{x['name']} {_pct(x['pick_rate'])}" for x in A["items"][:8]) + " … "
      + ", ".join(f"{x['name']} {_pct(x['pick_rate'])}" for x in A["items"][-5:]) + ".\n")
    w("**Papeles.** Clasificador de cada set (el primero que se cumple): **barredor con mejora** = un movimiento que sube "
      "Ataque, At. Especial o Velocidad (Danza Espada, Danza Dragón, Paz Mental, Corpulencia, Maldición, Agilidad, "
      "Tambor, Desarrollo...) y ≥ 2 ataques; **apoyo / estados** = ≥ 2 movimientos sin daño; **barredor rápido** = "
      "Velocidad base ≥ 90; **muro / tanque** = PS + (Def + DefEsp)/2 base ≥ 170 y máx(Atq, AtEsp) base < 100; "
      "**atacante lento / robusto** = el resto. **Lift** = fracción entre los elegidos / fracción entre los ofrecidos.\n")
    w("| papel | elegidos | ofrecidos | lift |\n|---|---|---|---|")
    for ro, d in A["roles"].items():
        w(f"| {ROLE_ES[ro]} | {_pct(d['chosen_share'])} | {_pct(d['offered_share'])} | "
          f"{'–' if d['lift'] is None else f'{d['lift']:.2f}'} |")
    w("\nFracción de cada papel en el equipo elegido, por ronda:\n")
    w(_round_table([(ROLE_ES[ro], d) for ro, d in A["roles_by_round"].items()], pct=True))
    w("\nComposiciones más frecuentes: " + "; ".join(
        "+".join(ROLE_ES[x] for x in rc["roles"]) + f" {_pct(rc['share'])}" for rc in A["role_compositions"][:6]) + ".\n")
    for cap, p in plots[1:2]:
        w(f"![{cap}]({p.split('/')[-2]}/{p.split('/')[-1]})\n")

    # ---- B swaps
    Bs = R["B_swaps"]
    w("## 3. Cambios tras ganar (sección B)\n")
    w("Definiciones: **cambio** = dar uno de los nuestros (sale) por uno del rival derrotado (entra); **quedarse** = no "
      "cambiar. Hay 10 opciones (quedarse + 3×3). **Tasa de salida** de una especie = veces que sale / veces que "
      "estaba en el equipo en una pantalla de cambio; **tasa de entrada** = veces que entra / veces que se ofrece. "
      "**Permanencia** = combates que un set pasa en el equipo (desde que llega por alquiler o cambio hasta que sale o "
      "acaba la racha; *censurada* si acaba la racha con él dentro).\n")
    w(_round_table([("tasa de cambio", Bs["swap_rate"]), ("entropía (nats)", Bs["entropy"])], pct=False))
    w("\n**Más veces sale**: " + ", ".join(f"{x['name']} {x['n']} ({_pct(x['rate'])})" for x in Bs["most_swapped_out"][:10]))
    w("\n**Más veces entra**: " + ", ".join(f"{x['name']} {x['n']} ({_pct(x['rate'])})" for x in Bs["most_swapped_in"][:10]))
    w("\n**Mayor tasa de entrada** (cuando se ofrece): " + ", ".join(f"{x['name']} {_pct(x['rate'])}"
                                                                     for x in Bs["swapped_in"][:10]))
    w("\n**Mayor tasa de salida**: " + ", ".join(f"{x['name']} {_pct(x['rate'])}" for x in Bs["swapped_out"][:10]) + "\n")
    tb = Bs["tenure_by_origin"]
    w(f"Permanencia media: alquilados {tb['rental']['mean_battles']:.2f} combates ({tb['rental']['n']}; censurada "
      f"{_pct(tb['rental']['censored_share'])}), llegados por cambio "
      + (f"{tb['swap']['mean_battles']:.2f} ({tb['swap']['n']}; censurada {_pct(tb['swap']['censored_share'])})"
         if tb["swap"]["mean_battles"] is not None else "–")
      + ". Por papel: " + ", ".join(f"{ROLE_ES[k]} {v:.2f}" for k, v in Bs["tenure_by_role"].items()) + ".")
    w("Permanencia más larga: " + ", ".join(f"{x['name']} {x['mean_battles']:.1f}" for x in Bs["tenure_species"][:8])
      + "; más corta: " + ", ".join(f"{x['name']} {x['mean_battles']:.1f}" for x in Bs["tenure_species"][-8:]) + ".\n")
    w(f"**Tipo de cambio por papel** (sale → entra; el papel del que entra con su set completo): mismo papel "
      f"{_pct(Bs['same_role_share'])}. Más frecuentes: " + ", ".join(
          f"{ROLE_ES[k.split('->')[0]]} → {ROLE_ES[k.split('->')[1]]} {_pct(v)}"
          for k, v in list(Bs["role_transitions"].items())[:8]) + ".\n")
    w("### Cobertura de tipos (B2–B3)\n")
    w("- **Cobertura defensiva**: para cada uno de los 17 tipos de ataque, el multiplicador sobre cada miembro (tabla de "
      "tipos y la habilidad si el jugador la conoce: Levitación, Absorbe Agua...). **Resistido** = algún miembro recibe "
      "×0,5 o menos; **nadie resiste** = 17 − resistidos; **debilidad compartida** = ≥ 2 miembros reciben ×2 o más. "
      "Se separa por la **categoría** del tipo atacante (en la 3.ª generación la categoría la da el tipo: normal, "
      "lucha, volador, veneno, tierra, roca, bicho, fantasma y acero son físicos; los otros 8, especiales).")
    w("- **Cobertura ofensiva**: de los 17 tipos simples, a cuántos golpea algún miembro ×2 con un ataque de su "
      "set (con tipo: no cuentan Sísmico, Furia Dragón, Contraataque...); también solo con ataques físicos o solo "
      "especiales. **Aguante físico / especial**: media de PS × Def (o DefEsp) base, en miles.")
    w("- Las opciones de cambio se ven como en la pantalla: del rival derrotado solo se conocen los movimientos que "
      "usó. **Percentil** de la elección = fracción de las otras opciones a las que supera (empates medio punto; "
      "0,5 = como una opción al azar, 1 = la mejor).\n")
    for kind, lab in (("rental", "Alquiler (20 tríos)"), ("swap", "Cambio (10 opciones)")):
        cov = Bs["coverage"][kind]
        w(f"**{lab}** — elegida / media de las opciones / mejor opción / percentil (todas las rondas):\n")
        w("| medida | elegida | media opciones | mejor opción | percentil |\n|---|---|---|---|---|")
        for k, d in cov.items():
            w(f"| {COV_ES[k]} | {_ci(d['chosen'].get('all'), nd=2)} | {_ci(d['options_mean'].get('all'), nd=2)} | "
              f"{_ci(d['best_option'].get('all'), nd=2)} | {_ci(d['percentile'].get('all'), nd=2)} |")
        w("\nPercentil por ronda:\n")
        w(_round_table([(COV_ES[k], d["percentile"]) for k, d in cov.items()], nd=2))
        w("")
    w("**Antes y después de cada cambio** (después − antes; el que entra con su set completo):\n")
    w(_round_table([(COV_ES[k], d) for k, d in Bs["before_after_swap"].items()], nd=2))
    w("\n### Físico / especial, encaje y velocidad (B4–B5)\n")
    w("- **Fracción física** = ataques físicos / ataques del equipo. **Encaje** de un miembro = fracción de sus ataques "
      "cuya categoría coincide con su mayor estadística de ataque base (Atq ≥ AtEsp: físicos); el del equipo es la "
      "media. **Desajustados** = miembros con encaje < 0,5.")
    w("- **Arquetipo** del equipo (medias de base): **rápido y frágil** = Velocidad ≥ 85 y PS × (Def + DefEsp)/2 "
      "< 7.500; **lento y robusto** = Velocidad < 70 y ese aguante ≥ 7.500; **mixto** = el resto.")
    w("- **Superar en velocidad con seguridad**: nuestra Velocidad real > la máxima posible del rival (cualquier IV, "
      "EV y naturaleza a nivel 100). **Fracción superada** de un miembro = fracción de los rivales vistos en esa ronda "
      "(en estos registros, ponderados por frecuencia) a los que supera con seguridad.\n")
    bal = Bs["balance_speed"]
    for kind in ("rental", "swap"):
        d = bal[kind]
        w(f"**{'Alquiler' if kind == 'rental' else 'Cambio'}** (elegida vs media de opciones, todas las rondas): "
          + "; ".join(f"{BAL_ES.get(k, k)} {_ci(v['chosen'].get('all'), nd=2)} vs {_ci(v['options_mean'].get('all'), nd=2)}"
                      for k, v in d.items() if k != "archetype")
          + ". Arquetipos (elegido / opciones): " + ", ".join(
              f"{ARCH_ES[a]} {_pct(x['chosen'])} / {_pct(x['options'])}" for a, x in d["archetype"].items()) + ".\n")
    st = Bs["speed_tiers"]
    w(_round_table([("fracción superada, elegidos", st["chosen"]), ("fracción superada, ofrecidos", st["offered"]),
                    ("máx. del equipo elegido", st["chosen_max"]), ("máx. de los 6 ofrecidos", st["offered_best3_max"])],
                   nd=2))
    w("\n**Correlaciones** (Spearman, equipos de cada combate) entre velocidad, aguantes, ataque y PS medios: "
      + ", ".join(f"{k} {v:+.2f}" for k, v in Bs["profile_correlations"].items()) + ".\n")
    kv = Bs["keep_vs_better"]
    th = kv["thresholds"]
    w("### Cuándo no cambia aunque otra opción parezca mejor (B6)\n")
    w(f"- **Q_plan** = valor del planificador (victorias esperadas, §20.3); **W(normal)** = probabilidad de ganar un "
      f"combate normal según W. En cada cambio: Δq = mejor Q_plan entre los cambios − Q_plan de quedarse; Δw igual con "
      f"W; Δbase = máx(suma de base del que entra − la del que sale). «El planificador dice cambiar» = Δq > "
      f"{th['q_margin']} victorias; «W dice» = Δw > {th['w_margin']}; «las estadísticas dicen» = Δbase ≥ "
      f"{th['bst_margin']}.\n")
    w(_round_table([("se queda", kv["keep_share"]), ("…aunque el planificador dice cambiar", kv["keep_though_planner"]),
                    ("…aunque W dice cambiar", kv["keep_though_w"]), ("…aunque las estadísticas dicen", kv["keep_though_bst"]),
                    ("cambia aunque quedarse es mejor (planificador)", kv["swap_though_planner_keep"]),
                    ("elige el máximo del planificador", kv["agree_planner_argmax"])], pct=True))
    w(f"\nCuando el planificador dice cambiar (n = {kv['planner_says_swap_n']}): victoria en el combate siguiente al "
      f"quedarse − al cambiar, controlado por ronda: {_diff(kv['planner_says_swap_outcome'], True)} puntos.\n")
    so = Bs["swap_outcome"]
    if so:
        w("### Calidad de los cambios por el resultado (B7)\n")
        w("- **Estados comparables**: misma ronda y mismo tercio de Q_plan(quedarse) dentro de la ronda (lo bueno que "
          "era ya el equipo según el planificador). Se compara cambiar con quedarse dentro de cada estrato.\n")
        w(f"- Victoria en el combate siguiente, cambiar − quedarse: {_diff(so['next_battle'], True)} puntos.")
        w(f"- Completar la ronda, cambiar − quedarse: {_diff(so['round_completed'], True)} puntos.")
        w("- Por ronda (combate siguiente): " + ", ".join(f"R{k} {_ci(v, True)}" for k, v in so["next_battle_by_round"].items()
                                                          if v.get("m") is not None) + ".\n")

    # ---- C synergies
    C = R["C_synergies"]
    w("## 4. Sinergias (sección C)\n")
    w("Definiciones (sobre el set completo de cada miembro; en las opciones de cambio, lo visto del rival):")
    for s in K.SYNERGIES:
        w(f"- **{s}**: {K.SYNERGY_NAMES_ES[s]}.")
    w("- **Línea base** de una decisión = fracción de sus opciones (20 tríos o 10 opciones de cambio) que tienen la "
      "sinergia. **Lift** = presencia en lo elegido / línea base, sobre las decisiones donde alguna opción la tiene "
      "(lift 1 = indiferente; > 1 la busca; < 1 la evita). **Usada** en combate: se eligió el movimiento que la "
      "activa (lluvia: Danza Lluvia; arena: Tormenta Arena o salir con Chorro Arena; Relevo; pantallas; Púas; "
      "Cascabel Cura). **Aprovechada**: lluvia / sol = un ataque de agua o Trueno (fuego, Rayo Solar, Meteorobola) "
      "elegido con ese clima, o Nado Rápido en lluvia; arena / granizo = decisiones con ese clima y el rival sin "
      "inmunidad; Relevo = relevar con alguna mejora o Sustituto; pantallas = alguna decisión con la pantalla puesta; "
      "Púas = el rival saca otro Pokémon con Púas puestas; Cascabel = usado con un miembro con estado.\n")
    for kind in ("rental", "swap"):
        w(f"**{'Alquiler' if kind == 'rental' else 'Cambio'}**:\n")
        w("| sinergia | decisiones donde existe | elegida | línea base | lift | elegida − base |\n|---|---|---|---|---|---|")
        for s, d in C["presence"][kind].items():
            w(f"| {s} | {_pct(d['available_share'])} | {_ci(d['chosen'], True)} | {_ci(d['baseline'], True)} | "
              f"{'–' if d['lift'] is None else f'{d['lift']:.2f}'} | "
              f"{_ci(d['chosen_minus_baseline'], True)} |")
        w("")
    w("**Uso en combate** (combates cuyo equipo tiene la sinergia):\n")
    w("| sinergia | combates | usada | aprovechada |\n|---|---|---|---|")
    for s, d in C["usage"].items():
        w(f"| {s} | {d['battles_with']} | {_ci(d['used'], True)} | {_ci(d['payoff'], True)} |")
    w("")

    # ---- D influence
    Dd = R["D_influence"]
    w("## 5. Influencia de la pista y de los rasgos W / planificador (sección D)\n")
    w("Sondas contrafactuales: en cada decisión del táctico se vuelve a pasar la red (sin jugar nada) con una entrada "
      "cambiada y se anota su elección:")
    w("- **pista → ninguna (entrada)**: la pista de la entrada de la red se cambia por «ninguna» (tipo 18, estilo 0); "
      "**(completa)**: además los rasgos de W y del planificador se recalculan sin pista (con los mismos sorteos de "
      "rivales del planificador, números aleatorios comunes).")
    w("- **pista → otro tipo t**: el tipo de la pista se cambia por t (3 tipos al azar por decisión), entrada o completa.")
    w("- **rasgos a cero**: los 7 rasgos por opción (W, Q_plan, Φ...) puestos a 0.")
    w("- **Cambia** = la opción elegida es otra; *equipo cambia* = el equipo resultante es otro (en el alquiler, otro "
      "trío; el líder puede cambiar sin cambiar el trío). **Δ resistencias** = miembros que resisten el tipo de la "
      "pista en el equipo con la pista − sin ella (para «otro tipo t»: con la pista t − con la real, contando "
      "resistencias a t): positivo = la pista hace añadir resistencias a ese tipo. **Δ x2** igual con miembros que "
      "golpean x2 a ese tipo. **ΔQ_plan** = Q_plan de la elección real − la de la sonda (victorias).\n")
    lab = {"hint_none_input": "pista → ninguna (entrada)", "hint_none_full": "pista → ninguna (completa)",
           "hint_t_input": "pista → otro tipo (entrada)", "hint_t_full": "pista → otro tipo (completa)",
           "feat_zero": "rasgos a cero"}
    for kind in ("rental", "swap"):
        d = Dd.get(kind) or {}
        w(f"**{'Alquiler' if kind == 'rental' else 'Cambio'}**:\n")
        w("| sonda | n | cambia | equipo cambia | Δ resistencias | Δ x2 | ΔQ_plan | ΔQ_plan si cambia |\n"
          "|---|---|---|---|---|---|---|---|")
        for p, x in d.items():
            if p not in lab:
                continue
            w(f"| {lab[p]} | {x['n']} | {_ci(x['changed'], True)} | {_ci(x['team_changed'], True)} | "
              f"{_ci(x['delta_resist'], nd=3)} | {_ci(x['delta_se'], nd=3)} | {_ci(x['delta_qplan'], nd=3)} | "
              f"{_ci(x['delta_qplan_changed'], nd=3)} |")
        w(f"\nElige el máximo de Q_plan: {_ci(d.get('planner_argmax_agreement'), True)} %; con los rasgos a cero: "
          f"{_ci(d.get('planner_argmax_agreement_feat_zero'), True)} %.\n")
        rows = [(lab[p], x["changed_by_round"]) for p, x in d.items() if p in lab]
        if rows:
            w("Fracción que cambia, por ronda:\n")
            w(_round_table(rows, pct=True))
            w("")

    # ---- E status / boosts
    E = R["E_status_boost"]
    w("## 6. Estados y mejoras (sección E)\n")
    w("- **Disponible** = decisión de combate en la que el Pokémon activo tiene ese movimiento utilizable. **Tasa de "
      "uso** = elegido / disponible. **Cuándo**: PS propios y del rival (barra) al usarlo frente a cuando estaba "
      "disponible; *amenaza KO* = algún movimiento ya visto del rival puede dejarnos KO (estimación de `rl/damage.py`). "
      "**Acierta** = en la decisión siguiente el rival muestra el estado nuevo. **Error** = el movimiento no podía "
      "funcionar, por lo que se veía: rival tras Sustituto, con Velo Sagrado, ya con un estado (o ya confuso / "
      "con Drenadoras), inmune por tipo (Onda Trueno contra tierra, Tóxico contra veneno / acero, Fuego Fatuo contra "
      "fuego, Drenadoras contra planta) o por una habilidad conocida.\n")
    w("| estado | disponible | usado | PS propios (uso / disp.) | PS rival (uso / disp.) | amenaza KO (uso / disp.) | "
      "rival ya con estado | acierta | error | motivos |\n|---|---|---|---|---|---|---|---|---|---|")
    for g in K.STATUS_GROUPS:
        x = E.get(g) or {}
        if not x.get("used"):
            w(f"| {STATUS_ES[g]} | {x.get('available', 0)} | 0 | | | | | | | |")
            continue
        w(f"| {STATUS_ES[g]} | {x['available']} | {x['used']} | {_ci(x['when_own_hp'], nd=2)} / "
          f"{_ci(x['avail_own_hp'], nd=2)} | {_ci(x['when_foe_hp'], nd=2)} / {_ci(x['avail_foe_hp'], nd=2)} | "
          f"{_ci(x['when_threat_ko'], nd=2)} / {_ci(x['avail_threat_ko'], nd=2)} | {_ci(x['when_foe_statused'], True)} | "
          f"{_ci(x['landed'], True)} | {_ci(x['error'], True)} | {x['error_reasons']} |")
    w("\nTasa de uso por ronda:\n")
    w(_round_table([(STATUS_ES[g], E[g]["use_rate"]) for g in K.STATUS_GROUPS if (E.get(g) or {}).get("used")],
                   pct=True))
    w("\n**Mejoras** (*ofensivas* suben Atq / AtEsp / Vel; *defensivas* Def / DefEsp / evasión...). *Cae después* = "
      "nuestro Pokémon se debilita ese turno; *tope* = todas las estadísticas que sube ya estaban en +6.\n")
    w("| mejoras | disp. | usadas | PS propios (uso / disp.) | amenaza KO (uso / disp.) | mejoras previas | cae después | "
      "tope |\n|---|---|---|---|---|---|---|---|")
    for name, x in E["boosts"].items():
        w(f"| {name} | {x['available']} | {x['used']} | {_ci(x['when_own_hp'], nd=2)} / {_ci(x['avail_own_hp'], nd=2)} | "
          f"{_ci(x['when_threat_ko'], nd=2)} / {_ci(x['avail_threat_ko'], nd=2)} | {_ci(x['when_own_boost'], nd=2)} | "
          f"{_ci(x['fainted_next'], True)} | {_ci(x['capped_error'], True)} |")
    w("\nPor movimiento (tasa de uso cuando está disponible): " + ", ".join(
        f"{x['move']} {_pct(x['rate'])} ({x['available']})" for x in E["per_move"][:20]) + ".\n")
    w("**Resultado** (victoria del combate con − sin, controlado por ronda; asociación): \n")
    w("| rasgo del combate | fracción de combates | con − sin (puntos) |\n|---|---|---|")
    for k, v in E["payoff"].items():
        w(f"| {k} | {_pct(v['share'])} | {_diff(v, True)} |")
    w("")

    # ---- F moves
    F = R["F_moves"]
    w("## 7. Elección de movimiento (sección F)\n")
    w("- **Eficacia** = multiplicador de tipo del movimiento sobre los tipos del rival activo (habilidad conocida "
      "incluida). **Daño estimado** = el de `rl/damage.py` (lo que calcula un jugador: estadísticas del rival acotadas "
      "por sus base, IV/EV/naturaleza posibles), en fracción de los PS máximos del rival; *máximo daño* = el movimiento "
      "utilizable con mayor estimación media. **KO seguro** = el daño mínimo estimado ≥ la barra de PS del rival. "
      "**Riesgo innecesario** = había otro ataque más preciso con ≥ 85 % del daño estimado del elegido. **Efecto "
      "secundario** = probabilidad de efecto > 0 o crítico alto; se compara con la fracción de esos movimientos entre "
      "los de daño parecido (≥ 85 % del máximo). **STAB** = movimiento del tipo del usuario (×1,5).\n")
    w(_round_table([("elige un movimiento (vs cambiar)", F["move_share"]), ("es un ataque", F["attack_share_of_moves"]),
                    ("eficacia elegida", F["eff_chosen"]), ("mejor eficacia disponible", F["eff_best_available"]),
                    ("elige la mejor eficacia", F["eff_is_best"]),
                    ("ataque resistido con uno neutro disponible", F["resisted_when_neutral_available"]),
                    ("elige el de máximo daño", F["max_damage_chosen"]), ("toma el KO seguro", F["sure_ko_taken"]),
                    ("KO seguro con el más preciso", F["sure_ko_most_accurate"]),
                    ("riesgo de precisión innecesario", F["unneeded_accuracy_risk"]), ("STAB", F["stab_share"]),
                    ("sin STAB pero más eficaz que el STAB", F["nonstab_more_effective"])], nd=3))
    w(f"\nHay KO seguro en {_pct(F['sure_ko_available_share'])} de las decisiones. Efecto secundario entre ataques de "
      f"daño parecido: elegido {_ci(F['secondary_chosen'], True)} frente a {_ci(F['secondary_baseline'], True)} al azar.")
    pr = F["priority_when_foe_low"]
    w(f"Prioridad con el rival en ≤ 25 % de PS (y un ataque con prioridad disponible): {_ci(pr['all'].get('all'), True)}; "
      f"si no es seguro que seamos más rápidos {_ci(pr['not_sure_faster'], True)}; si lo es {_ci(pr['sure_faster'], True)}.")
    w(f"Contacto contra habilidades que castigan el contacto (Elec. Estática, Punto Tóxico, Efecto Espora, Cuerpo "
      f"Llama, Piel Tosca, Gran Encanto, posibles en la especie), elección de contacto con − sin, controlado por ronda: "
      f"{_diff(F['contact_vs_punish'], True)} puntos.")
    for name, lab_ in (("ohko", "Golpes de KO directo (Fisura, Perforador, Guillotina, Frío Polar)"),
                       ("explosion", "Explosión / Autodestrucción")):
        x = F[name]
        w(f"- **{lab_}**: disponible {x['available']}, usado {x['used']}; PS propios {_ci(x['own_hp'], nd=2)}, PS rival "
          f"{_ci(x['foe_hp'], nd=2)}, nuestros vivos {_ci(x['alive'], nd=2)}, turno {_ci(x['turn'], nd=1)}, "
          f"amenaza KO {_ci(x['threat_ko'], nd=2)}.")
    w("\nMovimientos más elegidos: " + ", ".join(f"{m} {n}" for m, n in F["top_moves"][:20]) + ".\n")

    # ---- G switching
    G = R["G_switching"]
    w("## 8. Cambios dentro del combate (sección G)\n")
    w("- **Cambio voluntario** = cambiar de Pokémon pudiendo atacar. **Desventaja** = algún tipo con el que el rival "
      "puede atacar (sus ataques vistos y sus propios tipos, por STAB) nos hace ×2. **PS bajos** = < 1/3. **Rival "
      "mejorado** = suma de sus subidas ≥ 2. **Resistencia** de un Pokémon que entra = fracción de esos tipos que "
      "resiste. **Se queda y cae** = amenazado de KO, con cambio posible, ataca y se debilita ese turno "
      "(sacrificio o mala lectura).\n")
    w(_round_table([("tasa de cambio voluntario", G["switch_rate"])], pct=True))
    w("\n| situación | cambia si sí | cambia si no | diferencia (contr. ronda) |\n|---|---|---|---|")
    for k, d in G["switch_rate_given"].items():
        w(f"| {k} | {_ci(d['yes'], True)} | {_ci(d['no'], True)} | {_diff(d['diff_by_round'], True)} |")
    for k in ("voluntary", "forced"):
        x = G[k]
        w(f"\n**{'Voluntario' if k == 'voluntary' else 'Forzado (tras debilitarse)'}** (n = {x['n']}): resistencia del "
          f"que entra {_ci(x['resist_chosen'], nd=2)} frente a {_ci(x['resist_others'], nd=2)} la alternativa; elige el "
          f"que más resiste {_ci(x['chose_best_resist'], True)}.")
    t = G["threatened"]
    w(f"\nAmenazado de KO (n = {t['n']}): cambia {_ci(t['switched'], True)}; se queda y cae "
      f"{_ci(t['stayed_and_fainted'], True)}.\n")

    # ---- H confidence
    Hc = R["H_confidence"]
    w("## 9. Confianza (sección H)\n")
    w("- **Entropía** de una decisión = −Σ p log p sobre las opciones legales (0 = segura; log n = uniforme). "
      "**Prob. máxima** = la de la opción más probable. Solo decisiones con ≥ 2 opciones.\n")
    rows = []
    for kind in ("rental", "swap", "battle", "forced_switch"):
        rows.append((f"{kind}: entropía", Hc[kind]["entropy"]))
        rows.append((f"{kind}: prob. máxima", Hc[kind]["maxprob"]))
    w(_round_table(rows, nd=3))
    for kind in ("rental", "swap"):
        if "win_by_maxprob_tercile" in Hc[kind]:
            d = Hc[kind]["win_by_maxprob_tercile"]
            w(f"\n{kind}: victoria en el combate siguiente por tercios de prob. máxima (cortes "
              f"{', '.join(f'{x:.2f}' for x in Hc[kind]['maxprob_terciles'])}): "
              + ", ".join(f"{['bajo', 'medio', 'alto'][int(k)]} {_ci(v, True)}" for k, v in d.items()) + ".")
    if "win_by_mean_maxprob_tercile" in Hc["battle"]:
        d = Hc["battle"]["win_by_mean_maxprob_tercile"]
        w("\nCombate: victoria del tercio de prob. máxima media − el resto, controlado por ronda: "
          + ", ".join(f"{['bajo', 'medio', 'alto'][int(k)]} {_diff(v, True)}" for k, v in d.items()) + ".")
    cal = Hc.get("value_calibration") or []
    if cal:
        w("\n**Calibración** del valor del combatiente en su primera decisión (probabilidad de ganar que predice la red) "
          "frente a la fracción ganada:\n")
        w("| valor | n | media predicha | ganados |\n|---|---|---|---|")
        for x in cal:
            w(f"| {x['bin'][0]:.1f}–{x['bin'][1]:.1f} | {x['n']} | {x['pred']:.3f} | {x['won']:.3f} |")
    if "search_agreement" in Hc:
        w("\n**Búsqueda frente a la red**: fracción de decisiones en que la búsqueda elige lo mismo que el máximo de la "
          "red:\n")
        w(_round_table([("coincide", Hc["search_agreement"])], pct=True))
        w("\nPor prob. máxima de la red: " + ", ".join(f"{k}: {_ci(v, True)}" for k, v in
                                                        Hc["search_agreement_by_net_maxprob"].items()) + ".")
    w("\n**Suerte.** *Evento de mala suerte* = crítico del rival, fallo nuestro, parálisis total nuestra, congelación, "
      "sueño, confusión... (los de `rl/autopsy.py`); *balance de suerte* = eventos en contra − a favor.\n")
    w(_round_table([("derrotas con mala suerte", H["loss_with_bad_luck"]), ("victorias con mala suerte", H["win_with_bad_luck"]),
                    ("balance en derrotas", H["luck_balance_losses"]), ("balance en victorias", H["luck_balance_wins"])],
                   nd=2))
    ev = list(H["event_rates"].items())[:12]
    w("\nEventos por combate (derrotas / victorias): " + ", ".join(f"{k} {v['per_loss']:.2f}/{v['per_win']:.2f}"
                                                                   for k, v in ev) + ".\n")
    for cap, p in plots[2:]:
        w(f"![{cap}]({p.split('/')[-2]}/{p.split('/')[-1]})\n")

    T = R.get("truth") or {}
    if T:
        w("## 10. Análisis con información verdadera (MARCADO: el jugador no la tiene)\n")
        w("Usa el equipo verdadero del rival (campo `truth` del registro), solo para describir contra quién se pierde.\n")
        w("Rivales con más derrotas (fracción de combates perdidos cuando el rival incluye la especie): "
          + ", ".join(f"{x['name']} {_pct(x['loss_rate'])} ({x['faced']})" for x in T["hardest_opponents"][:15]) + ".")
        w("\nMás fáciles: " + ", ".join(f"{x['name']} {_pct(x['loss_rate'])} ({x['faced']})"
                                        for x in T["easiest_opponents"]) + ".\n")
    return "\n".join(L) + "\n"
