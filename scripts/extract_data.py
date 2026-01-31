#!/usr/bin/env python3
"""
Extract Pokemon data from pokeemerald decompilation and generate C++ data files.
"""

import re
import os

POKEEMERALD_DIR = "/home/apollo/Dev/pokeemerald"
OUTPUT_DIR = "/home/apollo/Dev/pokeemerald/simulator/src/data"
INCLUDE_DIR = "/home/apollo/Dev/pokeemerald/simulator/include"

# Type mappings
TYPE_MAP = {
    "TYPE_NORMAL": "Type::Normal", "TYPE_FIGHTING": "Type::Fighting", 
    "TYPE_FLYING": "Type::Flying", "TYPE_POISON": "Type::Poison",
    "TYPE_GROUND": "Type::Ground", "TYPE_ROCK": "Type::Rock",
    "TYPE_BUG": "Type::Bug", "TYPE_GHOST": "Type::Ghost",
    "TYPE_STEEL": "Type::Steel", "TYPE_MYSTERY": "Type::Mystery",
    "TYPE_FIRE": "Type::Fire", "TYPE_WATER": "Type::Water",
    "TYPE_GRASS": "Type::Grass", "TYPE_ELECTRIC": "Type::Electric",
    "TYPE_PSYCHIC": "Type::Psychic", "TYPE_ICE": "Type::Ice",
    "TYPE_DRAGON": "Type::Dragon", "TYPE_DARK": "Type::Dark",
}

NATURE_MAP = {
    "NATURE_HARDY": "Nature::Hardy", "NATURE_LONELY": "Nature::Lonely",
    "NATURE_BRAVE": "Nature::Brave", "NATURE_ADAMANT": "Nature::Adamant",
    "NATURE_NAUGHTY": "Nature::Naughty", "NATURE_BOLD": "Nature::Bold",
    "NATURE_DOCILE": "Nature::Docile", "NATURE_RELAXED": "Nature::Relaxed",
    "NATURE_IMPISH": "Nature::Impish", "NATURE_LAX": "Nature::Lax",
    "NATURE_TIMID": "Nature::Timid", "NATURE_HASTY": "Nature::Hasty",
    "NATURE_SERIOUS": "Nature::Serious", "NATURE_JOLLY": "Nature::Jolly",
    "NATURE_NAIVE": "Nature::Naive", "NATURE_MODEST": "Nature::Modest",
    "NATURE_MILD": "Nature::Mild", "NATURE_QUIET": "Nature::Quiet",
    "NATURE_BASHFUL": "Nature::Bashful", "NATURE_RASH": "Nature::Rash",
    "NATURE_CALM": "Nature::Calm", "NATURE_GENTLE": "Nature::Gentle",
    "NATURE_SASSY": "Nature::Sassy", "NATURE_CAREFUL": "Nature::Careful",
    "NATURE_QUIRKY": "Nature::Quirky",
}

def extract_move_effects():
    """Get all unique move effects from battle_moves.h"""
    filepath = os.path.join(POKEEMERALD_DIR, "src/data/battle_moves.h")
    with open(filepath, 'r') as f:
        content = f.read()
    
    effects = set()
    for m in re.finditer(r'\.effect\s*=\s*(\w+)', content):
        effects.add(m.group(1))
    
    effects = sorted(effects)
    
    # Create mapping: EFFECT_X -> MoveEffect::X
    effect_map = {}
    for e in effects:
        name = e.replace('EFFECT_', '')
        effect_map[e] = f"MoveEffect::{name}"
    
    return effects, effect_map

def extract_species_data():
    """Extract species base stats from species_info.h"""
    filepath = os.path.join(POKEEMERALD_DIR, "src/data/pokemon/species_info.h")
    with open(filepath, 'r') as f:
        content = f.read()
    
    entries = re.split(r'\n\s*\[SPECIES_', content)
    
    species_data = []
    for entry in entries[1:]:
        name_match = re.match(r'(\w+)\]\s*=', entry)
        if not name_match:
            continue
        name = name_match.group(1)
        if name == "NONE":
            continue
        
        try:
            hp = int(re.search(r'\.baseHP\s*=\s*(\d+)', entry).group(1))
            attack = int(re.search(r'\.baseAttack\s*=\s*(\d+)', entry).group(1))
            defense = int(re.search(r'\.baseDefense\s*=\s*(\d+)', entry).group(1))
            speed = int(re.search(r'\.baseSpeed\s*=\s*(\d+)', entry).group(1))
            sp_attack = int(re.search(r'\.baseSpAttack\s*=\s*(\d+)', entry).group(1))
            sp_defense = int(re.search(r'\.baseSpDefense\s*=\s*(\d+)', entry).group(1))
        except AttributeError:
            continue
        
        types_match = re.search(r'\.types\s*=\s*\{\s*(TYPE_\w+)\s*,\s*(TYPE_\w+)\s*\}', entry)
        if types_match:
            type1 = TYPE_MAP.get(types_match.group(1), "Type::Normal")
            type2 = TYPE_MAP.get(types_match.group(2), "Type::Normal")
        else:
            type1 = type2 = "Type::Normal"
        
        # Extract abilities
        abilities_match = re.search(r'\.abilities\s*=\s*\{\s*(ABILITY_\w+)\s*,\s*(ABILITY_\w+)\s*\}', entry)
        if abilities_match:
            ability1 = abilities_match.group(1)
            ability2 = abilities_match.group(2)
        else:
            ability1 = ability2 = "ABILITY_NONE"
        
        # Extract gender ratio
        # PERCENT_FEMALE(x) = (x * 255) / 100, MON_GENDERLESS = 255
        gender_match = re.search(r'\.genderRatio\s*=\s*(PERCENT_FEMALE\([\d.]+\)|\w+)', entry)
        gender_ratio = 127  # Default: 50% male/female
        if gender_match:
            gender_str = gender_match.group(1)
            if gender_str == "MON_GENDERLESS":
                gender_ratio = 255
            elif gender_str == "MON_MALE":
                gender_ratio = 0
            elif gender_str == "MON_FEMALE":
                gender_ratio = 254
            elif "PERCENT_FEMALE" in gender_str:
                pct_match = re.search(r'PERCENT_FEMALE\(([\d.]+)\)', gender_str)
                if pct_match:
                    pct = float(pct_match.group(1))
                    gender_ratio = min(254, int((pct * 255) / 100))
        
        species_data.append({
            'name': name, 'hp': hp, 'attack': attack, 'defense': defense,
            'speed': speed, 'sp_attack': sp_attack, 'sp_defense': sp_defense,
            'type1': type1, 'type2': type2, 'ability1': ability1, 'ability2': ability2,
            'gender_ratio': gender_ratio
        })
    
    return species_data

def extract_move_data(effect_map):
    """Extract move data from battle_moves.h"""
    filepath = os.path.join(POKEEMERALD_DIR, "src/data/battle_moves.h")
    with open(filepath, 'r') as f:
        content = f.read()
    
    pattern = r'\[MOVE_(\w+)\]\s*=\s*\{([^}]+)\}'
    
    moves = []
    for match in re.finditer(pattern, content):
        name = match.group(1)
        data = match.group(2)
        
        try:
            power = int(re.search(r'\.power\s*=\s*(\d+)', data).group(1))
            accuracy = int(re.search(r'\.accuracy\s*=\s*(\d+)', data).group(1))
            pp = int(re.search(r'\.pp\s*=\s*(\d+)', data).group(1))
        except AttributeError:
            continue
        
        type_match = re.search(r'\.type\s*=\s*(TYPE_\w+)', data)
        move_type = TYPE_MAP.get(type_match.group(1), "Type::Normal") if type_match else "Type::Normal"
        
        effect_match = re.search(r'\.effect\s*=\s*(\w+)', data)
        effect = effect_map.get(effect_match.group(1), "MoveEffect::HIT") if effect_match else "MoveEffect::HIT"
        
        chance = 0
        chance_match = re.search(r'\.secondaryEffectChance\s*=\s*(\d+)', data)
        if chance_match:
            chance = int(chance_match.group(1))
        
        priority = 0
        priority_match = re.search(r'\.priority\s*=\s*(-?\d+)', data)
        if priority_match:
            priority = int(priority_match.group(1))
        
        flags_match = re.search(r'\.flags\s*=\s*([^,\n]+)', data)
        flags = flags_match.group(1).strip() if flags_match else "0"
        is_physical = "FLAG_MAKES_CONTACT" in flags
        makes_contact = "FLAG_MAKES_CONTACT" in flags
        
        moves.append({
            'name': name, 'power': power, 'accuracy': accuracy, 'pp': pp,
            'type': move_type, 'effect': effect, 'chance': chance, 
            'priority': priority, 'is_physical': is_physical, 'makes_contact': makes_contact
        })
    
    return moves

def extract_frontier_mons():
    """Extract frontier mon data"""
    filepath = os.path.join(POKEEMERALD_DIR, "src/data/battle_frontier/battle_frontier_mons.h")
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by FRONTIER_MON entries (handles nested braces better)
    entries = re.split(r'\[FRONTIER_MON_', content)
    
    mons = []
    for entry in entries[1:]:  # Skip first (before any FRONTIER_MON_)
        # Get name
        name_match = re.match(r'(\w+)\]', entry)
        if not name_match:
            continue
        name = name_match.group(1)
        
        # Get species
        species_match = re.search(r'\.species\s*=\s*SPECIES_(\w+)', entry)
        species = species_match.group(1) if species_match else "NONE"
        
        # Get moves
        moves_match = re.search(r'\.moves\s*=\s*\{([^}]+)\}', entry)
        if moves_match:
            move_names = re.findall(r'MOVE_(\w+)', moves_match.group(1))
            while len(move_names) < 4:
                move_names.append("NONE")
        else:
            move_names = ["NONE", "NONE", "NONE", "NONE"]
        
        # Get item table ID (index into frontier item table)
        item_match = re.search(r'\.itemTableId\s*=\s*BATTLE_FRONTIER_ITEM_(\w+)', entry)
        item_name = item_match.group(1) if item_match else "NONE"
        
        # Get EV spread
        ev_match = re.search(r'\.evSpread\s*=\s*([^,\n]+)', entry)
        ev_spread = ev_match.group(1).strip() if ev_match else "0"
        
        # Get nature
        nature_match = re.search(r'\.nature\s*=\s*(NATURE_\w+)', entry)
        nature = NATURE_MAP.get(nature_match.group(1), "Nature::Hardy") if nature_match else "Nature::Hardy"
        
        mons.append({
            'name': name, 'species': species, 'moves': move_names,
            'item_name': item_name, 'ev_spread': ev_spread, 'nature': nature
        })
    
    return mons

def generate_move_effect_enum(effects):
    """Generate MoveEffect enum for types.hpp"""
    lines = ['// ============================================================================',
             '// Move Effects (auto-generated from battle_moves.h)',
             '// ============================================================================',
             'enum class MoveEffect : uint16_t {']
    
    for i, e in enumerate(effects):
        name = e.replace('EFFECT_', '')
        lines.append(f'    {name} = {i},')
    
    lines.append(f'    COUNT = {len(effects)}')
    lines.append('};')
    return '\n'.join(lines)

def generate_species_cpp(species_data):
    lines = ['#include "data.hpp"\n\nnamespace pkmn {\n\n// Species data array - extracted from species_info.h\n// Format: {HP, Atk, Def, Spd, SpA, SpD, Type1, Type2, {Ability1, Ability2}, GenderRatio}\nstatic const SpeciesData SPECIES_DATA[] = {\n']
    lines.append('    // SPECIES_NONE (0)\n')
    lines.append('    {0, 0, 0, 0, 0, 0, Type::Normal, Type::Normal, {ABILITY_NONE, ABILITY_NONE}, 255},\n')
    
    for i, s in enumerate(species_data):
        lines.append(f'    // SPECIES_{s["name"]} ({i+1})\n')
        lines.append(f'    {{{s["hp"]}, {s["attack"]}, {s["defense"]}, {s["speed"]}, {s["sp_attack"]}, {s["sp_defense"]}, ')
        lines.append(f'{s["type1"]}, {s["type2"]}, {{{s["ability1"]}, {s["ability2"]}}}, {s["gender_ratio"]}}},\n')
    
    lines.append('};\n\n')
    lines.append(f'const SpeciesData& getSpeciesData(uint16_t speciesId) {{\n')
    lines.append(f'    if (speciesId >= {len(species_data)+1}) return SPECIES_DATA[0];\n')
    lines.append('    return SPECIES_DATA[speciesId];\n')
    lines.append('}\n\n}  // namespace pkmn\n')
    
    return ''.join(lines)

def generate_moves_cpp(moves):
    lines = ['#include "data.hpp"\n\nnamespace pkmn {\nstatic const MoveData MOVE_DATA[] = {\n']
    
    for i, m in enumerate(moves):
        phys = "true" if m["is_physical"] else "false"
        contact = "true" if m["makes_contact"] else "false"
        lines.append(f'    // MOVE_{m["name"]} ({i})\n')
        lines.append(f'    {{{m["power"]}, {m["accuracy"]}, {m["pp"]}, {m["type"]}, {m["effect"]}, ')
        lines.append(f'{m["chance"]}, {m["priority"]}, {phys}, {contact}}},\n')
    
    lines.append('};\n\n')
    lines.append('const MoveData& getMoveData(uint16_t moveId) {\n')
    lines.append(f'    if (moveId >= {len(moves)}) return MOVE_DATA[0];\n')
    lines.append('    return MOVE_DATA[moveId];\n')
    lines.append('}\n\n}  // namespace pkmn\n')
    
    return ''.join(lines)

def generate_frontier_cpp(mons):
    # Build item name to ID mapping from the pokeemerald constants
    item_name_to_id = {
        "NONE": 0, "KINGS_ROCK": 1, "SITRUS_BERRY": 2, "ORAN_BERRY": 3,
        "CHESTO_BERRY": 4, "HARD_STONE": 5, "FOCUS_BAND": 6, "PERSIM_BERRY": 7,
        "MIRACLE_SEED": 8, "BERRY_JUICE": 9, "MACHO_BRACE": 10, "SILVER_POWDER": 11,
        "CHERI_BERRY": 12, "BLACK_GLASSES": 13, "BLACK_BELT": 14, "SOUL_DEW": 15,
        "CHOICE_BAND": 16, "MAGNET": 17, "SILK_SCARF": 18, "WHITE_HERB": 19,
        "DEEP_SEA_SCALE": 20, "DEEP_SEA_TOOTH": 21, "MYSTIC_WATER": 22, "SHARP_BEAK": 23,
        "QUICK_CLAW": 24, "LEFTOVERS": 25, "RAWST_BERRY": 26, "LIGHT_BALL": 27,
        "POISON_BARB": 28, "NEVER_MELT_ICE": 29, "ASPEAR_BERRY": 30, "SPELL_TAG": 31,
        "BRIGHT_POWDER": 32, "LEPPA_BERRY": 33, "SCOPE_LENS": 34, "TWISTED_SPOON": 35,
        "METAL_COAT": 36, "MENTAL_HERB": 37, "CHARCOAL": 38, "PECHA_BERRY": 39,
        "SOFT_SAND": 40, "LUM_BERRY": 41, "DRAGON_SCALE": 42, "DRAGON_FANG": 43,
        "IAPAPA_BERRY": 44, "WIKI_BERRY": 45, "SEA_INCENSE": 46, "SHELL_BELL": 47,
        "SALAC_BERRY": 48, "LANSAT_BERRY": 49, "APICOT_BERRY": 50, "STARF_BERRY": 51,
        "LIECHI_BERRY": 52, "GANLON_BERRY": 53, "PETAYA_BERRY": 54, "FIGY_BERRY": 55,
        "AGUAV_BERRY": 56, "MAGO_BERRY": 57, "LAX_INCENSE": 58, "AGUAV_BERRY_59": 59,
        "MAGO_BERRY_60": 60, "FIGY_BERRY_61": 61, "WIKI_BERRY_62": 62,
    }
    
    lines = ['#include "data.hpp"\n\nnamespace pkmn {\n\n// Frontier mons - extracted from battle_frontier_mons.h\n// Format: {Species, {Move1, Move2, Move3, Move4}, ItemTableId, EVSpread, Nature}\nstatic const FrontierMon FRONTIER_MONS[] = {\n']
    
    for i, m in enumerate(mons):
        move_ids = [f"MOVE_{mv}" for mv in m["moves"][:4]]
        move_str = ", ".join(move_ids)
        
        # Convert EV spread flags to bitfield
        ev_val = 0
        if "F_EV_SPREAD_HP" in m["ev_spread"]: ev_val |= 0x01
        if "F_EV_SPREAD_ATTACK" in m["ev_spread"]: ev_val |= 0x02
        if "F_EV_SPREAD_DEFENSE" in m["ev_spread"]: ev_val |= 0x04
        if "F_EV_SPREAD_SPEED" in m["ev_spread"]: ev_val |= 0x08
        if "F_EV_SPREAD_SP_ATTACK" in m["ev_spread"]: ev_val |= 0x10
        if "F_EV_SPREAD_SP_DEFENSE" in m["ev_spread"]: ev_val |= 0x20
        
        # Get item table ID
        item_id = item_name_to_id.get(m["item_name"], 0)
        
        lines.append(f'    // FRONTIER_MON_{m["name"]} ({i})\n')
        lines.append(f'    {{SPECIES_{m["species"]}, {{{move_str}}}, {item_id}, {ev_val}, {m["nature"]}}},\n')
    
    lines.append('};\n\n')
    
    # Full frontier item table (63 items)
    lines.append('// Frontier item table - maps itemTableId to actual ITEM_* constant\n')
    lines.append('static const uint16_t FRONTIER_ITEMS[] = {\n')
    lines.append('    ITEM_NONE, ITEM_KINGS_ROCK, ITEM_SITRUS_BERRY, ITEM_ORAN_BERRY,       // 0-3\n')
    lines.append('    ITEM_CHESTO_BERRY, ITEM_HARD_STONE, ITEM_FOCUS_BAND, ITEM_PERSIM_BERRY, // 4-7\n')
    lines.append('    ITEM_MIRACLE_SEED, ITEM_BERRY_JUICE, ITEM_MACHO_BRACE, ITEM_SILVER_POWDER, // 8-11\n')
    lines.append('    ITEM_CHERI_BERRY, ITEM_BLACK_GLASSES, ITEM_BLACK_BELT, ITEM_SOUL_DEW,  // 12-15\n')
    lines.append('    ITEM_CHOICE_BAND, ITEM_MAGNET, ITEM_SILK_SCARF, ITEM_WHITE_HERB,     // 16-19\n')
    lines.append('    ITEM_DEEP_SEA_SCALE, ITEM_DEEP_SEA_TOOTH, ITEM_MYSTIC_WATER, ITEM_SHARP_BEAK, // 20-23\n')
    lines.append('    ITEM_QUICK_CLAW, ITEM_LEFTOVERS, ITEM_RAWST_BERRY, ITEM_LIGHT_BALL,  // 24-27\n')
    lines.append('    ITEM_POISON_BARB, ITEM_NEVER_MELT_ICE, ITEM_ASPEAR_BERRY, ITEM_SPELL_TAG, // 28-31\n')
    lines.append('    ITEM_BRIGHT_POWDER, ITEM_LEPPA_BERRY, ITEM_SCOPE_LENS, ITEM_TWISTED_SPOON, // 32-35\n')
    lines.append('    ITEM_METAL_COAT, ITEM_MENTAL_HERB, ITEM_CHARCOAL, ITEM_PECHA_BERRY,  // 36-39\n')
    lines.append('    ITEM_SOFT_SAND, ITEM_LUM_BERRY, ITEM_DRAGON_SCALE, ITEM_DRAGON_FANG, // 40-43\n')
    lines.append('    ITEM_IAPAPA_BERRY, ITEM_WIKI_BERRY, ITEM_SEA_INCENSE, ITEM_SHELL_BELL, // 44-47\n')
    lines.append('    ITEM_SALAC_BERRY, ITEM_LANSAT_BERRY, ITEM_APICOT_BERRY, ITEM_STARF_BERRY, // 48-51\n')
    lines.append('    ITEM_LIECHI_BERRY, ITEM_GANLON_BERRY, ITEM_PETAYA_BERRY, ITEM_FIGY_BERRY, // 52-55\n')
    lines.append('    ITEM_AGUAV_BERRY, ITEM_MAGO_BERRY, ITEM_LAX_INCENSE,                  // 56-58\n')
    # Indices 59-62 are duplicates in original table, reuse existing items
    lines.append('    ITEM_AGUAV_BERRY, ITEM_MAGO_BERRY, ITEM_FIGY_BERRY, ITEM_WIKI_BERRY,  // 59-62\n')
    lines.append('};\n\n')
    
    lines.append('const FrontierMon& getFrontierMon(uint16_t id) {\n')
    lines.append(f'    if (id >= {len(mons)}) return FRONTIER_MONS[0];\n')
    lines.append('    return FRONTIER_MONS[id];\n')
    lines.append('}\n\n')
    
    lines.append('uint16_t getFrontierItem(uint8_t id) {\n')
    lines.append('    if (id >= 63) return ITEM_NONE;\n')
    lines.append('    return FRONTIER_ITEMS[id];\n')
    lines.append('}\n\n}  // namespace pkmn\n')
    
    return ''.join(lines)

if __name__ == "__main__":
    print("Extracting move effects...")
    effects, effect_map = extract_move_effects()
    print(f"  Found {len(effects)} unique effects")
    
    print("Extracting species data...")
    species = extract_species_data()
    print(f"  Found {len(species)} species")
    
    print("Extracting move data...")
    moves = extract_move_data(effect_map)
    print(f"  Found {len(moves)} moves")
    
    print("Extracting frontier mons...")
    frontier = extract_frontier_mons()
    print(f"  Found {len(frontier)} frontier mons")
    
    print("\nGenerating C++ files...")
    
    # Generate MoveEffect enum
    enum_code = generate_move_effect_enum(effects)
    print("  Generated MoveEffect enum (paste into types.hpp)")
    print(f"\n{enum_code}\n")
    
    with open(os.path.join(OUTPUT_DIR, "species_data.cpp"), 'w') as f:
        f.write(generate_species_cpp(species))
    print("  Generated species_data.cpp")
    
    with open(os.path.join(OUTPUT_DIR, "move_data.cpp"), 'w') as f:
        f.write(generate_moves_cpp(moves))
    print("  Generated move_data.cpp")
    
    with open(os.path.join(OUTPUT_DIR, "frontier_mons.cpp"), 'w') as f:
        f.write(generate_frontier_cpp(frontier))
    print("  Generated frontier_mons.cpp")
    
    print("\nDone! Copy the MoveEffect enum to types.hpp")
