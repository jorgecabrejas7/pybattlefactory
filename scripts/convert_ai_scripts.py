#!/usr/bin/env python3
import sys
import re
import os

# Opcode Mapping from asm/macros/battle_ai_script.inc
OPCODES = {
    'if_random_less_than': 0x0,
    'if_random_greater_than': 0x1,
    'if_random_equal': 0x2,
    'if_random_not_equal': 0x3,
    'score': 0x4,
    'if_hp_less_than': 0x5,
    'if_hp_more_than': 0x6,
    'if_hp_equal': 0x7,
    'if_hp_not_equal': 0x8,
    'if_status': 0x9,
    'if_not_status': 0xa,
    'if_status2': 0xb,
    'if_not_status2': 0xc,
    'if_status3': 0xd,
    'if_not_status3': 0xe,
    'if_side_affecting': 0xf,
    'if_not_side_affecting': 0x10,
    'if_less_than': 0x11,
    'if_more_than': 0x12,
    'if_equal': 0x13,
    'if_not_equal': 0x14,
    'if_less_than_ptr': 0x15,
    'if_more_than_ptr': 0x16,
    'if_equal_ptr': 0x17,
    'if_not_equal_ptr': 0x18,
    'if_move': 0x19,
    'if_not_move': 0x1a,
    'if_in_bytes': 0x1b,
    'if_not_in_bytes': 0x1c,
    'if_in_hwords': 0x1d,
    'if_not_in_hwords': 0x1e,
    'if_user_has_attacking_move': 0x1f,
    'if_user_has_no_attacking_moves': 0x20,
    'get_turn_count': 0x21,
    'get_type': 0x22,
    'get_considered_move_power': 0x23,
    'get_how_powerful_move_is': 0x24,
    'get_last_used_bank_move': 0x25,
    'if_equal_': 0x26,
    'if_not_equal_': 0x27,
    'if_user_goes': 0x28,
    'if_user_doesnt_go': 0x29,
    'nop_2A': 0x2a,
    'nop_2B': 0x2b,
    'count_usable_party_mons': 0x2c,
    'get_considered_move': 0x2d,
    'get_considered_move_effect': 0x2e,
    'get_ability': 0x2f,
    'get_highest_type_effectiveness': 0x30,
    'if_type_effectiveness': 0x31,
    'nop_32': 0x32,
    'nop_33': 0x33,
    'if_status_in_party': 0x34,
    'if_status_not_in_party': 0x35,
    'get_weather': 0x36,
    'if_effect': 0x37,
    'if_not_effect': 0x38,
    'if_stat_level_less_than': 0x39,
    'if_stat_level_more_than': 0x3a,
    'if_stat_level_equal': 0x3b,
    'if_stat_level_not_equal': 0x3c,
    'if_can_faint': 0x3d,
    'if_cant_faint': 0x3e,
    'if_has_move': 0x3f,
    'if_doesnt_have_move': 0x40,
    'if_has_move_with_effect': 0x41,
    'if_doesnt_have_move_with_effect': 0x42,
    'if_any_move_disabled_or_encored': 0x43,
    'if_curr_move_disabled_or_encored': 0x44,
    'flee': 0x45,
    'if_random_safari_flee': 0x46,
    'watch': 0x47,
    'get_hold_effect': 0x48,
    'get_gender': 0x49,
    'is_first_turn_for': 0x4a,
    'get_stockpile_count': 0x4b,
    'is_double_battle': 0x4c,
    'get_used_held_item': 0x4d,
    'get_move_type_from_result': 0x4e,
    'get_move_power_from_result': 0x4f,
    'get_move_effect_from_result': 0x50,
    'get_protect_count': 0x51,
    'nop_52': 0x52,
    'nop_53': 0x53,
    'nop_54': 0x54,
    'nop_55': 0x55,
    'nop_56': 0x56,
    'nop_57': 0x57,
    'call': 0x58,
    'goto': 0x59,
    'end': 0x5a,
    'if_level_cond': 0x5b,
    'if_target_taunted': 0x5c,
    'if_target_not_taunted': 0x5d,
    'if_target_is_ally': 0x5e,
    'is_of_type': 0x5f,
    'check_ability': 0x60,
    'if_flash_fired': 0x61,
    'if_holds_item': 0x62,
}

# Parameter types: 'b' (byte), 'h' (half/2byte), 'w' (word/4byte/ptr)
# Based on the macros
OP_PARAMS = {
    0x0: 'bw', 0x1: 'bw', 0x2: 'bw', 0x3: 'bw', # random checks
    0x4: 'b', # score
    0x5: 'bbw', 0x6: 'bbw', 0x7: 'bbw', 0x8: 'bbw', # hp checks
    0x9: 'bww', 0xa: 'bww', 0xb: 'bww', 0xc: 'bww', 0xd: 'bww', 0xe: 'bww', # status checks
    0xf: 'bww', 0x10: 'bww', # side status
    0x11: 'bw', 0x12: 'bw', 0x13: 'bw', 0x14: 'bw', # value checks
    0x15: 'ww', 0x16: 'ww', 0x17: 'ww', 0x18: 'ww', # ptr checks (addr, val)
    0x19: 'hw', 0x1a: 'hw', # move checks
    0x1b: 'ww', 0x1c: 'ww', 0x1d: 'ww', 0x1e: 'ww', # in list checks
    0x1f: 'w', 0x20: 'w', # user has move
    0x21: '', 0x22: 'b', # get turn, get type
    0x23: '', 0x24: '', # move power
    0x25: 'b', # last used move
    0x26: 'bw', 0x27: 'bw', # if_equal_
    0x28: 'bw', 0x29: 'bw', # user goes
    0x2a: '', 0x2b: '',
    0x2c: 'b', # count mons
    0x2d: '', 0x2e: '',
    0x2f: 'b', # get ability
    0x30: '', 
    0x31: 'bw', # if type eff
    0x32: '', 0x33: '',
    0x34: 'bww', 0x35: 'bww',
    0x36: '',
    0x37: 'bw', 0x38: 'bw', # if effect
    0x39: 'bbbw', 0x3a: 'bbbw', 0x3b: 'bbbw', 0x3c: 'bbbw', # stat level
    0x3d: 'w', 0x3e: 'w',
    0x3f: 'bhw', 0x40: 'bhw', # has move
    0x41: 'bbw', 0x42: 'bbw', # move with effect
    0x43: 'bbw', 0x44: 'bw', # disabled/encored
    0x45: '',
    0x46: 'w',
    0x47: '',
    0x48: 'b', 0x49: 'b', 0x4a: 'b', 0x4b: 'b', 0x4c: '', 0x4d: 'b',
    0x4e: '', 0x4f: '', 0x50: '', 0x51: 'b',
    0x52: '', 0x53: '', 0x54: '', 0x55: '', 0x56: '', 0x57: '',
    0x58: 'w', 0x59: 'w', # call, goto
    0x5a: '', # end
    0x5b: 'bw', # level cond
    0x5c: 'w', 0x5d: 'w',
    0x5e: 'w',
    0x5f: 'bb',
    0x60: 'bb',
    0x61: 'bw',
    0x62: 'bhw'
}

# Macros (high level that expand to others)
MACROS = {
    'get_curr_move_type': ['get_type', 'AI_TYPE_MOVE'],
    'get_user_type1': ['get_type', 'AI_TYPE1_USER'],
    'get_user_type2': ['get_type', 'AI_TYPE2_USER'],
    'get_target_type1': ['get_type', 'AI_TYPE1_TARGET'],
    'get_target_type2': ['get_type', 'AI_TYPE2_TARGET'],
    # Conditionals expanding to logic+jump
    'if_ability': lambda args: [['check_ability', args[0], args[1]], ['if_equal', '1', args[2]]],
    'if_no_ability': lambda args: [['check_ability', args[0], args[1]], ['if_equal', '0', args[2]]],
    'if_type': lambda args: [['is_of_type', args[0], args[1]], ['if_equal', '1', args[2]]],
    'if_no_type': lambda args: [['is_of_type', args[0], args[1]], ['if_equal', '0', args[2]]],
    'if_target_faster': lambda args: [['if_user_goes', '1', args[0]]],
    'if_user_faster': lambda args: [['if_user_goes', '0', args[0]]],
    'if_double_battle': lambda args: [['is_double_battle'], ['if_equal', '1', args[0]]],
    'if_not_double_battle': lambda args: [['is_double_battle'], ['if_equal', '0', args[0]]],
    'if_any_move_disabled': lambda args: [['if_any_move_disabled_or_encored', args[0], '0', args[1]]],
    'if_any_move_encored': lambda args: [['if_any_move_disabled_or_encored', args[0], '1', args[1]]],
}

def parse_line(line):
    line = line.strip()
    if not line or line.startswith('@') or line.startswith('//') or line.startswith('#'):
        return None
    # Remove comments
    line = line.split('@')[0].strip()
    
    # Check for labels
    if line.endswith(':'):
        return {'type': 'label', 'name': line[:-1]}
    
    # Directives
    if line.startswith('.'):
        parts = line.split(maxsplit=1)
        directive = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        return {'type': 'directive', 'name': directive, 'args': args}
    
    # Instructions
    parts = line.split(maxsplit=1)
    opcode = parts[0]
    args_str = parts[1] if len(parts) > 1 else ""
    # Split args by comma, ignoring space
    args = [a.strip() for a in args_str.split(',')]
    return {'type': 'instruction', 'name': opcode, 'args': args}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-cpp', required=True)
    parser.add_argument('--output-h', required=True)
    cmd_args = parser.parse_args()
    
    lines = []
    with open(cmd_args.input, 'r') as f:
        lines = f.readlines()
        
    symbol_table = {} # Label -> offset
    byte_code = []
    
    # Hardcoded table entries (Gen 3)
    table_entries = [
        'AI_CheckBadMove',          # 0
        'AI_TryToFaint',            # 1
        'AI_Viability',             # 2
        'AI_SetupFirstTurn',        # 3
        'AI_Risky',                 # 4
        'AI_PreferPowerExtremes',   # 5
        'AI_PreferBatonPass',       # 6
        'AI_DoubleBattle',          # 7
        'AI_HPAware',               # 8
        'AI_TrySunnyDayStart',      # 9
    ]
    # Fill 10-28 with None
    for i in range(10, 29):
        table_entries.append(None)
        
    table_entries.append('AI_Roaming')     # 29
    table_entries.append('AI_Safari')      # 30
    table_entries.append('AI_FirstBattle') # 31

    expanded_instructions = []
    for line in lines:
        parsed = parse_line(line)
        if not parsed: continue
        
        if parsed['type'] == 'label':
            expanded_instructions.append(parsed)
            
        elif parsed['type'] == 'directive':
             pass # Ignore directives
             
        elif parsed['type'] == 'instruction':

            name = parsed['name']
            if name in MACROS:
                definition = MACROS[name]
                if callable(definition):
                    # Lambda macro
                    expansion = definition(parsed['args'])
                    for instr in expansion:
                        expanded_instructions.append({'type': 'instruction', 'name': instr[0], 'args': instr[1:]})
                else:
                    # Simple alias macro
                    # e.g. get_curr_move_type -> get_type, AI_TYPE_MOVE
                    # Can have fixed args mixed with variable args?
                    # My constant dictionary handles fixed args.
                    # Assuming simple one-to-one or one-to-many without param mixing for now
                    expanded_instructions.append({'type': 'instruction', 'name': definition[0], 'args': definition[1:]})
            else:
                expanded_instructions.append(parsed)
                
    # Pass 2: Calculate Offsets
    current_offset = 0
    
    # Prepare map of label -> offset
    # But wait, instructions have variable length arguments.
    # We must size them.
    
    resolvable_instructions = [] # (offset, instruction)
    
    for item in expanded_instructions:
        if item['type'] == 'label':
            symbol_table[item['name']] = current_offset
        elif item['type'] == 'instruction':
            op_name = item['name']
            if op_name not in OPCODES:
                print(f"Unknown opcode: {op_name}")
                continue
            
            opcode = OPCODES[op_name]
            params = OP_PARAMS[opcode]
            
            size = 1 # opcode byte
            for p in params:
                if p == 'b': size += 1
                elif p == 'h': size += 2
                elif p == 'w': size += 4
            
            resolvable_instructions.append({'offset': current_offset, 'item': item, 'size': size})
            current_offset += size

    # Pass 3: Emit Bytes
    final_bytes = []
    
    for entry in resolvable_instructions:
        item = entry['item']
        offset = entry['offset']
        op_name = item['name']
        opcode = OPCODES[op_name]
        params = OP_PARAMS[opcode]
        args = item['args']
        
        final_bytes.append(opcode)
        
        arg_idx = 0
        for p in params:
            val_str = args[arg_idx]
            arg_idx += 1
            
            # Resolve value
            val = 0
            is_label = False
            
            if val_str in symbol_table:
                val = symbol_table[val_str]
                is_label = True
            else:
                try:
                    # Handle hex or int
                    val = int(val_str, 0)
                except ValueError:
                    # Constant? Use string value for C++
                    val = val_str
            
            # If it's a pointer (w) and it WAS a label, we use the offset.
            # If it's 'w' but not a label (e.g. 0xFFFFFFFF), use value.
            
            # Writing bytes
            # We will generate a C++ array initialization string.
            # e.g. 0x5, 0x0, 0x0, 0x0, 0x0 ...
            
            # Issues: 
            # 1. Constants (AI_USER).
            # 2. Labels need to be valid C++ if we use them? 
            #    No, labels are resolved to INTEGER OFFSETS by us.
            #    So 'w' param becomes 4 bytes of integer.
            
            pass 

    # Re-evaluating output format.
    # Since we can't easily resolve C constants like "MOVE_FISSURE" in python,
    # and we want a compiling C++ file.
    # We should generate an array of a struct or just bytes, but allow C++ compiler to resolve constants.
    # But how to pack "MOVE_FISSURE" (enum) into a byte array?
    # cast it: (uint8_t)(MOVE_FISSURE).
    #
    # So our output will be a textual C++ source file.
    # const uint8_t gBattleAI_Scripts[] = {
    #    0x5, (uint8_t)(AI_USER), ...
    # };
    #
    # For Labels, we define them as const uint32_t offsets?
    # No, we need to embed the offset in the byte stream.
    # 4 bytes: (offset & 0xFF), ((offset >> 8) & 0xFF), ...
    #
    # We should pre-calculate label OFFSETS (Pass 2), and when emitting, valid labels get turned into 4 bytes.
    # Unresolved constants (strings) get emitted as (uint8_t)(STRING) or (uint8_t)((STRING) >> 8) etc?
    # Multi-byte constants are hard! 
    # e.g. .4byte SOME_CONSTANT.
    # We need to split it into 4 bytes.
    #
    # Helper: MACRO(val) -> (val)&0xFF, (val>>8)&0xFF ...
    #
    # So the generated C++ will look like:
    # 0x13, (uint8_t)(AI_USER), SPLIT_WORD(Label_Offset), ...
    
    with open(cmd_args.output_cpp, 'w') as out:
        out.write('#include "ai_scripts.hpp"\n')
        out.write('#include "constants.hpp"\n')
        out.write('#include "types.hpp"\n\n')
        out.write('namespace pkmn {\n\n')
        
        # Helper macros for splitting
        out.write('#define B0(x) ((x) & 0xFF)\n')
        out.write('#define B1(x) (((x) >> 8) & 0xFF)\n')
        out.write('#define B2(x) (((x) >> 16) & 0xFF)\n')
        out.write('#define B3(x) (((x) >> 24) & 0xFF)\n\n')
        
        out.write('const uint8_t gBattleAI_Scripts[] = {\n')
        
        for entry in resolvable_instructions:
            item = entry['item']
            op_name = item['name']
            opcode = OPCODES[op_name]
            params = OP_PARAMS[opcode]
            args = item['args']
            
            # Emit Opcode
            out.write(f'    0x{opcode:02X}, ')
            
            arg_idx = 0
            for p in params:
                val = args[arg_idx]
                arg_idx += 1
                
                # Check format
                is_label = val in symbol_table
                
                if is_label:
                    # It's an offset!
                    offset = symbol_table[val]
                    # It must be 'w' (4 bytes)
                    out.write(f'B0({offset}), B1({offset}), B2({offset}), B3({offset}), ')
                else:
                    # It's a value/constant
                    # If p is 'b', emit 1 byte. 'h' 2 bytes, 'w' 4 bytes.
                    if p == 'b':
                        out.write(f'B0({val}), ')
                    elif p == 'h':
                        out.write(f'B0({val}), B1({val}), ')
                    elif p == 'w':
                        out.write(f'B0({val}), B1({val}), B2({val}), B3({val}), ')
            
            out.write(f' // {entry["offset"]:04X}: {op_name} {args}\n')
            
        out.write('};\n\n')
        
        # Table
        out.write('const uint32_t gBattleAI_ScriptsTable[] = {\n')
        for label in table_entries:
            if label in symbol_table:
                out.write(f'    {symbol_table[label]}, // {label}\n')
            else:
                out.write(f'    0, // ERROR: {label} not found\n')
        out.write('};\n\n')
        
        out.write('} // namespace pkmn\n')

    with open(cmd_args.output_h, 'w') as out:
        out.write('#pragma once\n#include <cstdint>\n\nnamespace pkmn {\n')
        out.write('extern const uint8_t gBattleAI_Scripts[];\n')
        out.write('extern const uint32_t gBattleAI_ScriptsTable[];\n')
        out.write('} // namespace pkmn\n')

if __name__ == '__main__':
    main()
