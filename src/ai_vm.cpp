#include "ai_context.hpp"
#include "ai_scripts.hpp"
#include "battle_engine.hpp"
#include "data.hpp"
#include <iostream>

namespace pkmn {

// Helper to read data (little endian)
static uint8_t readByte(const uint8_t*& ptr) { 
    return *ptr++; 
}

static uint16_t readWord(const uint8_t*& ptr) {
    uint16_t val = ptr[0] | (ptr[1] << 8);
    ptr += 2;
    return val;
}

static uint32_t readInt(const uint8_t*& ptr) {
    uint32_t val = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    ptr += 4;
    return val;
}

void AIContext::execute(uint32_t logicId) {
    // Lookup script
    uint32_t offset = gBattleAI_ScriptsTable[logicId];
    // offset 0 is "start of array". logic 0 is CheckBadMove.
    
    const uint8_t* ptr = gBattleAI_Scripts + offset;
    
    stack.clear();
    aiThinking.aiAction = 0;
    
    int instructions = 0;
    
    while ((aiThinking.aiAction & AI_ACTION_DONE) == 0 && instructions++ < 10000) {
        uint8_t opcode = readByte(ptr);
        
        switch (opcode) {
            case 0x00: // if_random_less_than
            {
                uint8_t val = readByte(ptr);
                uint32_t target = readInt(ptr);
                if (randomLessThan(val)) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x01: // if_random_greater_than
            {
                uint8_t val = readByte(ptr);
                uint32_t target = readInt(ptr);
                if (randomGreaterThan(val)) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x02: // if_random_equal
            case 0x03: // if_random_not_equal
            {
                // Not standard logic? Typically 50%.
                // Decomp: if (random % 256 == val)
                uint8_t val = readByte(ptr);
                uint32_t target = readInt(ptr);
                bool eq = ((engine.random() % 256) == val);
                if (opcode == 0x02) { if (eq) ptr = gBattleAI_Scripts + target; }
                else { if (!eq) ptr = gBattleAI_Scripts + target; }
                break;
            }
            case 0x04: // score
            {
                int8_t val = (int8_t)readByte(ptr); // Signed
                scoreOp(val);
                break;
            }
            case 0x05: // if_hp_less_than
            case 0x06: // if_hp_more_than
            case 0x07: // if_hp_equal
            case 0x08: // if_hp_not_equal
            {
                uint8_t battler = readByte(ptr);
                uint8_t val = readByte(ptr);
                uint32_t target = readInt(ptr);
                
                // Get HP %
                uint8_t id = getBattler(battler);
                const auto& mon = engine.getState().getActivePokemon(id);
                int hpPct = (mon.maxHP == 0) ? 0 : (mon.currentHP * 100 / mon.maxHP);
                
                bool cond = false;
                if (opcode == 0x05) cond = hpPct < val;
                else if (opcode == 0x06) cond = hpPct > val;
                else if (opcode == 0x07) cond = hpPct == val;
                else if (opcode == 0x08) cond = hpPct != val;
                
                if (cond) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x09: // if_status
            case 0x0A: // if_not_status
            {
                uint8_t battler = readByte(ptr);
                uint32_t mask = readInt(ptr);
                uint32_t target = readInt(ptr);
                bool has = hasStatus(battler, mask);
                if ((opcode == 0x09 && has) || (opcode == 0x0A && !has))
                    ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x0B: // if_status2
            case 0x0C: // if_not_status2
            {
                uint8_t battler = readByte(ptr);
                uint32_t mask = readInt(ptr);
                uint32_t target = readInt(ptr);
                bool has = hasStatus2(battler, mask);
                if ((opcode == 0x0B && has) || (opcode == 0x0C && !has))
                    ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x0D: // if_status3
            case 0x0E: // if_not_status3
            {
                uint8_t battler = readByte(ptr);
                uint32_t mask = readInt(ptr);
                uint32_t target = readInt(ptr);
                bool has = hasStatus3(battler, mask);
                if ((opcode == 0x0D && has) || (opcode == 0x0E && !has))
                    ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x0F: // if_side_affecting
            case 0x10: // if_not_side_affecting
            {
                uint8_t battler = readByte(ptr);
                uint32_t mask = readInt(ptr);
                uint32_t target = readInt(ptr);
                // "battler" usually maps to side (User side vs Target side)
                uint8_t side = (getBattler(battler) == battlerAI) ? 0 : 1; // 0=Player, 1=Opponent? 
                // Wait, getBattler returns 0-5.
                // Assuming side logic handled in hasSideStatus
                bool has = hasSideStatus(side, mask);
                if ((opcode == 0x0F && has) || (opcode == 0x10 && !has))
                    ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x11: // if_less_than
            case 0x12: // if_more_than
            case 0x13: // if_equal
            case 0x14: // if_not_equal
            {
                uint8_t val = readByte(ptr);
                uint32_t target = readInt(ptr);
                bool cond = false;
                // Compares funcResult
                if (opcode == 0x11) cond = aiThinking.funcResult < val;
                else if (opcode == 0x12) cond = aiThinking.funcResult > val;
                else if (opcode == 0x13) cond = aiThinking.funcResult == val;
                else if (opcode == 0x14) cond = aiThinking.funcResult != val;
                
                if (cond) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x19: // if_move
            case 0x1A: // if_not_move
            {
                uint16_t move = readWord(ptr);
                uint32_t target = readInt(ptr);
                bool isM = isMove(move);
                if ((opcode == 0x19 && isM) || (opcode == 0x1A && !isM))
                    ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x1B: // if_in_bytes
            case 0x1C: // if_not_in_bytes
            {
                uint32_t listOffset = readInt(ptr);
                uint32_t target = readInt(ptr);
                const uint8_t* list = gBattleAI_Scripts + listOffset;
                bool found = false;
                // what logic? compares move type?
                // Cmd_if_in_bytes:
                // T1_READ_PTR(list)
                // while(*list != 0xFF) { if (*list == val) found; list++; }
                // What is 'val'?
                // Cmd 0x1B usually compares a register?
                // Decomp: Cmd_if_in_bytes checks funcResult? Or 'val' from earlier?
                // Actually Cmd_if_in_bytes implementation:
                // while (*ptr != 0xFF) { if (*ptr == AI_THINKING_STRUCT->funcResult) ... }
                // Warning: funcResult is int, bytes are u8.
                while (*list != 0xFF) {
                    if (*list == (uint8_t)aiThinking.funcResult) { found = true; break; }
                    list++;
                }
                if ((opcode == 0x1B && found) || (opcode == 0x1C && !found))
                    ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x1D: // if_in_hwords (same but words)
            case 0x1E: 
            {
                uint32_t listOffset = readInt(ptr);
                uint32_t target = readInt(ptr);
                const uint8_t* list = gBattleAI_Scripts + listOffset;
                bool found = false;
                // while (*list != 0xFFFF)
                while (true) {
                    uint16_t val = list[0] | (list[1] << 8);
                    if (val == 0xFFFF) break;
                    if (val == (uint16_t)aiThinking.funcResult) { found = true; break; }
                    list += 2;
                }
                if ((opcode == 0x1D && found) || (opcode == 0x1E && !found))
                    ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x1F: // if_user_has_attacking_move
            {
                uint32_t target = readInt(ptr);
                // Check user moves
                bool hasAttacking = false;
                // MoveData access
                const auto& mon = engine.getState().getActivePokemon(battlerAI);
                for(int i=0; i<4; ++i) {
                     if(mon.moves[i] != 0) {
                         const MoveData& md = getMoveData(mon.moves[i]);
                         if(md.power > 1) { // >1 assumes 1 is variable power (could be attacking). 0 is status.
                             hasAttacking = true;
                             break;
                         }
                     }
                }
                
                if (hasAttacking) ptr = gBattleAI_Scripts + target; 
                break;
            }
            case 0x21: // get_turn_count
            {
                aiThinking.funcResult = engine.getTurnCount();
                break;
            }
            case 0x22: // get_type
            {
                uint8_t arg = readByte(ptr);
                // AI_TYPE1_TARGET etc. 
                // arg: 0=Move, 1=UserType1, 2=UserType2, 3=TargetType1, 4=TargetType2
                // Just implementing Move Type for now (most common)
                // TODO: Implement others
                if (arg == 0) { // MOVE
                     const MoveData& md = getMoveData(aiThinking.moveConsidered);
                     aiThinking.funcResult = (int)md.type;
                } else {
                     aiThinking.funcResult = 0; // Normal
                }
                break;
            }
            case 0x23: // get_considered_move_power
            {
                const MoveData& md = getMoveData(aiThinking.moveConsidered);
                aiThinking.funcResult = md.power;
                break;
            }
            case 0x24: // get_how_powerful_move_is
            {
                checkMostPowerfulMove();
                break;
            }
            case 0x26: // if_equal_ (compares funcResult with WORD/INT? No, opcode says 4 bytes param?)
            case 0x27: // if_not_equal_
            {
                // Macro: if_equal_ param0(byte), param1(ptr)
                // Wait, Macro 0x26: byte param0, 4byte param1.
                // Same as 0x13?
                // Decomp Cmd_if_equal_ (0x26):
                // matches Cmd_if_equal (0x13)?
                // check definition.
                // 0x13 param0 is 1 byte val.
                // 0x26 param0 is 1 byte val.
                // Difference?
                // Probably different source register?
                // Decomp: Cmd_if_equal checks funcResult.
                // 0x26? 
                // Assuming same for now.
                uint8_t val = readByte(ptr);
                uint32_t target = readInt(ptr);
                bool cond = (aiThinking.funcResult == val);
                if (opcode == 0x26 ? cond : !cond) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x28: // if_user_goes
            case 0x29: // if_user_doesnt_go
            {
                uint8_t bank = readByte(ptr); // arg: 0=user, 1=target?
                uint32_t target = readInt(ptr);
                // Check speed
                // Placeholder
                bool userFaster = true;
                bool cond = (bank == 0) ? userFaster : !userFaster;
                if (opcode == 0x28 ? cond : !cond) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x2C: // count_usable_party_mons
            {
                readByte(ptr); // battler
                aiThinking.funcResult = 3; // placeholder
                break;
            }
            case 0x2D: // get_considered_move
            {
                aiThinking.funcResult = aiThinking.moveConsidered;
                break;
            }
            case 0x2E: // get_considered_move_effect
            {
                const MoveData& md = getMoveData(aiThinking.moveConsidered);
                aiThinking.funcResult = (int)md.effect;
                break;
            }
            case 0x2F: // get_ability
            {
                uint8_t battler = readByte(ptr);
                aiThinking.funcResult = getAbility(battler);
                break;
            }
            case 0x31: // if_type_effectiveness
            {
                uint8_t eff = readByte(ptr);
                uint32_t target = readInt(ptr);
                if (typeEffectivenessEquals(eff)) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x37: // if_effect
            {
                uint8_t eff = readByte(ptr); // byte? Macro says byte?
                // check script data: 0x37, B0(EFFECT_SLEEP)...
                // Yes byte.
                uint32_t target = readInt(ptr);
                if (isEffect(eff)) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x38: // if_status
            {
                uint8_t battler = readByte(ptr);
                uint32_t statusMask = readInt(ptr);
                uint32_t target = readInt(ptr);
                if (hasStatus(battler, statusMask)) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x39: // stat level checks
            case 0x3A:
            case 0x3B:
            case 0x3C:
            {
                uint8_t battler = readByte(ptr);
                uint8_t stat = readByte(ptr);
                uint8_t val = readByte(ptr);
                uint32_t target = readInt(ptr);
                bool cond = false;
                if (opcode == 0x39) cond = statLevelLessThan(battler, stat, val);
                else if (opcode == 0x3A) cond = statLevelMoreThan(battler, stat, val);
                else if (opcode == 0x3B) cond = statLevelEqual(battler, stat, val);
                else if (opcode == 0x3C) cond = !statLevelEqual(battler, stat, val);
                if (cond) ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x3D: // if_can_faint
            {
                 uint32_t target = readInt(ptr); // jump target
                 int damage = engine.calculateDamage(battlerAI, battlerTarget, aiThinking.moveConsidered);
                 const auto& targetMon = engine.getState().getActivePokemon(battlerTarget);
                 if (damage >= targetMon.currentHP) {
                     ptr = gBattleAI_Scripts + target;
                 }
                 break;
            }
            case 0x3E: // if_cant_faint
            {
                 uint32_t target = readInt(ptr);
                 int damage = engine.calculateDamage(battlerAI, battlerTarget, aiThinking.moveConsidered);
                 const auto& targetMon = engine.getState().getActivePokemon(battlerTarget);
                 if (damage < targetMon.currentHP) {
                     ptr = gBattleAI_Scripts + target;
                 }
                 break;
            }
            case 0x49: // get_gender
            {
                readByte(ptr); 
                aiThinking.funcResult = 0; // Male
                break;
            }
            case 0x58: // call
            {
                uint32_t target = readInt(ptr);
                uint32_t returnOffset = (uint32_t)(ptr - gBattleAI_Scripts);
                stack.push_back(returnOffset);
                ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x59: // goto
            {
                uint32_t target = readInt(ptr);
                ptr = gBattleAI_Scripts + target;
                break;
            }
            case 0x5A: // end
            {
                if (stack.empty()) {
                    aiThinking.aiAction |= AI_ACTION_DONE;
                } else {
                    uint32_t ret = stack.back();
                    stack.pop_back();
                    ptr = gBattleAI_Scripts + ret;
                }
                break;
            }
            case 0x5E: // if_target_is_ally
            {
                uint32_t target = readInt(ptr);
                (void)target;
                // Singles: Never ally. Doubles: Maybe.
                // Assuming singles for now.
                // target is never ally.
                // So condition false.
                // But command checks if (Target == Ally) goto target.
                // Condition false -> fallthrough.
                break;
            }
            case 0x60: // check_ability
            {
                uint8_t battler = readByte(ptr);
                uint8_t ability = readByte(ptr);
                aiThinking.funcResult = hasAbility(battler, ability); // 1 or 0
                break;
            }
            default:
            {
                // Skip args? We don't know size!
                // This will desync. A crash is likely.
                // Ideally we implement everything.
                // Use OP_PARAMS from script?
                // For now, logging error and stop.
                std::cerr << "Unimplemented opcode: " << (int)opcode << std::endl;
                aiThinking.aiAction |= AI_ACTION_DONE;
                break;
            }
        }
    }
}

} // namespace pkmn
