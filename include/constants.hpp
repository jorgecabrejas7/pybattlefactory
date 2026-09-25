#pragma once

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#include <cstdint>

#include "gen/game_constants.hpp"

namespace pkmn {

// Game IDs (species/moves/items/abilities/hold effects) are generated from pokeemerald.elf.

constexpr uint8_t EV_HP = 0x01;
constexpr uint8_t EV_ATTACK = 0x02;
constexpr uint8_t EV_DEFENSE = 0x04;
constexpr uint8_t EV_SPEED = 0x08;
constexpr uint8_t EV_SP_ATK = 0x10;
constexpr uint8_t EV_SP_DEF = 0x20;

constexpr int MAX_PARTY_SIZE = 6;
constexpr int MAX_MOVES = 4;
constexpr int MAX_LEVEL = 100;

constexpr int STAT_STAGE_NUMERATORS[] = {2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8};
constexpr int STAT_STAGE_DENOMINATORS[] = {8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2};

}  // namespace pkmn
// Accuracy stage modifiers (index 0-12 for stages -6 to +6)
constexpr int ACC_STAGE_NUMERATORS[] = {33, 36, 43, 50, 60, 75, 100, 133, 166, 200, 250, 266, 300};
constexpr int ACC_STAGE_DENOMINATORS[] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};

// Critical hit chances (Gen 3: 1/16, 1/8, 1/4, 1/3, 1/2)  
constexpr int CRIT_CHANCE_NUMERATORS[] = {1, 1, 1, 1, 1};
constexpr int CRIT_CHANCE_DENOMINATORS[] = {16, 8, 4, 3, 2};

// ============================================================================
// Battle AI Constants
// ============================================================================

// Battlers
#define AI_TARGET 0
#define AI_USER 1
#define AI_TARGET_PARTNER 2
#define AI_USER_PARTNER 3

// Get Type Command
#define AI_TYPE1_TARGET 0
#define AI_TYPE1_USER 1
#define AI_TYPE2_TARGET 2
#define AI_TYPE2_USER 3
#define AI_TYPE_MOVE 4

// Type Effectiveness
#define AI_EFFECTIVENESS_x4     160
#define AI_EFFECTIVENESS_x2     80
#define AI_EFFECTIVENESS_x1     40
#define AI_EFFECTIVENESS_x0_5   20
#define AI_EFFECTIVENESS_x0_25  10
#define AI_EFFECTIVENESS_x0     0

// AI Weather
#define AI_WEATHER_NONE 0xFFFFFFFF
#define AI_WEATHER_SUN 0
#define AI_WEATHER_RAIN 1
#define AI_WEATHER_SANDSTORM 2
#define AI_WEATHER_HAIL 3

// Power checks
#define MOVE_POWER_OTHER        0
#define MOVE_NOT_MOST_POWERFUL  1
#define MOVE_MOST_POWERFUL      2

// Script Flags
#define AI_SCRIPT_CHECK_BAD_MOVE        (1 << 0)
#define AI_SCRIPT_TRY_TO_FAINT          (1 << 1)
#define AI_SCRIPT_CHECK_VIABILITY       (1 << 2)
#define AI_SCRIPT_SETUP_FIRST_TURN      (1 << 3)
#define AI_SCRIPT_RISKY                 (1 << 4)
#define AI_SCRIPT_PREFER_POWER_EXTREMES (1 << 5)
#define AI_SCRIPT_PREFER_BATON_PASS     (1 << 6)
#define AI_SCRIPT_DOUBLE_BATTLE         (1 << 7)
#define AI_SCRIPT_HP_AWARE              (1 << 8)
#define AI_SCRIPT_TRY_SUNNY_DAY_START   (1 << 9)
#define AI_SCRIPT_ROAMING               (1 << 29)
#define AI_SCRIPT_SAFARI                (1 << 30)
#define AI_SCRIPT_FIRST_BATTLE          (1 << 31)

// AI Actions
#define AI_ACTION_DONE          (1 << 0)
#define AI_ACTION_FLEE          (1 << 1)
#define AI_ACTION_WATCH         (1 << 2)
#define AI_ACTION_DO_NOT_ATTACK (1 << 3)

// Status Constants
// Non-volatile status conditions
#define STATUS1_NONE             0
#define STATUS1_SLEEP            (1 << 0 | 1 << 1 | 1 << 2) // First 3 bits (Number of turns to sleep)
#define STATUS1_SLEEP_TURN(num)  ((num) << 0)
#define STATUS1_POISON           (1 << 3)
#define STATUS1_BURN             (1 << 4)
#define STATUS1_FREEZE           (1 << 5)
#define STATUS1_PARALYSIS        (1 << 6)
#define STATUS1_TOXIC_POISON     (1 << 7)
#define STATUS1_TOXIC_COUNTER    (1 << 8 | 1 << 9 | 1 << 10 | 1 << 11)
#define STATUS1_TOXIC_TURN(num)  ((num) << 8)
#define STATUS1_PSN_ANY          (STATUS1_POISON | STATUS1_TOXIC_POISON)
#define STATUS1_ANY              (STATUS1_SLEEP | STATUS1_POISON | STATUS1_BURN | STATUS1_FREEZE | STATUS1_PARALYSIS | STATUS1_TOXIC_POISON)

// Volatile status ailments
#define STATUS2_CONFUSION             (1 << 0 | 1 << 1 | 1 << 2)
#define STATUS2_CONFUSION_TURN(num)   ((num) << 0)
#define STATUS2_FLINCHED              (1 << 3)
#define STATUS2_UPROAR                (1 << 4 | 1 << 5 | 1 << 6)
#define STATUS2_UPROAR_TURN(num)      ((num) << 4)
#define STATUS2_UNUSED                (1 << 7)
#define STATUS2_BIDE                  (1 << 8 | 1 << 9)
#define STATUS2_BIDE_TURN(num)        (((num) << 8) & STATUS2_BIDE)
#define STATUS2_LOCK_CONFUSE          (1 << 10 | 1 << 11)
#define STATUS2_LOCK_CONFUSE_TURN(num)((num) << 10)
#define STATUS2_MULTIPLETURNS         (1 << 12)
#define STATUS2_WRAPPED               (1 << 13 | 1 << 14 | 1 << 15)
#define STATUS2_WRAPPED_TURN(num)     ((num) << 13)
#define STATUS2_INFATUATION           (1 << 16 | 1 << 17 | 1 << 18 | 1 << 19)
#define STATUS2_INFATUATED_WITH(battler) ((1 << battler) << 16)
#define STATUS2_FOCUS_ENERGY          (1 << 20)
#define STATUS2_TRANSFORMED           (1 << 21)
#define STATUS2_RECHARGE              (1 << 22)
#define STATUS2_RAGE                  (1 << 23)
#define STATUS2_SUBSTITUTE            (1 << 24)
#define STATUS2_DESTINY_BOND          (1 << 25)
#define STATUS2_ESCAPE_PREVENTION     (1 << 26)
#define STATUS2_NIGHTMARE             (1 << 27)
#define STATUS2_CURSED                (1 << 28)
#define STATUS2_FORESIGHT             (1 << 29)
#define STATUS2_DEFENSE_CURL          (1 << 30)
#define STATUS2_TORMENT               (1 << 31)

// Status 3
#define STATUS3_LEECHSEED_BATTLER       (1 << 0 | 1 << 1)
#define STATUS3_LEECHSEED               (1 << 2)
#define STATUS3_ALWAYS_HITS             (1 << 3 | 1 << 4)
#define STATUS3_ALWAYS_HITS_TURN(num)   (((num) << 3) & STATUS3_ALWAYS_HITS)
#define STATUS3_PERISH_SONG             (1 << 5)
#define STATUS3_ON_AIR                  (1 << 6)
#define STATUS3_UNDERGROUND             (1 << 7)
#define STATUS3_MINIMIZED               (1 << 8)
#define STATUS3_CHARGED_UP              (1 << 9)
#define STATUS3_ROOTED                  (1 << 10)
#define STATUS3_YAWN                    (1 << 11 | 1 << 12)
#define STATUS3_YAWN_TURN(num)          (((num) << 11) & STATUS3_YAWN)
#define STATUS3_IMPRISONED_OTHERS       (1 << 13)
#define STATUS3_GRUDGE                  (1 << 14)
#define STATUS3_CANT_SCORE_A_CRIT       (1 << 15)
#define STATUS3_MUDSPORT                (1 << 16)
#define STATUS3_WATERSPORT              (1 << 17)
#define STATUS3_UNDERWATER              (1 << 18)
#define STATUS3_INTIMIDATE_POKES        (1 << 19)
#define STATUS3_TRACE                   (1 << 20)
#define STATUS3_SEMI_INVULNERABLE       (STATUS3_UNDERGROUND | STATUS3_ON_AIR | STATUS3_UNDERWATER)

// Side Status
#define SIDE_STATUS_REFLECT          (1 << 0)
#define SIDE_STATUS_LIGHTSCREEN      (1 << 1)
#define SIDE_STATUS_X4               (1 << 2)
#define SIDE_STATUS_SPIKES           (1 << 4)
#define SIDE_STATUS_SAFEGUARD        (1 << 5)
#define SIDE_STATUS_FUTUREATTACK     (1 << 6)
#define SIDE_STATUS_MIST             (1 << 8)
#define SIDE_STATUS_SPIKES_DAMAGED   (1 << 9)

// AI Script Mappings
// Types
#define TYPE_NORMAL   0
#define TYPE_FIGHTING 1
#define TYPE_FLYING   2
#define TYPE_POISON   3
#define TYPE_GROUND   4
#define TYPE_ROCK     5
#define TYPE_BUG      6
#define TYPE_GHOST    7
#define TYPE_STEEL    8
#define TYPE_MYSTERY  9
#define TYPE_FIRE     10
#define TYPE_WATER    11
#define TYPE_GRASS    12
#define TYPE_ELECTRIC 13
#define TYPE_PSYCHIC  14
#define TYPE_ICE      15
#define TYPE_DRAGON   16
#define TYPE_DARK     17

// Stats (BattleStat mapping)

#define DEFAULT_STAT_STAGE 6
#define MAX_STAT_STAGE 12
#define MIN_STAT_STAGE 0

// Gender Constants
#define MON_MALE       0x00
#define MON_FEMALE     0xFE
#define MON_GENDERLESS 0xFF

// Effects (Auto-generated from types.hpp)
#define EFFECT_SPECIAL_DEFENSE_UP 199 // Missing in types.hpp

