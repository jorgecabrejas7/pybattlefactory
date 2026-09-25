#pragma once

#include <cstdint>
#include <array>

namespace pkmn {

// ============================================================================
// Pokemon Types
// ============================================================================
enum class Type : uint8_t {
    Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost, Steel,
    Mystery,  // ??? type
    Fire, Water, Grass, Electric, Psychic, Ice, Dragon, Dark,
    COUNT
};

// ============================================================================
// Status Conditions
// ============================================================================
enum class Status : uint8_t {
    None = 0,
    Sleep1, Sleep2, Sleep3, Sleep4, Sleep5, Sleep6, Sleep7,  // 1-7 turns
    Poison,
    Burn,
    Freeze,
    Paralysis,
    BadPoison,  // Toxic
};

inline bool isSleep(Status s) { return s >= Status::Sleep1 && s <= Status::Sleep7; }

// ============================================================================
// Stat Indices
// ============================================================================
enum Stat : uint8_t {
    HP = 0, Attack, Defense, Speed, SpAttack, SpDefense,
    STAT_COUNT = 6
};

enum BattleStat : uint8_t {
    ATK = 0, DEF, SPA, SPD, SPE, ACC, EVA,
    BATTLE_STAT_COUNT = 7
};

// ============================================================================
// Nature
// ============================================================================
enum class Nature : uint8_t {
    Hardy, Lonely, Brave, Adamant, Naughty,
    Bold, Docile, Relaxed, Impish, Lax,
    Timid, Hasty, Serious, Jolly, Naive,
    Modest, Mild, Quiet, Bashful, Rash,
    Calm, Gentle, Sassy, Careful, Quirky,
    COUNT = 25
};

// ============================================================================
// Move Target
// ============================================================================
enum class MoveTarget : uint8_t {
    Selected,       // Single target
    User,           // Self
    AllOpponents,   // Hits all opponents
    AllExceptUser,  // Earthquake, etc.
    // ... add more as needed
};

// ============================================================================
// Species Info (Base Stats)
// 
// Core species data extracted from pokeemerald/src/data/pokemon/species_info.h
// This represents the immutable base stats for each species.
// ============================================================================
struct SpeciesData {
    uint8_t baseHP;        // Base HP stat (1-255)
    uint8_t baseAttack;    // Base Attack stat (1-255)
    uint8_t baseDefense;   // Base Defense stat (1-255)
    uint8_t baseSpeed;     // Base Speed stat (1-255)
    uint8_t baseSpAttack;  // Base Special Attack stat (1-255)
    uint8_t baseSpDefense; // Base Special Defense stat (1-255)
    Type type1;            // Primary type (always set)
    Type type2;            // Secondary type (same as type1 if mono-type)
    uint8_t abilities[2];  // Possible abilities [slot1, slot2]. ABILITY_NONE if no second ability
    uint8_t genderRatio;   // Gender ratio encoding:
                           //   0   = 100% male (MON_MALE)
                           //   31  = 87.5% male / 12.5% female (starters)
                           //   63  = 75% male / 25% female
                           //   127 = 50% male / 50% female
                           //   191 = 25% male / 75% female
                           //   254 = 100% female (MON_FEMALE)
                           //   255 = genderless (MON_GENDERLESS)
};

// ============================================================================
// Move Effects (auto-generated from battle_moves.h)
// ============================================================================
enum class MoveEffect : uint8_t {
#include "gen/move_effects.inc"
};

// ============================================================================
// Move Data
// 
// Core move data extracted from pokeemerald/src/data/battle_moves.h
// This represents the immutable properties of each move.
// ============================================================================
struct MoveData {
    uint8_t power;        // Base power (0 for status moves, 1 for OHKO, etc)
    uint8_t accuracy;     // Accuracy 0-100 (0 = never misses, like Swift/Aerial Ace)
    uint8_t pp;           // Base PP (before PP Ups)
    Type type;            // Move type (for STAB and effectiveness)
    MoveEffect effect;    // Effect ID - determines move behavior (see MoveEffect enum)
    uint8_t effectChance; // Chance for secondary effect (0-100, 0 = always/never applies)
    int8_t priority;      // Move priority (-7 to +5, 0 = normal)
                          //   +5: Helping Hand
                          //   +4: Magic Coat, Snatch
                          //   +3: Fake Out, Follow Me
                          //   +2: Extreme Speed
                          //   +1: Quick Attack, Mach Punch
                          //    0: Most moves
                          //   -1: Vital Throw
                          //   -5: Counter, Mirror Coat
                          //   -6: Roar, Whirlwind
    bool isPhysical;      // true = Physical (uses Attack/Defense), false = Special
    bool makesContact;    // true = triggers contact abilities (Rough Skin, etc)
    uint8_t target;       // MOVE_TARGET_* bits
    uint8_t flags;        // FLAG_* bits (contact, protect, magic coat, snatch, mirror move, king's rock)
};

// ============================================================================
// Item Data (gItems hold effects)
// ============================================================================
struct ItemData {
    uint8_t holdEffect;       // HOLD_EFFECT_*
    uint8_t holdEffectParam;
};

// ============================================================================
// Pokemon Instance (in party)
// ============================================================================
struct Pokemon {
    uint16_t species;
    uint16_t heldItem;
    uint8_t ability;
    uint8_t level;
    Nature nature;
    
    uint16_t moves[4];
    uint8_t pp[4];
    
    // IVs (0-31)
    uint8_t ivs[6];
    
    // EVs (0-255 each, 510 total max)
    uint8_t evs[6];
    
    // Current HP
    uint16_t currentHP;
    uint16_t maxHP;
    
    Status status;
    
    // Calculated stats
    uint16_t stats[6];
    
    // Calculate stats from base + IVs + EVs + nature
    void calculateStats(const SpeciesData& base);
};

// ============================================================================
// Active Battler (in-battle state)
// ============================================================================
struct ActiveMon {
    uint8_t partyIndex;  // Which party slot this mon came from
    
    // Stat stages (-6 to +6)
    int8_t statStages[BATTLE_STAT_COUNT];
    
    // Volatile status (cleared on switch)
    bool isConfused;
    uint8_t confusionTurns;
    bool isTaunted;
    uint8_t tauntTurns;
    bool isSeeded;      // Leech Seed
    bool hasSubstitute;
    uint8_t substituteHP;

    // Additional Volatile Status
    bool isFlinched;
    bool isEncored;
    uint8_t encoreTurns;
    uint16_t encoredMove;
    bool isInfatuated; // Attract
    uint8_t infatuationSource; // bitmask or side?
    bool isCursed; // Ghost Curse
    bool isNightmare;
    
    // Counters
    uint8_t toxicCounter; // For Bad Poison damage calculation
    uint8_t sleepCounter; // For Sleep Talk / Snore / Wake up
    uint8_t stockPileCount; 
    
    // Move History / Locking
    uint16_t lastMoveUsed;
    uint16_t lastMoveTarget; // For Counter/Mirror Coat
    uint16_t choiceLockedMove; // For Choice Band/Specs/Scarf
    
    // Protect/Detect tracking
    bool isProtected;
    uint8_t protectUses;  // For diminishing returns
    bool protectedThisTurn;
    bool enduredThisTurn; // Endure
    
    // Multi-Turn statuses
    bool isRecharging;      // Hyper Beam
    bool isCharging;        // Solar Beam / Skull Bash
    uint16_t chargingMove;  // The move being charged
    bool isInvulnerable;    // Fly / Dig
    uint16_t invulnerableMove; // To distinguish Fly vs Dig vs Dive
    bool isLockedIntoMove;  // Thrash / Outrage / Petal Dance
    uint8_t lockedMoveTurns;
    bool isTrapped; // Mean Look / Spider Web / Arena Trap
    
    // Damage History (Counter/Mirror Coat)
    int lastDamageTaken;
    uint16_t lastMoveTaken;
    
    // Transform
    bool isTransformed;
    uint16_t originalSpecies;
    
    // Baton Pass
    bool batonPassing;
    
    // Type changes
    Type types[2];
    bool typesOverridden;
    
    // Ability Activation Flags (e.g. Flash Fire active)
    bool isFlashFireActive;
    
    void reset();
};

// ============================================================================
// Battle State
// ============================================================================
enum class Weather : uint8_t {
    None, Sun, Rain, Sandstorm, Hail
};

struct SideState {
    bool hasReflect;
    uint8_t reflectTurns;
    bool hasLightScreen;
    uint8_t lightScreenTurns;
    bool hasSpikes;
    uint8_t spikesLayers;  // 1-3
    bool hasToxicSpikes;
    uint8_t toxicSpikesLayers;  // 1-2
    bool hasStealthRock;
};

struct BattleState {
    // Teams (0 = player, 1 = opponent)
    std::array<Pokemon, 6> teams[2];
    uint8_t teamSizes[2];
    
    // Active battlers (singles: 1 each side)
    ActiveMon active[2];
    
    // Field conditions
    Weather weather;
    uint8_t weatherTurns;
    SideState sides[2];
    
    // Turn state
    uint16_t turnNumber;
    
    // RNG state (for deterministic replay)
    uint32_t rngState;
    
    // Get the Pokemon reference for an active battler
    Pokemon& getActivePokemon(uint8_t side) {
        return teams[side][active[side].partyIndex];
    }
    const Pokemon& getActivePokemon(uint8_t side) const {
        return teams[side][active[side].partyIndex];
    }
    
    // Count remaining (non-fainted) mons
    uint8_t countRemaining(uint8_t side) const;
    
    // Check if battle is over
    bool isTerminal() const;
    int getWinner() const;  // -1 if not terminal, 0 or 1 otherwise
};

// ============================================================================
// Actions
// ============================================================================
enum class ActionType : uint8_t {
    Move1, Move2, Move3, Move4,
    Switch1, Switch2, Switch3, Switch4, Switch5,
    Struggle,  // Forced when out of PP
};

struct Action {
    ActionType type;
    
    bool isMove() const { return type <= ActionType::Move4 || type == ActionType::Struggle; }
    bool isSwitch() const { return type >= ActionType::Switch1 && type <= ActionType::Switch5; }
    uint8_t getMoveIndex() const { return static_cast<uint8_t>(type); }
    uint8_t getSwitchTarget() const { return static_cast<uint8_t>(type) - static_cast<uint8_t>(ActionType::Switch1); }
};

// ============================================================================
// Step Result
// ============================================================================
struct StepResult {
    bool done;
    int winner;  // 0 = player, 1 = opponent, -1 = not done
    float reward;
};

}  // namespace pkmn
