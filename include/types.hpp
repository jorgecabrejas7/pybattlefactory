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
enum class MoveEffect : uint16_t {
    ABSORB = 0, ACCURACY_DOWN = 1, ACCURACY_DOWN_HIT = 2, ALL_STATS_UP_HIT = 3,
    ALWAYS_HIT = 4, ASSIST = 5, ATTACK_DOWN = 6, ATTACK_DOWN_2 = 7,
    ATTACK_DOWN_HIT = 8, ATTACK_UP = 9, ATTACK_UP_2 = 10, ATTACK_UP_HIT = 11,
    ATTRACT = 12, BATON_PASS = 13, BEAT_UP = 14, BELLY_DRUM = 15,
    BIDE = 16, BLAZE_KICK = 17, BRICK_BREAK = 18, BULK_UP = 19,
    BURN_HIT = 20, CALM_MIND = 21, CAMOUFLAGE = 22, CHARGE = 23,
    CONFUSE = 24, CONFUSE_HIT = 25, CONVERSION = 26, CONVERSION_2 = 27,
    COSMIC_POWER = 28, COUNTER = 29, CURSE = 30, DEFENSE_CURL = 31,
    DEFENSE_DOWN = 32, DEFENSE_DOWN_2 = 33, DEFENSE_DOWN_HIT = 34, DEFENSE_UP = 35,
    DEFENSE_UP_2 = 36, DEFENSE_UP_HIT = 37, DESTINY_BOND = 38, DISABLE = 39,
    DOUBLE_EDGE = 40, DOUBLE_HIT = 41, DRAGON_DANCE = 42, DRAGON_RAGE = 43,
    DREAM_EATER = 44, EARTHQUAKE = 45, ENCORE = 46, ENDEAVOR = 47,
    ENDURE = 48, ERUPTION = 49, EVASION_DOWN = 50, EVASION_UP = 51,
    EXPLOSION = 52, FACADE = 53, FAKE_OUT = 54, FALSE_SWIPE = 55,
    FLAIL = 56, FLATTER = 57, FLINCH_HIT = 58, FLINCH_MINIMIZE_HIT = 59,
    FOCUS_ENERGY = 60, FOCUS_PUNCH = 61, FOLLOW_ME = 62, FORESIGHT = 63,
    FREEZE_HIT = 64, FRUSTRATION = 65, FURY_CUTTER = 66, FUTURE_SIGHT = 67,
    GRUDGE = 68, GUST = 69, HAIL = 70, HAZE = 71,
    HEAL_BELL = 72, HELPING_HAND = 73, HIDDEN_POWER = 74, HIGH_CRITICAL = 75,
    HIT = 76, IMPRISON = 77, INGRAIN = 78, KNOCK_OFF = 79,
    LEECH_SEED = 80, LEVEL_DAMAGE = 81, LIGHT_SCREEN = 82, LOCK_ON = 83,
    LOW_KICK = 84, MAGIC_COAT = 85, MAGNITUDE = 86, MEAN_LOOK = 87,
    MEMENTO = 88, METRONOME = 89, MIMIC = 90, MINIMIZE = 91,
    MIRROR_COAT = 92, MIRROR_MOVE = 93, MIST = 94, MOONLIGHT = 95,
    MORNING_SUN = 96, MUD_SPORT = 97, MULTI_HIT = 98, NATURE_POWER = 99,
    NIGHTMARE = 100, OHKO = 101, OVERHEAT = 102, PAIN_SPLIT = 103,
    PARALYZE = 104, PARALYZE_HIT = 105, PAY_DAY = 106, PERISH_SONG = 107,
    POISON = 108, POISON_FANG = 109, POISON_HIT = 110, POISON_TAIL = 111,
    PRESENT = 112, PROTECT = 113, PSYCH_UP = 114, PSYWAVE = 115,
    PURSUIT = 116, QUICK_ATTACK = 117, RAGE = 118, RAIN_DANCE = 119,
    RAMPAGE = 120, RAPID_SPIN = 121, RAZOR_WIND = 122, RECHARGE = 123,
    RECOIL = 124, RECOIL_IF_MISS = 125, RECYCLE = 126, REFLECT = 127,
    REFRESH = 128, REST = 129, RESTORE_HP = 130, RETURN = 131,
    REVENGE = 132, ROAR = 133, ROLE_PLAY = 134, ROLLOUT = 135,
    SAFEGUARD = 136, SANDSTORM = 137, SECRET_POWER = 138, SEMI_INVULNERABLE = 139,
    SKETCH = 140, SKILL_SWAP = 141, SKULL_BASH = 142, SKY_ATTACK = 143,
    SKY_UPPERCUT = 144, SLEEP = 145, SLEEP_TALK = 146, SMELLINGSALT = 147,
    SNATCH = 148, SNORE = 149, SOFTBOILED = 150, SOLAR_BEAM = 151,
    SONICBOOM = 152, SPECIAL_ATTACK_DOWN_HIT = 153, SPECIAL_ATTACK_UP = 154, SPECIAL_ATTACK_UP_2 = 155,
    SPECIAL_DEFENSE_DOWN_2 = 156, SPECIAL_DEFENSE_DOWN_HIT = 157, SPECIAL_DEFENSE_UP_2 = 158, SPEED_DOWN = 159,
    SPEED_DOWN_2 = 160, SPEED_DOWN_HIT = 161, SPEED_UP_2 = 162, SPIKES = 163,
    SPITE = 164, SPIT_UP = 165, SPLASH = 166, STOCKPILE = 167,
    SUBSTITUTE = 168, SUNNY_DAY = 169, SUPERPOWER = 170, SUPER_FANG = 171,
    SWAGGER = 172, SWALLOW = 173, SYNTHESIS = 174, TAUNT = 175,
    TEETER_DANCE = 176, TELEPORT = 177, THAW_HIT = 178, THIEF = 179,
    THUNDER = 180, TICKLE = 181, TORMENT = 182, TOXIC = 183,
    TRANSFORM = 184, TRAP = 185, TRICK = 186, TRIPLE_KICK = 187,
    TRI_ATTACK = 188, TWINEEDLE = 189, TWISTER = 190, UPROAR = 191,
    VITAL_THROW = 192, WATER_SPORT = 193, WEATHER_BALL = 194, WILL_O_WISP = 195,
    WISH = 196, YAWN = 197,
    COUNT = 198
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
    
    // Protect/Detect tracking
    uint8_t protectUses;  // For diminishing returns
    bool protectedThisTurn;
    
    // Type changes
    Type types[2];
    bool typesOverridden;
    
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
