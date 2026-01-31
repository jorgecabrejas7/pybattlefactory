#include "data.hpp"

namespace pkmn {

// Move data array - extracted from battle_moves.h
// Format: {Power, Accuracy, PP, Type, Effect, EffectChance, Priority, IsPhysical, MakesContact}
// Power: 0 for status moves, 1 for variable-power moves (OHKO, Seismic Toss, etc.)
// Accuracy: 0 means always-hit moves (Swift, Aerial Ace, etc.)
// Priority: -7 to +5 (e.g., -6=Roar, -5=Counter, +1=Quick Attack, +3=Protect, +5=Helping Hand)
// IsPhysical: true for physical, false for special (Gen 3 uses type-based split)
// MakesContact: true if the move makes physical contact
static const MoveData MOVE_DATA[] = {
    // MOVE_NONE (0)
    {0, 0, 0, Type::Normal, MoveEffect::HIT, 0, 0, false, false},
    // MOVE_POUND (1)
    {40, 100, 35, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_KARATE_CHOP (2)
    {50, 100, 25, Type::Fighting, MoveEffect::HIGH_CRITICAL, 0, 0, true, true},
    // MOVE_DOUBLE_SLAP (3)
    {15, 85, 10, Type::Normal, MoveEffect::MULTI_HIT, 0, 0, true, true},
    // MOVE_COMET_PUNCH (4)
    {18, 85, 15, Type::Normal, MoveEffect::MULTI_HIT, 0, 0, true, true},
    // MOVE_MEGA_PUNCH (5)
    {80, 85, 20, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_PAY_DAY (6)
    {40, 100, 20, Type::Normal, MoveEffect::PAY_DAY, 100, 0, false, false},
    // MOVE_FIRE_PUNCH (7)
    {75, 100, 15, Type::Fire, MoveEffect::BURN_HIT, 10, 0, true, true},
    // MOVE_ICE_PUNCH (8)
    {75, 100, 15, Type::Ice, MoveEffect::FREEZE_HIT, 10, 0, true, true},
    // MOVE_THUNDER_PUNCH (9)
    {75, 100, 15, Type::Electric, MoveEffect::PARALYZE_HIT, 10, 0, true, true},
    // MOVE_SCRATCH (10)
    {40, 100, 35, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_VICE_GRIP (11)
    {55, 100, 30, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_GUILLOTINE (12)
    {1, 30, 5, Type::Normal, MoveEffect::OHKO, 0, 0, true, true},
    // MOVE_RAZOR_WIND (13)
    {80, 100, 10, Type::Normal, MoveEffect::RAZOR_WIND, 0, 0, false, false},
    // MOVE_SWORDS_DANCE (14)
    {0, 0, 30, Type::Normal, MoveEffect::ATTACK_UP_2, 0, 0, false, false},
    // MOVE_CUT (15)
    {50, 95, 30, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_GUST (16)
    {40, 100, 35, Type::Flying, MoveEffect::GUST, 0, 0, false, false},
    // MOVE_WING_ATTACK (17)
    {60, 100, 35, Type::Flying, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_WHIRLWIND (18)
    {0, 100, 20, Type::Normal, MoveEffect::ROAR, 0, -6, false, false},
    // MOVE_FLY (19)
    {70, 95, 15, Type::Flying, MoveEffect::SEMI_INVULNERABLE, 0, 0, true, true},
    // MOVE_BIND (20)
    {15, 75, 20, Type::Normal, MoveEffect::TRAP, 100, 0, true, true},
    // MOVE_SLAM (21)
    {80, 75, 20, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_VINE_WHIP (22)
    {35, 100, 10, Type::Grass, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_STOMP (23)
    {65, 100, 20, Type::Normal, MoveEffect::FLINCH_MINIMIZE_HIT, 30, 0, true, true},
    // MOVE_DOUBLE_KICK (24)
    {30, 100, 30, Type::Fighting, MoveEffect::DOUBLE_HIT, 0, 0, true, true},
    // MOVE_MEGA_KICK (25)
    {120, 75, 5, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_JUMP_KICK (26)
    {70, 95, 25, Type::Fighting, MoveEffect::RECOIL_IF_MISS, 0, 0, true, true},
    // MOVE_ROLLING_KICK (27)
    {60, 85, 15, Type::Fighting, MoveEffect::FLINCH_HIT, 30, 0, true, true},
    // MOVE_SAND_ATTACK (28)
    {0, 100, 15, Type::Ground, MoveEffect::ACCURACY_DOWN, 0, 0, false, false},
    // MOVE_HEADBUTT (29)
    {70, 100, 15, Type::Normal, MoveEffect::FLINCH_HIT, 30, 0, true, true},
    // MOVE_HORN_ATTACK (30)
    {65, 100, 25, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_FURY_ATTACK (31)
    {15, 85, 20, Type::Normal, MoveEffect::MULTI_HIT, 0, 0, true, true},
    // MOVE_HORN_DRILL (32)
    {1, 30, 5, Type::Normal, MoveEffect::OHKO, 0, 0, true, true},
    // MOVE_TACKLE (33)
    {35, 95, 35, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_BODY_SLAM (34)
    {85, 100, 15, Type::Normal, MoveEffect::PARALYZE_HIT, 30, 0, true, true},
    // MOVE_WRAP (35)
    {15, 85, 20, Type::Normal, MoveEffect::TRAP, 100, 0, true, true},
    // MOVE_TAKE_DOWN (36)
    {90, 85, 20, Type::Normal, MoveEffect::RECOIL, 0, 0, true, true},
    // MOVE_THRASH (37)
    {90, 100, 20, Type::Normal, MoveEffect::RAMPAGE, 100, 0, true, true},
    // MOVE_DOUBLE_EDGE (38)
    {120, 100, 15, Type::Normal, MoveEffect::DOUBLE_EDGE, 0, 0, true, true},
    // MOVE_TAIL_WHIP (39)
    {0, 100, 30, Type::Normal, MoveEffect::DEFENSE_DOWN, 0, 0, false, false},
    // MOVE_POISON_STING (40)
    {15, 100, 35, Type::Poison, MoveEffect::POISON_HIT, 30, 0, false, false},
    // MOVE_TWINEEDLE (41)
    {25, 100, 20, Type::Bug, MoveEffect::TWINEEDLE, 20, 0, false, false},
    // MOVE_PIN_MISSILE (42)
    {14, 85, 20, Type::Bug, MoveEffect::MULTI_HIT, 0, 0, false, false},
    // MOVE_LEER (43)
    {0, 100, 30, Type::Normal, MoveEffect::DEFENSE_DOWN, 0, 0, false, false},
    // MOVE_BITE (44)
    {60, 100, 25, Type::Dark, MoveEffect::FLINCH_HIT, 30, 0, true, true},
    // MOVE_GROWL (45)
    {0, 100, 40, Type::Normal, MoveEffect::ATTACK_DOWN, 0, 0, false, false},
    // MOVE_ROAR (46)
    {0, 100, 20, Type::Normal, MoveEffect::ROAR, 0, -6, false, false},
    // MOVE_SING (47)
    {0, 55, 15, Type::Normal, MoveEffect::SLEEP, 0, 0, false, false},
    // MOVE_SUPERSONIC (48)
    {0, 55, 20, Type::Normal, MoveEffect::CONFUSE, 0, 0, false, false},
    // MOVE_SONIC_BOOM (49)
    {1, 90, 20, Type::Normal, MoveEffect::SONICBOOM, 0, 0, false, false},
    // MOVE_DISABLE (50)
    {0, 55, 20, Type::Normal, MoveEffect::DISABLE, 0, 0, false, false},
    // MOVE_ACID (51)
    {40, 100, 30, Type::Poison, MoveEffect::DEFENSE_DOWN_HIT, 10, 0, false, false},
    // MOVE_EMBER (52)
    {40, 100, 25, Type::Fire, MoveEffect::BURN_HIT, 10, 0, false, false},
    // MOVE_FLAMETHROWER (53)
    {95, 100, 15, Type::Fire, MoveEffect::BURN_HIT, 10, 0, false, false},
    // MOVE_MIST (54)
    {0, 0, 30, Type::Ice, MoveEffect::MIST, 0, 0, false, false},
    // MOVE_WATER_GUN (55)
    {40, 100, 25, Type::Water, MoveEffect::HIT, 0, 0, false, false},
    // MOVE_HYDRO_PUMP (56)
    {120, 80, 5, Type::Water, MoveEffect::HIT, 0, 0, false, false},
    // MOVE_SURF (57)
    {95, 100, 15, Type::Water, MoveEffect::HIT, 0, 0, false, false},
    // MOVE_ICE_BEAM (58)
    {95, 100, 10, Type::Ice, MoveEffect::FREEZE_HIT, 10, 0, false, false},
    // MOVE_BLIZZARD (59)
    {120, 70, 5, Type::Ice, MoveEffect::FREEZE_HIT, 10, 0, false, false},
    // MOVE_PSYBEAM (60)
    {65, 100, 20, Type::Psychic, MoveEffect::CONFUSE_HIT, 10, 0, false, false},
    // MOVE_BUBBLE_BEAM (61)
    {65, 100, 20, Type::Water, MoveEffect::SPEED_DOWN_HIT, 10, 0, false, false},
    // MOVE_AURORA_BEAM (62)
    {65, 100, 20, Type::Ice, MoveEffect::ATTACK_DOWN_HIT, 10, 0, false, false},
    // MOVE_HYPER_BEAM (63)
    {150, 90, 5, Type::Normal, MoveEffect::RECHARGE, 0, 0, false, false},
    // MOVE_PECK (64)
    {35, 100, 35, Type::Flying, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_DRILL_PECK (65)
    {80, 100, 20, Type::Flying, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_SUBMISSION (66)
    {80, 80, 25, Type::Fighting, MoveEffect::RECOIL, 0, 0, true, true},
    // MOVE_LOW_KICK (67)
    {1, 100, 20, Type::Fighting, MoveEffect::LOW_KICK, 0, 0, true, true},
    // MOVE_COUNTER (68)
    {1, 100, 20, Type::Fighting, MoveEffect::COUNTER, 0, -5, true, true},
    // MOVE_SEISMIC_TOSS (69)
    {1, 100, 20, Type::Fighting, MoveEffect::LEVEL_DAMAGE, 0, 0, true, true},
    // MOVE_STRENGTH (70)
    {80, 100, 15, Type::Normal, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_ABSORB (71)
    {20, 100, 20, Type::Grass, MoveEffect::ABSORB, 0, 0, false, false},
    // MOVE_MEGA_DRAIN (72)
    {40, 100, 10, Type::Grass, MoveEffect::ABSORB, 0, 0, false, false},
    // MOVE_LEECH_SEED (73)
    {0, 90, 10, Type::Grass, MoveEffect::LEECH_SEED, 0, 0, false, false},
    // MOVE_GROWTH (74)
    {0, 0, 40, Type::Normal, MoveEffect::SPECIAL_ATTACK_UP, 0, 0, false, false},
    // MOVE_RAZOR_LEAF (75)
    {55, 95, 25, Type::Grass, MoveEffect::HIGH_CRITICAL, 0, 0, false, false},
    // MOVE_SOLAR_BEAM (76)
    {120, 100, 10, Type::Grass, MoveEffect::SOLAR_BEAM, 0, 0, false, false},
    // MOVE_POISON_POWDER (77)
    {0, 75, 35, Type::Poison, MoveEffect::POISON, 0, 0, false, false},
    // MOVE_STUN_SPORE (78)
    {0, 75, 30, Type::Grass, MoveEffect::PARALYZE, 0, 0, false, false},
    // MOVE_SLEEP_POWDER (79)
    {0, 75, 15, Type::Grass, MoveEffect::SLEEP, 0, 0, false, false},
    // MOVE_PETAL_DANCE (80)
    {70, 100, 20, Type::Grass, MoveEffect::RAMPAGE, 100, 0, true, true},
    // MOVE_STRING_SHOT (81)
    {0, 95, 40, Type::Bug, MoveEffect::SPEED_DOWN, 0, 0, false, false},
    // MOVE_DRAGON_RAGE (82)
    {1, 100, 10, Type::Dragon, MoveEffect::DRAGON_RAGE, 0, 0, false, false},
    // MOVE_FIRE_SPIN (83)
    {15, 70, 15, Type::Fire, MoveEffect::TRAP, 100, 0, false, false},
    // MOVE_THUNDER_SHOCK (84)
    {40, 100, 30, Type::Electric, MoveEffect::PARALYZE_HIT, 10, 0, false, false},
    // MOVE_THUNDERBOLT (85)
    {95, 100, 15, Type::Electric, MoveEffect::PARALYZE_HIT, 10, 0, false, false},
    // MOVE_THUNDER_WAVE (86)
    {0, 100, 20, Type::Electric, MoveEffect::PARALYZE, 0, 0, false, false},
    // MOVE_THUNDER (87)
    {120, 70, 10, Type::Electric, MoveEffect::THUNDER, 30, 0, false, false},
    // MOVE_ROCK_THROW (88)
    {50, 90, 15, Type::Rock, MoveEffect::HIT, 0, 0, false, false},
    // MOVE_EARTHQUAKE (89)
    {100, 100, 10, Type::Ground, MoveEffect::EARTHQUAKE, 0, 0, false, false},
    // MOVE_FISSURE (90)
    {1, 30, 5, Type::Ground, MoveEffect::OHKO, 0, 0, false, false},
    // MOVE_DIG (91)
    {60, 100, 10, Type::Ground, MoveEffect::SEMI_INVULNERABLE, 0, 0, true, true},
    // MOVE_TOXIC (92)
    {0, 85, 10, Type::Poison, MoveEffect::TOXIC, 100, 0, false, false},
    // MOVE_CONFUSION (93)
    {50, 100, 25, Type::Psychic, MoveEffect::CONFUSE_HIT, 10, 0, false, false},
    // MOVE_PSYCHIC (94)
    {90, 100, 10, Type::Psychic, MoveEffect::SPECIAL_DEFENSE_DOWN_HIT, 10, 0, false, false},
    // MOVE_HYPNOSIS (95)
    {0, 60, 20, Type::Psychic, MoveEffect::SLEEP, 0, 0, false, false},
    // MOVE_MEDITATE (96)
    {0, 0, 40, Type::Psychic, MoveEffect::ATTACK_UP, 0, 0, false, false},
    // MOVE_AGILITY (97)
    {0, 0, 30, Type::Psychic, MoveEffect::SPEED_UP_2, 0, 0, false, false},
    // MOVE_QUICK_ATTACK (98)
    {40, 100, 30, Type::Normal, MoveEffect::QUICK_ATTACK, 0, 1, true, true},
    // MOVE_RAGE (99)
    {20, 100, 20, Type::Normal, MoveEffect::RAGE, 0, 0, true, true},
    // MOVE_TELEPORT (100)
    {0, 0, 20, Type::Psychic, MoveEffect::TELEPORT, 0, 0, false, false},
    // MOVE_NIGHT_SHADE (101)
    {1, 100, 15, Type::Ghost, MoveEffect::LEVEL_DAMAGE, 0, 0, false, false},
    // MOVE_MIMIC (102)
    {0, 100, 10, Type::Normal, MoveEffect::MIMIC, 0, 0, false, false},
    // MOVE_SCREECH (103)
    {0, 85, 40, Type::Normal, MoveEffect::DEFENSE_DOWN_2, 0, 0, false, false},
    // MOVE_DOUBLE_TEAM (104)
    {0, 0, 15, Type::Normal, MoveEffect::EVASION_UP, 0, 0, false, false},
    // MOVE_RECOVER (105)
    {0, 0, 20, Type::Normal, MoveEffect::RESTORE_HP, 0, 0, false, false},
    // MOVE_HARDEN (106)
    {0, 0, 30, Type::Normal, MoveEffect::DEFENSE_UP, 0, 0, false, false},
    // MOVE_MINIMIZE (107)
    {0, 0, 20, Type::Normal, MoveEffect::MINIMIZE, 0, 0, false, false},
    // MOVE_SMOKESCREEN (108)
    {0, 100, 20, Type::Normal, MoveEffect::ACCURACY_DOWN, 0, 0, false, false},
    // MOVE_CONFUSE_RAY (109)
    {0, 100, 10, Type::Ghost, MoveEffect::CONFUSE, 0, 0, false, false},
    // MOVE_WITHDRAW (110)
    {0, 0, 40, Type::Water, MoveEffect::DEFENSE_UP, 0, 0, false, false},
    // MOVE_DEFENSE_CURL (111)
    {0, 0, 40, Type::Normal, MoveEffect::DEFENSE_CURL, 0, 0, false, false},
    // MOVE_BARRIER (112)
    {0, 0, 30, Type::Psychic, MoveEffect::DEFENSE_UP_2, 0, 0, false, false},
    // MOVE_LIGHT_SCREEN (113)
    {0, 0, 30, Type::Psychic, MoveEffect::LIGHT_SCREEN, 0, 0, false, false},
    // MOVE_HAZE (114)
    {0, 0, 30, Type::Ice, MoveEffect::HAZE, 0, 0, false, false},
    // MOVE_REFLECT (115)
    {0, 0, 20, Type::Psychic, MoveEffect::REFLECT, 0, 0, false, false},
    // MOVE_FOCUS_ENERGY (116)
    {0, 0, 30, Type::Normal, MoveEffect::FOCUS_ENERGY, 0, 0, false, false},
    // MOVE_BIDE (117)
    {1, 100, 10, Type::Normal, MoveEffect::BIDE, 0, 0, true, true},
    // MOVE_METRONOME (118)
    {0, 0, 10, Type::Normal, MoveEffect::METRONOME, 0, 0, false, false},
    // MOVE_MIRROR_MOVE (119)
    {0, 0, 20, Type::Flying, MoveEffect::MIRROR_MOVE, 0, 0, false, false},
    // MOVE_SELF_DESTRUCT (120)
    {200, 100, 5, Type::Normal, MoveEffect::EXPLOSION, 0, 0, false, false},
    // MOVE_EGG_BOMB (121)
    {100, 75, 10, Type::Normal, MoveEffect::HIT, 0, 0, false, false},
    // MOVE_LICK (122)
    {20, 100, 30, Type::Ghost, MoveEffect::PARALYZE_HIT, 30, 0, true, true},
    // MOVE_SMOG (123)
    {20, 70, 20, Type::Poison, MoveEffect::POISON_HIT, 40, 0, false, false},
    // MOVE_SLUDGE (124)
    {65, 100, 20, Type::Poison, MoveEffect::POISON_HIT, 30, 0, false, false},
    // MOVE_BONE_CLUB (125)
    {65, 85, 20, Type::Ground, MoveEffect::FLINCH_HIT, 10, 0, false, false},
    // MOVE_FIRE_BLAST (126)
    {120, 85, 5, Type::Fire, MoveEffect::BURN_HIT, 10, 0, false, false},
    // MOVE_WATERFALL (127)
    {80, 100, 15, Type::Water, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_CLAMP (128)
    {35, 75, 10, Type::Water, MoveEffect::TRAP, 100, 0, true, true},
    // MOVE_SWIFT (129)
    {60, 0, 20, Type::Normal, MoveEffect::ALWAYS_HIT, 0, 0, false, false},
    // MOVE_SKULL_BASH (130)
    {100, 100, 15, Type::Normal, MoveEffect::SKULL_BASH, 0, 0, true, true},
    // MOVE_SPIKE_CANNON (131)
    {20, 100, 15, Type::Normal, MoveEffect::MULTI_HIT, 0, 0, false, false},
    // MOVE_CONSTRICT (132)
    {10, 100, 35, Type::Normal, MoveEffect::SPEED_DOWN_HIT, 10, 0, true, true},
    // MOVE_AMNESIA (133)
    {0, 0, 20, Type::Psychic, MoveEffect::SPECIAL_DEFENSE_UP_2, 0, 0, false, false},
    // MOVE_KINESIS (134)
    {0, 80, 15, Type::Psychic, MoveEffect::ACCURACY_DOWN, 0, 0, false, false},
    // MOVE_SOFT_BOILED (135)
    {0, 100, 10, Type::Normal, MoveEffect::SOFTBOILED, 0, 0, false, false},
    // MOVE_HI_JUMP_KICK (136)
    {85, 90, 20, Type::Fighting, MoveEffect::RECOIL_IF_MISS, 0, 0, true, true},
    // MOVE_GLARE (137)
    {0, 75, 30, Type::Normal, MoveEffect::PARALYZE, 0, 0, false, false},
    // MOVE_DREAM_EATER (138)
    {100, 100, 15, Type::Psychic, MoveEffect::DREAM_EATER, 0, 0, false, false},
    // MOVE_POISON_GAS (139)
    {0, 55, 40, Type::Poison, MoveEffect::POISON, 0, 0, false, false},
    // MOVE_BARRAGE (140)
    {15, 85, 20, Type::Normal, MoveEffect::MULTI_HIT, 0, 0, false, false},
    // MOVE_LEECH_LIFE (141)
    {20, 100, 15, Type::Bug, MoveEffect::ABSORB, 0, 0, true, true},
    // MOVE_LOVELY_KISS (142)
    {0, 75, 10, Type::Normal, MoveEffect::SLEEP, 0, 0, false, false},
    // MOVE_SKY_ATTACK (143)
    {140, 90, 5, Type::Flying, MoveEffect::SKY_ATTACK, 30, 0, false, false},
    // MOVE_TRANSFORM (144)
    {0, 0, 10, Type::Normal, MoveEffect::TRANSFORM, 0, 0, false, false},
    // MOVE_BUBBLE (145)
    {20, 100, 30, Type::Water, MoveEffect::SPEED_DOWN_HIT, 10, 0, false, false},
    // MOVE_DIZZY_PUNCH (146)
    {70, 100, 10, Type::Normal, MoveEffect::CONFUSE_HIT, 20, 0, true, true},
    // MOVE_SPORE (147)
    {0, 100, 15, Type::Grass, MoveEffect::SLEEP, 0, 0, false, false},
    // MOVE_FLASH (148)
    {0, 70, 20, Type::Normal, MoveEffect::ACCURACY_DOWN, 0, 0, false, false},
    // MOVE_PSYWAVE (149)
    {1, 80, 15, Type::Psychic, MoveEffect::PSYWAVE, 0, 0, false, false},
    // MOVE_SPLASH (150)
    {0, 0, 40, Type::Normal, MoveEffect::SPLASH, 0, 0, false, false},
    // MOVE_ACID_ARMOR (151)
    {0, 0, 40, Type::Poison, MoveEffect::DEFENSE_UP_2, 0, 0, false, false},
    // MOVE_CRABHAMMER (152)
    {90, 85, 10, Type::Water, MoveEffect::HIGH_CRITICAL, 0, 0, true, true},
    // MOVE_EXPLOSION (153)
    {250, 100, 5, Type::Normal, MoveEffect::EXPLOSION, 0, 0, false, false},
    // MOVE_FURY_SWIPES (154)
    {18, 80, 15, Type::Normal, MoveEffect::MULTI_HIT, 0, 0, true, true},
    // MOVE_BONEMERANG (155)
    {50, 90, 10, Type::Ground, MoveEffect::DOUBLE_HIT, 0, 0, false, false},
    // MOVE_REST (156)
    {0, 0, 10, Type::Psychic, MoveEffect::REST, 0, 0, false, false},
    // MOVE_ROCK_SLIDE (157)
    {75, 90, 10, Type::Rock, MoveEffect::FLINCH_HIT, 30, 0, false, false},
    // MOVE_HYPER_FANG (158)
    {80, 90, 15, Type::Normal, MoveEffect::FLINCH_HIT, 10, 0, true, true},
    // MOVE_SHARPEN (159)
    {0, 0, 30, Type::Normal, MoveEffect::ATTACK_UP, 0, 0, false, false},
    // MOVE_CONVERSION (160)
    {0, 0, 30, Type::Normal, MoveEffect::CONVERSION, 0, 0, false, false},
    // MOVE_TRI_ATTACK (161)
    {80, 100, 10, Type::Normal, MoveEffect::TRI_ATTACK, 20, 0, false, false},
    // MOVE_SUPER_FANG (162)
    {1, 90, 10, Type::Normal, MoveEffect::SUPER_FANG, 0, 0, true, true},
    // MOVE_SLASH (163)
    {70, 100, 20, Type::Normal, MoveEffect::HIGH_CRITICAL, 0, 0, true, true},
    // MOVE_SUBSTITUTE (164)
    {0, 0, 10, Type::Normal, MoveEffect::SUBSTITUTE, 0, 0, false, false},
    // MOVE_STRUGGLE (165)
    {50, 100, 1, Type::Normal, MoveEffect::RECOIL, 0, 0, true, true},
    // MOVE_SKETCH (166)
    {0, 0, 1, Type::Normal, MoveEffect::SKETCH, 0, 0, false, false},
    // MOVE_TRIPLE_KICK (167)
    {10, 90, 10, Type::Fighting, MoveEffect::TRIPLE_KICK, 0, 0, true, true},
    // MOVE_THIEF (168)
    {40, 100, 10, Type::Dark, MoveEffect::THIEF, 100, 0, true, true},
    // MOVE_SPIDER_WEB (169)
    {0, 100, 10, Type::Bug, MoveEffect::MEAN_LOOK, 0, 0, false, false},
    // MOVE_MIND_READER (170)
    {0, 100, 5, Type::Normal, MoveEffect::LOCK_ON, 0, 0, false, false},
    // MOVE_NIGHTMARE (171)
    {0, 100, 15, Type::Ghost, MoveEffect::NIGHTMARE, 0, 0, false, false},
    // MOVE_FLAME_WHEEL (172)
    {60, 100, 25, Type::Fire, MoveEffect::THAW_HIT, 10, 0, true, true},
    // MOVE_SNORE (173)
    {40, 100, 15, Type::Normal, MoveEffect::SNORE, 30, 0, false, false},
    // MOVE_CURSE (174)
    {0, 0, 10, Type::Mystery, MoveEffect::CURSE, 0, 0, false, false},
    // MOVE_FLAIL (175)
    {1, 100, 15, Type::Normal, MoveEffect::FLAIL, 0, 0, true, true},
    // MOVE_CONVERSION_2 (176)
    {0, 100, 30, Type::Normal, MoveEffect::CONVERSION_2, 0, 0, false, false},
    // MOVE_AEROBLAST (177)
    {100, 95, 5, Type::Flying, MoveEffect::HIGH_CRITICAL, 0, 0, false, false},
    // MOVE_COTTON_SPORE (178)
    {0, 85, 40, Type::Grass, MoveEffect::SPEED_DOWN_2, 0, 0, false, false},
    // MOVE_REVERSAL (179)
    {1, 100, 15, Type::Fighting, MoveEffect::FLAIL, 0, 0, true, true},
    // MOVE_SPITE (180)
    {0, 100, 10, Type::Ghost, MoveEffect::SPITE, 0, 0, false, false},
    // MOVE_POWDER_SNOW (181)
    {40, 100, 25, Type::Ice, MoveEffect::FREEZE_HIT, 10, 0, false, false},
    // MOVE_PROTECT (182)
    {0, 0, 10, Type::Normal, MoveEffect::PROTECT, 0, 3, false, false},
    // MOVE_MACH_PUNCH (183)
    {40, 100, 30, Type::Fighting, MoveEffect::QUICK_ATTACK, 0, 1, true, true},
    // MOVE_SCARY_FACE (184)
    {0, 90, 10, Type::Normal, MoveEffect::SPEED_DOWN_2, 0, 0, false, false},
    // MOVE_FAINT_ATTACK (185)
    {60, 0, 20, Type::Dark, MoveEffect::ALWAYS_HIT, 0, 0, false, false},
    // MOVE_SWEET_KISS (186)
    {0, 75, 10, Type::Normal, MoveEffect::CONFUSE, 0, 0, false, false},
    // MOVE_BELLY_DRUM (187)
    {0, 0, 10, Type::Normal, MoveEffect::BELLY_DRUM, 0, 0, false, false},
    // MOVE_SLUDGE_BOMB (188)
    {90, 100, 10, Type::Poison, MoveEffect::POISON_HIT, 30, 0, false, false},
    // MOVE_MUD_SLAP (189)
    {20, 100, 10, Type::Ground, MoveEffect::ACCURACY_DOWN_HIT, 100, 0, false, false},
    // MOVE_OCTAZOOKA (190)
    {65, 85, 10, Type::Water, MoveEffect::ACCURACY_DOWN_HIT, 50, 0, false, false},
    // MOVE_SPIKES (191)
    {0, 0, 20, Type::Ground, MoveEffect::SPIKES, 0, 0, false, false},
    // MOVE_ZAP_CANNON (192)
    {100, 50, 5, Type::Electric, MoveEffect::PARALYZE_HIT, 100, 0, false, false},
    // MOVE_FORESIGHT (193)
    {0, 100, 40, Type::Normal, MoveEffect::FORESIGHT, 0, 0, false, false},
    // MOVE_DESTINY_BOND (194)
    {0, 0, 5, Type::Ghost, MoveEffect::DESTINY_BOND, 0, 0, false, false},
    // MOVE_PERISH_SONG (195)
    {0, 0, 5, Type::Normal, MoveEffect::PERISH_SONG, 0, 0, false, false},
    // MOVE_ICY_WIND (196)
    {55, 95, 15, Type::Ice, MoveEffect::SPEED_DOWN_HIT, 100, 0, false, false},
    // MOVE_DETECT (197)
    {0, 0, 5, Type::Fighting, MoveEffect::PROTECT, 0, 3, false, false},
    // MOVE_BONE_RUSH (198)
    {25, 80, 10, Type::Ground, MoveEffect::MULTI_HIT, 0, 0, false, false},
    // MOVE_LOCK_ON (199)
    {0, 100, 5, Type::Normal, MoveEffect::LOCK_ON, 0, 0, false, false},
    // MOVE_OUTRAGE (200)
    {90, 100, 15, Type::Dragon, MoveEffect::RAMPAGE, 100, 0, true, true},
    // MOVE_SANDSTORM (201)
    {0, 0, 10, Type::Rock, MoveEffect::SANDSTORM, 0, 0, false, false},
    // MOVE_GIGA_DRAIN (202)
    {60, 100, 5, Type::Grass, MoveEffect::ABSORB, 0, 0, false, false},
    // MOVE_ENDURE (203)
    {0, 0, 10, Type::Normal, MoveEffect::ENDURE, 0, 3, false, false},
    // MOVE_CHARM (204)
    {0, 100, 20, Type::Normal, MoveEffect::ATTACK_DOWN_2, 0, 0, false, false},
    // MOVE_ROLLOUT (205)
    {30, 90, 20, Type::Rock, MoveEffect::ROLLOUT, 0, 0, true, true},
    // MOVE_FALSE_SWIPE (206)
    {40, 100, 40, Type::Normal, MoveEffect::FALSE_SWIPE, 0, 0, true, true},
    // MOVE_SWAGGER (207)
    {0, 90, 15, Type::Normal, MoveEffect::SWAGGER, 100, 0, false, false},
    // MOVE_MILK_DRINK (208)
    {0, 0, 10, Type::Normal, MoveEffect::SOFTBOILED, 0, 0, false, false},
    // MOVE_SPARK (209)
    {65, 100, 20, Type::Electric, MoveEffect::PARALYZE_HIT, 30, 0, true, true},
    // MOVE_FURY_CUTTER (210)
    {10, 95, 20, Type::Bug, MoveEffect::FURY_CUTTER, 0, 0, true, true},
    // MOVE_STEEL_WING (211)
    {70, 90, 25, Type::Steel, MoveEffect::DEFENSE_UP_HIT, 10, 0, true, true},
    // MOVE_MEAN_LOOK (212)
    {0, 100, 5, Type::Normal, MoveEffect::MEAN_LOOK, 0, 0, false, false},
    // MOVE_ATTRACT (213)
    {0, 100, 15, Type::Normal, MoveEffect::ATTRACT, 0, 0, false, false},
    // MOVE_SLEEP_TALK (214)
    {0, 0, 10, Type::Normal, MoveEffect::SLEEP_TALK, 0, 0, false, false},
    // MOVE_HEAL_BELL (215)
    {0, 0, 5, Type::Normal, MoveEffect::HEAL_BELL, 0, 0, false, false},
    // MOVE_RETURN (216)
    {1, 100, 20, Type::Normal, MoveEffect::RETURN, 0, 0, true, true},
    // MOVE_PRESENT (217)
    {1, 90, 15, Type::Normal, MoveEffect::PRESENT, 0, 0, false, false},
    // MOVE_FRUSTRATION (218)
    {1, 100, 20, Type::Normal, MoveEffect::FRUSTRATION, 0, 0, true, true},
    // MOVE_SAFEGUARD (219)
    {0, 0, 25, Type::Normal, MoveEffect::SAFEGUARD, 0, 0, false, false},
    // MOVE_PAIN_SPLIT (220)
    {0, 100, 20, Type::Normal, MoveEffect::PAIN_SPLIT, 0, 0, false, false},
    // MOVE_SACRED_FIRE (221)
    {100, 95, 5, Type::Fire, MoveEffect::THAW_HIT, 50, 0, false, false},
    // MOVE_MAGNITUDE (222)
    {1, 100, 30, Type::Ground, MoveEffect::MAGNITUDE, 0, 0, false, false},
    // MOVE_DYNAMIC_PUNCH (223)
    {100, 50, 5, Type::Fighting, MoveEffect::CONFUSE_HIT, 100, 0, true, true},
    // MOVE_MEGAHORN (224)
    {120, 85, 10, Type::Bug, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_DRAGON_BREATH (225)
    {60, 100, 20, Type::Dragon, MoveEffect::PARALYZE_HIT, 30, 0, false, false},
    // MOVE_BATON_PASS (226)
    {0, 0, 40, Type::Normal, MoveEffect::BATON_PASS, 0, 0, false, false},
    // MOVE_ENCORE (227)
    {0, 100, 5, Type::Normal, MoveEffect::ENCORE, 0, 0, false, false},
    // MOVE_PURSUIT (228)
    {40, 100, 20, Type::Dark, MoveEffect::PURSUIT, 0, 0, true, true},
    // MOVE_RAPID_SPIN (229)
    {20, 100, 40, Type::Normal, MoveEffect::RAPID_SPIN, 0, 0, true, true},
    // MOVE_SWEET_SCENT (230)
    {0, 100, 20, Type::Normal, MoveEffect::EVASION_DOWN, 0, 0, false, false},
    // MOVE_IRON_TAIL (231)
    {100, 75, 15, Type::Steel, MoveEffect::DEFENSE_DOWN_HIT, 30, 0, true, true},
    // MOVE_METAL_CLAW (232)
    {50, 95, 35, Type::Steel, MoveEffect::ATTACK_UP_HIT, 10, 0, true, true},
    // MOVE_VITAL_THROW (233)
    {70, 100, 10, Type::Fighting, MoveEffect::VITAL_THROW, 0, -1, true, true},
    // MOVE_MORNING_SUN (234)
    {0, 0, 5, Type::Normal, MoveEffect::MORNING_SUN, 0, 0, false, false},
    // MOVE_SYNTHESIS (235)
    {0, 0, 5, Type::Grass, MoveEffect::SYNTHESIS, 0, 0, false, false},
    // MOVE_MOONLIGHT (236)
    {0, 0, 5, Type::Normal, MoveEffect::MOONLIGHT, 0, 0, false, false},
    // MOVE_HIDDEN_POWER (237)
    {1, 100, 15, Type::Normal, MoveEffect::HIDDEN_POWER, 0, 0, false, false},
    // MOVE_CROSS_CHOP (238)
    {100, 80, 5, Type::Fighting, MoveEffect::HIGH_CRITICAL, 0, 0, true, true},
    // MOVE_TWISTER (239)
    {40, 100, 20, Type::Dragon, MoveEffect::TWISTER, 20, 0, false, false},
    // MOVE_RAIN_DANCE (240)
    {0, 0, 5, Type::Water, MoveEffect::RAIN_DANCE, 0, 0, false, false},
    // MOVE_SUNNY_DAY (241)
    {0, 0, 5, Type::Fire, MoveEffect::SUNNY_DAY, 0, 0, false, false},
    // MOVE_CRUNCH (242)
    {80, 100, 15, Type::Dark, MoveEffect::SPECIAL_DEFENSE_DOWN_HIT, 20, 0, true, true},
    // MOVE_MIRROR_COAT (243)
    {1, 100, 20, Type::Psychic, MoveEffect::MIRROR_COAT, 0, -5, false, false},
    // MOVE_PSYCH_UP (244)
    {0, 0, 10, Type::Normal, MoveEffect::PSYCH_UP, 0, 0, false, false},
    // MOVE_EXTREME_SPEED (245)
    {80, 100, 5, Type::Normal, MoveEffect::QUICK_ATTACK, 0, 1, true, true},
    // MOVE_ANCIENT_POWER (246)
    {60, 100, 5, Type::Rock, MoveEffect::ALL_STATS_UP_HIT, 10, 0, true, true},
    // MOVE_SHADOW_BALL (247)
    {80, 100, 15, Type::Ghost, MoveEffect::SPECIAL_DEFENSE_DOWN_HIT, 20, 0, false, false},
    // MOVE_FUTURE_SIGHT (248)
    {80, 90, 15, Type::Psychic, MoveEffect::FUTURE_SIGHT, 0, 0, false, false},
    // MOVE_ROCK_SMASH (249)
    {20, 100, 15, Type::Fighting, MoveEffect::DEFENSE_DOWN_HIT, 50, 0, true, true},
    // MOVE_WHIRLPOOL (250)
    {15, 70, 15, Type::Water, MoveEffect::TRAP, 100, 0, false, false},
    // MOVE_BEAT_UP (251)
    {10, 100, 10, Type::Dark, MoveEffect::BEAT_UP, 0, 0, false, false},
    // MOVE_FAKE_OUT (252)
    {40, 100, 10, Type::Normal, MoveEffect::FAKE_OUT, 0, 1, false, false},
    // MOVE_UPROAR (253)
    {50, 100, 10, Type::Normal, MoveEffect::UPROAR, 100, 0, false, false},
    // MOVE_STOCKPILE (254)
    {0, 0, 10, Type::Normal, MoveEffect::STOCKPILE, 0, 0, false, false},
    // MOVE_SPIT_UP (255)
    {100, 100, 10, Type::Normal, MoveEffect::SPIT_UP, 0, 0, false, false},
    // MOVE_SWALLOW (256)
    {0, 0, 10, Type::Normal, MoveEffect::SWALLOW, 0, 0, false, false},
    // MOVE_HEAT_WAVE (257)
    {100, 90, 10, Type::Fire, MoveEffect::BURN_HIT, 10, 0, false, false},
    // MOVE_HAIL (258)
    {0, 0, 10, Type::Ice, MoveEffect::HAIL, 0, 0, false, false},
    // MOVE_TORMENT (259)
    {0, 100, 15, Type::Dark, MoveEffect::TORMENT, 0, 0, false, false},
    // MOVE_FLATTER (260)
    {0, 100, 15, Type::Dark, MoveEffect::FLATTER, 0, 0, false, false},
    // MOVE_WILL_O_WISP (261)
    {0, 75, 15, Type::Fire, MoveEffect::WILL_O_WISP, 0, 0, false, false},
    // MOVE_MEMENTO (262)
    {0, 100, 10, Type::Dark, MoveEffect::MEMENTO, 0, 0, false, false},
    // MOVE_FACADE (263)
    {70, 100, 20, Type::Normal, MoveEffect::FACADE, 0, 0, true, true},
    // MOVE_FOCUS_PUNCH (264)
    {150, 100, 20, Type::Fighting, MoveEffect::FOCUS_PUNCH, 0, -3, true, true},
    // MOVE_SMELLING_SALT (265)
    {60, 100, 10, Type::Normal, MoveEffect::SMELLINGSALT, 0, 0, true, true},
    // MOVE_FOLLOW_ME (266)
    {0, 100, 20, Type::Normal, MoveEffect::FOLLOW_ME, 0, 3, false, false},
    // MOVE_NATURE_POWER (267)
    {0, 95, 20, Type::Normal, MoveEffect::NATURE_POWER, 0, 0, false, false},
    // MOVE_CHARGE (268)
    {0, 100, 20, Type::Electric, MoveEffect::CHARGE, 0, 0, false, false},
    // MOVE_TAUNT (269)
    {0, 100, 20, Type::Dark, MoveEffect::TAUNT, 0, 0, false, false},
    // MOVE_HELPING_HAND (270)
    {0, 100, 20, Type::Normal, MoveEffect::HELPING_HAND, 0, 5, false, false},
    // MOVE_TRICK (271)
    {0, 100, 10, Type::Psychic, MoveEffect::TRICK, 0, 0, false, false},
    // MOVE_ROLE_PLAY (272)
    {0, 100, 10, Type::Psychic, MoveEffect::ROLE_PLAY, 0, 0, false, false},
    // MOVE_WISH (273)
    {0, 100, 10, Type::Normal, MoveEffect::WISH, 0, 0, false, false},
    // MOVE_ASSIST (274)
    {0, 100, 20, Type::Normal, MoveEffect::ASSIST, 0, 0, false, false},
    // MOVE_INGRAIN (275)
    {0, 100, 20, Type::Grass, MoveEffect::INGRAIN, 0, 0, false, false},
    // MOVE_SUPERPOWER (276)
    {120, 100, 5, Type::Fighting, MoveEffect::SUPERPOWER, 0, 0, true, true},
    // MOVE_MAGIC_COAT (277)
    {0, 100, 15, Type::Psychic, MoveEffect::MAGIC_COAT, 0, 4, false, false},
    // MOVE_RECYCLE (278)
    {0, 100, 10, Type::Normal, MoveEffect::RECYCLE, 0, 0, false, false},
    // MOVE_REVENGE (279)
    {60, 100, 10, Type::Fighting, MoveEffect::REVENGE, 0, -4, true, true},
    // MOVE_BRICK_BREAK (280)
    {75, 100, 15, Type::Fighting, MoveEffect::BRICK_BREAK, 0, 0, true, true},
    // MOVE_YAWN (281)
    {0, 100, 10, Type::Normal, MoveEffect::YAWN, 0, 0, false, false},
    // MOVE_KNOCK_OFF (282)
    {20, 100, 20, Type::Dark, MoveEffect::KNOCK_OFF, 100, 0, true, true},
    // MOVE_ENDEAVOR (283)
    {1, 100, 5, Type::Normal, MoveEffect::ENDEAVOR, 0, 0, true, true},
    // MOVE_ERUPTION (284)
    {150, 100, 5, Type::Fire, MoveEffect::ERUPTION, 0, 0, false, false},
    // MOVE_SKILL_SWAP (285)
    {0, 100, 10, Type::Psychic, MoveEffect::SKILL_SWAP, 0, 0, false, false},
    // MOVE_IMPRISON (286)
    {0, 100, 10, Type::Psychic, MoveEffect::IMPRISON, 0, 0, false, false},
    // MOVE_REFRESH (287)
    {0, 100, 20, Type::Normal, MoveEffect::REFRESH, 0, 0, false, false},
    // MOVE_GRUDGE (288)
    {0, 100, 5, Type::Ghost, MoveEffect::GRUDGE, 0, 0, false, false},
    // MOVE_SNATCH (289)
    {0, 100, 10, Type::Dark, MoveEffect::SNATCH, 0, 4, false, false},
    // MOVE_SECRET_POWER (290)
    {70, 100, 20, Type::Normal, MoveEffect::SECRET_POWER, 30, 0, false, false},
    // MOVE_DIVE (291)
    {60, 100, 10, Type::Water, MoveEffect::SEMI_INVULNERABLE, 0, 0, true, true},
    // MOVE_ARM_THRUST (292)
    {15, 100, 20, Type::Fighting, MoveEffect::MULTI_HIT, 0, 0, true, true},
    // MOVE_CAMOUFLAGE (293)
    {0, 100, 20, Type::Normal, MoveEffect::CAMOUFLAGE, 0, 0, false, false},
    // MOVE_TAIL_GLOW (294)
    {0, 100, 20, Type::Bug, MoveEffect::SPECIAL_ATTACK_UP_2, 0, 0, false, false},
    // MOVE_LUSTER_PURGE (295)
    {70, 100, 5, Type::Psychic, MoveEffect::SPECIAL_DEFENSE_DOWN_HIT, 50, 0, false, false},
    // MOVE_MIST_BALL (296)
    {70, 100, 5, Type::Psychic, MoveEffect::SPECIAL_ATTACK_DOWN_HIT, 50, 0, false, false},
    // MOVE_FEATHER_DANCE (297)
    {0, 100, 15, Type::Flying, MoveEffect::ATTACK_DOWN_2, 0, 0, false, false},
    // MOVE_TEETER_DANCE (298)
    {0, 100, 20, Type::Normal, MoveEffect::TEETER_DANCE, 0, 0, false, false},
    // MOVE_BLAZE_KICK (299)
    {85, 90, 10, Type::Fire, MoveEffect::BLAZE_KICK, 10, 0, true, true},
    // MOVE_MUD_SPORT (300)
    {0, 100, 15, Type::Ground, MoveEffect::MUD_SPORT, 0, 0, false, false},
    // MOVE_ICE_BALL (301)
    {30, 90, 20, Type::Ice, MoveEffect::ROLLOUT, 0, 0, true, true},
    // MOVE_NEEDLE_ARM (302)
    {60, 100, 15, Type::Grass, MoveEffect::FLINCH_MINIMIZE_HIT, 30, 0, true, true},
    // MOVE_SLACK_OFF (303)
    {0, 100, 10, Type::Normal, MoveEffect::RESTORE_HP, 0, 0, false, false},
    // MOVE_HYPER_VOICE (304)
    {90, 100, 10, Type::Normal, MoveEffect::HIT, 0, 0, false, false},
    // MOVE_POISON_FANG (305)
    {50, 100, 15, Type::Poison, MoveEffect::POISON_FANG, 30, 0, true, true},
    // MOVE_CRUSH_CLAW (306)
    {75, 95, 10, Type::Normal, MoveEffect::DEFENSE_DOWN_HIT, 50, 0, true, true},
    // MOVE_BLAST_BURN (307)
    {150, 90, 5, Type::Fire, MoveEffect::RECHARGE, 0, 0, false, false},
    // MOVE_HYDRO_CANNON (308)
    {150, 90, 5, Type::Water, MoveEffect::RECHARGE, 0, 0, false, false},
    // MOVE_METEOR_MASH (309)
    {100, 85, 10, Type::Steel, MoveEffect::ATTACK_UP_HIT, 20, 0, true, true},
    // MOVE_ASTONISH (310)
    {30, 100, 15, Type::Ghost, MoveEffect::FLINCH_MINIMIZE_HIT, 30, 0, true, true},
    // MOVE_WEATHER_BALL (311)
    {50, 100, 10, Type::Normal, MoveEffect::WEATHER_BALL, 0, 0, false, false},
    // MOVE_AROMATHERAPY (312)
    {0, 0, 5, Type::Grass, MoveEffect::HEAL_BELL, 0, 0, false, false},
    // MOVE_FAKE_TEARS (313)
    {0, 100, 20, Type::Dark, MoveEffect::SPECIAL_DEFENSE_DOWN_2, 0, 0, false, false},
    // MOVE_AIR_CUTTER (314)
    {55, 95, 25, Type::Flying, MoveEffect::HIGH_CRITICAL, 0, 0, false, false},
    // MOVE_OVERHEAT (315)
    {140, 90, 5, Type::Fire, MoveEffect::OVERHEAT, 100, 0, true, true},
    // MOVE_ODOR_SLEUTH (316)
    {0, 100, 40, Type::Normal, MoveEffect::FORESIGHT, 0, 0, false, false},
    // MOVE_ROCK_TOMB (317)
    {50, 80, 10, Type::Rock, MoveEffect::SPEED_DOWN_HIT, 100, 0, false, false},
    // MOVE_SILVER_WIND (318)
    {60, 100, 5, Type::Bug, MoveEffect::ALL_STATS_UP_HIT, 10, 0, false, false},
    // MOVE_METAL_SOUND (319)
    {0, 85, 40, Type::Steel, MoveEffect::SPECIAL_DEFENSE_DOWN_2, 0, 0, false, false},
    // MOVE_GRASS_WHISTLE (320)
    {0, 55, 15, Type::Grass, MoveEffect::SLEEP, 0, 0, false, false},
    // MOVE_TICKLE (321)
    {0, 100, 20, Type::Normal, MoveEffect::TICKLE, 0, 0, false, false},
    // MOVE_COSMIC_POWER (322)
    {0, 0, 20, Type::Psychic, MoveEffect::COSMIC_POWER, 0, 0, false, false},
    // MOVE_WATER_SPOUT (323)
    {150, 100, 5, Type::Water, MoveEffect::ERUPTION, 0, 0, false, false},
    // MOVE_SIGNAL_BEAM (324)
    {75, 100, 15, Type::Bug, MoveEffect::CONFUSE_HIT, 10, 0, false, false},
    // MOVE_SHADOW_PUNCH (325)
    {60, 0, 20, Type::Ghost, MoveEffect::ALWAYS_HIT, 0, 0, true, true},
    // MOVE_EXTRASENSORY (326)
    {80, 100, 30, Type::Psychic, MoveEffect::FLINCH_MINIMIZE_HIT, 10, 0, false, false},
    // MOVE_SKY_UPPERCUT (327)
    {85, 90, 15, Type::Fighting, MoveEffect::SKY_UPPERCUT, 0, 0, true, true},
    // MOVE_SAND_TOMB (328)
    {15, 70, 15, Type::Ground, MoveEffect::TRAP, 100, 0, false, false},
    // MOVE_SHEER_COLD (329)
    {1, 30, 5, Type::Ice, MoveEffect::OHKO, 0, 0, false, false},
    // MOVE_MUDDY_WATER (330)
    {95, 85, 10, Type::Water, MoveEffect::ACCURACY_DOWN_HIT, 30, 0, false, false},
    // MOVE_BULLET_SEED (331)
    {10, 100, 30, Type::Grass, MoveEffect::MULTI_HIT, 0, 0, false, false},
    // MOVE_AERIAL_ACE (332)
    {60, 0, 20, Type::Flying, MoveEffect::ALWAYS_HIT, 0, 0, true, true},
    // MOVE_ICICLE_SPEAR (333)
    {10, 100, 30, Type::Ice, MoveEffect::MULTI_HIT, 0, 0, false, false},
    // MOVE_IRON_DEFENSE (334)
    {0, 0, 15, Type::Steel, MoveEffect::DEFENSE_UP_2, 0, 0, false, false},
    // MOVE_BLOCK (335)
    {0, 100, 5, Type::Normal, MoveEffect::MEAN_LOOK, 0, 0, false, false},
    // MOVE_HOWL (336)
    {0, 0, 40, Type::Normal, MoveEffect::ATTACK_UP, 0, 0, false, false},
    // MOVE_DRAGON_CLAW (337)
    {80, 100, 15, Type::Dragon, MoveEffect::HIT, 0, 0, true, true},
    // MOVE_FRENZY_PLANT (338)
    {150, 90, 5, Type::Grass, MoveEffect::RECHARGE, 0, 0, false, false},
    // MOVE_BULK_UP (339)
    {0, 0, 20, Type::Fighting, MoveEffect::BULK_UP, 0, 0, false, false},
    // MOVE_BOUNCE (340)
    {85, 85, 5, Type::Flying, MoveEffect::SEMI_INVULNERABLE, 30, 0, true, true},
    // MOVE_MUD_SHOT (341)
    {55, 95, 15, Type::Ground, MoveEffect::SPEED_DOWN_HIT, 100, 0, false, false},
    // MOVE_POISON_TAIL (342)
    {50, 100, 25, Type::Poison, MoveEffect::POISON_TAIL, 10, 0, true, true},
    // MOVE_COVET (343)
    {40, 100, 40, Type::Normal, MoveEffect::THIEF, 100, 0, false, false},
    // MOVE_VOLT_TACKLE (344)
    {120, 100, 15, Type::Electric, MoveEffect::DOUBLE_EDGE, 0, 0, true, true},
    // MOVE_MAGICAL_LEAF (345)
    {60, 0, 20, Type::Grass, MoveEffect::ALWAYS_HIT, 0, 0, false, false},
    // MOVE_WATER_SPORT (346)
    {0, 100, 15, Type::Water, MoveEffect::WATER_SPORT, 0, 0, false, false},
    // MOVE_CALM_MIND (347)
    {0, 0, 20, Type::Psychic, MoveEffect::CALM_MIND, 0, 0, false, false},
    // MOVE_LEAF_BLADE (348)
    {70, 100, 15, Type::Grass, MoveEffect::HIGH_CRITICAL, 0, 0, true, true},
    // MOVE_DRAGON_DANCE (349)
    {0, 0, 20, Type::Dragon, MoveEffect::DRAGON_DANCE, 0, 0, false, false},
    // MOVE_ROCK_BLAST (350)
    {25, 80, 10, Type::Rock, MoveEffect::MULTI_HIT, 0, 0, false, false},
    // MOVE_SHOCK_WAVE (351)
    {60, 0, 20, Type::Electric, MoveEffect::ALWAYS_HIT, 0, 0, false, false},
    // MOVE_WATER_PULSE (352)
    {60, 100, 20, Type::Water, MoveEffect::CONFUSE_HIT, 20, 0, false, false},
    // MOVE_DOOM_DESIRE (353)
    {120, 85, 5, Type::Steel, MoveEffect::FUTURE_SIGHT, 0, 0, false, false},
    // MOVE_PSYCHO_BOOST (354)
    {140, 90, 5, Type::Psychic, MoveEffect::OVERHEAT, 100, 0, false, false},
};

const MoveData& getMoveData(uint16_t moveId) {
    if (moveId >= 355) return MOVE_DATA[0];
    return MOVE_DATA[moveId];
}

}  // namespace pkmn
