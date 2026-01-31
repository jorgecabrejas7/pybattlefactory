#include "data.hpp"

namespace pkmn {

// Type effectiveness chart (Gen 3)
// Rows = attacking type, Columns = defending type
// Values: 0 = immune, 50 = not very effective, 100 = normal, 200 = super effective
// Order: Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost, Steel, ???, Fire, Water, Grass, Electric, Psychic, Ice, Dragon, Dark
static constexpr int TYPE_CHART[18][18] = {
    // Nor  Fig  Fly  Poi  Gro  Roc  Bug  Gho  Ste  ???  Fir  Wat  Gra  Ele  Psy  Ice  Dra  Dar
    { 100, 100, 100, 100, 100,  50, 100,   0,  50, 100, 100, 100, 100, 100, 100, 100, 100, 100}, // Normal
    { 200, 100,  50,  50, 100, 200,  50,   0, 200, 100, 100, 100, 100, 100,  50, 200, 100, 200}, // Fighting
    { 100, 200, 100, 100, 100,  50, 200, 100,  50, 100, 100, 100, 200,  50, 100, 100, 100, 100}, // Flying
    { 100, 100, 100,  50,  50,  50, 100,  50,   0, 100, 100, 100, 200, 100, 100, 100, 100, 100}, // Poison
    { 100, 100,   0, 200, 100, 200,  50, 100, 200, 100, 200, 100,  50, 200, 100, 100, 100, 100}, // Ground
    { 100,  50, 200, 100,  50, 100, 200, 100,  50, 100, 200, 100, 100, 100, 100, 200, 100, 100}, // Rock
    { 100,  50,  50,  50, 100, 100, 100,  50,  50, 100,  50, 100, 200, 100, 200, 100, 100, 200}, // Bug
    {   0, 100, 100, 100, 100, 100, 100, 200,  50, 100, 100, 100, 100, 100, 200, 100, 100,  50}, // Ghost
    { 100, 100, 100, 100, 100, 200, 100, 100,  50, 100,  50,  50, 100,  50, 100, 200, 100, 100}, // Steel
    { 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100}, // ??? (Mystery)
    { 100, 100, 100, 100, 100,  50, 200, 100, 200, 100,  50,  50, 200, 100, 100, 200,  50, 100}, // Fire
    { 100, 100, 100, 100, 200, 200, 100, 100, 100, 100, 200,  50,  50, 100, 100, 100,  50, 100}, // Water
    { 100, 100,  50,  50, 200, 200,  50, 100,  50, 100,  50, 200,  50, 100, 100, 100,  50, 100}, // Grass
    { 100, 100, 200, 100,   0, 100, 100, 100, 100, 100, 100, 200,  50,  50, 100, 100,  50, 100}, // Electric
    { 100, 200, 100, 200, 100, 100, 100, 100,  50, 100, 100, 100, 100, 100,  50, 100, 100,   0}, // Psychic
    { 100, 100, 200, 100, 200, 100, 100, 100,  50, 100,  50,  50, 200, 100, 100,  50, 200, 100}, // Ice
    { 100, 100, 100, 100, 100, 100, 100, 100,  50, 100, 100, 100, 100, 100, 100, 100, 200, 100}, // Dragon
    { 100,  50, 100, 100, 100, 100, 100, 200,  50, 100, 100, 100, 100, 100, 200, 100, 100,  50}, // Dark
};

int getTypeEffectiveness(Type attackType, Type defendType) {
    return TYPE_CHART[static_cast<int>(attackType)][static_cast<int>(defendType)];
}

int getTypeEffectivenessDual(Type attackType, Type defType1, Type defType2) {
    int eff1 = getTypeEffectiveness(attackType, defType1);
    if (defType1 == defType2) {
        return eff1;
    }
    int eff2 = getTypeEffectiveness(attackType, defType2);
    // Multiply and normalize: (eff1/100) * (eff2/100) * 100 = eff1 * eff2 / 100
    return (eff1 * eff2) / 100;
}

// Nature modifier table
// Rows = nature, Columns = stat (Atk, Def, Spe, SpA, SpD)
// HP is not affected by nature
// Values: 90 = decrease, 100 = neutral, 110 = increase
static constexpr int NATURE_MODIFIERS[25][5] = {
    // Atk   Def   Spe   SpA   SpD
    { 100, 100, 100, 100, 100}, // Hardy
    { 110, 100, 100,  90, 100}, // Lonely
    { 110, 100,  90, 100, 100}, // Brave
    { 110,  90, 100, 100, 100}, // Adamant
    { 110, 100, 100, 100,  90}, // Naughty
    {  90, 110, 100, 100, 100}, // Bold
    { 100, 100, 100, 100, 100}, // Docile
    { 100, 110,  90, 100, 100}, // Relaxed
    { 100, 110, 100,  90, 100}, // Impish
    { 100, 110, 100, 100,  90}, // Lax
    {  90, 100, 110, 100, 100}, // Timid
    { 100,  90, 110, 100, 100}, // Hasty
    { 100, 100, 100, 100, 100}, // Serious
    { 100, 100, 110,  90, 100}, // Jolly
    { 100, 100, 110, 100,  90}, // Naive
    {  90, 100, 100, 110, 100}, // Modest
    { 100,  90, 100, 110, 100}, // Mild
    { 100, 100,  90, 110, 100}, // Quiet
    { 100, 100, 100, 100, 100}, // Bashful
    { 100, 100, 100, 110,  90}, // Rash
    {  90, 100, 100, 100, 110}, // Calm
    { 100,  90, 100, 100, 110}, // Gentle
    { 100, 100,  90, 100, 110}, // Sassy
    { 100, 100, 100,  90, 110}, // Careful
    { 100, 100, 100, 100, 100}, // Quirky
};

int getNatureModifier(Nature nature, Stat stat) {
    if (stat == Stat::HP) return 100;  // HP not affected
    // Map stat to column: Attack=0, Defense=1, Speed=2, SpAttack=3, SpDefense=4
    int col = static_cast<int>(stat) - 1;  // Stat::Attack = 1
    return NATURE_MODIFIERS[static_cast<int>(nature)][col];
}

// EV spread calculation
void calculateEVsFromSpread(uint8_t evSpread, uint8_t* outEVs) {
    // Count how many stats are flagged
    int count = 0;
    for (int i = 0; i < 6; i++) {
        if (evSpread & (1 << i)) count++;
    }
    
    if (count == 0) {
        for (int i = 0; i < 6; i++) outEVs[i] = 0;
        return;
    }
    
    // Distribute 510 EVs evenly among flagged stats
    // Max 255 per stat
    uint8_t perStat = (count <= 2) ? 255 : (510 / count);
    if (perStat > 255) perStat = 255;
    
    for (int i = 0; i < 6; i++) {
        outEVs[i] = (evSpread & (1 << i)) ? perStat : 0;
    }
}

}  // namespace pkmn
