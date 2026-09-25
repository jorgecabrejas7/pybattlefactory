#include "gen3_game.hpp"

extern "C" {
#include "gen3/host.h"
#include "gen3/search_host.h"
}

#include <stdexcept>

namespace pkmn {

namespace {

// The instance whose state currently occupies the live battle RAM section.
Gen3Game* s_live = nullptr;

const std::vector<uint8_t>& pristineState() {
    static std::vector<uint8_t> state = [] {
        Gen3_InitSystem();
        std::vector<uint8_t> s(Gen3_StateSize());
        Gen3_SaveState(s.data());
        return s;
    }();
    return state;
}

}  // namespace

Gen3Game::Gen3Game() : m_state(pristineState()) {}

Gen3Game::Gen3Game(const Gen3Game& other) {
    if (s_live == &other) Gen3_SaveState(const_cast<uint8_t*>(other.m_state.data()));
    m_state = other.m_state;
    m_determinized = other.m_determinized;
}

Gen3Game& Gen3Game::operator=(const Gen3Game& other) {
    if (this != &other) {
        if (s_live == &other) Gen3_SaveState(const_cast<uint8_t*>(other.m_state.data()));
        m_state = other.m_state;
        m_determinized = other.m_determinized;
        if (s_live == this) Gen3_LoadState(m_state.data());
    }
    return *this;
}

Gen3Game::~Gen3Game() {
    if (s_live == this) s_live = nullptr;
}

void Gen3Game::activate() {
    if (s_live == this) return;
    if (s_live) Gen3_SaveState(s_live->m_state.data());
    Gen3_LoadState(m_state.data());
    s_live = this;
}

void Gen3Game::setBattle(uint32_t flags, uint16_t trainerId) {
    activate();
    Gen3_SetBattleParams(flags, trainerId);
}

void Gen3Game::writeParty(int side, const std::string& raw) {
    activate();
    Gen3_WriteParty(side, raw.data(), raw.size());
}

void Gen3Game::writeSaveBlock2(uint32_t offset, const std::string& data) {
    activate();
    Gen3_WriteSaveBlock2(offset, data.data(), data.size());
}

std::string Gen3Game::readSaveBlock2(uint32_t offset, size_t size) {
    activate();
    std::string out(size, '\0');
    Gen3_ReadSaveBlock2(offset, out.data(), size);
    return out;
}

void Gen3Game::setFlag(uint16_t id, bool on) {
    activate();
    Gen3_SetFlag(id, on ? 1 : 0);
}

void Gen3Game::setVar(uint16_t id, uint16_t value) {
    activate();
    Gen3_SetVar(id, value);
}

bool Gen3Game::start(uint32_t rng) {
    activate();
    if (!Gen3_StartBattle(10000)) return false;
    Gen3_SetRng(rng);
    return true;
}

int Gen3Game::run(uint32_t maxFrames) {
    activate();
    return Gen3_RunUntilDecision(maxFrames);
}

void Gen3Game::chooseMove(int slot) {
    activate();
    if (Gen3_Host()->pendingDecision != ACTION) throw std::runtime_error("not at an action decision");
    if (slot < 0 || slot > 3) throw std::out_of_range("move slot must be 0-3");
    if (!Gen3_MoveSlotValid(static_cast<uint8_t>(slot))) throw std::invalid_argument("no move in that slot");
    Gen3_ChooseMove(static_cast<uint8_t>(slot));
}

void Gen3Game::chooseSwitch(int partyIndex) {
    activate();
    int pending = Gen3_Host()->pendingDecision;
    if (pending != ACTION && pending != SWITCH) throw std::runtime_error("not at a decision");
    if (partyIndex < 0 || partyIndex > 2) throw std::out_of_range("party index must be 0-2");
    if (!Gen3_SwitchTargetValid(static_cast<uint8_t>(partyIndex)))
        throw std::invalid_argument("not a switch target (empty, fainted or already out)");
    if (pending == ACTION && !Gen3_CanSwitch(0)) throw std::runtime_error("cannot switch: trapped");
    Gen3_ChooseSwitch(static_cast<uint8_t>(partyIndex));
}

namespace {
uint8_t checkBattler(int battler) {
    if (battler < 0 || battler > 1) throw std::out_of_range("battler must be 0 (player) or 1 (opponent)");
    return static_cast<uint8_t>(battler);
}
uint8_t checkRental(int i) {
    if (i < 0 || i > 5) throw std::out_of_range("rental index must be 0-5");
    return static_cast<uint8_t>(i);
}
}  // namespace

uint8_t Gen3Game::unusableMoves(int battler) {
    uint8_t b = checkBattler(battler);
    activate();
    return Gen3_UnusableMoves(b);
}

void Gen3Game::forfeit() {
    activate();
    if (Gen3_Host()->pendingDecision != ACTION) throw std::runtime_error("not at an action decision");
    Gen3_Forfeit();
}

bool Gen3Game::canSwitch(int battler) {
    uint8_t b = checkBattler(battler);
    activate();
    return Gen3_CanSwitch(b) != 0;
}

uint16_t Gen3Game::choicedMove(int battler) {
    uint8_t b = checkBattler(battler);
    activate();
    return Gen3_ChoicedMove(b);
}

uint32_t Gen3Game::rng() {
    activate();
    return Gen3_GetRng();
}

std::string Gen3Game::readRam(uint32_t addr, size_t size) {
    activate();
    std::string out(size, '\0');
    if (!Gen3_ReadRam(addr, out.data(), size)) throw std::out_of_range("address not in host battle RAM");
    return out;
}

void Gen3Game::factoryBegin(bool openLevel, uint16_t winStreak, uint16_t rentsCount, uint32_t seed) {
    activate();
    Gen3Factory_Begin(openLevel ? 1 : 0, winStreak, rentsCount, seed);
}

int Gen3Game::factoryPhase() {
    activate();
    return Gen3Factory_Phase();
}

std::string Gen3Game::factoryRental(int i) {
    uint8_t k = checkRental(i);
    activate();
    std::string out(100, '\0');
    Gen3Factory_ReadRental(k, out.data());
    return out;
}

uint16_t Gen3Game::factoryRentalMonId(int i) {
    uint8_t k = checkRental(i);
    activate();
    return Gen3Factory_RentalFrontierMonId(k);
}

bool Gen3Game::factoryRent(int a, int b, int c) {
    uint8_t x = checkRental(a), y = checkRental(b), z = checkRental(c);
    activate();
    return Gen3Factory_Rent(x, y, z) != 0;
}

bool Gen3Game::factorySwap(int playerSlot, int enemySlot) {
    if (playerSlot > 2 || (playerSlot >= 0 && (enemySlot < 0 || enemySlot > 2)))
        throw std::out_of_range("swap: player slot must be 0-2 (or < 0 to keep), enemy slot 0-2");
    activate();
    return Gen3Factory_Swap(playerSlot < 0 ? -1 : playerSlot, enemySlot) != 0;
}

int Gen3Game::factoryRunBattle(uint32_t maxFrames) {
    activate();
    return Gen3Factory_RunBattle(maxFrames);
}

Gen3Game::FactoryInfo Gen3Game::factoryInfo() {
    activate();
    const Gen3FactoryState* s = Gen3Factory_State();
    return {s->phase, s->lvlMode, s->hintType, s->hintStyle, s->brainStatus, s->lastOutcome,
            s->trainerId, s->wins, s->swaps, s->challengesWon};
}

void Gen3Game::writeRam(uint32_t addr, const std::string& data) {
    activate();
    if (!Gen3_WriteRam(addr, data.data(), data.size())) throw std::out_of_range("address not in host battle RAM");
}

uint32_t Gen3Game::frames() {
    activate();
    return Gen3_Host()->frames;
}

void Gen3Game::traceRandom(bool enable) {
    Gen3_TraceRandom(enable ? 1 : 0);
}

std::pair<int, int> Gen3Game::simStep(int kind, int index) {
    activate();
    if (Gen3_BattleOutcome() != 0) throw std::runtime_error("battle is over");
    if (kind == 0) {
        chooseMove(index);
    } else if (kind == 1) {
        chooseSwitch(index);
    } else {
        throw std::invalid_argument("kind must be 0 (move) or 1 (switch)");
    }
    // Same frame budget as Gen3Factory_RunBattle (SimBackend), without its FinishBattle
    int decision = Gen3_RunUntilDecision(400000);
    return {decision, Gen3_BattleOutcome()};
}

void Gen3Game::setRng(uint32_t value) {
    activate();
    Gen3_SetRng(value);
}

int Gen3Game::battleOutcome() {
    activate();
    return Gen3_BattleOutcome();
}

bool Gen3Game::determinize(const std::vector<DetMon>& mons, const DetHidden* hidden, int64_t hiddenSeed) {
    std::vector<Gen3DetMon> c;
    c.reserve(mons.size());
    for (const DetMon& m : mons) {
        Gen3DetMon d{};
        d.partySlot = m.partySlot;
        d.species = m.species;
        for (int i = 0; i < 4; i++) d.moves[i] = m.moves[i];
        d.item = m.item;
        for (int i = 0; i < 6; i++) {
            d.ivs[i] = m.ivs[i];
            d.evs[i] = m.evs[i];
        }
        d.nature = m.nature;
        d.abilityBit = m.abilityBit;
        d.hpFraction = m.hpFraction;
        c.push_back(d);
    }
    Gen3DetHidden h{};
    if (hidden) {
        for (int s = 0; s < 2; s++) {
            for (int i = 0; i < 3; i++) h.sleepElapsed[s][i] = hidden->sleepElapsed[s][i];
            h.confusionElapsed[s] = hidden->confusionElapsed[s];
            h.wrapElapsed[s] = hidden->wrapElapsed[s];
            h.uproarElapsed[s] = hidden->uproarElapsed[s];
            h.rampageElapsed[s] = hidden->rampageElapsed[s];
        }
    }
    activate();
    int r = Gen3Search_Determinize(c.data(), static_cast<int>(c.size()), hidden ? &h : nullptr, hiddenSeed);
    if (r < 0) throw std::invalid_argument("bad determinization spec");
    if (r == 1) m_determinized = true;
    return r == 1;
}

std::string Gen3Game::stateBytes() {
    activate();
    Gen3_SaveState(m_state.data());
    return std::string(m_state.begin(), m_state.end());
}

long Gen3Game::stateOffset(uint32_t gbaAddress, bool deref) {
    activate();
    return Gen3_StateOffset(gbaAddress, deref ? 1 : 0);
}

void Gen3Game::redrawTurn(uint32_t seed) {
    activate();
    Gen3Search_RedrawTurn(seed);
}

}  // namespace pkmn
